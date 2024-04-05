# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Dict, Iterator, List, Tuple
from absl import logging
import torch
import torch.distributed.nn as dist_nn

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP = pytorch_setup()[0]
HPARAMS = {
    "dropout_rate": 0.1,
    "learning_rate": 0.0025,
    "one_minus_beta1": 0.1,
    "beta2": 0.9955159689799007,
    "weight_decay": 0.08121616522670176,
    "warmup_factor": 0.02,
    "weight_lr_power": 2,
    "label_smoothing": 0.2,
    "r": 0.75,
    "conformer_bs": 192,
}


class AdamWScheduleFree(torch.optim.Optimizer):
    r"""Schedule Free AdamW
    """
    def __init__(self, params, 
                 lr=1e-3, 
                 betas=(0.9, 0.999), 
                 eps=1e-8,
                 weight_decay=0,
                 weight_lr_power=2,
                 warmup_steps=0,
                 r=0,
                 ):
        defaults = dict(lr=lr, 
                        betas=betas, 
                        eps=eps,
                        r=r,
                        k=0,
                        weight_sum=0.0,
                        lr_max=0.0,
                        warmup_steps=warmup_steps,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay)

        super().__init__(params, defaults)

    def reset(self):
        for group in self.param_groups:
            group['k'] = 0
            group['lr_max'] = 0 
            group['weight_sum'] = 0

            for p in group['params']:
                # State initialization
                state = self.state[p]
                state['z'].copy_(state['x0'])
                p.data.copy_(state['x0'])
                state['exp_avg_sq'].zero_()

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # Swap to extrapolated point:
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            r = group['r']
            k = group['k']

            for p in group['params']:
                # State initialization
                state = self.state[p]
                if 'z' not in state:
                    state['z'] = torch.clone(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    state['x0'] = p.data.cpu()                    

                z = state['z']

                # Extrapolate
                #p = p + (1-beta1)*(z-p)
                #p.data.mul_(beta1).add_(z, alpha=1-beta1)
                p.data.lerp_(end=z, weight=1-beta1)

        # Evaluate gradient at extrapolated point
        loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            k = group['k']
            warmup_steps = group['warmup_steps']
            
            if k < warmup_steps:
              sched = (k+1) / warmup_steps
            else:
              sched = 1.0
            annealed_lr = group['lr']*sched

            lr = max(annealed_lr, eps)

            decay = group['weight_decay']
            beta1, beta2 = group['betas']
            weight_lr_power = group['weight_lr_power']
            
            r = group['r']
            lr_max = group['lr_max'] = max(lr, group['lr_max'])
            
            weight = ((k+1)**r) * (lr_max**weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            ckp1 = weight/weight_sum

            bias_correction2 = 1 - beta2 ** (k+1)
            step_size = lr * math.sqrt(bias_correction2)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                state = self.state[p]

                exp_avg_sq = state['exp_avg_sq']
                z = state['z']

                # Unextrapolate
                #p = (p - (1-beta1)*z)/beta1
                #p.data.sub_(z, alpha=1-beta1).div_(beta1)
                p.data.lerp_(end=z, weight=1-1/beta1)

                # Decay the first and second moment running average coefficient
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                denom = exp_avg_sq.sqrt().add_(eps)

                z.addcdiv_(grad, denom, value=-step_size)

                # Decay
                z.sub_(p.data, alpha=step_size*decay)

                ### Take step
                #p.data.mul_(1-ckp1).add_(z, alpha=ckp1)
                p.data.lerp_(end=z, weight=ckp1)

            group['k'] = k+1
        return loss

def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del model_state

  optimizer = AdamWScheduleFree(
    model_params.parameters(),
    lr=HPARAMS['learning_rate'],
    betas=(1.0 - HPARAMS['one_minus_beta1'], HPARAMS['beta2']),
    warmup_steps=int(HPARAMS['warmup_factor'] * workload.step_hint * 0.75),
    weight_decay=HPARAMS['weight_decay'],
    weight_lr_power=HPARAMS['weight_lr_power'],
    r=HPARAMS['r'])

  optimizer_state = {'optimizer':optimizer, 'max_checked_eval_step': -1, 'has_forced_reset': False, 'first_eval': False, }
  return optimizer_state

def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del hyperparameters


  metric_name = workload.target_metric_name
  # TODO - remove force_reset
  eval_step = len(eval_results)
  if (global_step > workload.step_hint*0.10) and optimizer_state['max_checked_eval_step'] < eval_step:
    optimizer_state['max_checked_eval_step'] = eval_step
    # Don't do resetting on workloads that don't run eval often enough
    if len(eval_results) >= 4: # and optimizer_state["first_eval_is_far_from_target"]:
      val_metric = f"validation/{metric_name}"
      initial_eval = eval_results[0][1][val_metric]
      latest_eval = eval_results[-1][1][val_metric]
      second_latest_eval = eval_results[-2][1][val_metric]
      third_latest_eval = eval_results[-3][1][val_metric]
      fourth_latest_eval = eval_results[-4][1][val_metric]
      MARGIN = 0.01
      if metric_name in ["loss", "wer"]:
        # Decreasing eval workloads should be flipped
        initial_eval = -initial_eval
        latest_eval = -latest_eval
        second_latest_eval = -second_latest_eval
        third_latest_eval = -third_latest_eval
        fourth_latest_eval = -fourth_latest_eval
      # Higher is better
      # scale as a curve from 0 --> 1
      # if the eval values are far from the target (i.e. - worse than initial) and stays far from the target for 4 evals
      if (latest_eval - initial_eval < MARGIN) and (latest_eval - second_latest_eval < MARGIN) and (second_latest_eval - third_latest_eval < MARGIN) and (third_latest_eval - fourth_latest_eval < MARGIN):
        # Reset parameters since we appear to have diverged
        logging.info("Reseting All Weights ")
        logging.info(f"Global Step: {global_step}")
        optimizer_state['has_forced_reset'] = True

        # Perform reset
        del model_state
        model_state = None
        optimizer_state['optimizer'].reset()

        # Decrease learning rate by 2x if it diverged.
        for param_group in optimizer_state['optimizer'].param_groups:
          param_group['lr'] = param_group['lr']/2.0

  ###########

  current_model = current_param_container
  current_model.train()

  new_model_state = None

  def closure():
    nonlocal new_model_state
    optimizer_state['optimizer'].zero_grad()

    logits_batch, new_model_state = workload.model_fn(
        params=current_model,
        augmented_and_preprocessed_input_batch=batch,
        model_state=model_state,
        mode=spec.ForwardPassMode.TRAIN,
        rng=rng,
        update_batch_norm=True)

    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits_batch,
        mask_batch=batch.get('weights'),
        label_smoothing=HPARAMS['label_smoothing'])
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    if USE_PYTORCH_DDP:
      # Use dist_nn.all_reduce to ensure correct loss and gradient scaling.
      summed_loss = dist_nn.all_reduce(summed_loss)
      n_valid_examples = dist_nn.all_reduce(n_valid_examples)
    loss = summed_loss / n_valid_examples

    loss.backward()
    return loss

  loss = optimizer_state['optimizer'].step(closure)

  return (optimizer_state, current_param_container, new_model_state)


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144
  elif workload_name == 'fastmri':
    return 16 # 32
  elif workload_name == 'imagenet_resnet':
    return 1024
  elif workload_name == 'imagenet_vit':
    return 1024
  elif workload_name == 'librispeech_conformer':
    return 224
  elif workload_name == 'librispeech_deepspeech':
    return 128 # 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 128
  elif workload_name == 'mnist':
    return 16
  elif workload_name == 'imagenet_resnet_gelu':
    return 512
  elif workload_name == 'imagenet_resnet_silu':
    return 512
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')

def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   model_state: spec.ModelAuxiliaryState,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.
  Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  batch = next(input_queue)
  return batch
