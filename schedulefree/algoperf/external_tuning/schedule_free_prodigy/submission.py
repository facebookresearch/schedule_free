# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from math import sqrt
from typing import Dict, Iterator, List, Tuple

from absl import logging
import torch
import torch.distributed.nn as dist_nn
from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP = pytorch_setup()[0]

class ProdigyScheduleFree(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.9955),
                 r=0.75,
                 eps=1e-8, weight_decay=0, 
                 d0=1e-6, d_coef=1):

        defaults = dict(lr=lr, betas=betas,
                        eps=eps, weight_decay=weight_decay,
                        d=d0, d0=d0, d_max=d0,
                        d_numerator=0.0, d_coef=d_coef,
                        k=0,
                        r=r,
                        weight_sum=0.0)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        beta1, beta2 = group['betas']
        k = group['k']

        d = group['d']
        d_coef = group['d_coef']
        lr = max(group['lr'] for group in self.param_groups)

        r = group['r']
        weight_sum = group['weight_sum']

        dlr = d*lr
        sqrt_beta2 = sqrt(beta2)

        d_numerator = group['d_numerator']
        d_numerator *= sqrt_beta2
        d_numerator_acum = torch.tensor(0.0, device='cuda')
        d_denom = torch.tensor(0.0, device='cuda')

        # Swap to extrapolated point:
        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                # State initialization
                state = self.state[p]
                if 'z' not in state:
                    state['z'] = torch.clone(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    state['s'] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    state['p0'] = p.detach().to(torch.bfloat16, copy=True)

                # Extrapolate
                p.data.mul_(beta1).add_(state['z'], alpha=1-beta1)

        loss = closure()

        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
               
                grad = p.grad.data

                state = self.state[p]

                exp_avg_sq = state['exp_avg_sq']
                s = state['s']
                z = state['z']
                p0 = state['p0']

                # Unextrapolate:
                p.data.sub_(z, alpha=1-beta1).div_(beta1)

                x0_minus_z = (p0.data - z.data).flatten()

                d_numerator_acum.add_(torch.dot(grad.flatten(), x0_minus_z).mul(d*dlr))
                del x0_minus_z

                # Adam EMA update
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1-beta2))

                s.mul_(sqrt_beta2).add_(grad, alpha=d*dlr)
                d_denom += s.abs().sum().item()

            ######

        d_numerator += d_numerator_acum.item()
        d_denom_item = d_denom.item()
        if d_denom_item == 0:
            return loss

        d_hat = d_coef * d_numerator / d_denom_item
        d = max(d, d_hat)

        for group in self.param_groups:
            group['d_numerator'] = d_numerator
            group['d'] = d
            group['d_hat'] = d_hat

            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            weight = ((k+1)**r)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight
            ckp1 = weight/weight_sum

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                exp_avg_sq = state['exp_avg_sq']
                z = state['z']

                denom = exp_avg_sq.sqrt().add_(d * eps)

                # Apply weight decay (decoupled variant)
                z.data.sub_(p.data, alpha=decay * dlr)

                # Z Step
                z.data.addcdiv_(grad, denom, value=-d*dlr)

                ### p step
                p.data.mul_(1-ckp1).add_(z, alpha=ckp1)

                del denom

            group['k'] = k + 1

        return loss
        
def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del model_state
  del rng

  optimizer_state = {
      'optimizer':
          ProdigyScheduleFree(
              model_params.parameters(),
              weight_decay=hyperparameters.weight_decay),
              
  }

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
  del eval_results
  del hyperparameters

  #### Set LR
  warmup = int(0.075 * workload.step_hint)
  if global_step < warmup:
    sched_lr = (global_step+1) / warmup
  else:
    sched_lr = 1.0
  
  for param_group in optimizer_state['optimizer'].param_groups:
      lr = param_group['lr'] = sched_lr

  current_model = current_param_container
  current_model.train()
  optimizer_state['optimizer'].zero_grad()
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
        label_smoothing=0.2)
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

  # Log training metrics - loss, grad_norm, batch_size.
  if global_step <= 100 or global_step % 500 == 0:
    if workload.metrics_logger is not None:
      workload.metrics_logger.append_scalar_metrics(
          {
              'loss': loss.item(),
              'lr' : lr,
          }, global_step)
    logging.info('%d) loss = %0.3f',
                 global_step,
                 loss.item())

  return (optimizer_state, current_param_container, new_model_state)


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144 # Decreased from 262_144
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
    return 1024
  elif workload_name == 'imagenet_vit':
    return 1024
  elif workload_name == 'librispeech_conformer':
    return 224 # Fix
  elif workload_name == 'librispeech_deepspeech':
    return 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 128
  elif workload_name == 'mnist':
    return 16
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
