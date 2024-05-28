# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.optim
import math
import torch.distributed as dist

class AdamWScheduleFreeReference(torch.optim.Optimizer):
    r"""
    Schedule-Free AdamW (Simplified Reference Implementation)
    
    This version of the optimizer uses more memory but has a much simpler
    implementation that closely follows the theoretical presentation.
    This is useful as a basis for adapting or modifying or testing the method.
    
    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining 
            parameter groups.
        lr (float): 
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): 
            Term added to the denominator outside of the root operation to 
            improve numerical stability. (default: 1e-8).
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        warmup_steps (int): Enables a linear learning rate warmup (default 0).
        r (float): Use polynomial weighting in the average 
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
        decay_at_z (bool): Apply weight decay calculated at the z sequence
            instead of the y sequence (default False).
    """
    def __init__(self, params, 
                 lr=0.0025, 
                 betas=(0.9, 0.999), 
                 eps=1e-8,
                 weight_decay=0,
                 warmup_steps=0,
                 r=0,
                 weight_lr_power=2,
                 decay_at_z=False,
                 foreach=False, # Ignored
                 ):

        defaults = dict(lr=lr, 
                        betas=betas, 
                        eps=eps,
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        train_mode = True,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        weight_lr_power=weight_lr_power,
                        decay_at_z=decay_at_z,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    def eval(self):
        first = True

        for group in self.param_groups:
            train_mode = group['train_mode']

            if train_mode:
                if first:
                    print(f"Switching to eval mode")
                for p in group['params']:
                    state = self.state[p]

                    if 'x' in state:
                        # Switch p.data to x
                        p.data.copy_(state['x'])
                group['train_mode'] = False
                first = False

    def train(self):
        first = True

        for group in self.param_groups:
            train_mode = group['train_mode']
            if not train_mode:
                if first:
                    print(f"Switching to train mode")
                for p in group['params']:
                    state = self.state[p]
                    if 'y' in state:
                        p.data.copy_(state['y'])
                group['train_mode'] = True
                first = False

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            eps = group['eps']
            lr = max(group['lr'], eps)
            decay = group['weight_decay']
            beta1, beta2 = group['betas']
            k = group['k']
            r = group['r']
            warmup_steps = group['warmup_steps']
            weight_lr_power = group['weight_lr_power']
            decay_at_z = group['decay_at_z']
            
            if k < warmup_steps:
              sched = (k+1) / warmup_steps
            else:
              sched = 1.0
            
            bias_correction2 = 1 - beta2 ** (k+1)
            lr = group['lr']*sched*math.sqrt(bias_correction2)
            
            lr_max = group['lr_max'] = max(lr, group['lr_max'])
            
            weight = ((k+1)**r) * (lr_max**weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            try:
                ckp1 = weight/weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            if not group['train_mode']:
                raise Exception("Not in train mode!")

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['x'] = torch.clone(p.data)
                    state['y'] = torch.clone(p.data) 

                exp_avg_sq = state['exp_avg_sq']
                z = state['z']
                x = state['x']
                y = state['y']

                # Weight decay
                if decay != 0:
                    decay_point = z if decay_at_z else y
                    z.data.sub_(decay_point, alpha=lr*decay)

                # Decay the first and second moment running average coefficient
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                denom = exp_avg_sq.sqrt().add_(eps)

                z.data.addcdiv_(grad, denom, value=-lr)

                ### Take step
                x.mul_(1-ckp1).add_(z, alpha=ckp1)

                # Compute y
                y.copy_(x.mul(beta1).add_(z, alpha=1-beta1))
                p.data.copy_(y)

            group['k'] = k+1
        return loss
