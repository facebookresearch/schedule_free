# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.optim

class SGDScheduleFreeClosure(torch.optim.Optimizer):
    r"""
    Schedule-Free SGD
    As the name suggests, no scheduler is needed with this optimizer. 
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.

    This "closure" version requires that step be called with a closure, see 
    https://pytorch.org/docs/stable/optim.html#optimizer-step-closure. 
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining 
            parameter groups.
        lr (float): 
            Learning rate parameter (default 1e-3)
        momentum (float): momentum factor, must be between 0 and 1 exclusive
            (default: 0.9)
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        warmup_steps (int): Enables a linear learning rate warmup (default 0).
        r (float): Use polynomial weighting in the average 
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
    """
    def __init__(self, 
                 params, 
                 lr, 
                 weight_decay=0,
                 momentum=0.9, 
                 warmup_steps=0,
                 r=0.0,
                 weight_lr_power=2.0,
                 ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if momentum <= 0 or momentum >= 1:
            raise ValueError("Momentum must be between 0 and 1 exclusive: {}".format(momentum))

        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay,
            r=r,
            warmup_steps=warmup_steps,
            k=0,
            weight_sum=0.0,
            weight_lr_power=weight_lr_power,
            lr_max=0.0,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # Swap to extrapolated point:
        for group in self.param_groups:
            momentum = group['momentum']

            for p in group['params']:
                # State initialization
                state = self.state[p]
                if 'z' not in state:
                    state['z'] = torch.clone(p.data)

                z = state['z']

                # Extrapolate, replacing p with y temporarily
                p.data.lerp_(end=z, weight=1-momentum)

        # Evaluate gradient at extrapolated point
        loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            k = group['k']
            warmup_steps = group['warmup_steps']

            if k < warmup_steps:
              sched = (k+1) / warmup_steps
            else:
              sched = 1.0
            lr = group['lr']*sched

            weight_lr_power = group['weight_lr_power']
            
            r = group['r']
            lr_max = group['lr_max'] = max(lr, group['lr_max'])
            
            weight = ((k+1)**r) * (lr_max**weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            ckp1 = weight/weight_sum

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                y = p.data # Notation to match theory
                
                # Weight decay calculated at y.
                if weight_decay != 0:
                    grad.add_(y, alpha=weight_decay)

                state = self.state[p]
                z = state['z']
                
                # Unextrapolate, changing p back to x
                x = y.lerp_(end=z, weight=1-1/momentum)

                # Main step
                z.data.sub_(grad, alpha=lr)

                # x moving average update
                x.lerp_(end=z, weight=ckp1)

            group['k'] = k+1
        return loss
