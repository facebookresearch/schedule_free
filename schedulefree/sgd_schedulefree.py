# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.optim

class SGDScheduleFree(torch.optim.Optimizer):
    r"""
    Schedule-Free SGD
    As the name suggests, no scheduler is needed with this optimizer. 
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.

    This optimizer requires that .train() and .val() be called before the
    beginning of training and evaluation respectively.
    
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
                 momentum=0.9, 
                 weight_decay=0,
                 warmup_steps=0,
                 r=0.0,
                 weight_lr_power=2,
                 ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if momentum <= 0 or momentum >= 1:
            raise ValueError("Momentum must be between 0 and 1 exclusive: {}".format(momentum))

        defaults = dict(lr=lr, 
                        momentum=momentum, 
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        train_mode = True,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            momentum = group['momentum']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to x
                        p.data.lerp_(end=state['z'], weight=1-1/momentum)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            momentum = group['momentum']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to y
                        p.data.lerp_(end=state['z'], weight=1-momentum)
                group['train_mode'] = True

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
            momentum = group['momentum']
            lr = group['lr']
            weight_decay = group['weight_decay']
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

            if not group['train_mode']:
                raise Exception("Not in train mode!")

            for p in group['params']:
                if p.grad is None:
                    continue

                y = p.data # Notation to match theory
                grad = p.grad.data

                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(y)

                z = state['z']

                # Apply weight decay
                if weight_decay != 0:
                    grad.add_(y, alpha=weight_decay)

                # These operations update y in-place,
                # without computing x explicitly.
                y.lerp_(end=z, weight=ckp1)
                y.add_(grad, alpha=lr*(momentum*(1-ckp1)-1))

                # SGD step
                z.sub_(grad, alpha=lr)

            group['k'] = k+1
        return loss
