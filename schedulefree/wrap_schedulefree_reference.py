# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.optim
from typing import Optional, Callable

class ScheduleFreeWrapperReference:
    r""" 
        Simplified reference implementation of ScheduleFreeWrapper.
        This version is easy to modify and numerically stable, it's a good
        place to start when doing more Research-orientated experimentation
        combining Schedule-Free with other optimizers.

        base (torch.optim.Optimizer): 
            PyTorch optimizer object
        momentum (float): Apply momentum on the outer optimizer (default 0.9)
        weight_decay_at_y (float): 
            Weight decay calculated at the y point. Set weight decay on the 
            inner optimizer to instead calculate at z (default: 0).
        r (float): Use polynomial weighting in the average 
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
    """
    def __init__(self, 
                 base : torch.optim.Optimizer, 
                 momentum : float = 0.9,
                 weight_decay_at_y : float = 0.0,
                 r : float = 0,
                 weight_lr_power : float = 2):

        self.base = base

        self.momentum = momentum
        self.weight_lr_power = weight_lr_power
        self.r = r
        self.train_mode = False

        for group in self.param_groups:
            group['weight_decay_at_y'] = weight_decay_at_y

    def add_param_group(self, param_group):
        return self.base.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        return self.base.load_state_dict(state_dict)

    def state_dict(self):
        return self.base.state_dict()
        
    def zero_grad(self, set_to_none=True):
        return self.base.zero_grad(set_to_none)

    @property
    def param_groups(self):
        return self.base.param_groups

    @property
    def state(self):
        return self.base.state

    @torch.no_grad()
    def eval(self):
        if self.train_mode:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if 'x' in state:
                        # Switch p to x
                        p.copy_(state['x'])
        self.train_mode = False

    @torch.no_grad()
    def train(self):
        if not self.train_mode:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if 'y' in state:
                        p.copy_(state['y'])

        self.train_mode = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.train_mode:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay_at_y = group['weight_decay_at_y']
            for p in group['params']:
                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(p, memory_format=torch.preserve_format)
                    state['x'] = torch.clone(p, memory_format=torch.preserve_format)
                    state['y'] = torch.clone(p, memory_format=torch.preserve_format) 

                y = state['y']
                z = state['z']

                # Apply weight decay to z, computed at point y
                if weight_decay_at_y != 0.0:
                    z.sub_(y, alpha=lr*weight_decay_at_y)

                # Copy z into param buffer
                p.copy_(z)

        #######

        # Run base optimizer, acting on params p, which are a copy of z
        self.base.step()

        ######
        for group in self.param_groups:
            weight_lr_power = self.weight_lr_power
            r = self.r
            k = group.get('k', 0)
            d = group.get('d', 1.0)
            lr = group['lr']*d
            lr_max = group['lr_max'] = max(lr, group.get('lr_max', 0))
            
            weight = ((k+1)**r) * (lr_max**weight_lr_power)
            weight_sum = group['weight_sum'] = group.get('weight_sum', 0.0) + weight

            ckp1 = weight/weight_sum

            for p in group['params']:
                state = self.state[p]
                
                y = state['y']
                z = state['z']
                x = state['x']

                # Update z to new value after base optimizer step
                z.copy_(p)

                # Update x
                x.lerp_(end=z, weight=ckp1)

                # Compute y
                y.copy_(x).lerp_(end=z, weight=1-self.momentum)

                # Copy y into param buffer
                p.copy_(y)

            group['k'] = k+1

        return loss