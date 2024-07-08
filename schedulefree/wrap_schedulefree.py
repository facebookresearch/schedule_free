# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.optim

class ScheduleFreeWrapper:
    r"""
        Wrap any optimizer to make it Schedule-Free. 
        
        This version uses a memory-efficient swap operation but may be slightly
        slower than other versions, and less numerically stable. In most cases
        the performance difference is negligible.
        For the best possible performance and memory-usage, Schedule-Free needs 
        to be directly integrated with the base optimizer.

        When using this version, you can disable the base optimizer's 
        momentum, as it's no longer necessary when using our wrapper's 
        momentum (although you can use both types of momentum if you want).


        Example usage:
        ```
        base_optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0025)
        optimizer = ScheduleFreeWrapper(
            base_optimizer, momentum=0.9, weight_decay_at_y=0.1)
        ```

        If you set weight decay on the base optimizer, it computes weight decay
        at $z$. We offer the option to compute weight decay at $y$, via the 
        `weight_decay_at_y` parameter, which seems to give better results in 
        our experiments. This approach to decay only works correctly if the base
        optimizer uses group["lr"] as the current learning rate. 

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
                 weight_decay_at_y : float = 0.0,
                 momentum : float = 0.9,
                 weight_lr_power : float = 2,
                 r : float = 0):

        self.base = base
        self.weight_decay_at_y = weight_decay_at_y
        self.weight_lr_power = weight_lr_power
        self.r = r
        self.momentum = momentum
        self.train_mode = False


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
                    if 'z' in state:
                        # Set p to x
                        p.lerp_(end=state['z'], weight=1-1/self.momentum)
        self.train_mode = False

    @torch.no_grad()
    def train(self):
        if not self.train_mode:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to y
                        p.lerp_(end=state['z'], weight=1-self.momentum)
        self.train_mode = True

    @staticmethod
    def swap(x, y):
        # Memory efficient but potentially unstable
        x.add_(y)
        torch.sub(x, y, out=y)
        x.sub_(y)

    @torch.no_grad()
    def step(self, closure=None):
        if not self.train_mode:
            raise Exception("Please insert .train() and .eval() calls for the "
                            "optimizer. See documentation for details")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(p)

                z = state['z']

                # Apply weight_decay_at_y
                if self.weight_decay_at_y != 0.0:
                    z.sub_(p, alpha=lr*self.weight_decay_at_y)    
                    p.sub_(p, alpha=lr*self.weight_decay_at_y*(1-self.momentum))

                # Unextrapolate p converting from y -> x
                p.lerp_(end=z, weight=1-1/self.momentum)

                # Swap x into z buffer temporarily
                self.swap(z, p)

                # Now state['z'] is x and p is z.

        #######
        # Apply step to z
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
                if p.grad is None:
                    continue
                
                state = self.state[p]
                z = state['z']

                # Swap x back out of z buffer, leaving p as x
                self.swap(z, p)

                # Update x
                p.lerp_(end=z, weight=ckp1)

                # Now set p to y
                p.lerp_(end=state['z'], weight=1-self.momentum)

            group['k'] = k+1

        return loss