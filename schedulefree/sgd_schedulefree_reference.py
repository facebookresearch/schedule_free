# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union, Optional, Iterable, Dict, Any
from typing_extensions import TypeAlias
import torch
import torch.optim
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

class SGDScheduleFreeReference(torch.optim.Optimizer):
    r"""
    Schedule-Free SGD
    As the name suggests, no scheduler is needed with this optimizer. 
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.

    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.

    This is the reference implementation, which is simpler but uses more memory.
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining 
            parameter groups.
        lr (float): 
            Learning rate parameter (default 1.0)
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
        foreach (bool): Use a foreach-backed implementation of the optimizer.
            Ignored for the reference implementation
    """
    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 1.0,
                 momentum: float = 0.9,
                 weight_decay: float = 0,
                 warmup_steps: int = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2,
                 foreach: Optional[bool] = hasattr(torch, "_foreach_mul_"),
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
                        train_mode=True,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        scheduled_lr=0.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        foreach=foreach)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'x' in state:
                        # Set p to x
                        p.copy_(state['x'])
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'y' in state:
                        # Set p to y
                        p.copy_(state['y'])
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.param_groups[0]['train_mode']:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")

        loss = None
        if closure is not None:
            with torch.enable_grad():
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
            group['scheduled_lr'] = lr # For logging purposes

            weight_lr_power = group['weight_lr_power']
            
            r = group['r']
            lr_max = group['lr_max'] = max(lr, group['lr_max'])
            
            weight = ((k+1)**r) * (lr_max**weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            try:
                ckp1 = weight/weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(p, memory_format=torch.preserve_format)
                    state['x'] = torch.clone(p, memory_format=torch.preserve_format)
                    state['y'] = torch.clone(p, memory_format=torch.preserve_format) 

                z = state['z']
                x = state['x']
                y = state['y']

                # Apply weight decay
                if weight_decay != 0:
                    grad.add_(y, alpha=weight_decay)

                # SGD step
                z.sub_(grad, alpha=lr)

                ### Take step
                x.mul_(1-ckp1).add_(z, alpha=ckp1)

                # Compute y
                y.copy_(x.mul(momentum).add_(z, alpha=1-momentum))
                p.copy_(y)

            group['k'] = k+1
        return loss
