# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union, Optional, Iterable, Dict, Callable, Any
from typing_extensions import TypeAlias
import torch
import torch.optim
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

class SGDScheduleFree(torch.optim.Optimizer):
    r"""
    Schedule-Free SGD
    As the name suggests, no scheduler is needed with this optimizer. 
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.

    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.
    
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
            Should be significantly faster, but will have higher peak memory
            usage (default True if supported in your PyTorch version).
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
                        train_mode=False,
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
            momentum = group['momentum']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to x
                        p.lerp_(end=state['z'].to(p.device), weight=1-1/momentum)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            momentum = group['momentum']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to y
                        p.lerp_(end=state['z'].to(p.device), weight=1-momentum)
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
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

            active_p = [p for p in group['params'] if p.grad is not None]

            for p in active_p:
                if 'z' not in self.state[p]:
                    self.state[p]['z'] = torch.clone(p, memory_format=torch.preserve_format)

            if group['foreach'] and len(active_p) > 0:
                y, grad, z = zip(*[(p, p.grad, self.state[p]['z']) 
                                for p in active_p])

                # Apply weight decay
                if weight_decay != 0:
                    torch._foreach_add_(grad, y, alpha=weight_decay)

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(y, z, weight=ckp1)
                torch._foreach_add_(y, grad, alpha=lr*(momentum*(1-ckp1)-1))

                # SGD step
                torch._foreach_sub_(z, grad, alpha=lr)
            else:
                for p in active_p:
                    y = p # Notation to match theory
                    grad = p.grad
                    z = self.state[p]['z']

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
