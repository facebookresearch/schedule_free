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
                 weight_decay: float = 0,
                 momentum: float = 0.9,
                 warmup_steps: int = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2.0,
                 foreach: Optional[bool] = hasattr(torch, "_foreach_mul_"),
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
            scheduled_lr=0.0,
            foreach=foreach
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float]) -> Optional[float]:
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
                if 'z' not in self.state[p]:
                    self.state[p]['z'] = torch.clone(p, memory_format=torch.preserve_format)

            if group['foreach']:
                if len(group['params']) > 0:
                    y, z = zip(*[(p, self.state[p]['z']) for p in group['params']])

                    torch._foreach_lerp_(y, z, weight=1-momentum)

            else:
                for p in group['params']:
                    z = self.state[p]['z']

                    # Extrapolate, replacing p with y temporarily
                    p.lerp_(end=z, weight=1-momentum)

        # Evaluate gradient at extrapolated point
        with torch.enable_grad():
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

            if group['foreach'] and len(active_p) > 0:
                y, grad, z = zip(*[(p, 
                                    p.grad, 
                                    self.state[p]['z']) 
                                    for p in active_p])

                # Weight decay calculated at y.
                if weight_decay != 0:
                    torch._foreach_add_(grad, y, alpha=weight_decay)

                # Unextrapolate, changing p back to x
                torch._foreach_lerp_(y, z, weight=1-1/momentum)

                # Main step
                torch._foreach_sub_(z, grad, alpha=lr)

                # x moving average update
                torch._foreach_lerp_(y, z, weight=ckp1)
            else:
                for p in active_p:
                    grad = p.grad
                    y = p # Notation to match theory
                    
                    # Weight decay calculated at y.
                    if weight_decay != 0:
                        grad.add_(y, alpha=weight_decay)

                    state = self.state[p]
                    z = state['z']
                    
                    # Unextrapolate, changing p back to x
                    x = y.lerp_(end=z, weight=1-1/momentum)

                    # Main step
                    z.sub_(grad, alpha=lr)

                    # x moving average update
                    x.lerp_(end=z, weight=ckp1)

            group['k'] = k+1
        return loss
