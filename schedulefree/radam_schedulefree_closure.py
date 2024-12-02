# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any
from typing_extensions import TypeAlias
import torch.optim
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
import math

class RAdamScheduleFreeClosure(torch.optim.Optimizer):
    r"""
    Schedule-Free RAdam
    Neither warmup hyperparameter nor scheduler is needed with this optimizer.

    This "closure" version requires that step be called with a closure, see 
    https://pytorch.org/docs/stable/optim.html#optimizer-step-closure. 
    
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
        r (float): Use polynomial weighting in the average
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
        foreach (bool): Use a foreach-backed implementation of the optimizer.
            Should be significantly faster, but will have higher peak memory
            usage (default True if supported in your PyTorch version).
        silent_sgd_phase (bool): If True, the optimizer will not use the first SGD phase of RAdam.
            This means that the optimizer will not update model parameters during the early training
            steps (e.g., < 5 when β_2 = 0.999), but just update the momentum values of the optimizer.
            This helps stabilize training by ensuring smoother warmup behavior and more reliable
            calculation of the moving average coefficient (`ckp1`). Recommended to set to True
            (default True).
    """
    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 0.0025,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 r: float = 0,
                 weight_lr_power: float = 2.0,
                 foreach: Optional[bool] = hasattr(torch, "_foreach_mul_"),
                 silent_sgd_phase: bool = True
                 ):
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        r=r,
                        k=0,
                        weight_sum=0.0,
                        lr_max=0.0,
                        scheduled_lr=0.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        foreach=foreach,
                        silent_sgd_phase=silent_sgd_phase)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float]) -> Optional[float]:
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
                    state['z'] = torch.clone(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            if group['foreach']:
                if len(group['params']) > 0:
                    y, z = zip(*[(p, self.state[p]['z']) for p in group['params']])

                    torch._foreach_lerp_(y, z, weight=1-beta1)
            else:
                for p in group['params']:
                    z = self.state[p]['z']

                    # Extrapolate p to equal y
                    p.lerp_(end=z, weight=1-beta1)

        # Evaluate gradient at extrapolated point
        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            k = group['k']
            step = k + 1
            r = group['r']
            silent_sgd_phase = group["silent_sgd_phase"]
            decay = group['weight_decay']
            beta1, beta2 = group['betas']
            weight_lr_power = group['weight_lr_power']

            beta2_t = beta2**step
            bias_correction2 = 1 - beta2_t

            # maximum length of the approximated SMA
            rho_inf = 2 / (1 - beta2) - 1
            # compute the length of the approximated SMA
            rho_t = rho_inf - 2 * step * beta2_t / bias_correction2
            rect = (
                ((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
                if rho_t > 4.0
                else float(not silent_sgd_phase)
            )

            lr = group["lr"] * rect
            group['scheduled_lr'] = lr # For logging purposes
            
            lr_max = group['lr_max'] = max(lr, group['lr_max'])
            
            weight = (step**r) * (lr_max**weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            try:
                ckp1 = weight/weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            adaptive_y_lr = lr * (beta1 * (1 - ckp1) - 1)
            active_p = [p for p in group['params'] if p.grad is not None]

            if group['foreach'] and len(active_p) > 0:
                y, grad, exp_avg_sq, z = zip(*[(p, 
                                                p.grad, 
                                                self.state[p]['exp_avg_sq'], 
                                                self.state[p]['z']) 
                                                for p in active_p])

                # Decay the first and second moment running average coefficient
                torch._foreach_mul_(exp_avg_sq, beta2)
                torch._foreach_addcmul_(exp_avg_sq, grad, grad, value=1-beta2)

                if rho_t > 4.0:
                    # Adam step
                    denom = torch._foreach_div(exp_avg_sq, bias_correction2)
                    torch._foreach_sqrt_(denom)
                    torch._foreach_add_(denom, eps)

                    # normalize grad in-place
                    torch._foreach_div_(grad, denom)

                # Weight decay calculated at y
                if decay != 0:
                    torch._foreach_add_(grad, y, alpha=decay)

                # Unextrapolate
                torch._foreach_lerp_(y, z, weight=1-1/beta1)

                torch._foreach_sub_(z, grad, alpha=lr)

                ### Take step
                torch._foreach_lerp_(y, z, weight=ckp1)
            else:
                for p in active_p:
                    grad = p.grad
                    y = p # Notation to match theory
                    
                    state = self.state[p]

                    exp_avg_sq = state['exp_avg_sq']
                    z = state['z']

                    # Decay the first and second moment running average coefficient
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                    if rho_t > 4.0:
                        # Adam step
                        denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)
        
                        grad_normalized = grad.div_(denom)
                    else:
                        # Fall back to SGD (or nothing)
                        grad_normalized = grad

                    # Weight decay calculated at y
                    if decay != 0:
                        grad_normalized.add_(y, alpha=decay)

                    # Unextrapolate
                    x = y.lerp_(end=z, weight=1-1/beta1)

                    z.sub_(grad_normalized, alpha=lr)

                    ### Take step
                    x.lerp_(end=z, weight=ckp1)

            group['k'] = k+1
        return loss
