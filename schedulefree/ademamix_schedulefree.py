# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any
from typing_extensions import TypeAlias
import torch
import torch.optim
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

class AdEMAMixScheduleFree(torch.optim.Optimizer):
    r"""
    Schedule-Free AdEMAMix.

    As the name suggests, no scheduler is needed with this optimizer.
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.

    The outer loop is schedule-free averaging (mirroring AdoptScheduleFree):
    iterates y with z buffers and the ckp1 weighting. The inner update is
    AdEMAMix (https://arxiv.org/abs/2310.00064): a fast momentum, a slow
    momentum with its own schedule, and a second moment, combined as
    (m + alpha_t * s) / denom. betas[0] serves both as the schedule-free
    interpolation coefficient and the fast momentum factor.

    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float, float], optional): fast momentum, second
            moment decay, and slow momentum base (default: (0.9, 0.999,
            0.9999)).
        eps (float):
            Term added to the denominator (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        alpha (float): slow-momentum mixing coefficient (default: 5.0).
        T_alpha_beta3 (int, optional): warmup horizon for the alpha/beta3
            schedule. With None (default) alpha and beta3 stay constant.
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
                 lr: Union[float, torch.Tensor] = 0.0025,
                 betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 alpha: float = 5.0,
                 T_alpha_beta3: Optional[int] = None,
                 warmup_steps: int = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2.0,
                 foreach: Optional[bool] = hasattr(torch, "_foreach_mul_")
                 ):

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        train_mode=False,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        scheduled_lr=0.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        alpha=alpha,
                        T_alpha_beta3=T_alpha_beta3,
                        foreach=foreach)
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _, _ = group['betas']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to x
                        p.lerp_(end=state['z'].to(p.device), weight=1-1/beta1)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _, _ = group['betas']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to y
                        p.lerp_(end=state['z'].to(p.device), weight=1-beta1)
                group['train_mode'] = True

    @torch.no_grad()
    def _alpha_beta3(self, step, alpha, beta1, beta3, T_alpha_beta3):
        if T_alpha_beta3 is not None:
            alpha_t = min(step * alpha / T_alpha_beta3, alpha)
            beta3_t = min(math.exp(math.log(beta1) * math.log(beta3) /
                          ((1 - step / T_alpha_beta3) * math.log(beta3) +
                           (step / T_alpha_beta3) * math.log(beta1))), beta3)
        else:
            alpha_t = alpha
            beta3_t = beta3
        return alpha_t, beta3_t

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
            eps = group['eps']
            beta1, beta2, beta3 = group['betas']
            decay = group['weight_decay']
            alpha = group['alpha']
            T_alpha_beta3 = group['T_alpha_beta3']
            k = group['k']
            r = group['r']
            warmup_steps = group['warmup_steps']
            weight_lr_power = group['weight_lr_power']

            if k < warmup_steps:
              sched = (k+1) / warmup_steps
            else:
              sched = 1.0

            lr = group['lr']*sched
            group['scheduled_lr'] = lr # For logging purposes

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
                    self.state[p]['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    self.state[p]['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    self.state[p]['exp_avg_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    self.state[p]['step'] = 0

            if group['foreach'] and len(active_p) > 0:
                y, grad, exp_avg, exp_avg_sq, exp_avg_slow, z = zip(*[(p,
                                                p.grad,
                                                self.state[p]['exp_avg'],
                                                self.state[p]['exp_avg_sq'],
                                                self.state[p]['exp_avg_slow'],
                                                self.state[p]['z'])
                                                for p in active_p])
                steps = [self.state[p]['step'] + 1 for p in active_p]
                for p, s in zip(active_p, steps):
                    self.state[p]['step'] = s

                # Fast and slow momenta share scalar coefficients across the
                # group only through beta1/beta2; alpha/beta3 schedules are
                # per-parameter (they depend on each step count).
                torch._foreach_mul_(list(exp_avg), beta1)
                torch._foreach_add_(list(exp_avg), list(grad), alpha=1-beta1)
                torch._foreach_mul_(list(exp_avg_sq), beta2)
                torch._foreach_addcmul_(list(exp_avg_sq), list(grad), list(grad), value=1-beta2)
                upds = []
                for i, p in enumerate(active_p):
                    alpha_t, beta3_t = self._alpha_beta3(
                        steps[i], alpha, beta1, beta3, T_alpha_beta3)
                    s_buf = exp_avg_slow[i]
                    s_buf.mul_(beta3_t).add_(grad[i], alpha=1-beta3_t)
                    bc1 = 1 - beta1**steps[i]
                    bc2 = 1 - beta2**steps[i]
                    denom = exp_avg_sq[i].sqrt().div_(bc2**0.5).add_(eps)
                    mixed = exp_avg[i].add(s_buf, alpha=alpha_t)
                    upds.append(mixed.div(denom).mul_(lr / bc1))

                if decay != 0:
                    torch._foreach_mul_(list(y), 1 - lr*decay)

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(list(y), list(z), weight=ckp1)
                torch._foreach_add_(list(y), upds, alpha=(beta1*(1-ckp1)-1))

                # z step
                torch._foreach_sub_(list(z), upds, alpha=1.0)
            else:
                for p in active_p:
                    y = p # Notation to match theory
                    grad = p.grad

                    state = self.state[p]

                    z = state['z']
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_slow = state['exp_avg_slow']
                    state['step'] += 1
                    step = state['step']

                    exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                    alpha_t, beta3_t = self._alpha_beta3(step, alpha, beta1, beta3, T_alpha_beta3)
                    exp_avg_slow.mul_(beta3_t).add_(grad, alpha=1-beta3_t)
                    bc1 = 1 - beta1**step
                    bc2 = 1 - beta2**step
                    denom = (exp_avg_sq.sqrt() / bc2**0.5).add_(eps)
                    upd = (exp_avg + alpha_t * exp_avg_slow).div(denom).mul_(lr / bc1)
                    if decay != 0:
                        y.mul_(1 - lr*decay)

                    # These operations update y in-place,
                    # without computing x explicitly.
                    y.lerp_(end=z, weight=ckp1)
                    y.add_(upd, alpha=(beta1*(1-ckp1)-1))

                    # z step
                    z.sub_(upd)

            group['k'] = k+1
        return loss
