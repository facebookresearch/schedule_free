# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any
from typing_extensions import TypeAlias
import torch
import torch.optim
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
import math


def _record_param_update(acc, y, z, grad_normalized, ckp1, alpha):
    """Square-sum of y's step, from tensors that already exist.

    y_new = (1 - ckp1) * y + ckp1 * z + alpha * grad_normalized, so
    y_new - y = ckp1 * (z - y) + alpha * grad_normalized. This does not
    clone the parameter.
    """
    delta = torch.lerp(y, z, ckp1).add(grad_normalized, alpha=alpha).sub(y)
    slot = acc.get(delta.device)
    if slot is None:
        slot = torch.zeros((), dtype=torch.float64, device=delta.device)
        acc[delta.device] = slot
    slot.add_(delta.detach().square().sum())


class AdamWScheduleFree(torch.optim.Optimizer):
    r"""
    Schedule-Free AdamW
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
        inner_momentum (float): Momentum applied to the gradient inside the
            AdamW update (i.e. an exponential moving average of the gradient,
            equivalent to AdamW's first-moment beta1). When set to 0 (default)
            no inner momentum is used and no memory is allocated for the
            running average buffer. A recommended value when enabling this
            feature is 0.9.
        foreach (bool): Use a foreach-backed implementation of the optimizer.
            Should be significantly faster, but will have higher peak memory
            usage (default True if supported in your PyTorch version).
    """
    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 0.0025,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 warmup_steps: int = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2.0,
                 inner_momentum: float = 0.0,
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
                        inner_momentum=inner_momentum,
                        foreach=foreach)
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
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
            beta1, _ = group['betas']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p to y
                        p.lerp_(end=state['z'].to(p.device), weight=1-beta1)
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

        self._update_sq_acc = {}
        self._update_numel = 0

        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
            decay = group['weight_decay']
            k = group['k']
            r = group['r']
            warmup_steps = group['warmup_steps']
            weight_lr_power = group['weight_lr_power']
            inner_momentum = group['inner_momentum']

            if k < warmup_steps:
              sched = (k+1) / warmup_steps
            else:
              sched = 1.0

            bias_correction2 = 1 - beta2 ** (k+1)
            if inner_momentum != 0:
                bias_correction1 = 1 - inner_momentum ** (k+1)
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
                    self.state[p]['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if inner_momentum != 0:
                        self.state[p]['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            if group['foreach'] and len(active_p) > 0:
                y, grad, exp_avg_sq, z = zip(*[(p,
                                                p.grad,
                                                self.state[p]['exp_avg_sq'],
                                                self.state[p]['z'])
                                                for p in active_p])

                # Decay the first and second moment running average coefficient
                torch._foreach_mul_(exp_avg_sq, beta2)
                torch._foreach_addcmul_(exp_avg_sq, grad, grad, value=1-beta2)
                denom = torch._foreach_div(exp_avg_sq, bias_correction2)
                torch._foreach_sqrt_(denom)
                torch._foreach_add_(denom, eps)

                if inner_momentum != 0:
                    exp_avg = tuple(self.state[p]['exp_avg'] for p in active_p)
                    # exp_avg = inner_momentum*exp_avg + (1-inner_momentum)*grad
                    torch._foreach_mul_(exp_avg, inner_momentum)
                    torch._foreach_add_(exp_avg, grad, alpha=1-inner_momentum)
                    # grad_normalized = (exp_avg / bias_correction1) / denom
                    grad_normalized = torch._foreach_div(exp_avg, bias_correction1)
                    torch._foreach_div_(grad_normalized, denom)
                else:
                    # Normalize grad in-place for memory efficiency
                    torch._foreach_div_(grad, denom)
                    grad_normalized = grad

                # Weight decay calculated at y
                if decay != 0:
                    torch._foreach_add_(grad_normalized, y, alpha=decay)

                alpha = lr * (beta1 * (1 - ckp1) - 1)
                for yi, zi, gi in zip(y, z, grad_normalized):
                    _record_param_update(self._update_sq_acc, yi, zi, gi, ckp1, alpha)
                    self._update_numel += yi.numel()

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(y, z, weight=ckp1)
                torch._foreach_add_(y, grad_normalized, alpha=alpha)

                # z step
                torch._foreach_sub_(z, grad_normalized, alpha=lr)
            else:
                for p in active_p:
                    y = p # Notation to match theory
                    grad = p.grad

                    state = self.state[p]

                    z = state['z']
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                    if inner_momentum != 0:
                        exp_avg = state['exp_avg']
                        exp_avg.mul_(inner_momentum).add_(grad, alpha=1-inner_momentum)
                        grad_normalized = exp_avg.div(bias_correction1).div_(denom)
                    else:
                        # Reuse grad buffer for memory efficiency
                        grad_normalized = grad.div_(denom)

                    # Weight decay calculated at y
                    if decay != 0:
                        grad_normalized.add_(y, alpha=decay)

                    alpha = lr * (beta1 * (1 - ckp1) - 1)
                    _record_param_update(self._update_sq_acc, y, z, grad_normalized, ckp1, alpha)
                    self._update_numel += y.numel()

                    # These operations update y in-place,
                    # without computing x explicitly.
                    y.lerp_(end=z, weight=ckp1)
                    y.add_(grad_normalized, alpha=alpha)

                    # z step
                    z.sub_(grad_normalized, alpha=lr)

            group['k'] = k+1

        total_sq = sum((v.item() for v in self._update_sq_acc.values()), 0.0)
        n = self._update_numel
        self._update_rms = math.sqrt(total_sq / n) if n else 0.0
        return loss

    @property
    def update_rms(self):
        """Root-mean-square of the last step's change in the training parameters.

        None before the first step. This is the size of the parameter update,
        not the gradient. See issue #59.
        """
        return getattr(self, "_update_rms", None)
