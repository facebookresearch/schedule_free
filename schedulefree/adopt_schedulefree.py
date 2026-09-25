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

class AdoptScheduleFree(torch.optim.Optimizer):
    r"""
    Schedule-Free ADOPT.

    As the name suggests, no scheduler is needed with this optimizer.
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.

    The outer loop is schedule-free averaging (this file mirrors
    AdamWScheduleFree): iterates y with z buffers and the ckp1 weighting.
    The inner update is ADOPT (https://arxiv.org/pdf/2411.02853): the
    gradient is normalized by a clamped root-mean-square estimate, clipped
    by clip_lambda(step), and tracked with momentum. betas[0] serves both
    as the schedule-free interpolation coefficient and the ADOPT momentum
    factor; betas[1] decays the second-moment estimate.

    Like ADOPT, the first step() call only initializes the second-moment
    estimate and does not move the parameters.

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
            running averages of the normalized gradient and its square
            (default: (0.9, 0.9999)).
        eps (float):
            Floor of the denominator (default: 1e-6).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        warmup_steps (int): Enables a linear learning rate warmup (default 0).
        r (float): Use polynomial weighting in the average
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
        decouple (bool): Use decoupled weight decay (default False).
        clip_lambda (Callable[[int], float], optional): gradient clipping
            schedule applied to the normalized gradient, as a function of
            the per-parameter step (default: step**0.25).
        foreach (bool): Use a foreach-backed implementation of the optimizer.
            Should be significantly faster, but will have higher peak memory
            usage (default True if supported in your PyTorch version).
    """
    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 0.0025,
                 betas: Tuple[float, float] = (0.9, 0.9999),
                 eps: float = 1e-6,
                 weight_decay: float = 0,
                 warmup_steps: int = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2.0,
                 decouple: bool = False,
                 clip_lambda: Optional[Callable[[int], float]] = None,
                 foreach: Optional[bool] = hasattr(torch, "_foreach_mul_")
                 ):

        if clip_lambda is None:
            clip_lambda = lambda step: step**0.25

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
                        decouple=decouple,
                        clip_lambda=clip_lambda,
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

        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
            decay = group['weight_decay']
            decouple = group['decouple']
            clip_lambda = group['clip_lambda']
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
                    self.state[p]['step'] = 0

            # ADOPT initializes the second moment on the first step and does
            # not move the parameters. Mirror that here: parameters with
            # step == 0 refresh exp_avg_sq and advance to step 1.
            first_p = [p for p in active_p if self.state[p]['step'] == 0]
            rest_p = [p for p in active_p if self.state[p]['step'] != 0]

            if group['foreach'] and len(first_p) > 0:
                first_sq = tuple(self.state[p]['exp_avg_sq'] for p in first_p)
                first_grad = tuple(p.grad for p in first_p)
                torch._foreach_addcmul_(first_sq, first_grad, first_grad)
                for p in first_p:
                    self.state[p]['step'] = 1
            else:
                for p in first_p:
                    self.state[p]['exp_avg_sq'].addcmul_(p.grad, p.grad, value=1)
                    self.state[p]['step'] = 1

            if group['foreach'] and len(rest_p) > 0:
                y, grad, exp_avg, exp_avg_sq, z = zip(*[(p,
                                                p.grad,
                                                self.state[p]['exp_avg'],
                                                self.state[p]['exp_avg_sq'],
                                                self.state[p]['z'])
                                                for p in rest_p])
                steps = [self.state[p]['step'] for p in rest_p]

                if decay != 0 and not decouple:
                    grad = tuple(g.add(y_p, alpha=decay) for g, y_p in zip(grad, y))

                # ADOPT normalization with the current (pre-update) estimate
                denom = torch._foreach_sqrt(list(exp_avg_sq))
                torch._foreach_maximum_(denom, eps)
                normed = torch._foreach_div(list(grad), denom)
                clip = clip_lambda(steps[0])
                torch._foreach_maximum_(normed, -clip)
                torch._foreach_minimum_(normed, clip)
                torch._foreach_lerp_(list(exp_avg), normed, 1 - beta1)
                grad_normalized = exp_avg

                if decay != 0 and decouple:
                    torch._foreach_mul_(list(y), 1 - lr*decay)

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(list(y), list(z), weight=ckp1)
                torch._foreach_add_(list(y), list(grad_normalized), alpha=lr*(beta1*(1-ckp1)-1))

                # z step
                torch._foreach_sub_(list(z), list(grad_normalized), alpha=lr)

                # Refresh the second moment with the current gradient
                torch._foreach_mul_(list(exp_avg_sq), beta2)
                torch._foreach_addcmul_(list(exp_avg_sq), list(grad), list(grad), value=1-beta2)
                for p in rest_p:
                    self.state[p]['step'] += 1
            else:
                for p in rest_p:
                    y = p # Notation to match theory
                    grad = p.grad

                    state = self.state[p]

                    z = state['z']
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    step = state['step']

                    if decay != 0 and not decouple:
                        grad = grad.add(y, alpha=decay)

                    denom = exp_avg_sq.sqrt().clamp_(min=eps)
                    normed = grad.div(denom)
                    clip = clip_lambda(step)
                    normed.clamp_(-clip, clip)
                    exp_avg.lerp_(normed, 1 - beta1)
                    grad_normalized = exp_avg

                    if decay != 0 and decouple:
                        y.mul_(1 - lr*decay)

                    # These operations update y in-place,
                    # without computing x explicitly.
                    y.lerp_(end=z, weight=ckp1)
                    y.add_(grad_normalized, alpha=lr*(beta1*(1-ckp1)-1))

                    # z step
                    z.sub_(grad_normalized, alpha=lr)

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                    state['step'] = step + 1

            group['k'] = k+1
        return loss
