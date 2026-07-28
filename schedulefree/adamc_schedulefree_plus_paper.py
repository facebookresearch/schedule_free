# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any
from typing_extensions import TypeAlias
import torch
import torch.distributed as dist

import torch.optim
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
import math


class AdamCScheduleFreePlusPaper(torch.optim.Optimizer):
    r"""
    AdamC + Schedule-Free + Polyak (Reference / Research Implementation)

    This code uses a non-standard optimizer step interface: it requires the
    rank-local function value to be passed into the step. This, together
    with gradient norm information, will be all-reduced to compute the
    Polyak step size.

    This optimizer combines three ideas into a single update rule:

    1. **AdamC**: an AdamW-style adaptive optimizer. Decoupled weight decay
       is applied to the ``z`` sequence at the ``y`` point and is scaled by
       ``lr**2`` (the "C" variant), so that the effective decay does not
       blow up when the learning rate is adapted online.
       See "Why Gradients Rapidly Increase Near the End of Training"
       https://arxiv.org/abs/2506.02285
       Because it's decoupled, weight-decay values DO NOT correspond to
       standard AdamW values, values should be more like 20 or 50 than 0.1.
    2. **Schedule-Free**: maintains the three iterate sequences ``x`` (the
       averaged / evaluation iterate), ``y`` (the gradient query point used
       during training) and ``z`` (the unaveraged AdamW iterate). The
       averaging weight ``ckp1`` follows the standard Schedule-Free
       schedule parameterised by ``r`` and ``weight_lr_power``. The
       momentum used for the ``x -> y`` extrapolation is the
       *Schedule-Free* momentum ``sf_beta1`` (distinct from Adam's
       ``beta1``), and it can optionally be annealed from ``sf_beta1``
       to ``sf_beta1_max`` over
       ``sf_beta1_anneal_steps`` (using a log-space interpolation on
       ``1 - sf_beta1_k`` so that the effective averaging window grows
       smoothly).
    3. **Polyak step size**: the per-step learning rate is set adaptively
       using a Polyak-style rule of the form
       ``polyak_lr = max(0, f + ip_term) / grad_l1_ema``,
       where ``f`` is a (globally reduced) function value supplied by the
       caller, ``ip_term`` is a correction term equal to
       ``sf_beta1_k * <grad, z - x>`` summed over all parameters, and
       ``grad_l1_ema`` is an EMA of the L1 norm of the gradient, scaled by
       ``sqrt(pi/2)`` (the expected ratio of L1 to L2 norms for Gaussian
       vectors) and bias-corrected with ``polyak_beta``. The effective
       per-group learning rate is ``group_lr = lr * polyak_lr``.

    This is a *reference* implementation intended for research and
    experimentation. It is written for clarity rather than performance: it
    allocates the full set of buffers (``x``, ``y``, ``z``, ``exp_avg``,
    ``exp_avg_sq``) and does not use the foreach or fused kernels.

    Distributed training: this optimizer calls ``torch.distributed.all_reduce``
    on the L1 norm of the gradient, the inner-product correction term, and
    the function value, but only when at least one parameter's gradient is a
    DTensor (i.e. sharded across ranks). When gradients are regular
    ``torch.Tensor`` instances (e.g. single-process training, or DDP where
    gradients are already replicated), the all_reduce calls are skipped and
    the local values are used directly. Gradients are accessed via
    ``.to_local()`` when they are DTensors so the optimizer is compatible
    with DTensor-sharded parameters.

    Train / eval mode: as with all Schedule-Free optimizers, ``.train()``
    and ``.eval()`` must be called before training and evaluation
    respectively (and before saving checkpoints), so that the parameter
    buffer ``p`` is set to ``y`` during training and to ``x`` during
    evaluation.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float or Tensor):
            Base learning rate (default 1.0). The effective step size is
            ``lr * polyak_lr`` where ``polyak_lr`` is computed online from
            the function value passed to ``step_func``.
        betas (Tuple[float, float]):
            Coefficients used for computing running averages of the gradient
            (``beta1``) and its square (``beta2``) inside the AdamW-style
            update (default: ``(0.9, 0.95)``).
        sf_beta1 (float):
            Schedule-Free outer momentum used for the ``x -> y``
            extrapolation (default 0.9). This is distinct from Adam's
            ``beta1``.
        eps (float):
            Term added to the denominator of the AdamW update for
            numerical stability, and used as a floor on ``lr`` to avoid
            zero step sizes (default 1e-8).
        weight_decay (float):
            Decoupled weight decay coefficient applied to the ``z`` sequence
            at the ``y`` point. Scaled by ``group_lr ** 2`` (the AdamC
            scaling) so that adapting ``polyak_lr`` does not change the
            effective decay strength inappropriately (default 0).
        r (float):
            Polynomial weighting power used in the Schedule-Free averaging
            weights (default 0). Try ``r=2`` as well -- it sometimes gives
            better results, particularly for long-duration runs.
        polyak_beta (float):
            EMA decay used for the running estimate of the gradient L1
            norm in the Polyak step-size rule. ``0`` means no averaging --
            the instantaneous (bias-corrected) L1 norm is used
            (default 0).
        c_warmup (int):
            Number of initial steps during which the averaging weight
            ``ckp1`` is forced to ``1.0`` (i.e. ``x`` tracks ``z``
            exactly). After this warmup, the standard Schedule-Free
            weighting kicks in. It is recommended to set ``c_warmup`` to
            1x or 2x the lr warmup (default 0).
        sf_beta1_anneal_steps (int):
            If greater than zero, ``sf_beta1`` is annealed from
            ``sf_beta1`` to ``sf_beta1_max`` over this many steps, using a
            log-space schedule on ``1 - sf_beta1_k``. Set to ``0`` to
            disable annealing (default 0).
        sf_beta1_max (float):
            Target value for ``sf_beta1`` at the end of the annealing
            schedule (default 0.965).
        weight_lr_power (float):
            Exponent used to weight the per-step contribution to the
            Schedule-Free average by ``lr_max ** weight_lr_power``
            (default 2).

    State maintained per parameter:
        x, y, z (Tensors): the Schedule-Free iterate sequences.
        exp_avg (Tensor): AdamW first-moment buffer.
        exp_avg_sq (Tensor): AdamW second-moment buffer.

    Group state (populated each step for logging / inspection):
        scheduled_lr: the per-group effective learning rate used this step
            (``lr * polyak_lr``).
        lr_max: running max of ``group_lr``.
        weight_sum: running sum of Schedule-Free averaging weights.
        grad_l1_ema, grad_l1_ema_corr: raw and bias-corrected EMA of the L1
            norm of the gradient.
        function_value_ema: ``global_function_value + ip_term`` used in the
            Polyak rule.
        ip_term: the Schedule-Free inner-product correction.
    """
    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 1.0,
                 betas: Tuple[float, float] = (0.9, 0.95),
                 sf_beta1: float = 0.9,
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 r: float = 0,
                 polyak_beta: float = 0,
                 c_warmup: int = 0,
                 sf_beta1_anneal_steps: int = 0,
                 sf_beta1_max: float = 0.965,
                 weight_lr_power: float = 2,
                 ):

        defaults = dict(lr=lr,
                        betas=betas,
                        sf_beta1=sf_beta1,
                        eps=eps,
                        r=r,
                        k=0,
                        train_mode = False,
                        weight_sum=0.0,
                        lr_max=eps,
                        scheduled_lr=0.0,
                        polyak_beta=polyak_beta,
                        sf_beta1_anneal_steps=sf_beta1_anneal_steps,
                        sf_beta1_max=sf_beta1_max,
                        grad_l1_ema=0.0,
                        polyak_denom_ema=0.0,
                        c_warmup=c_warmup,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for gidx, group in enumerate(self.param_groups):
            train_mode = group['train_mode']

            if train_mode:
                for p in group['params']:
                    state = self.state[p]

                    if 'x' in state:
                        # Switch p to x
                        p.detach().copy_(state['x'])
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for gidx, group in enumerate(self.param_groups):
            train_mode = group['train_mode']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'y' in state:
                        p.detach().copy_(state['y'])
                group['train_mode'] = True

    @torch.no_grad()
    def step_func(self, function_value=None) -> Optional[float]:
        """Performs a single optimization step.

        Unlike a standard PyTorch optimizer ``step``, this method takes the
        current value of the loss function as a scalar (``function_value``)
        instead of a closure.  The value is used by the Polyak step-size
        rule.  When at least one parameter has a DTensor gradient, the
        function value is averaged across all ranks via
        ``dist.all_reduce(..., AVG)`` to form ``global_function_value``;
        otherwise (regular ``torch.Tensor`` gradients) the local value is
        used directly.

        Arguments:
            function_value (float): The current (local-rank) value of the
                objective at the current parameters ``y``.  Must not be
                ``None`` — it is required by the Polyak step-size rule.

        Returns:
            The (un-reduced, local) ``function_value`` that was passed in,
            for convenience.
        """
        if not self.param_groups[0]['train_mode']:
            raise Exception("Optimizer was not in train mode when step is called. "
                            "Please insert .train() and .eval() calls on the "
                            "optimizer. See documentation for details.")


        grad_l1_ema = self.param_groups[0]['grad_l1_ema']
        polyak_denom_ema = self.param_groups[0]['polyak_denom_ema']
        k = self.param_groups[0]['k']
        beta1, beta2 = self.param_groups[0]['betas']
        eps = self.param_groups[0]['eps']
        polyak_beta = self.param_groups[0]['polyak_beta']
        sf_beta1 = self.param_groups[0]['sf_beta1']
        sf_beta1_max = self.param_groups[0]['sf_beta1_max']
        sf_beta1_anneal_steps = self.param_groups[0]['sf_beta1_anneal_steps']

        if sf_beta1_anneal_steps > 0:
            progress = min(k/sf_beta1_anneal_steps, 1.0)
            sf_beta1_k = 1 - math.exp(math.log(1-sf_beta1) * (1-progress) + math.log(1-sf_beta1_max) * progress)
        else:
            sf_beta1_k = sf_beta1

        grad_l1_list = []
        ip_term_list = []

        is_distributed = False

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data

                # Detect DTensor-sharded gradients: only DTensors require an
                # all_reduce to obtain the global L1 / inner-product values.
                # Regular tensors (e.g. under DDP) are already replicated, so
                # the local values are already the global ones.
                if hasattr(grad, 'to_local'):
                    is_distributed = True
                    local_grad = grad.to_local()
                else:
                    local_grad = grad

                # 1. Compute LOCALLY to avoid network syncs inside the loop
                grad_l1_p = torch.linalg.vector_norm(local_grad, ord=1)
                grad_l1_list.append(grad_l1_p)

                if 'z' in state:
                    z = state['z']
                    x = state['x']
                    if hasattr(z, 'to_local'):
                        local_z = z.to_local()
                    else:
                        local_z = z
                    if hasattr(x, 'to_local'):
                        local_x = x.to_local()
                    else:
                        local_x = x
                    ip_term_p = sf_beta1_k * (local_grad.mul(local_z - local_x)).sum()
                    ip_term_list.append(ip_term_p)

        # 2. Stack and sum the local scalars into single tensors on this device
        local_grad_l1 = torch.stack(grad_l1_list).sum() if grad_l1_list else torch.tensor(0.0, device=grad.device)
        local_ip_term = torch.stack(ip_term_list).sum() if ip_term_list else torch.tensor(0.0, device=grad.device)

        # 3. Perform ONE global all_reduce over the network to get the final
        # values, but only when at least one parameter is sharded across ranks
        # (i.e. its gradient is a DTensor). For replicated (non-DTensor)
        # gradients, the local sums are already the global sums and an
        # all_reduce would over-count by the world size.
        if is_distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_grad_l1, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_ip_term, op=dist.ReduceOp.SUM)

        grad_l1 = local_grad_l1.item()
        ip_term = local_ip_term.item()

        if is_distributed and dist.is_available() and dist.is_initialized():
            dist_tensor = torch.zeros(1).cuda()
            dist_tensor[0] = function_value
            dist.all_reduce(dist_tensor, op=dist.ReduceOp.AVG)
            global_function_value = dist_tensor[0].item()
        else:
            global_function_value = float(function_value)

        grad_l1_ema = polyak_beta*(grad_l1_ema) + (1-polyak_beta)*grad_l1*math.sqrt(math.pi/2)
        grad_l1_ema_corr = grad_l1_ema/(1 - polyak_beta ** (k+1))

        polyak_lr = max(0, global_function_value+ip_term)/grad_l1_ema_corr

        for group in self.param_groups:
            eps = group['eps']
            lr = max(group['lr'], eps)
            decay = group['weight_decay']
            beta1, beta2 = group['betas']
            k = group['k']
            r = group['r']
            weight_lr_power = group['weight_lr_power']
            c_warmup = group['c_warmup']

            bias_correction1 = 1 - beta1 ** (k+1)
            bias_correction2 = 1 - beta2 ** (k+1)

            # Apply any warmup that's part of the lr sequence
            group_lr = lr*polyak_lr

            # For plotting
            group['grad_l1_ema'] = group['grad_l1_ema_corr'] = grad_l1_ema_corr
            group['function_value_ema'] = global_function_value+ip_term
            group['ip_term'] = ip_term
            group['scheduled_lr'] = group_lr # For logging purposes

            lr_max = group['lr_max'] = max(group_lr, group['lr_max'])

            if k < c_warmup:
                ckp1 = 1.0
            else:
                weight = ((k+1)**r) * (lr_max**weight_lr_power)
                weight_sum = group['weight_sum'] = group['weight_sum'] + weight

                ckp1 = weight/weight_sum

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(p.detach(), memory_format=torch.preserve_format)
                    state['x'] = torch.clone(p.detach(), memory_format=torch.preserve_format)
                    state['y'] = torch.clone(p.detach(), memory_format=torch.preserve_format)

                    state['exp_avg_sq'] = torch.zeros_like(p.detach(), memory_format=torch.preserve_format)
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg_sq = state['exp_avg_sq']
                exp_avg = state['exp_avg']
                z = state['z']
                x = state['x']
                y = state['y']

                # Weight decay
                z.sub_(y, alpha=group_lr*group_lr*decay)

                # Momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_corr = exp_avg.div(bias_correction1)

                # Decay the first and second moment running average coefficient
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                z.addcdiv_(exp_avg_corr, denom, value=-group_lr)

                ### Take step
                x.mul_(1-ckp1).add_(z, alpha=ckp1)

                # Compute y
                y.copy_(x.mul(sf_beta1_k).add_(z, alpha=1-sf_beta1_k))
                p.detach().copy_(y)

            group['k'] = k+1
        return function_value
