# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from schedulefree import (SGDScheduleFree, SGDScheduleFreeClosure,
    AdamWScheduleFree, AdamWScheduleFreeClosure, AdamWScheduleFreeReference,
    RAdamScheduleFree, RAdamScheduleFreeClosure,
    ScheduleFreeWrapper, ScheduleFreeWrapperReference, SGDScheduleFreeReference,
    AdamCScheduleFreePlusPaper)

def allclose(x, y):
    assert torch.allclose(x, y, rtol=1e-05, atol=1e-06)

def test_schedulefree_wrapper():
    lr = 0.3
    decay = 0.1
    weight1 = torch.randn(3, 2).requires_grad_()
    weight2 = torch.clone(weight1.detach()).requires_grad_()
    optimizer1 = SGDScheduleFree(
        [weight1], lr=lr,
        weight_decay=decay, momentum=0.9, foreach=False)

    optimizer2 = ScheduleFreeWrapper(
        torch.optim.SGD([weight2], lr=lr, momentum=0.0),
        momentum=0.9,
        weight_decay_at_y=decay)

    compare_schedulefree_versions(weight1, optimizer1, weight2, optimizer2)


def test_schedulefree_wrapper_reference():
    lr = 0.3
    decay = 0.1
    weight1 = torch.randn(3, 2).requires_grad_()
    weight2 = torch.clone(weight1.detach()).requires_grad_()
    optimizer1 = SGDScheduleFree(
        [weight1], lr=lr,
        weight_decay=decay, momentum=0.9, foreach=False)

    optimizer2 = ScheduleFreeWrapperReference(
        torch.optim.SGD([weight2], lr=lr, momentum=0.0),
        momentum=0.9,
        weight_decay_at_y=decay)

    compare_schedulefree_versions(weight1, optimizer1, weight2, optimizer2)

def compare_schedulefree_versions(weight1, optimizer1, weight2, optimizer2):
    assert torch.allclose(weight1, weight2)

    for step_idx in range(100):
        if step_idx % 10 == 0:
            print(step_idx)
        optimizer1.train()
        optimizer2.train()
        grad = torch.rand_like(weight1)

        weight1.grad = torch.clone(grad)
        weight2.grad = torch.clone(grad)

        optimizer1.step()
        optimizer2.step()

        allclose(weight1, weight2)

        optimizer1.eval()
        optimizer2.eval()

        allclose(weight1, weight2)


def test_adamw_reference_inner_momentum_matches_wrap():
    """Verify AdamWScheduleFreeReference with inner_momentum behaves identically
    to ScheduleFreeWrapperReference wrapping torch.optim.AdamW with the same
    parameters."""
    lr = 0.3
    inner_momentum = 0.9
    outer_momentum = 0.95
    beta2 = 0.999
    eps = 1e-8

    weight1 = torch.randn(3, 2).requires_grad_()
    weight2 = torch.clone(weight1.detach()).requires_grad_()

    optimizer1 = AdamWScheduleFreeReference(
        [weight1],
        lr=lr,
        betas=(outer_momentum, beta2),
        eps=eps,
        weight_decay=0.0,
        warmup_steps=0,
        inner_momentum=inner_momentum,
    )

    optimizer2 = ScheduleFreeWrapperReference(
        torch.optim.AdamW(
            [weight2],
            lr=lr,
            betas=(inner_momentum, beta2),
            eps=eps,
            weight_decay=0.0,
        ),
        momentum=outer_momentum,
        weight_decay_at_y=0.0,
    )

    # Pre-initialize AdamW's state. ScheduleFreeWrapperReference shares the
    # state dict with the inner optimizer, so when the wrapper adds 'z', 'x',
    # 'y' to the state, AdamW's `len(state) == 0` guard would otherwise skip
    # initializing 'step', 'exp_avg', and 'exp_avg_sq'.
    for p in [weight2]:
        adamw_state = optimizer2.base.state[p]
        adamw_state["step"] = torch.tensor(0.0)
        adamw_state["exp_avg"] = torch.zeros_like(
            p, memory_format=torch.preserve_format)
        adamw_state["exp_avg_sq"] = torch.zeros_like(
            p, memory_format=torch.preserve_format)

    compare_schedulefree_versions(weight1, optimizer1, weight2, optimizer2)


def test_adamw_reference_no_inner_momentum_no_buffer():
    """When inner_momentum=0 (default), no exp_avg buffer should be allocated."""
    weight = torch.randn(3, 2).requires_grad_()
    optimizer = AdamWScheduleFreeReference([weight], lr=0.1, weight_decay=0.0)
    optimizer.train()
    weight.grad = torch.rand_like(weight)
    optimizer.step()
    state = optimizer.state[weight]
    assert 'exp_avg' not in state
    assert 'exp_avg_sq' in state
    assert 'z' in state


def test_adamw_inner_momentum_matches_reference():
    """Verify AdamWScheduleFree with inner_momentum matches
    AdamWScheduleFreeReference with the same inner_momentum, for both the
    foreach and non-foreach code paths."""
    lr = 0.3
    decay = 0.5
    warmup = 5
    inner_momentum = 0.85

    weight_ref = torch.randn(3, 2).requires_grad_()
    weight = torch.clone(weight_ref.data).requires_grad_()
    weight_foreach = torch.clone(weight_ref.data).requires_grad_()

    optimizer_ref = AdamWScheduleFreeReference(
        [weight_ref], lr=lr, warmup_steps=warmup, weight_decay=decay,
        inner_momentum=inner_momentum)
    optimizer = AdamWScheduleFree(
        [weight], lr=lr, warmup_steps=warmup, weight_decay=decay,
        inner_momentum=inner_momentum, foreach=False)
    optimizer_foreach = AdamWScheduleFree(
        [weight_foreach], lr=lr, warmup_steps=warmup, weight_decay=decay,
        inner_momentum=inner_momentum, foreach=True)

    for step_idx in range(10):
        print(step_idx)
        optimizer.train()
        optimizer_ref.train()
        optimizer_foreach.train()

        grad = torch.rand_like(weight)
        weight.grad = torch.clone(grad)
        weight_ref.grad = torch.clone(grad)
        weight_foreach.grad = torch.clone(grad)

        optimizer.step()
        optimizer_ref.step()
        optimizer_foreach.step()

        optimizer.eval()
        optimizer_ref.eval()
        optimizer_foreach.eval()

        state = optimizer.state[weight]
        state_ref = optimizer_ref.state[weight_ref]
        state_foreach = optimizer_foreach.state[weight_foreach]

        # Inner-momentum buffers should be allocated.
        assert 'exp_avg' in state
        assert 'exp_avg' in state_ref
        assert 'exp_avg' in state_foreach

        # Param value in eval mode is x; check equivalence.
        allclose(weight, weight_ref)
        allclose(weight, weight_foreach)

        # z and exp_avg should also match.
        allclose(state['z'], state_ref['z'])
        allclose(state['z'], state_foreach['z'])
        allclose(state['exp_avg'], state_ref['exp_avg'])
        allclose(state['exp_avg'], state_foreach['exp_avg'])


def test_adamw_no_inner_momentum_no_buffer():
    """When inner_momentum=0 (default), AdamWScheduleFree should not allocate
    an exp_avg buffer (for both foreach and non-foreach paths)."""
    for use_foreach in [False, True]:
        weight = torch.randn(3, 2).requires_grad_()
        optimizer = AdamWScheduleFree(
            [weight], lr=0.1, weight_decay=0.0, foreach=use_foreach)
        optimizer.train()
        weight.grad = torch.rand_like(weight)
        optimizer.step()
        state = optimizer.state[weight]
        assert 'exp_avg' not in state, f"foreach={use_foreach}"
        assert 'exp_avg_sq' in state
        assert 'z' in state


def test_schedulefree_sgd():
    decay = 0.5
    warmup = 5
    weight_closure = torch.randn(3, 2).requires_grad_()
    weight = torch.clone(weight_closure.data).requires_grad_()
    weight_ref = torch.clone(weight_closure.data).requires_grad_()
    optimizer_closure = SGDScheduleFreeClosure([weight_closure], lr=0.3, warmup_steps=warmup, weight_decay=decay)
    optimizer = SGDScheduleFree([weight], lr=0.3, warmup_steps=warmup, weight_decay=decay)
    optimizer_ref = SGDScheduleFreeReference([weight_ref], lr=0.3, warmup_steps=warmup, weight_decay=decay)


    for step_idx in range(10):
        print(step_idx)
        optimizer.train()
        optimizer_ref.train()

        grad = torch.rand_like(weight)

        weight.grad = torch.clone(grad)
        weight_ref.grad = torch.clone(grad)

        def closure():
            weight_closure.grad = torch.clone(grad)

        optimizer.step()
        optimizer_closure.step(closure=closure)
        optimizer_ref.step()

        optimizer.eval()
        optimizer_ref.eval()

        for group_closure, group, group_ref in zip(
                optimizer_closure.param_groups,
                optimizer.param_groups,
                optimizer_ref.param_groups):
            for p_closure, p, p_ref in zip(
                    group_closure['params'],
                    group['params'],
                    group_ref['params']):

                state_closure = optimizer_closure.state[p_closure]
                state_ref = optimizer_ref.state[p_ref]
                state = optimizer.state[p]

                assert torch.allclose(p, p_closure)
                assert torch.allclose(p, p_ref)

                z_closure = state_closure['z']
                z_ref = state_ref['z']
                z = state['z']
                assert torch.allclose(z, z_closure)
                assert torch.allclose(z, z_ref)

def test_schedulefree_adam():
    decay = 0.5
    warmup = 5
    weight_closure = torch.randn(3, 2).requires_grad_()
    weight = torch.clone(weight_closure.data).requires_grad_()
    weight_reference = torch.clone(weight_closure.data).requires_grad_()
    optimizer_closure = AdamWScheduleFreeClosure([weight_closure], lr=0.3, warmup_steps=warmup, weight_decay=decay)
    optimizer = AdamWScheduleFree([weight], lr=0.3, warmup_steps=warmup, weight_decay=decay)
    optimizer_reference = AdamWScheduleFreeReference([weight_reference], lr=0.3, warmup_steps=warmup, weight_decay=decay)

    for step_idx in range(10):
        print(step_idx)
        optimizer.train()
        optimizer_reference.train()
        grad = torch.rand_like(weight)

        weight.grad = torch.clone(grad)
        weight_reference.grad = torch.clone(grad)

        def closure():
            weight_closure.grad = torch.clone(grad)

        optimizer.step()
        optimizer_closure.step(closure=closure)
        optimizer_reference.step()

        optimizer.eval()
        optimizer_reference.eval()

        for group_closure, group, group_reference in zip(optimizer_closure.param_groups, optimizer.param_groups, optimizer_reference.param_groups):
            for p_closure, p, p_reference in zip(group_closure['params'], group['params'], group_reference['params']):
                state_closure = optimizer_closure.state[p_closure]
                state = optimizer.state[p]
                state_reference = optimizer_reference.state[p_reference]

                z_closure = state_closure['z']
                z = state['z']
                z_reference = state_reference['z']

                allclose(p, p_closure)
                allclose(p, p_reference)
                allclose(z, z_closure)
                allclose(z, z_reference)

        optimizer.train()
        optimizer_reference.train()

        for group_closure, group, group_reference in zip(optimizer_closure.param_groups, optimizer.param_groups, optimizer_reference.param_groups):
            for p_closure, p, p_reference in zip(group_closure['params'], group['params'], group_reference['params']):
                state_closure = optimizer_closure.state[p_closure]
                state = optimizer.state[p]
                state_reference = optimizer_reference.state[p_reference]

                z_closure = state_closure['z']
                z = state['z']
                z_reference = state_reference['z']

                # Extrapolate p.data to equal y
                y = p.data
                y_closure = p_closure.lerp(end=z_closure, weight=1-0.9)


                allclose(y, y_closure)
                allclose(y, p_reference.data)

def test_schedulefree_radam():
    decay = 0.5
    weight_closure = torch.randn(3, 2).requires_grad_()
    weight = torch.clone(weight_closure.data).requires_grad_()
    optimizer_closure = RAdamScheduleFreeClosure([weight_closure], lr=0.3, weight_decay=decay)
    optimizer = RAdamScheduleFree([weight], lr=0.3, weight_decay=decay)

    for step_idx in range(20):
        print(step_idx)
        optimizer.train()
        grad = torch.rand_like(weight)

        weight.grad = torch.clone(grad)

        def closure():
            weight_closure.grad = torch.clone(grad)

        optimizer.step()
        optimizer_closure.step(closure=closure)

        optimizer.eval()

        for group_closure, group in zip(optimizer_closure.param_groups, optimizer.param_groups):
            for p_closure, p in zip(group_closure['params'], group['params']):
                state_closure = optimizer_closure.state[p_closure]
                state = optimizer.state[p]

                z_closure = state_closure['z']
                z = state['z']

                allclose(p, p_closure)
                allclose(z, z_closure)

        optimizer.train()

        for group_closure, group in zip(optimizer_closure.param_groups, optimizer.param_groups):
            for p_closure, p in zip(group_closure['params'], group['params']):
                state_closure = optimizer_closure.state[p_closure]
                state = optimizer.state[p]

                z_closure = state_closure['z']
                z = state['z']

                # Extrapolate p.data to equal y
                y = p.data
                y_closure = p_closure.lerp(end=z_closure, weight=1-0.9)

                allclose(y, y_closure)

def test_foreach():
    decay = 0.5
    warmup = 5
    weight_foreach = torch.randn(3, 2).requires_grad_()
    weight_foreach2 = torch.randn(1, 1).requires_grad_()
    weight_foreach_nograd = torch.randn(1, 2).requires_grad_()

    weight = torch.clone(weight_foreach.data).requires_grad_()
    weight2 = torch.clone(weight_foreach2.data).requires_grad_()
    weight_nograd = torch.clone(weight_foreach_nograd.data).requires_grad_()
    optimizer_foreach = AdamWScheduleFree([
        {'params': [weight_foreach, weight_foreach2]},
        {'params': [weight_foreach_nograd]},
        {'params': []}],
        lr=0.3, warmup_steps=warmup, weight_decay=decay, foreach=True)
    optimizer = AdamWScheduleFree([
        {'params': [weight, weight2]},
        {'params': [weight_nograd]},
        {'params': []}], lr=0.3, warmup_steps=warmup, weight_decay=decay, foreach=False)

    for step_idx in range(10):
        optimizer.train()
        optimizer_foreach.train()
        grad = torch.rand_like(weight)
        grad2 = torch.rand_like(weight2)

        weight.grad = torch.clone(grad)
        weight2.grad = torch.clone(grad2)
        weight_foreach.grad = torch.clone(grad)
        weight_foreach2.grad = torch.clone(grad2)

        optimizer.step()
        optimizer_foreach.step()

        optimizer.eval()
        optimizer_foreach.eval()

        for group_foreach, group in zip(optimizer_foreach.param_groups, optimizer.param_groups):
            for p_foreach, p in zip(group_foreach['params'], group['params']):
                if p.grad is not None or p_foreach.grad is not None:
                    state_foreach = optimizer_foreach.state[p_foreach]
                    state = optimizer.state[p]
                    z_foreach = state_foreach['z']
                    z = state['z']

                    assert torch.allclose(p, p_foreach)
                    assert torch.allclose(z, z_foreach)


def test_equiv():
    model1 = torch.tensor([1.0])
    model2 = torch.tensor([1.0])
    z = torch.tensor([2.0])
    momentum = 0.9
    ckp1 = 0.05

    # y -> x
    model1.lerp_(end=z, weight=1-1/momentum)

    # x update
    model1.lerp_(end=z, weight=ckp1)

    # x -> y
    model1.lerp_(end=z, weight=1-momentum)

    model2.lerp_(end=z, weight=ckp1)

    assert torch.allclose(model1, model2)

    # Update x
    model1.lerp_(end=z, weight=ckp1)

    # Convert Model x -> y
    model1.lerp_(end=z, weight=1-momentum)

    # Update x then convert x -> y using a fused operation
    model2.lerp_(end=z, weight=1-momentum*(1-ckp1))

    assert torch.allclose(model1, model2)

def test_compile():
    #torch._dynamo.config.verbose=True

    lr = 0.3
    decay = 0.5
    warmup = 5
    weight = torch.randn(3, 2).requires_grad_()
    weight_uncompiled = torch.clone(weight.data).requires_grad_()
    optimizer = AdamWScheduleFree([weight], lr=lr, warmup_steps=warmup, weight_decay=decay)
    optimizer_uncompiled = AdamWScheduleFree([weight_uncompiled], lr=lr, warmup_steps=warmup, weight_decay=decay)

    #@torch.compile(fullgraph=False)
    def opt_train():
        optimizer.train()

    #@torch.compile(fullgraph=False)
    def opt_eval():
        optimizer.eval()

    #@torch.compile(fullgraph=False)
    def opt_step():
        optimizer.step()

    for step_idx in range(10):
        print(step_idx)
        opt_train()
        optimizer_uncompiled.train()

        grad = torch.rand_like(weight)
        weight.grad = torch.clone(grad)
        weight_uncompiled.grad = torch.clone(grad)

        opt_step()
        optimizer_uncompiled.step()

        assert torch.allclose(weight, weight_uncompiled)

        opt_eval()
        optimizer_uncompiled.eval()

        assert torch.allclose(weight, weight_uncompiled)


def test_adamc_schedulefree_polyak_runs():
    """Smoke test: AdamCScheduleFreePlusPaper takes several steps
    without errors in a non-distributed environment (DTensor detection
    should skip all_reduce calls)."""
    torch.manual_seed(42)
    weight = torch.randn(3, 2).requires_grad_()
    optimizer = AdamCScheduleFreePlusPaper(
        [weight], lr=1.0, weight_decay=0.0)

    initial = weight.detach().clone()
    optimizer.train()

    for step_idx in range(5):
        grad = torch.rand_like(weight)
        weight.grad = grad
        function_value = float(grad.pow(2).sum().item())  # positive scalar
        optimizer.step_func(function_value=function_value)

    # The parameter should have moved.
    assert not torch.allclose(weight, initial)

    # State should contain all the expected buffers.
    state = optimizer.state[weight]
    for key in ['x', 'y', 'z', 'exp_avg', 'exp_avg_sq']:
        assert key in state, f"missing state key: {key}"

    # Group state populated for logging.
    group = optimizer.param_groups[0]
    for key in ['scheduled_lr', 'grad_l1_ema', 'grad_l1_ema_corr',
                'function_value_ema', 'ip_term', 'lr_max']:
        assert key in group, f"missing group key: {key}"


def test_adamc_schedulefree_polyak_train_eval():
    """Verify p is set to y in train mode and x in eval mode."""
    torch.manual_seed(0)
    weight = torch.randn(3, 2).requires_grad_()
    optimizer = AdamCScheduleFreePlusPaper([weight], lr=1.0)

    optimizer.train()
    weight.grad = torch.rand_like(weight)
    optimizer.step_func(function_value=1.0)

    optimizer.eval()
    # In eval mode, p should equal state['x'].
    allclose(weight, optimizer.state[weight]['x'])

    optimizer.train()
    # Back in train mode, p should equal state['y'].
    allclose(weight, optimizer.state[weight]['y'])


def test_adamc_schedulefree_polyak_requires_train_mode():
    """step_func should raise when called without first entering train
    mode."""
    weight = torch.randn(3, 2).requires_grad_()
    optimizer = AdamCScheduleFreePlusPaper([weight], lr=1.0)
    weight.grad = torch.rand_like(weight)
    raised = False
    try:
        optimizer.step_func(function_value=1.0)
    except Exception as exc:
        raised = True
        assert "train mode" in str(exc)
    assert raised, "step_func did not raise when not in train mode"


def test_adamc_schedulefree_polyak_no_dtensor_no_allreduce():
    """When no gradient is a DTensor, step_func must not invoke
    torch.distributed.all_reduce, even if distributed happens to be
    initialized.  We verify this by monkey-patching dist.all_reduce to
    raise if called."""
    import torch.distributed as dist

    torch.manual_seed(1)
    weight = torch.randn(3, 2).requires_grad_()
    optimizer = AdamCScheduleFreePlusPaper([weight], lr=1.0)

    original_all_reduce = dist.all_reduce
    original_is_initialized = dist.is_initialized

    called = {'all_reduce': False}

    def fake_all_reduce(*args, **kwargs):
        called['all_reduce'] = True
        return original_all_reduce(*args, **kwargs)

    dist.all_reduce = fake_all_reduce
    # Pretend distributed is initialized, so the only thing that should
    # gate the call is the DTensor detection.
    dist.is_initialized = lambda: True
    try:
        optimizer.train()
        weight.grad = torch.rand_like(weight)
        optimizer.step_func(function_value=1.0)
    finally:
        dist.all_reduce = original_all_reduce
        dist.is_initialized = original_is_initialized

    assert not called['all_reduce'], (
        "all_reduce was invoked for non-DTensor gradients")


def test_adamc_schedulefree_polyak_multiple_param_groups():
    """Optimizer should handle multiple param groups."""
    torch.manual_seed(2)
    w1 = torch.randn(3, 2).requires_grad_()
    w2 = torch.randn(2, 4).requires_grad_()
    optimizer = AdamCScheduleFreePlusPaper(
        [{'params': [w1]}, {'params': [w2]}], lr=1.0)

    optimizer.train()
    for _ in range(3):
        w1.grad = torch.rand_like(w1)
        w2.grad = torch.rand_like(w2)
        optimizer.step_func(function_value=1.0)
    optimizer.eval()

    # Both parameters should have buffered state.
    assert 'z' in optimizer.state[w1]
    assert 'z' in optimizer.state[w2]


def test_adamc_schedulefree_polyak_bias_correction():
    """Verify that group['grad_l1_ema'] stores the raw uncorrected EMA,
    while group['grad_l1_ema_corr'] stores the debiased value."""
    torch.manual_seed(42)
    weight = torch.randn(3, 2).requires_grad_()
    optimizer = AdamCScheduleFreePlusPaper([weight], lr=1.0, polyak_beta=0.9)
    optimizer.train()

    weight.grad = torch.ones_like(weight)
    optimizer.step_func(function_value=1.0)

    group = optimizer.param_groups[0]
    raw = group['grad_l1_ema']
    corr = group['grad_l1_ema_corr']
    # At step 1 (k=0), bias correction factor is 1 - 0.9^1 = 0.1
    assert raw > 0
    assert torch.isclose(torch.tensor(corr * (1.0 - 0.9)), torch.tensor(raw))
    assert raw != corr


if __name__ == "__main__":
    torch.manual_seed(1)

    test_compile()

    test_equiv()

    test_schedulefree_wrapper()

    test_schedulefree_wrapper_reference()

    test_foreach()

    test_schedulefree_adam()
    test_adamw_reference_inner_momentum_matches_wrap()
    test_adamw_reference_no_inner_momentum_no_buffer()
    test_adamw_inner_momentum_matches_reference()
    test_adamw_no_inner_momentum_no_buffer()
    test_schedulefree_sgd()
    test_schedulefree_radam()

    test_adamc_schedulefree_polyak_runs()
    test_adamc_schedulefree_polyak_train_eval()
    test_adamc_schedulefree_polyak_requires_train_mode()
    test_adamc_schedulefree_polyak_no_dtensor_no_allreduce()
    test_adamc_schedulefree_polyak_multiple_param_groups()
    test_adamc_schedulefree_polyak_bias_correction()
