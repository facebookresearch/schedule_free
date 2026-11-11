# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy

import pytest
import torch

from schedulefree import ScheduleFreeWrapper, ScheduleFreeWrapperReference


@pytest.fixture(params=[ScheduleFreeWrapper, ScheduleFreeWrapperReference])
def wrapper(request):
    return request.param


def keys(wrapper):
    return ('z',) if wrapper is ScheduleFreeWrapper else ('x', 'y', 'z')


def parameter(dtype=torch.float64):
    return torch.tensor([1., -2.], dtype=dtype, requires_grad=True)


def step(opt, params):
    for p in params:
        p.grad = p.detach().sin()
    opt.step()


@pytest.mark.parametrize('first_step', [True, False])
def test_buffers_survive_base_exception(wrapper, first_step):
    class FailingSGD(torch.optim.SGD):
        fail = False

        def step(self, closure=None):
            if self.fail:
                raise RuntimeError('base step failed')
            return super().step(closure)

    p = parameter()
    base = FailingSGD([p], lr=.01)
    opt = wrapper(base)
    opt.train()
    if not first_step:
        step(opt, [p])
    base.fail = True
    with pytest.raises(RuntimeError, match='base step failed'):
        step(opt, [p])
    for key in keys(wrapper):
        assert key in opt.state[p]
        assert key not in base.state[p]


def test_buffers_survive_later_parameter_preparation_failure(wrapper):
    p, q = parameter(), parameter()
    base = torch.optim.SGD([{'params': [p]}, {'params': [q]}], lr=.01)
    opt = wrapper(base, weight_decay_at_y=.1)
    opt.train()
    step(opt, [p, q])
    before = {key: opt.state[p][key] for key in keys(wrapper)}
    # Fail in preparation of the second group after processing the first.
    base.param_groups[1]['lr'] = object()
    with pytest.raises(TypeError):
        step(opt, [p, q])
    for key, tensor in before.items():
        assert opt.state[p][key] is tensor


@pytest.mark.parametrize('optimizer', [torch.optim.SGD, torch.optim.AdamW])
def test_step_hooks_can_inspect_both_states(wrapper, optimizer):
    p = parameter()
    base = optimizer([p], lr=.01)
    opt = wrapper(base)
    observations = []

    def inspect(optimizer, args, kwargs):
        for key in keys(wrapper):
            assert key in opt.state[p]
            assert key not in optimizer.state[p]
        for key, value in optimizer.state[p].items():
            assert opt.state[p][key] is value
        observations.append(1)

    handles = [base.register_step_pre_hook(inspect),
               base.register_step_post_hook(inspect)]
    opt.train()
    try:
        for _ in range(3):
            step(opt, [p])
    finally:
        for handle in handles:
            handle.remove()
    assert len(observations) == 6


def test_failing_hook_preserves_buffers(wrapper):
    p = parameter()
    base = torch.optim.SGD([p], lr=.01)
    opt = wrapper(base)
    opt.train()
    step(opt, [p])
    buffers = {key: opt.state[p][key] for key in keys(wrapper)}

    def fail(optimizer, args, kwargs):
        raise RuntimeError('hook failed')

    handle = base.register_step_post_hook(fail)
    try:
        with pytest.raises(RuntimeError, match='hook failed'):
            step(opt, [p])
    finally:
        handle.remove()
    for key, tensor in buffers.items():
        assert opt.state[p][key] is tensor


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_legacy_checkpoint_load_casts_wrapper_buffers(wrapper, dtype):
    # Construct the old flat format independently from wrapper.state_dict().
    p = parameter(torch.float32)
    base = torch.optim.AdamW([p], lr=.01)
    p.grad = p.detach().sin()
    base.step()
    checkpoint = copy.deepcopy(base.state_dict())
    if wrapper is ScheduleFreeWrapperReference:
        checkpoint['param_groups'][0]['weight_decay_at_y'] = 0.
    for key in keys(wrapper):
        checkpoint['state'][0][key] = p.detach().clone()
    original_keys = set(checkpoint['state'][0])
    q = p.detach().to(dtype).requires_grad_()
    resumed = wrapper(torch.optim.AdamW([q], lr=.01))
    resumed.load_state_dict(checkpoint)
    assert set(checkpoint['state'][0]) == original_keys
    for key in keys(wrapper):
        assert resumed.state[q][key].dtype == dtype
        assert resumed.state[q][key].device == q.device
        torch.testing.assert_close(resumed.state[q][key], p.detach().to(dtype))
        assert key not in resumed.base.state[q]
    resumed.train()
    step(resumed, [q])
    assert torch.isfinite(q).all()


def test_checkpoint_with_only_wrapper_state_initializes_base(wrapper):
    p = parameter()
    opt = wrapper(torch.optim.AdamW([p], lr=.01))
    checkpoint = opt.base.state_dict()
    checkpoint['state'][0] = {key: p.detach().clone() for key in keys(wrapper)}
    opt.load_state_dict(checkpoint)
    assert not opt.base.state[p]
    opt.train()
    step(opt, [p])
    assert 'exp_avg' in opt.base.state[p]
    assert opt.base.state[p]['step'].item() == 1


def test_state_dict_preserves_flat_format_without_polluting_base(wrapper):
    p = parameter()
    opt = wrapper(torch.optim.AdamW([p], lr=.01))
    opt.train()
    step(opt, [p])
    before = set(opt.base.state[p])
    checkpoint = opt.state_dict()
    assert set(checkpoint) == {'state', 'param_groups'}
    assert set(checkpoint['state'][0]) == before | set(keys(wrapper))
    assert set(opt.base.state[p]) == before
    # Editing the returned field dictionary must not modify either live state.
    del checkpoint['state'][0]['z']
    assert 'z' in opt.state[p]
    del checkpoint['state'][0]['exp_avg']
    assert 'exp_avg' in opt.base.state[p]


def test_combined_state_writes_and_deletes_reach_correct_storage(wrapper):
    p = parameter()
    opt = wrapper(torch.optim.SGD([p], lr=.01))
    state = opt.state[p]
    z = p.detach().clone()
    state['z'] = z
    state['custom_base_value'] = 3
    assert state.copy() == {'z': z, 'custom_base_value': 3}
    assert opt.base.state[p] == {'custom_base_value': 3}
    assert state.pop('custom_base_value') == 3
    assert not opt.base.state[p]
    opt.state[p] = {'z': z, 'custom_base_value': 4}
    assert opt.state[p]['z'] is z
    assert opt.base.state[p]['custom_base_value'] == 4
    del opt.state[p]
    assert p not in opt.state
    assert p not in opt.base.state


def test_invalid_checkpoint_does_not_discard_wrapper_buffers(wrapper):
    p = parameter()
    opt = wrapper(torch.optim.AdamW([p], lr=.01))
    opt.train()
    step(opt, [p])
    z = opt.state[p]['z']
    checkpoint = opt.state_dict()
    checkpoint['param_groups'] = []
    with pytest.raises(ValueError):
        opt.load_state_dict(checkpoint)
    assert opt.state[p]['z'] is z


def test_state_lookup_helpers_do_not_create_missing_entries(wrapper):
    p = parameter()
    opt = wrapper(torch.optim.SGD([p], lr=.01))
    marker = object()
    assert opt.state.get(p, marker) is marker
    assert opt.state.pop(p, marker) is marker
    assert p not in opt.state
    with pytest.raises(KeyError):
        opt.state.pop(p)
    state = opt.state.setdefault(p, {'custom_base_value': 2})
    assert state['custom_base_value'] == 2
    assert opt.base.state[p]['custom_base_value'] == 2
    assert opt.state.setdefault(p, {'custom_base_value': 3})['custom_base_value'] == 2
    assert opt.state.pop(p) == {'custom_base_value': 2}
    assert p not in opt.state
