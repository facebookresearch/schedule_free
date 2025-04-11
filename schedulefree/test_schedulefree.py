# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from schedulefree import (SGDScheduleFree, SGDScheduleFreeClosure, 
    AdamWScheduleFree, AdamWScheduleFreeClosure, AdamWScheduleFreeReference, 
    RAdamScheduleFree, RAdamScheduleFreeClosure,
    ScheduleFreeWrapper, ScheduleFreeWrapperReference, SGDScheduleFreeReference)

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


if __name__ == "__main__":
    torch.manual_seed(1)

    test_compile()

    test_equiv()

    test_schedulefree_wrapper()

    test_schedulefree_wrapper_reference()

    test_foreach()

    test_schedulefree_adam()
    test_schedulefree_sgd()
    test_schedulefree_radam()