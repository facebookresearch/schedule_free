# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from schedulefree import SGDScheduleFree, SGDScheduleFreeClosure, AdamWScheduleFree, AdamWScheduleFreeClosure

def test_schedulefree_sgd():
    decay = 0.5
    warmup = 5
    weight_closure = torch.randn(3, 2).cuda().requires_grad_()
    weight = torch.clone(weight_closure.data).requires_grad_()
    optimizer_closure = SGDScheduleFreeClosure([weight_closure], lr=0.3, warmup_steps=warmup, weight_decay=decay)
    optimizer = SGDScheduleFree([weight], lr=0.3, warmup_steps=warmup, weight_decay=decay)

    for step_idx in range(50):
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

                assert torch.allclose(p, p_closure)

                z_closure = state_closure['z']
                z = state['z']
                assert torch.allclose(z, z_closure)
 
def test_schedulefree_adam():
    decay = 0.5
    warmup = 5
    weight_closure = torch.randn(3, 2).cuda().requires_grad_()
    weight = torch.clone(weight_closure.data).requires_grad_()
    optimizer_closure = AdamWScheduleFreeClosure([weight_closure], lr=0.3, warmup_steps=warmup, weight_decay=decay)
    optimizer = AdamWScheduleFree([weight], lr=0.3, warmup_steps=warmup, weight_decay=decay)

    for step_idx in range(50):
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

                #print(f"p: {p}")
                #print(f"p: {p_closure}")


                #print(f"z: {z}")
                #print(f"z: {z_closure}")

                assert torch.allclose(p, p_closure)
                assert torch.allclose(z, z_closure)
 
        optimizer.train()

        for group_closure, group in zip(optimizer_closure.param_groups, optimizer.param_groups):
            for p_closure, p in zip(group_closure['params'], group['params']):
                state_closure = optimizer_closure.state[p_closure]
                state = optimizer.state[p]

                z_closure = state_closure['z']

                # Extrapolate p.data to equal y
                y = p.data
                y_closure = p_closure.lerp(end=z_closure, weight=1-0.9)

                #print(f"y: {y}")
                #print(f"y closure: {y_closure}")

                assert torch.allclose(y, y_closure)

def test_foreach():
    decay = 0.5
    warmup = 5
    weight_foreach = torch.randn(3, 2).cuda().requires_grad_()
    weight_foreach2 = torch.randn(1, 1).cuda().requires_grad_()
    weight_foreach_nograd = torch.randn(1, 2).cuda().requires_grad_()

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

    for step_idx in range(50):
        optimizer.train()
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
    
        optimizer.train()
        optimizer_foreach.train()


if __name__ == "__main__":
    test_foreach()

    test_schedulefree_adam()
    test_schedulefree_sgd()