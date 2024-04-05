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

if __name__ == "__main__":
    test_schedulefree_adam()
    test_schedulefree_sgd()