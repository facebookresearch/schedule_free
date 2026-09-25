# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from schedulefree import AdoptScheduleFree


def _drive(opt, weight, steps, gen):
    opt.train()
    for _ in range(steps):
        weight.grad = gen(weight).clone()
        opt.step()
    opt.eval()


def test_foreach_matches_single_tensor():
    torch.manual_seed(61)
    w1 = torch.randn(4, 3)
    w2 = w1.clone()
    o1 = AdoptScheduleFree([w1], lr=1e-3, foreach=True)
    o2 = AdoptScheduleFree([w2], lr=1e-3, foreach=False)
    g = torch.randn(4, 3)
    o1.train()
    o2.train()
    for _ in range(50):
        w1.grad = g.clone()
        w2.grad = g.clone()
        o1.step()
        o2.step()
    o1.eval()
    o2.eval()
    assert torch.equal(w1, w2)
    assert torch.equal(o1.state[w1]["z"], o2.state[w2]["z"])
    assert torch.equal(o1.state[w1]["exp_avg"], o2.state[w2]["exp_avg"])
    assert torch.equal(o1.state[w1]["exp_avg_sq"], o2.state[w2]["exp_avg_sq"])


def test_first_step_initializes_only():
    w = torch.randn(3, 2)
    before = w.clone()
    opt = AdoptScheduleFree([w], lr=1e-3, foreach=False)
    opt.train()
    w.grad = torch.randn(3, 2)
    opt.step()
    assert torch.equal(w, before)
    assert opt.state[w]["step"] == 1
    assert torch.equal(
        opt.state[w]["z"], before
    )


def test_second_step_matches_hand_computation():
    torch.manual_seed(62)
    w = torch.randn(2, 2)
    opt = AdoptScheduleFree(
        [w], lr=1e-3, betas=(0.9, 0.9999), eps=1e-6, foreach=False
    )
    opt.train()
    g1 = torch.randn(2, 2)
    w.grad = g1.clone()
    opt.step()
    g2 = torch.randn(2, 2)
    w.grad = g2.clone()
    before = w.clone()
    z0 = opt.state[w]["z"].clone()
    opt.step()
    # Hand computation. k=1 here: sched=1, lr=1e-3, lr_max=1e-3,
    # weight=1, weight_sum=2, ckp1=0.5.
    v = g1 * g1
    denom = v.sqrt().clamp(min=1e-6)
    normed = (g2 / denom).clamp(-(1**0.25), 1**0.25)
    m = 0.1 * normed
    y = before.lerp(z0, 0.5) + 1e-3 * (0.9 * 0.5 - 1) * m
    assert torch.allclose(w, y, atol=1e-6)
    assert torch.equal(
        opt.state[w]["z"], z0 - 1e-3 * m
    )


def test_converges_linear_regression():
    torch.manual_seed(63)
    x = torch.randn(40, 4)
    truth = torch.randn(4, 1)
    y = x @ truth
    w = torch.zeros(4, 1, requires_grad=True)
    opt = AdoptScheduleFree([w], lr=0.1, foreach=False)
    opt.train()
    with torch.no_grad():
        init = ((x @ w - y) ** 2).mean().item()
    for _ in range(300):
        opt.zero_grad()
        loss = ((x @ w - y) ** 2).mean()
        loss.backward()
        opt.step()
        prev = loss.item()
    opt.eval()
    # Sign-like updates orbit the optimum instead of settling into it.
    assert prev < init / 100
    assert prev < 0.01
