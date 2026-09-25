# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
from schedulefree import AdEMAMixScheduleFree


def test_foreach_matches_single_tensor():
    torch.manual_seed(46)
    w1 = torch.randn(4, 3)
    w2 = w1.clone()
    o1 = AdEMAMixScheduleFree([w1], lr=1e-3, foreach=True)
    o2 = AdEMAMixScheduleFree([w2], lr=1e-3, foreach=False)
    o1.train()
    o2.train()
    for t in range(40):
        g = torch.randn(4, 3)
        w1.grad = g.clone()
        w2.grad = g.clone()
        o1.step()
        o2.step()
    o1.eval()
    o2.eval()
    assert torch.equal(w1, w2)
    for key in ("z", "exp_avg", "exp_avg_sq", "exp_avg_slow"):
        assert torch.equal(o1.state[w1][key], o2.state[w2][key])
    assert o1.state[w1]["step"] == o2.state[w2]["step"] == 40


def test_step_matches_hand_computation():
    torch.manual_seed(47)
    w = torch.randn(2, 2)
    opt = AdEMAMixScheduleFree(
        [w], lr=1e-3, betas=(0.9, 0.999, 0.9999), eps=1e-8,
        alpha=5.0, foreach=False,
    )
    opt.train()
    g1 = torch.randn(2, 2)
    w.grad = g1.clone()
    before = w.clone()
    z0 = opt.state[w]["z"].clone() if "z" in opt.state[w] else before.clone()
    opt.step()
    # k=0: sched=1, lr=1e-3, lr_max=1e-3, weight=1, weight_sum=1, ckp1=1.
    # y.step1: y.lerp(z, 1) = z0 = before; y += upd*(0.9*0-1) = before - upd.
    m = 0.1 * g1
    v = 0.001 * g1 * g1
    a1, b1 = 1 - 0.9, 1 - 0.999
    s = (1 - 0.9999) * g1
    alpha_t, beta3_t = 5.0, 0.9999
    denom = (v / (1 - 0.999**1)).sqrt() + 1e-8
    upd = ((m + alpha_t * s) / denom) * (1e-3 / (1 - 0.9**1))
    assert torch.allclose(w, before - upd, atol=1e-6)
    assert torch.equal(opt.state[w]["z"], z0 - upd)


def test_alpha_schedule_warms_up():
    w = torch.randn(2, 2)
    opt = AdEMAMixScheduleFree(
        [w], lr=1e-3, alpha=5.0, T_alpha_beta3=10, foreach=False
    )
    got = [
        opt._alpha_beta3(s, 5.0, 0.9, 0.9999, 10) for s in (1, 5, 10, 11)
    ]
    assert got[0][0] == 0.5
    assert got[2][0] == 5.0
    assert got[3][0] == 5.0
    assert got[3][1] == 0.9999


def test_converges_linear_regression():
    torch.manual_seed(48)
    x = torch.randn(40, 4)
    truth = torch.randn(4, 1)
    y = x @ truth
    w = torch.zeros(4, 1, requires_grad=True)
    opt = AdEMAMixScheduleFree([w], lr=0.1, foreach=False)
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
