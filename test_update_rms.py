"""update_rms matches the actual change in the training parameters."""
import torch
from schedulefree import AdamWScheduleFree


def rms_of(before, after):
    sq = sum(((a.detach() - b) ** 2).sum().item() for a, b in zip(after, before))
    n = sum(p.numel() for p in after)
    return (sq / n) ** 0.5


def check(foreach):
    torch.manual_seed(0)
    params = [
        torch.nn.Parameter(torch.randn(32, 16)),
        torch.nn.Parameter(torch.randn(8)),
    ]
    opt = AdamWScheduleFree(params, lr=0.1, weight_decay=0.01, foreach=foreach)
    assert opt.update_rms is None
    opt.train()
    before = [p.detach().clone() for p in params]
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    got = opt.update_rms
    ref = rms_of(before, params)
    assert abs(got - ref) < 1e-6, (foreach, got, ref)
    assert all(torch.isfinite(p).all() for p in params)
    print("ok", foreach, got)


if __name__ == "__main__":
    check(False)
    check(True)
