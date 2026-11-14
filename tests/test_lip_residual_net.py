# -*- coding: utf-8 -*-
"""CPU self-tests for LipResidualMotionNetwork; no TG CUDA extensions required."""
import torch
from lip_residual_net import LipResidualMotionNetwork


def run():
    torch.manual_seed(7)
    n, vocab = 257, 92
    model = LipResidualMotionNetwork(vocab_size=vocab, max_delta=0.003)
    xyz = torch.randn(n, 3) * 0.05
    tg = torch.randn(n, 3) * 0.002
    ids = torch.randint(4, vocab, (8,), dtype=torch.long)
    valid = torch.ones(8)
    weight = torch.rand(n)
    au = torch.randn(6)

    out = model(xyz, tg, ids, valid, weight, au, return_gate=True)
    delta, gate = out["delta_xyz"], out["gate"]
    assert delta.shape == (n, 3)
    assert gate.shape == (n, 64)
    assert torch.allclose(delta, torch.zeros_like(delta)), "zero-init must preserve TG"
    assert torch.isfinite(delta).all()

    # Train one step: head gets gradients first; all gradients remain finite.
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = (delta - torch.full_like(delta, 0.001)).pow(2).mean()
    loss.backward()
    assert model.residual_head[-1].weight.grad is not None
    assert torch.isfinite(model.residual_head[-1].weight.grad).all()
    opt.step()

    out2 = model(xyz, tg, ids, valid, weight, au, return_gate=True)
    d2 = out2["delta_xyz"]
    assert d2.shape == (n, 3)
    assert d2.abs().max().item() <= 0.0030001

    # Backpropagate through spatial/phone inputs and ensure encoder gradients
    # become active after the zero-initialized final head has started learning.
    opt.zero_grad()
    target = torch.randn_like(d2) * 0.001
    ((model(xyz, tg, ids, valid, weight, au)["delta_xyz"] - target) ** 2).mean().backward()
    assert model.phone_encoder.embedding.weight.grad is not None
    assert torch.isfinite(model.phone_encoder.embedding.weight.grad).all()

    # Fully invalid windows must be accepted and yield finite outputs.
    invalid = torch.zeros(8)
    out3 = model(xyz, tg, torch.zeros(8, dtype=torch.long), invalid, weight, au)
    assert torch.isfinite(out3["delta_xyz"]).all()
    print("[PASS] shape, zero-init identity, bounded residual, gradients, invalid-window handling")


if __name__ == "__main__":
    run()
