#!/usr/bin/env python3
"""CPU smoke tests for E1 fusion without importing CUDA hash-grid extensions."""
import ast
from pathlib import Path

import torch
from torch import nn


def load_fusion_class():
    src = Path(__file__).resolve().parents[1] / "scene" / "motion_net.py"
    tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
    target = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "PhonemeFusion"
    )
    module = ast.Module(body=[target], type_ignores=[])
    namespace = {"nn": nn, "torch": torch}
    exec(compile(module, str(src), "exec"), namespace)
    return namespace["PhonemeFusion"]


def main():
    torch.manual_seed(0)
    Fusion = load_fusion_class()
    net = Fusion(num_phones=92, audio_dim=32)
    audio = torch.randn(8, 32)
    ids = torch.tensor([0, 5, 15, 3, 8, 8, 0, 91], dtype=torch.long)
    mask = torch.tensor([0, 1, 1, 0, 1, 1, 0, 1], dtype=torch.float32)

    # Zero-initialized projection must leave the original TG audio unchanged.
    out = net(audio, ids, mask)
    assert out.shape == (8, 32)
    assert torch.equal(out, audio), "E1 must initially be an identity residual"

    with torch.no_grad():
        net.phone_proj.weight.fill_(0.01)
    out = net(audio, ids, mask)
    assert torch.equal(out[mask == 0], audio[mask == 0]), "Masked phones changed audio"
    assert not torch.equal(out[mask == 1], audio[mask == 1]), "Valid phones had no effect"

    loss = out.sum()
    loss.backward()
    assert net.phone_proj.weight.grad is not None
    assert net.phone_proj.weight.grad.abs().sum().item() > 0
    assert net.phone_embedding.weight.grad is not None

    try:
        net(audio, ids[:-1], mask)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a shape error for bad phone_ids")

    print("[PASS] E1 CPU fusion: identity init, masks, gradient, shape checks")


if __name__ == "__main__":
    main()
