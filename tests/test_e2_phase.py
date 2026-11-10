"""CPU-only tests for generic E2 phase features and temporal learning.

Run from project root: python tests/test_e2_phase.py
"""
import numpy as np
import torch

from data_utils.phoneme_phase import compute_phone_phase, get_phone_phase_window
from scene.motion_net import AudioAttNet, PhaseAwareTemporal


def test_phase():
    # Includes two phones, a two-frame phoneme, silence, and a masked SPN.
    ids = np.array([2, 4, 4, 4, 5, 5, 2, 3, 6, 6, 6, 6])
    valid = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.float32)
    phase = compute_phone_phase({"phone_ids": ids, "phone_valid": valid})
    assert phase.shape == (len(ids), 4)
    assert not phase[0].any() and not phase[6].any() and not phase[7].any()
    assert np.all(np.diff(phase[1:4, 0]) > 0)
    assert phase[1, 1] > phase[3, 1]  # strongest onset at first frame
    assert phase[3, 2] > phase[1, 2]  # strongest offset at last frame
    assert phase[4, 1] == 1 and phase[5, 2] == 1
    window = get_phone_phase_window(phase, 0)
    assert window.shape == (8, 4)
    assert torch.all(window[:4] == 0)
    assert torch.allclose(window[5], torch.from_numpy(phase[1]))
    assert torch.allclose(get_phone_phase_window(phase, 8)[4], torch.from_numpy(phase[8]))


def test_module():
    torch.manual_seed(42)
    attention = AudioAttNet(dim_aud=32, seq_len=8)
    stage = PhaseAwareTemporal(dim_aud=32)
    tokens = torch.randn(8, 32, requires_grad=True)
    phase = torch.tensor([
        [0., 0., 0., 0.],
        [0.2, 1., 0.1, 0.6],
        [0.4, 0.7, 0.2, 0.6],
        [0.6, 0.3, 0.7, 0.6],
        [0.8, 0.1, 1., 0.6],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
    ], dtype=torch.float32)
    valid = torch.ones(8)
    modified, bias, gain = stage(tokens, phase, valid)
    output = attention(modified.unsqueeze(0), bias)
    output = output + gain * (modified[4].unsqueeze(0) - output)
    reference = attention(tokens.unsqueeze(0))
    assert torch.allclose(output, reference, atol=1e-6), "E2 must start as E1"
    loss = output.square().sum()
    loss.backward()
    assert stage.phase_net[-1].weight.grad is not None
    assert stage.logit_net.weight.grad is not None
    assert stage.center_scale.grad is not None
    assert stage.phase_net[-1].weight.grad.abs().sum() > 0
    assert stage.logit_net.weight.grad.abs().sum() > 0
    assert stage.center_scale.grad.abs().sum() > 0

    with torch.no_grad():
        stage.phase_net[-1].weight.fill_(0.03)
        stage.logit_net.weight.fill_(0.02)
        stage.center_scale.fill_(0.15)
        modified, bias, gain = stage(tokens, phase, valid)
        output = attention(modified.unsqueeze(0), bias)
        output = output + gain * (modified[4].unsqueeze(0) - output)
        assert (output - reference).abs().max() > 1e-6

    # Completely inactive phase should have no impact, even after training.
    with torch.no_grad():
        zeros = torch.zeros_like(phase)
        modified, bias, gain = stage(tokens, zeros, valid)
        output = attention(modified.unsqueeze(0), bias)
        output = output + gain * (modified[4].unsqueeze(0) - output)
        assert torch.allclose(output, reference, atol=1e-6)


if __name__ == "__main__":
    test_phase()
    test_module()
    print("[PASS] E2 phase boundaries, silence masking, identity init, and gradients")
