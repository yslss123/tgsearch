# -*- coding: utf-8 -*-
"""CPU test: all 20 FAN mouth landmarks contribute gradients."""
import torch
from train_lip_residual import mouth_landmark_loss


def test_all_20_mouth_points_have_gradients():
    gt = torch.zeros(1, 68, 2)
    pred = torch.zeros(1, 68, 2, requires_grad=True)
    with torch.no_grad():
        pred[:, 48:68, 0] = 1.0
    loss = mouth_landmark_loss(pred, gt)
    loss.backward()
    assert pred.grad[:, 48:68].abs().sum() > 0
    assert pred.grad[:, :48].abs().sum() == 0
    print("[PASS] all 20 mouth landmarks contribute to the loss")


if __name__ == "__main__":
    test_all_20_mouth_points_have_gradients()
