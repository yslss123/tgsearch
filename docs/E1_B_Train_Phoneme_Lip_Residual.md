# E1-B: Train phoneme-conditioned Gaussian lip residual

## What this stage trains

train_lip_residual.py loads the original Fuse checkpoint, S1 alpha-contribution weights, frame-aligned phone labels, and FAN. It freezes the original Face/Mouth Gaussian models and both original TG motion networks. Only LipResidualMotionNetwork is optimized.

The forward path is:
1. Original Face MotionNetwork predicts the base Face Gaussian motion from TG audio/AU.
2. The residual network receives canonical Face Gaussian XYZ, frozen TG base XYZ motion, S1 lip contribution weight, an 8-frame phone context, and optional AU.
3. Its bounded per-Gaussian XYZ residual is multiplied by the S1 lip weight and added to frozen TG Face positions.
4. The frozen Mouth branch is rendered unchanged and composited with the original Fuse alpha formula.
5. The final RGB is compared with the same-frame GT mouth ROI and passed through frozen FAN.

## Loss

- L_geo: S2b mouth height/width loss from FAN landmarks 48, 54, 62, 66.
- L_landmark: normalized SmoothL1 over all 20 FAN mouth landmarks 48..67.
- L_rgb: Charbonnier RGB loss on the same GT-defined mouth ROI.
- L_delta: regularizer on the actually applied lip-weighted displacement.

The FAN target is FAN output on GT RGB. The .lms points define the fixed same-frame crop and audit FAN calibration; frames above --max-gt-fan-error are skipped. The crop is only for paired training/validation and must not be used as a future-frame inference input.

## CPU checks

    python train_lip_residual.py --self-test
    PYTHONPATH=. python tests/test_lip_residual_losses.py

These checks do not load TG CUDA extensions, FAN weights, or test the real rasterizer gradient.

## Real CUDA smoke test

    python train_lip_residual.py \
      -s data/zh01 -m output/zh02 \
      --weights output/lip_stage1_contributions/lip_contribution_weights.npz \
      --phone-csv frame_phone.csv \
      --out output/lip_residual_smoke \
      --iterations 10 --eval-every 5 --save-every 5 --val-frames 2

This tests the RTX 3090, TG rasterizer, FAN, phone CSV, S1 weights, and final Fuse RGB path together. Inspect the logs and output/lip_residual_smoke/previews/. The initial residual should be near zero because the residual head is zero-initialized.

## Short training after smoke passes

    python train_lip_residual.py \
      -s data/zh01 -m output/zh02 \
      --weights output/lip_stage1_contributions/lip_contribution_weights.npz \
      --phone-csv frame_phone.csv \
      --out output/lip_residual_e1 \
      --iterations 1000 --eval-every 100 --save-every 100 --val-frames 6

Resume:

    python train_lip_residual.py \
      -s data/zh01 -m output/zh02 \
      --weights output/lip_stage1_contributions/lip_contribution_weights.npz \
      --phone-csv frame_phone.csv \
      --out output/lip_residual_e1 \
      --iterations 2000 --resume output/lip_residual_e1/lip_residual_latest.pth

## Outputs

- lip_residual_latest.pth: latest residual model and optimizer state.
- lip_residual_best.pth: lowest mean validation total loss.
- train_metrics.csv: per-step losses, residual magnitude, and gradient diagnostics.
- val_metrics.jsonl: baseline-vs-trained validation metrics.
- previews/step_*_frame_*.jpg: GT, frozen TG baseline, and trained Fuse RGB with all 20 FAN mouth points.

The original Fuse checkpoint is read-only and is never overwritten.
