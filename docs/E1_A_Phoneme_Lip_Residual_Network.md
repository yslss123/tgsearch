# E1-A: Phoneme-conditioned lip residual network

This commit implements the first trainable E1 model component and opt-in phoneme data loading.

## What is included

- `scene/lip_residual_net.py`: 8-frame phoneme context encoder, per-Gaussian spatial encoder, gated feature fusion, and bounded XYZ residual head.
- `tests/test_lip_residual_net.py`: CPU self-test without TG rasterizer/CUDA extensions.
- Optional phoneme attachment in `scene/dataset_readers.py`: enabled only when the dataset args object has `phone_csv`; vanilla TG behavior is unchanged otherwise.

## Model interface

```python
from scene.lip_residual_net import LipResidualMotionNetwork

net = LipResidualMotionNetwork(vocab_size=len(phone_vocab), max_delta=0.003)
out = net(
    xyz=face_gaussians.get_xyz,       # [N, 3]
    tg_d_xyz=face_motion["d_xyz"],    # [N, 3], frozen TG prediction
    phone_ids=view.talking_dict["phone_ids"],       # [8]
    phone_valid=view.talking_dict["phone_valid"],   # [8]
    lip_weight=lip_weights,           # [N] or [N, 1]
    au_exp=view.talking_dict["au_exp"],              # optional [6]
)
delta = out["delta_xyz"]              # [N, 3]
final_xyz = face_gaussians.get_xyz + face_motion["d_xyz"] + lip_weights[:, None] * delta
```

The residual head is zero-initialized, so the branch starts as an identity correction. Its output is bounded by `max_delta` using tanh.

## Data loading

Set `args.phone_csv` before constructing `Scene` in a dedicated training/evaluation script. Optionally set `args.phone_vocab` and `args.phone_fps` (default 25). The existing `data_utils/phoneme_utils.py` remains the source of the fixed vocabulary and 8-frame window.

## Self-test

```bash
python -m tests.test_lip_residual_net
```

This validates tensor shapes, zero-initialized identity behavior, bounded output, gradients, and all-invalid phone windows on CPU.

## Scope and limitations

This is E1-A, not a complete training pipeline. It does not yet load the Fuse checkpoint, render the final Face+Mouth RGB, compute FAN losses, train on the 5,609 training frames, or evaluate generalization. The next implementation step is the training/integration script that freezes the TG branches and optimizes only this residual network through the existing differentiable Fuse RGB/FAN path.
