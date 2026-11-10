# E2 — Generic phoneme-phase-aware lip-motion training

This is an **experimental** extension of E1, designed to test whether
short onset/offset articulatory motions are weakened by the original
8-frame temporal pooling. It does **not** hardcode /f/, /w/, or any other
phoneme label, and no improvement is claimed before a controlled evaluation.

## Source branch

- Base: `exp/e1-phoneme-fusion-20261110`
- E2: `exp/e2-phase-aware-onset-20261110`
- TG / E1 behavior remains unchanged **without** `--use_phone_phase`.
- No CUDA, rasterizer, Gaussian representation, HashGrid, or DeepSpeech
  implementation changed.

## Changes

1. `data_utils/phoneme_phase.py` computes each phoneme *run* in the original
   whole-audio frame sequence and associates each frame with 4 values:
   [relative_progress, onset_proximity, offset_proximity, log_duration].
   Reserved labels (<sil>, spn, <pad>, <unk>) produce zero features.
   Note: consecutive identical labels cannot be split into distinct
   occurrences unless the source CSV has explicit boundaries.
2. `scene/dataset_readers.py` loads these fixed [8,4] phase windows for
   both train and val cameras, aligned to the existing E1 [8] phone IDs.
3. `scene/motion_net.py` adds `PhaseAwareTemporal` before `AudioAttNet`:
   - phase-to-token residual;
   - phase-conditioned attention-logit bias, explicitly changing the
     time aggregation weights;
   - a trainable, zero-initialized central-frame preservation path.
   All new outputs are zero at initialization: fixed E1 model weights
   produce identical outputs when E2 is first enabled.
4. `gaussian_renderer/__init__.py` feeds the optional phase window into
   Face and Mouth; rendering itself is unchanged.
5. `train_face.py` adds an optional *outer-lip* L1 loss weighted by the
   central frame's generic phoneme onset strength. Interior Mouth pixels
   are excluded from this Face-only term.
6. `arguments/__init__.py` adds `--use_phone_phase` and
   `--e2_lip_weight` (default 0.02).

The phase module is attached to both Face and Mouth optimizers through
`get_params()`. The Fuse stage still freezes both motion networks.

## Assumptions and limits

- The current dataset's `frame_phone.csv` represents a frame-aligned
  25 FPS source audio and uses the same 92-entry vocabulary as E1.
- E2 phase durations require *knowledge of the end of each phoneme*.
  This offline training/inference setup is **not yet a streaming real-time
  method**; later streaming adaptation would require predictions or
  bounded look-ahead.
- `wen` does not necessarily contain a separate `w` token in the phone
  CSV. This module uses whatever phoneme labeling exists instead of
  assuming a Pinyin-to-phoneme one-to-one mapping.
- The lip region is taken from TG's existing per-frame `lips_rect`.
  If landmarks/parsing are inaccurate, onset-lip loss can hurt output.
- Although new components start at zero for fixed E1 weights, training
  E2 **from scratch** is not equivalent to fine-tuning E1. Use independent
  runs for proper comparisons.

## Test commands (server)

Run from the repo root, in the existing `talking_gaussian` Conda env.

```bash
export CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9

python -m py_compile \
    arguments/__init__.py \
    data_utils/phoneme_phase.py \
    scene/dataset_readers.py \
    scene/motion_net.py \
    gaussian_renderer/__init__.py \
    train_face.py

python tests/test_e1_phoneme_fusion.py
python tests/test_e2_phase.py
```

## Fresh E2 training

**Do not reuse output/zh01_e1 or output/zh02.**

```bash
python train_face.py -s data/zh01 -m output/zh01_e2 \
  --use_phoneme --use_phone_phase \
  --phone_csv frame_phone.csv \
  --phone_vocab phoneme_input_test/phone_vocab.json \
  --e2_lip_weight 0.02

python train_mouth.py -s data/zh01 -m output/zh01_e2 \
  --use_phoneme --use_phone_phase \
  --phone_csv frame_phone.csv \
  --phone_vocab phoneme_input_test/phone_vocab.json

python train_fuse.py -s data/zh01 -m output/zh01_e2 \
  --use_phoneme --use_phone_phase \
  --phone_csv frame_phone.csv \
  --phone_vocab phoneme_input_test/phone_vocab.json
```

To isolate phase aggregation from the onset-lip loss, run a separate E2
experiment with `--e2_lip_weight 0` in Face training; keep `--use_phone_phase`.
To isolate lip weighting from stage-aware temporal modeling, a separate
ablative version could use the onset-weighted loss but disable the E2 module
in code; this is not provided as an existing CLI switch.

## Synthesis after training

```bash
python synthesize_fuse.py -s data/zh01 -m output/zh01_e2 --eval \
  --use_phoneme --use_phone_phase \
  --phone_csv frame_phone.csv \
  --phone_vocab phoneme_input_test/phone_vocab.json
```

Test video: `output/zh01_e2/test/ours_None/renders/out.mp4` (actual
folder may depend on the Scene iteration number). Uses the existing
video's poses/expressions, not yet audio-only streaming.

## Controlled evaluation

E0 = original TG `output/zh02`; E1 = basic phone fusion
`output/zh01_e1`; E2 = phase aggregation + onset loss
`output/zh01_e2`. The current 561 validation frames are the end of
the original 6170-frame sequence. Use those identical frames and GT
for paired comparison; also test independent Chinese utterances before
claiming generalization.

Recommended analysis: lip crop L1/PSNR, onset-specific lip L1, plots
of lip landmarks and per-frame errors, qualitative closeups for *many*
phoneme types. A phoneme's acoustic label is not itself a perfect
articulatory action label.

## Checkpoints

E2 adds new parameter groups. E1 checkpoint + optimizer state is not
compatible with `--start_checkpoint` as-is. Start a fresh E2 run;
do not combine E1/E2 face and mouth checkpoints. Keep all existing
E0/E1 checkpoints and directories unchanged.
