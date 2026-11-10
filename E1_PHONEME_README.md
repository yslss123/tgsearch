# E1 — DeepSpeech + frame-aligned Chinese phoneme fusion

This experiment is opt-in and preserves the original TalkingGaussian behavior
when `--use_phoneme` is omitted. It does not implement phoneme phase/duration
modeling (those are separate experiments).

## Changes

- Dataset loader calls the existing `load_phone_csv` and `get_phone_features`
  and puts `phone_ids: LongTensor[8]`, `phone_valid: FloatTensor[8]` into
  each Camera's `talking_dict`.
- The renderer passes these inputs to both the Face and Mouth motion networks.
- Each `AudioNet` produces 8 audio embeddings `[8,32]`. E1 adds a
  masked 16-dimensional phoneme embedding projected to 32 dimensions,
  then uses the **original** `AudioAttNet` to aggregate the 8 audio/phone tokens.
- The projection starts at zero, so before training the E1 network produces
  the same features as the original TG network. During training, phone weights
  learn residual improvements.

## Requirements

- Aligned 25 FPS phone CSV, columns: `frame_idx,audio_time_s,phone`.
- A **fixed** phone vocabulary JSON. For the current dataset use
  `phoneme_input_test/phone_vocab.json` (92 IDs), with the current
  `frame_phone.csv`.
- The CSV must represent the union of train+val frames when training on the
  native original audio. The loader validates these IDs.
- For external audio (`--audio`), provide a **new** CSV aligned to that
  audio's frame indices (not the original training CSV); keep the
  **same vocabulary JSON** from training. New symbols map to `<unk>`
  and are masked by the existing loader.

## Commands

Run commands from the repository root after switching to this branch.

```bash
python -m py_compile arguments/__init__.py scene/dataset_readers.py \
  scene/motion_net.py gaussian_renderer/__init__.py
python tests/test_e1_phoneme_fusion.py
```

E1 Face training (use a **new** output directory, do not overwrite the TG baseline):

```bash
python train_face.py -s data/zh01 -m output/zh01_e1 \
  --use_phoneme --phone_csv frame_phone.csv \
  --phone_vocab phoneme_input_test/phone_vocab.json
```

E1 Mouth training (same new E1 output directory, after face or in a separate run):

```bash
python train_mouth.py -s data/zh01 -m output/zh01_e1 \
  --use_phoneme --phone_csv frame_phone.csv \
  --phone_vocab phoneme_input_test/phone_vocab.json
```

Fuse training (after Face and Mouth checkpoints exist):

```bash
python train_fuse.py -s data/zh01 -m output/zh01_e1 \
  --use_phoneme --phone_csv frame_phone.csv \
  --phone_vocab phoneme_input_test/phone_vocab.json
```

Rendering on native audio, explicitly enable E1 even when `cfg_args` exists:

```bash
python synthesize_fuse.py -s data/zh01 -m output/zh01_e1 --eval \
  --use_phoneme --phone_csv frame_phone.csv \
  --phone_vocab phoneme_input_test/phone_vocab.json
```

For external audio, pass `--audio path/to/new_aud_ds.npy` and a NEW
frame-aligned `--phone_csv path/to/new_frame_phone.csv`. The existing TG
inference still uses the training video camera poses and expressions.

**Checkpoint warning:** E1 adds trainable parameters and optimizer groups.
Do not resume E1 training directly from an acoustic-only TG checkpoint using
`--start_checkpoint`. Start in a fresh output directory. Do not combine
face/mouth checkpoints from different E1 settings.

**Research status:** This is an engineering baseline for testing whether
phoneme information improves lip accuracy. It makes no claim of superior
synchronization or Chinese phoneme generalization until evaluated.
