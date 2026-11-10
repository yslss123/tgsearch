"""Generic phoneme-phase features for E2; independent of specific phoneme labels.

Returns [progress, onset_proximity, offset_proximity, log_duration].
Phases are computed from contiguous runs in the *full* aligned CSV sequence,
never separately from shuffled train/validation subsets.
"""
import math
import numpy as np
import torch


def compute_phone_phase(phone_data):
    ids = np.asarray(phone_data["phone_ids"], dtype=np.int64)
    valid = np.asarray(phone_data["phone_valid"], dtype=np.float32)
    phase = np.zeros((len(ids), 4), dtype=np.float32)
    i = 0
    while i < len(ids):
        j = i + 1
        while j < len(ids) and ids[j] == ids[i] and valid[j] == valid[i]:
            j += 1
        # The reserved classes pad/unknown/silence/spn do not carry articulation.
        if valid[i] > 0 and ids[i] >= 4:
            n = j - i
            k = np.arange(n, dtype=np.float32)
            phase[i:j, 0] = (k + 0.5) / n
            phase[i:j, 1] = np.exp(-k / 2.0)
            phase[i:j, 2] = np.exp(-(n - 1 - k) / 2.0)
            phase[i:j, 3] = min(1.0, math.log1p(n) / math.log1p(50.0))
        i = j
    return phase


def get_phone_phase_window(phone_phase, frame_id, window=8, left_context=4):
    if window != 8 or left_context != 4:
        raise ValueError("E2 requires TG's 8-frame window and left context 4")
    frame_id = int(frame_id)
    if frame_id < 0 or frame_id >= len(phone_phase):
        raise IndexError("Frame {} out of range".format(frame_id))
    left = frame_id - left_context
    right = left + window
    out = np.zeros((window, 4), dtype=np.float32)
    lo, hi = max(left, 0), min(right, len(phone_phase))
    out[lo-left:hi-left] = phone_phase[lo:hi]
    return torch.from_numpy(out.copy())
