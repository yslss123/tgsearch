
"""Load frame-aligned MFA phone labels for TalkingGaussian."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

RESERVED_PHONE_IDS = {
    "<pad>": 0,
    "<unk>": 1,
    "<sil>": 2,
    "spn": 3,
}

PHONE_WINDOW = 8
PHONE_LEFT_CONTEXT = 4


def load_phone_csv(phone_csv, phone_vocab="",
                   expected_frame_ids=None, fps=25.0,
                   time_tolerance=0.021):

    csv_path = Path(phone_csv).expanduser().resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(str(csv_path))

    required = {"frame_idx", "audio_time_s", "phone"}
    rows = {}

    with csv_path.open("r", encoding="utf-8-sig",
                       newline="") as f:
        reader = csv.DictReader(f)

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("CSV missing columns: {}".format(missing))

        for line_no, row in enumerate(reader, start=2):
            try:
                frame_id = int(row["frame_idx"])
                audio_time = float(row["audio_time_s"])
                symbol = (row.get("phone") or "").strip() or "<unk>"
            except Exception as exc:
                raise ValueError(
                    "Invalid CSV row {}: {}".format(line_no, row)
                ) from exc

            if frame_id < 0:
                raise ValueError("Negative frame_idx")

            if frame_id in rows:
                raise ValueError(
                    "Duplicate frame_idx {}".format(frame_id)
                )

            expected_time = frame_id / float(fps)

            if abs(audio_time - expected_time) > time_tolerance:
                raise ValueError(
                    "CSV time/frame mismatch at frame_idx {}: "
                    "audio_time_s={:.6f}, expected {:.6f}".format(
                        frame_id, audio_time, expected_time
                    )
                )

            rows[frame_id] = symbol

    if not rows:
        raise ValueError("CSV contains no frame rows")

    csv_ids = set(rows)
    max_id = max(csv_ids)

    if csv_ids != set(range(max_id + 1)):
        raise ValueError(
            "CSV frame_idx must be continuous from 0"
        )

    # Verify CSV IDs against the union of TG train/val img_ids.
    if expected_frame_ids is not None:
        expected_frame_ids = {
            int(x) for x in expected_frame_ids
        }

        missing = sorted(expected_frame_ids - csv_ids)
        extra = sorted(csv_ids - expected_frame_ids)

        if missing or extra:
            raise ValueError(
                "CSV frame_idx and TG img_id are not one-to-one. "
                "Missing: {}; extra: {}".format(
                    missing[:20], extra[:20]
                )
            )

    vocab_path = (
        Path(phone_vocab).expanduser().resolve()
        if phone_vocab
        else csv_path.parent / "phone_vocab.json"
    )

    symbols = set(rows.values())

    if vocab_path.is_file():
        with vocab_path.open("r", encoding="utf-8") as f:
            vocab = {
                str(k): int(v)
                for k, v in json.load(f).items()
            }

        for symbol, idx in RESERVED_PHONE_IDS.items():
            if vocab.get(symbol) != idx:
                raise ValueError(
                    "Reserved ID mismatch: {}".format(symbol)
                )

        unseen = sorted(
            symbols - set(vocab) - {"<outside>", "<unk>"}
        )

        if unseen:
            print(
                "[Phone][WARN] Unseen symbols mapped to <unk>:",
                unseen[:20]
            )

    else:
        vocab = dict(RESERVED_PHONE_IDS)

        for symbol in sorted(
            symbols - set(vocab) - {"<outside>", "<unk>"}
        ):
            vocab[symbol] = len(vocab)

        vocab_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        with vocab_path.open("w", encoding="utf-8") as f:
            json.dump(
                vocab, f, ensure_ascii=False, indent=2
            )

        print("[Phone] Created vocabulary:", vocab_path)

    num_frames = max_id + 1

    phone_ids = np.zeros(
        (num_frames,), dtype=np.int64
    )

    phone_valid = np.zeros(
        (num_frames,), dtype=np.float32
    )

    spn_frames = 0
    unk_frames = 0

    for frame_id, symbol in rows.items():

        if symbol == "spn":
            phone_ids[frame_id] = RESERVED_PHONE_IDS["spn"]
            phone_valid[frame_id] = 0.0
            spn_frames += 1

        elif symbol in ("<unk>", "<outside>") or symbol not in vocab:
            phone_ids[frame_id] = RESERVED_PHONE_IDS["<unk>"]
            phone_valid[frame_id] = 0.0
            unk_frames += 1

        else:
            phone_ids[frame_id] = vocab[symbol]
            phone_valid[frame_id] = 1.0

    print("[Phone] CSV:", csv_path)
    print("[Phone] Frames:", num_frames)
    print("[Phone] Vocab size:", len(vocab))
    print("[Phone] SPN frames masked:", spn_frames)
    print("[Phone] Unknown frames masked:", unk_frames)

    if expected_frame_ids is not None:
        print(
            "[Phone] Verified exact mapping: "
            "CSV frame_idx == TG img_id"
        )

    return {
        "phone_ids": phone_ids,
        "phone_valid": phone_valid,
        "vocab": vocab,
        "vocab_path": str(vocab_path),
        "num_frames": num_frames,
    }


def get_phone_features(phone_data, frame_id, window=8):

    frame_id = int(frame_id)

    if not 0 <= frame_id < phone_data["num_frames"]:
        raise IndexError(
            "TG img_id {} outside phone CSV range".format(frame_id)
        )

    if window != 8:
        raise ValueError("Phone window must be 8")

    left = frame_id - PHONE_LEFT_CONTEXT
    right = left + window

    ids = np.zeros((window,), dtype=np.int64)
    valid = np.zeros((window,), dtype=np.float32)

    src_left = max(0, left)
    src_right = min(phone_data["num_frames"], right)

    dst_left = src_left - left
    dst_right = dst_left + (src_right - src_left)

    ids[dst_left:dst_right] = (
        phone_data["phone_ids"][src_left:src_right]
    )

    valid[dst_left:dst_right] = (
        phone_data["phone_valid"][src_left:src_right]
    )

    return {
        "phone_ids": torch.from_numpy(ids.copy()),
        "phone_valid": torch.from_numpy(valid.copy()),
    }
