#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# 直接调用 TalkingGaussian 原始数据加载代码
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import cameraList_from_camInfos


RESERVED = {
    "<pad>": 0,
    "<unk>": 1,
    "<sil>": 2,
    "spn": 3
}

WINDOW = 8
LEFT = 4


# ==========================================================
# 1. 读取逐帧音素 CSV
# ==========================================================

def read_phone_csv(path):

    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError("CSV not found: {}".format(path))

    records = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        missing = {"frame_idx", "phone"} - set(reader.fieldnames or [])

        if missing:
            raise ValueError("CSV missing columns: {}".format(missing))

        for line_no, row in enumerate(reader, 2):

            fid = int(row["frame_idx"])
            phone = (row.get("phone") or "").strip() or "<unk>"

            if fid in records:
                raise ValueError("Duplicate frame_idx: {}".format(fid))

            records[fid] = phone

    if not records:
        raise ValueError("CSV has no data")

    if set(records) != set(range(max(records) + 1)):
        raise ValueError("CSV frame_idx must be continuous from 0")

    print("[CSV] Loaded {} frames: 0..{}".format(
        len(records), max(records)
    ))

    return records


# ==========================================================
# 2. 读取 TG 原始 transforms，检查 img_id
# ==========================================================

def read_transform_ids(source):

    split_ids = {}
    all_ids = set()

    for name in (
        "transforms_train.json",
        "transforms_val.json"
    ):

        path = source / name

        if not path.is_file():
            raise FileNotFoundError("Missing TG file: {}".format(path))

        with path.open("r", encoding="utf-8") as f:
            frames = json.load(f)["frames"]

        ids = [int(x["img_id"]) for x in frames]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate img_id in {}".format(name))

        if all_ids.intersection(ids):
            raise ValueError("Train/val img_id overlap")

        split_ids[name] = set(ids)
        all_ids.update(ids)

        print("[TG JSON] {}: {} frames".format(name, len(ids)))

    return split_ids, all_ids


# ==========================================================
# 3. 读取固定音素词典
# ==========================================================

def load_vocab(records, path):

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():

        with path.open("r", encoding="utf-8") as f:
            vocab = {
                str(k): int(v)
                for k, v in json.load(f).items()
            }

        for k, v in RESERVED.items():

            if vocab.get(k) != v:
                raise ValueError("Reserved vocab ID mismatch: {}".format(k))

        unknown = set(records.values()) - set(vocab) - {"<outside>"}

        if unknown:
            raise ValueError(
                "Phones absent from fixed vocab: {}".format(
                    sorted(unknown)[:20]
                )
            )

        print("[VOCAB] Reuse {} ({} entries)".format(
            path, len(vocab)
        ))

    else:

        vocab = dict(RESERVED)

        for p in sorted(
            set(records.values()) - set(vocab) - {"<outside>"}
        ):
            vocab[p] = len(vocab)

        with path.open("w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        print("[VOCAB] Created {} ({} entries)".format(
            path, len(vocab)
        ))

    return vocab


# ==========================================================
# 4. 生成音素 ID 和 8 帧上下文
# ==========================================================

def make_windows(records, vocab):

    n = len(records)

    ids = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=np.float32)

    for fid, phone in records.items():

        if phone == "spn":

            ids[fid] = RESERVED["spn"]
            valid[fid] = 0.0

        elif phone in ("<unk>", "<outside>"):

            ids[fid] = RESERVED["<unk>"]
            valid[fid] = 0.0

        else:

            ids[fid] = vocab[phone]
            valid[fid] = 1.0

    window_ids = np.zeros((n, WINDOW), dtype=np.int64)
    window_valid = np.zeros((n, WINDOW), dtype=np.float32)

    for t in range(n):

        for j in range(WINDOW):

            src = t + j - LEFT

            if 0 <= src < n:

                window_ids[t, j] = ids[src]
                window_valid[t, j] = valid[src]

    return window_ids, window_valid


# ==========================================================
# 5. 将音素加入 TG 原始 CameraInfo.talking_dict
# ==========================================================

def inject_phone(cam_infos, window_ids, window_valid, split):

    seen = set()

    for info in cam_infos:

        td = info.talking_dict

        if "img_id" not in td:
            raise KeyError("TG talking_dict has no img_id")

        fid = int(td["img_id"])

        if fid in seen:
            raise ValueError("Duplicate {} img_id {}".format(split, fid))

        if fid < 0 or fid >= len(window_ids):
            raise IndexError("img_id {} outside CSV range".format(fid))

        seen.add(fid)

        # CPU Tensor，与 TG 原始音频/AU 数据保持一致的存放方式
        td["phone_ids"] = torch.from_numpy(
            window_ids[fid].copy()
        ).long()

        td["phone_valid"] = torch.from_numpy(
            window_valid[fid].copy()
        ).float()

    print("[INJECT] {}: {} CameraInfo objects updated".format(
        split, len(cam_infos)
    ))

    return seen


# ==========================================================
# 6. 检查 TG 真正创建的 Camera 对象
# ==========================================================

def verify_cameras(cameras, expected_ids, split):

    actual_ids = set()

    for cam in cameras:

        td = cam.talking_dict

        fid = int(td["img_id"])

        actual_ids.add(fid)

        if "auds" not in td:
            raise KeyError(
                "TG Camera img_id {} missing original auds".format(fid)
            )

        if "phone_ids" not in td or "phone_valid" not in td:
            raise KeyError(
                "TG Camera img_id {} missing phone data".format(fid)
            )

        pids = td["phone_ids"]
        valid = td["phone_valid"]
        auds = td["auds"]

        if tuple(pids.shape) != (8,) or pids.dtype != torch.long:
            raise TypeError(
                "img_id {} phone_ids must be LongTensor[8]".format(fid)
            )

        if tuple(valid.shape) != (8,) or valid.dtype != torch.float32:
            raise TypeError(
                "img_id {} phone_valid must be FloatTensor[8]".format(fid)
            )

        if not torch.is_tensor(auds) or auds.shape[0] != 8:
            raise TypeError(
                "img_id {} original TG auds must have 8-frame window".format(fid)
            )

    if actual_ids != expected_ids:
        raise AssertionError(
            "{} Camera img_id set mismatch".format(split)
        )

    print(
        "[PASS] {} TG Camera objects: {} frames".format(
            split, len(cameras)
        )
    )

    print(
        "[PASS] Every Camera has auds + phone_ids[8] + phone_valid[8]"
    )


# ==========================================================
# 7. 主程序
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", default="./data/zh01")
    parser.add_argument("--csv", default="./frame_phone.csv")

    parser.add_argument(
        "--audio-extractor",
        default="deepspeech",
        choices=["deepspeech", "esperanto", "hubert"]
    )

    parser.add_argument("--audio", default="")

    parser.add_argument(
        "--vocab",
        default="./phoneme_input_test/phone_vocab.json"
    )

    parser.add_argument(
        "--data-device",
        default="cpu",
        choices=["cpu", "cuda"]
    )

    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip image/mask preloading to reduce RAM"
    )

    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()

    if not source.is_dir():
        raise FileNotFoundError(
            "TG dataset folder not found: {}".format(source)
        )

    # 检查 TG 原始音频特征文件
    if args.audio:

        audio_path = Path(args.audio).expanduser().resolve()

    else:

        postfix = {
            "deepspeech": "ds",
            "esperanto": "eo",
            "hubert": "hu"
        }[args.audio_extractor]

        audio_path = source / "aud_{}.npy".format(postfix)

    if not audio_path.is_file():
        raise FileNotFoundError(
            "TG audio feature file not found: {}".format(audio_path)
        )

    # ------------------------------------------------------
    print("\n========== 1. 检查 CSV 与 TG 帧编号 ==========")

    records = read_phone_csv(args.csv)

    split_ids, all_tg_ids = read_transform_ids(source)

    if set(records) != all_tg_ids:

        raise ValueError(
            "CSV/TG ID mismatch:\n"
            "CSV-only: {}\n"
            "TG-only: {}".format(
                sorted(set(records) - all_tg_ids)[:20],
                sorted(all_tg_ids - set(records))[:20]
            )
        )

    print("[PASS] CSV frame_idx == TG img_id")
    print("[PASS] Total frames:", len(all_tg_ids))

    # ------------------------------------------------------
    print("\n========== 2. 准备音素特征 ==========")

    vocab = load_vocab(records, args.vocab)

    window_ids, window_valid = make_windows(records, vocab)

    print("[PHONE] Vocab size:", len(vocab))

    print(
        "[PHONE] SPN frames:",
        sum(p == "spn" for p in records.values())
    )

    print(
        "[PHONE] Valid frames:",
        int(window_valid[:, 4].sum())
    )

    # ------------------------------------------------------
    print("\n========== 3. 调用 TG 原始数据加载函数 ==========")

    preload = not args.no_preload

    print("[TG] Dataset:", source)
    print("[TG] Audio:", audio_path)
    print("[TG] Preload:", preload)

    # 这里直接调用 TG 的原始 readCamerasFromTransforms
    train_infos = readCamerasFromTransforms(
        str(source),
        "transforms_train.json",
        False,
        ".jpg",
        str(audio_path) if args.audio else "",
        args.audio_extractor,
        preload=preload
    )

    val_infos = readCamerasFromTransforms(
        str(source),
        "transforms_val.json",
        False,
        ".jpg",
        str(audio_path) if args.audio else "",
        args.audio_extractor,
        preload=preload
    )

    if len(train_infos) != len(split_ids["transforms_train.json"]):
        raise RuntimeError("TG train loader frame count mismatch")

    if len(val_infos) != len(split_ids["transforms_val.json"]):
        raise RuntimeError("TG val loader frame count mismatch")

    print("[PASS] TG original loader:")
    print("       train:", len(train_infos))
    print("       val:  ", len(val_infos))

    # ------------------------------------------------------
    print("\n========== 4. 将音素加入 TG talking_dict ==========")

    train_seen = inject_phone(
        train_infos, window_ids, window_valid, "train"
    )

    val_seen = inject_phone(
        val_infos, window_ids, window_valid, "val"
    )

    if train_seen != split_ids["transforms_train.json"]:
        raise AssertionError("Train img_id mismatch")

    if val_seen != split_ids["transforms_val.json"]:
        raise AssertionError("Val img_id mismatch")

    # ------------------------------------------------------
    print("\n========== 5. 调用 TG 原始 Camera 创建函数 ==========")

    # TG 的 Camera.__init__ 内部无条件调用 .cuda()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TG Camera requires CUDA. "
            "Please run in the CUDA-enabled talking_gaussian environment."
        )

    camera_args = SimpleNamespace(
        data_device=args.data_device
    )

    train_cameras = cameraList_from_camInfos(
        train_infos, 1.0, camera_args
    )

    val_cameras = cameraList_from_camInfos(
        val_infos, 1.0, camera_args
    )

    verify_cameras(
        train_cameras,
        split_ids["transforms_train.json"],
        "train"
    )

    verify_cameras(
        val_cameras,
        split_ids["transforms_val.json"],
        "val"
    )

    # ------------------------------------------------------
    print("\n========== 6. 检查真实 Camera.talking_dict ==========")

    cameras_by_id = {
        int(c.talking_dict["img_id"]): c
        for c in train_cameras + val_cameras
    }

    sample_ids = sorted(set([
        0,
        10,
        len(records) // 2,
        len(records) - 1
    ]))

    for fid in sample_ids:

        cam = cameras_by_id[fid]
        td = cam.talking_dict

        print("\nimg_id:", fid)
        print("Camera.uid:", cam.uid)
        print("auds.shape:", tuple(td["auds"].shape))
        print("phone_ids:", td["phone_ids"].tolist())
        print(
            "phone_valid:",
            td["phone_valid"].to(torch.int).tolist()
        )

    # ------------------------------------------------------
    print("\n========== 测试结果 ==========")

    print("[PASS] TG 原始 readCamerasFromTransforms 已执行")
    print("[PASS] TG 原始 cameraList_from_camInfos 已执行")
    print("[PASS] train/val Camera 均成功创建")
    print("[PASS] 每个 Camera.talking_dict 均包含原始 auds")
    print("[PASS] 每个 Camera.talking_dict 均包含 phone_ids")
    print("[PASS] 每个 Camera.talking_dict 均包含 phone_valid")
    print("[PASS] 音素与 img_id 一一对应")

    print("\n[STOP] 没有调用以下模块：")
    print("       GaussianModel")
    print("       MotionNetwork")
    print("       MouthMotionNetwork")
    print("       gaussian_renderer")
    print("       train_mouth")

    print("\n[NOTE] 本次只在内存中添加音素，没有修改 TG 源码或数据集文件。")


if __name__ == "__main__":
    main()

