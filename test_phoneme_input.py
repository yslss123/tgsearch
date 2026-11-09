import os
import csv
import json
import argparse
import numpy as np

from pathlib import Path
from collections import Counter


RESERVED = {
    "<pad>": 0,
    "<unk>": 1,
    "<sil>": 2,
    "spn": 3
}


def read_csv_file(path):

    records = {}

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required = {"frame_idx", "audio_time_s", "phone"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV缺少必要字段")

        for row in reader:

            idx = int(row["frame_idx"])
            phone = (row["phone"] or "").strip()

            if not phone:
                phone = "<unk>"

            if idx in records:
                raise ValueError("重复 frame_idx: {}".format(idx))

            records[idx] = {
                "time": float(row["audio_time_s"]),
                "phone": phone,
                "word": row.get("word", "")
            }

    if not records:
        raise ValueError("CSV没有数据")

    expected = set(range(max(records) + 1))

    if set(records) != expected:
        raise ValueError("frame_idx不连续，无法直接映射")

    return records


def read_tg_ids(dataset):

    expected_ids = set()

    for filename in [
        "transforms_train.json",
        "transforms_val.json",
        "transforms_test.json"
    ]:

        path = Path(dataset) / filename

        if not path.exists():
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for frame in data["frames"]:

            if "img_id" not in frame:
                raise ValueError(
                    "{} 缺少 img_id".format(filename)
                )

            idx = int(frame["img_id"])

            if idx in expected_ids:
                raise ValueError(
                    "不同数据划分出现重复 img_id: {}".format(idx)
                )

            expected_ids.add(idx)

        print(
            "[TG] {}: {} frames".format(
                filename, len(data["frames"])
            )
        )

    if not expected_ids:
        raise ValueError("没有找到TG transforms文件")

    return expected_ids


def build_features(records, out_dir):

    n = len(records)

    symbols = {
        item["phone"] for item in records.values()
    }

    vocab = dict(RESERVED)

    for symbol in sorted(
        symbols - set(vocab) - {"<outside>", "<unk>"}
    ):
        vocab[symbol] = len(vocab)

    phone_ids = np.zeros(n, dtype=np.int64)
    phone_valid = np.zeros(n, dtype=np.float32)

    for idx, item in records.items():

        phone = item["phone"]

        if phone == "spn":
            phone_ids[idx] = RESERVED["spn"]
            phone_valid[idx] = 0

        elif phone in ("<unk>", "<outside>"):
            phone_ids[idx] = RESERVED["<unk>"]
            phone_valid[idx] = 0

        elif phone in vocab:
            phone_ids[idx] = vocab[phone]
            phone_valid[idx] = 1

        else:
            phone_ids[idx] = RESERVED["<unk>"]
            phone_valid[idx] = 0

    # 生成8帧窗口：t-4 到 t+3
    window_ids = np.zeros((n, 8), dtype=np.int64)
    window_valid = np.zeros((n, 8), dtype=np.float32)

    for t in range(n):

        for j in range(8):

            source_idx = t + j - 4

            if 0 <= source_idx < n:

                window_ids[t, j] = phone_ids[source_idx]

                window_valid[t, j] = phone_valid[source_idx]

    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "phone_ids.npy", phone_ids)
    np.save(out_dir / "phone_valid.npy", phone_valid)

    np.save(out_dir / "phone_window_ids.npy", window_ids)
    np.save(out_dir / "phone_window_valid.npy", window_valid)

    with open(
        out_dir / "phone_vocab.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 输出人工检查表
    reverse_vocab = {
        v: k for k, v in vocab.items()
    }

    with open(
        out_dir / "phone_preview.csv",
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "frame_idx",
            "audio_time_s",
            "word",
            "phone",
            "phone_id",
            "phone_valid",
            "window_labels",
            "window_valid"
        ])

        for t in range(n):

            labels = [
                reverse_vocab[int(x)]
                for x in window_ids[t]
            ]

            writer.writerow([
                t,
                records[t]["time"],
                records[t]["word"],
                records[t]["phone"],
                int(phone_ids[t]),
                int(phone_valid[t]),
                " ".join(labels),
                " ".join(
                    map(str, window_valid[t].astype(int))
                )
            ])

    return phone_ids, phone_valid, window_ids, window_valid, vocab


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="./frame_phone.csv"
    )

    parser.add_argument(
        "--dataset",
        default="./data/zh01"
    )

    parser.add_argument(
        "--out",
        default="./phoneme_input_test"
    )

    args = parser.parse_args()

    print("\n========== 1. 读取CSV ==========")

    records = read_csv_file(args.csv)

    print("[CSV] 总帧数:", len(records))
    print("[CSV] 第一帧:", records[0])
    print("[CSV] 最后一帧:", records[len(records)-1])

    # 检查25 FPS时间间隔
    times = np.array([
        records[i]["time"] for i in range(len(records))
    ])

    if len(times) > 1:

        diffs = np.diff(times)

        abnormal = np.where(
            np.abs(diffs - 0.04) > 0.005
        )[0]

        print("[CSV] 非正常时间间隔数量:", len(abnormal))

        if len(abnormal):
            print("[WARN] 时间间隔需要检查")
        else:
            print("[PASS] 25 FPS时间间隔正常")

    print("\n========== 2. 检查TG帧编号 ==========")

    tg_ids = read_tg_ids(args.dataset)

    csv_ids = set(records.keys())

    missing = tg_ids - csv_ids
    extra = csv_ids - tg_ids

    if missing or extra:

        print("[FAIL] CSV和TG img_id不完全对应")
        print("TG缺少的CSV编号:", sorted(missing)[:20])
        print("CSV多出的编号:", sorted(extra)[:20])

        raise ValueError("帧编号不匹配，停止测试")

    print("[PASS] CSV frame_idx == TG img_id")
    print("[PASS] 一一对应，总帧数:", len(tg_ids))

    print("\n========== 3. 音素编码 ==========")

    out_dir = Path(args.out)

    ids, valid, window_ids, window_valid, vocab = \
        build_features(records, out_dir)

    counts = Counter(
        item["phone"] for item in records.values()
    )

    print("[PHONE] 音素类别数量:", len(vocab))
    print("[PHONE] SPN帧数:", counts.get("spn", 0))
    print("[PHONE] 有效音素帧数:", int(valid.sum()))
    print("[PHONE] SIL帧数:", counts.get("<sil>", 0))

    print("\n========== 4. 检查时间窗口 ==========")

    for t in sorted(set([
        0,
        10,
        len(ids)//4,
        len(ids)//2,
        len(ids)-1
    ])):

        print("\nFrame:", t)
        print("当前音素:", records[t]["phone"])
        print("当前ID:", ids[t])
        print("当前Valid:", valid[t])
        print("8帧ID:", window_ids[t].tolist())
        print(
            "8帧Valid:",
            window_valid[t].astype(int).tolist()
        )

    print("\n========== 5. 检查Tensor ==========")

    try:
        import torch

        x = torch.from_numpy(window_ids)
        y = torch.from_numpy(window_valid)

        print("phone_ids:", x.shape, x.dtype)
        print("phone_valid:", y.shape, y.dtype)

        assert x.shape == (len(ids), 8)
        assert y.shape == (len(ids), 8)

        print("[PASS] PyTorch Tensor转换正常")

    except ImportError:
        print("[WARN] 当前环境没有PyTorch，跳过Tensor测试")

    print("\n========== 测试完成 ==========")

    print("输出目录:", out_dir.resolve())
    print("phone_ids.npy:", ids.shape)
    print("phone_valid.npy:", valid.shape)
    print("phone_window_ids.npy:", window_ids.shape)
    print("phone_window_valid.npy:", window_valid.shape)

    print("\n[PASS] 音素数据预处理测试成功")


if __name__ == "__main__":
    main()
