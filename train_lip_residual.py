#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train phoneme-conditioned lip Gaussian residuals through frozen TG Fuse RGB.

Only LipResidualMotionNetwork is optimized. The original Face/Mouth Gaussian
models, both original TG motion networks, and FAN are frozen.
"""
import argparse
import copy
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data_utils.phoneme_utils import load_phone_csv
from scene.lip_residual_net import LipResidualMotionNetwork
from tg_lip_stage2b_rgb_fan_probe import (
    face_bbox, mouth_bbox, fan_landmarks, mouth_geometry, geometry_loss,
    load_fan, render_final_fuse,
)
from tg_lip_stage2_geometry_probe import choose_frames


def mouth_landmark_loss(pred_points, gt_points):
    """Normalized SmoothL1 on all 20 FAN mouth landmarks, indices 48..67."""
    if pred_points.shape != gt_points.shape or pred_points.shape[-2:] != (68, 2):
        raise ValueError("Expected matching FAN landmarks [B,68,2]")
    width = torch.linalg.norm(
        gt_points[:, 54] - gt_points[:, 48], dim=-1
    ).detach().clamp_min(8.0)
    diff = (pred_points[:, 48:68] - gt_points[:, 48:68]) / width[:, None, None]
    return F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=0.02)


def compute_losses(pred_rgb, gt_rgb, pred_points, gt_points, roi):
    pred_h, pred_w = mouth_geometry(pred_points)
    gt_h, gt_w = mouth_geometry(gt_points)
    geo = geometry_loss(pred_h, pred_w, gt_h, gt_w)
    landmark = mouth_landmark_loss(pred_points, gt_points)
    x0, y0, x1, y1 = roi
    rgb_diff = pred_rgb[:, y0:y1, x0:x1] - gt_rgb[:, y0:y1, x0:x1]
    rgb = torch.sqrt(rgb_diff.square() + 1e-6).mean()
    return {
        "geo": geo, "landmark": landmark, "rgb": rgb,
        "pred_h": pred_h, "pred_w": pred_w, "gt_h": gt_h, "gt_w": gt_w,
    }


def combine_losses(losses, delta, lip_weight, args):
    applied = lip_weight[:, None] * delta / max(args.max_delta, 1e-8)
    reg = applied.square().sum() / (3.0 * lip_weight.sum().clamp_min(1.0))
    total = (
        args.lambda_geo * losses["geo"]
        + args.lambda_landmark * losses["landmark"]
        + args.lambda_rgb * losses["rgb"]
        + args.lambda_delta * reg
    )
    return total, reg


def read_expected_frame_ids(source_path):
    ids = set()
    for name in ("transforms_train.json", "transforms_val.json"):
        path = Path(source_path) / name
        if not path.is_file():
            raise FileNotFoundError("Missing TG transforms: {}".format(path))
        with path.open("r", encoding="utf-8") as f:
            frames = json.load(f)["frames"]
        current = [int(x["img_id"]) for x in frames]
        if len(current) != len(set(current)):
            raise ValueError("Duplicate img_id in {}".format(name))
        overlap = ids.intersection(current)
        if overlap:
            raise ValueError("Train/val img_id overlap: {}".format(sorted(overlap)[:10]))
        ids.update(current)
    return ids


def freeze_gaussians(gs):
    for name in (
        "_xyz", "_features_dc", "_features_rest", "_identity",
        "_rotation", "_scaling", "_opacity",
    ):
        value = getattr(gs, name, None)
        if torch.is_tensor(value):
            value.requires_grad_(False)


def build_frame(view_base, source_path, gs_face, net_face, gs_mouth,
                net_mouth, fan, pipe, bg, beta, device):
    """Load a single frame and cache all frozen TG/GT-side quantities."""
    from utils.camera_utils import loadCamOnTheFly
    from gaussian_renderer import render_motion_mouth

    view = loadCamOnTheFly(copy.deepcopy(view_base))
    fid = int(view.talking_dict["img_id"])
    td = view.talking_dict
    if "phone_ids" not in td or "phone_valid" not in td:
        raise KeyError("Frame {} lacks phone_ids/phone_valid".format(fid))

    lm_path = Path(source_path) / "ori_imgs" / ("{}.lms".format(fid))
    lm = np.loadtxt(str(lm_path)).astype(np.float32)
    if lm.shape != (68, 2):
        raise ValueError("Expected 68-point .lms for frame {}".format(fid))
    _, height, width = view.original_image.shape
    fbox = face_bbox(lm, height, width)
    mbox = mouth_bbox(lm, height, width)
    gt_rgb = (view.original_image.to(device).float() / 255.0).clamp(0, 1)

    with torch.no_grad():
        gt_points = fan_landmarks(fan, gt_rgb, fbox, beta)
        gt_points_np = gt_points.cpu().numpy()[0]
        fan_error = float(
            np.linalg.norm(gt_points_np[48:68] - lm[48:68], axis=1).mean()
        )
        face_motion = net_face(
            gs_face.get_xyz, td["auds"].to(device), td["au_exp"].to(device), None
        )
        mouth_pkg = render_motion_mouth(view, gs_mouth, net_mouth, pipe, bg)
        alpha_mouth = mouth_pkg["alpha"]
        mouth_image = (
            mouth_pkg["render"] - bg[:, None, None] * (1.0 - alpha_mouth)
            + view.background.to(device).float() / 255.0 * (1.0 - alpha_mouth)
        ).detach()

    return {
        "view": view, "frame_id": fid, "face_bbox": fbox, "mouth_bbox": mbox,
        "gt_rgb": gt_rgb, "gt_points": gt_points, "fan_error": fan_error,
        "face_motion": face_motion, "mouth_image": mouth_image,
    }


def predict_delta(model, frame, gs_face, lip_weight, use_au, device):
    td = frame["view"].talking_dict
    au = td["au_exp"].to(device) if use_au else None
    result = model(
        xyz=gs_face.get_xyz.detach(),
        tg_d_xyz=frame["face_motion"]["d_xyz"].detach(),
        phone_ids=td["phone_ids"].to(device),
        phone_valid=td["phone_valid"].to(device),
        lip_weight=lip_weight,
        au_exp=au,
    )
    return result["delta_xyz"]


def render_fuse(frame, gs_face, lip_weight, delta, pipe, bg):
    return render_final_fuse(
        frame["view"], gs_face, frame["face_motion"], lip_weight,
        delta, frame["mouth_image"], pipe, bg
    ).clamp(0.0, 1.0)


def draw_mouth_points(image, points, roi, target_points=None,
                      pred_color=(255, 80, 40), target_color=(40, 230, 80)):
    x0, y0, x1, y1 = roi
    arr = (
        image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
    ).astype(np.uint8)
    crop = arr[y0:y1, x0:x1].copy()

    def paint(pts, color):
        if pts is None:
            return
        if torch.is_tensor(pts):
            pts = pts.detach().cpu().numpy()[0]
        for idx in range(48, 68):
            x, y = pts[idx]
            cv2.circle(
                crop, (int(round(x - x0)), int(round(y - y0))),
                2, color, -1, cv2.LINE_AA
            )

    if target_points is not None:
        paint(target_points, target_color)
    paint(points, pred_color)
    return cv2.resize(
        crop, (max(1, crop.shape[1] * 4), max(1, crop.shape[0] * 4)),
        interpolation=cv2.INTER_NEAREST
    )


def save_preview(path, frame, base_rgb, pred_rgb, base_pts, pred_pts,
                 base_loss, pred_loss):
    gt_panel = draw_mouth_points(
        frame["gt_rgb"], frame["gt_points"], frame["mouth_bbox"],
        target_points=frame["gt_points"], pred_color=(40, 230, 80)
    )
    base_panel = draw_mouth_points(
        base_rgb, base_pts, frame["mouth_bbox"], target_points=frame["gt_points"]
    )
    pred_panel = draw_mouth_points(
        pred_rgb, pred_pts, frame["mouth_bbox"], target_points=frame["gt_points"]
    )
    panel_w = min(gt_panel.shape[1], base_panel.shape[1], pred_panel.shape[1])
    panels = [
        cv2.resize(x, (panel_w, x.shape[0]))
        for x in (gt_panel, base_panel, pred_panel)
    ]
    panel_h = min(x.shape[0] for x in panels)
    panels = [x[:panel_h] for x in panels]
    bar = np.zeros((58, panel_w * 3, 3), dtype=np.uint8)
    for i, label in enumerate(("GT RGB + FAN 20", "TG Fuse baseline", "TG Fuse + residual")):
        cv2.putText(
            bar, label, (i * panel_w + 8, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
        )
    line = (
        "base geo/lm/rgb %.4f/%.4f/%.4f | trained %.4f/%.4f/%.4f"
        % (
            float(base_loss["geo"].item()), float(base_loss["landmark"].item()),
            float(base_loss["rgb"].item()), float(pred_loss["geo"].item()),
            float(pred_loss["landmark"].item()), float(pred_loss["rgb"].item())
        )
    )
    cv2.putText(
        bar, line, (8, 43), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (255, 255, 255), 1, cv2.LINE_AA
    )
    canvas = np.concatenate([bar, np.concatenate(panels, axis=1)], axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def evaluate(model, val_views, selected, source_path, gs_face, net_face,
             gs_mouth, net_mouth, lip_weight, fan, pipe, bg, args, step):
    model.eval()
    keys = (
        "base_total", "trained_total", "base_geo", "trained_geo",
        "base_landmark", "trained_landmark", "base_rgb", "trained_rgb",
    )
    sums = dict((key, 0.0) for key in keys)
    rows = []
    preview_dir = Path(args.out) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for _, index, fid in selected:
        frame = build_frame(
            val_views[int(index)], source_path, gs_face, net_face,
            gs_mouth, net_mouth, fan, pipe, bg,
            args.softargmax_beta, torch.device("cuda")
        )
        if frame["fan_error"] > args.max_gt_fan_error:
            print(
                "[VAL][SKIP] id={} FAN-vs-.lms={:.2f}px > {:.2f}px".format(
                    fid, frame["fan_error"], args.max_gt_fan_error
                ), flush=True
            )
            continue

        with torch.no_grad():
            zero = torch.zeros_like(gs_face.get_xyz)
            base_rgb = render_fuse(frame, gs_face, lip_weight, zero, pipe, bg)
            base_pts = fan_landmarks(
                fan, base_rgb, frame["face_bbox"], args.softargmax_beta
            )
            base_loss = compute_losses(
                base_rgb, frame["gt_rgb"], base_pts, frame["gt_points"],
                frame["mouth_bbox"]
            )
            delta = predict_delta(
                model, frame, gs_face, lip_weight, args.use_au, torch.device("cuda")
            )
            pred_rgb = render_fuse(frame, gs_face, lip_weight, delta, pipe, bg)
            pred_pts = fan_landmarks(
                fan, pred_rgb, frame["face_bbox"], args.softargmax_beta
            )
            pred_loss = compute_losses(
                pred_rgb, frame["gt_rgb"], pred_pts, frame["gt_points"],
                frame["mouth_bbox"]
            )
            base_total, _ = combine_losses(base_loss, zero, lip_weight, args)
            pred_total, _ = combine_losses(pred_loss, delta, lip_weight, args)

        def val(x):
            return float(x.detach().item())

        row = {
            "frame_id": int(fid), "fan_gt_vs_lms_mean_px": frame["fan_error"],
            "base_total": val(base_total), "trained_total": val(pred_total),
            "base_geo": val(base_loss["geo"]), "trained_geo": val(pred_loss["geo"]),
            "base_landmark": val(base_loss["landmark"]),
            "trained_landmark": val(pred_loss["landmark"]),
            "base_rgb": val(base_loss["rgb"]), "trained_rgb": val(pred_loss["rgb"]),
            "gt_height_px": val(pred_loss["gt_h"]),
            "base_height_px": val(base_loss["pred_h"]),
            "trained_height_px": val(pred_loss["pred_h"]),
            "gt_width_px": val(pred_loss["gt_w"]),
            "base_width_px": val(base_loss["pred_w"]),
            "trained_width_px": val(pred_loss["pred_w"]),
        }
        rows.append(row)
        for key in keys:
            sums[key] += row[key]
        save_preview(
            preview_dir / ("step_%06d_frame_%05d.jpg" % (step, fid)),
            frame, base_rgb, pred_rgb, base_pts, pred_pts, base_loss, pred_loss
        )
        print(
            "[VAL] step={} id={} FANerr={:.2f}px total {:.5f}->{:.5f} "
            "geo {:.5f}->{:.5f} lm20 {:.5f}->{:.5f} rgb {:.5f}->{:.5f}".format(
                step, fid, frame["fan_error"], row["base_total"], row["trained_total"],
                row["base_geo"], row["trained_geo"],
                row["base_landmark"], row["trained_landmark"],
                row["base_rgb"], row["trained_rgb"]
            ), flush=True
        )

    if not rows:
        raise RuntimeError(
            "No valid validation frames: FAN GT landmarks failed calibration. "
            "Do not train with unreliable FAN targets."
        )
    result = dict((key, value / len(rows)) for key, value in sums.items())
    result.update({"step": int(step), "valid_frames": len(rows), "frames": rows})
    model.train()
    return result


def save_checkpoint(path, model, optimizer, step, best_val, phone_data, args):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(step),
        "best_val": float(best_val),
        "phone_vocab": phone_data["vocab"],
        "vocab_size": int(max(phone_data["vocab"].values()) + 1),
        "max_delta": float(args.max_delta),
        "use_au": bool(args.use_au),
        "config": {
            "source_path": args.source_path, "model_path": args.model_path,
            "weights": args.weights, "phone_csv": args.phone_csv,
            "lambda_geo": args.lambda_geo, "lambda_landmark": args.lambda_landmark,
            "lambda_rgb": args.lambda_rgb, "lambda_delta": args.lambda_delta,
        },
    }, str(path))


def run_self_test():
    torch.manual_seed(11)
    model = LipResidualMotionNetwork(vocab_size=24, max_delta=0.002)
    n = 96
    xyz = torch.randn(n, 3) * 0.03
    tg = torch.randn(n, 3) * 0.001
    ids = torch.randint(4, 24, (8,))
    valid = torch.ones(8)
    weights = torch.rand(n)
    au = torch.randn(6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = model(xyz, tg, ids, valid, weights, au)
    assert out["delta_xyz"].shape == (n, 3)
    assert torch.allclose(out["delta_xyz"], torch.zeros_like(out["delta_xyz"]))

    gt = torch.randn(1, 68, 2)
    pred = (gt + 0.2).clone().requires_grad_(True)
    loss = mouth_landmark_loss(pred, gt)
    grad = torch.autograd.grad(loss, pred)[0]
    assert torch.isfinite(grad).all()
    assert grad[:, 48:68].abs().sum() > 0
    assert grad[:, :48].abs().sum() == 0

    optimizer.zero_grad()
    target = torch.full_like(out["delta_xyz"], 0.0005)
    (out["delta_xyz"] - target).square().mean().backward()
    optimizer.step()
    out2 = model(xyz, tg, ids, valid, weights, au)
    assert out2["delta_xyz"].abs().max().item() <= 0.002001
    print("[SELF-TEST][PASS] residual network, 20-point mouth loss, gradients")


def parse_args():
    from arguments import ModelParams, PipelineParams
    parser = argparse.ArgumentParser(
        description="Train phoneme-conditioned residual motion on frozen TG Fuse"
    )
    mp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--weights", default="output/lip_stage1_contributions/lip_contribution_weights.npz")
    parser.add_argument("--phone-csv", default="frame_phone.csv")
    parser.add_argument("--phone-vocab", default="")
    parser.add_argument("--phone-fps", type=float, default=25.0)
    parser.add_argument("--out", default="output/lip_residual_e1")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-delta", type=float, default=0.0015)
    parser.add_argument("--lambda-geo", type=float, default=1.0)
    parser.add_argument("--lambda-landmark", type=float, default=0.5)
    parser.add_argument("--lambda-rgb", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=1e-4)
    parser.add_argument("--softargmax-beta", type=float, default=25.0)
    parser.add_argument("--max-gt-fan-error", type=float, default=5.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-frames", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--resume", default="")
    parser.add_argument("--no-au", dest="use_au", action="store_false", default=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1 or args.lr <= 0 or args.max_delta <= 0:
        parser.error("iterations, lr, and max-delta must be positive")
    if min(args.eval_every, args.save_every, args.val_frames) < 1:
        parser.error("eval-every, save-every, and val-frames must be >= 1")
    if min(args.lambda_geo, args.lambda_landmark, args.lambda_rgb, args.lambda_delta) < 0:
        parser.error("Loss weights must be non-negative")
    return args, mp.extract(args), pp.extract(args)


def main():
    args, cfg, pipe = parse_args()
    if args.self_test:
        run_self_test()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA for TG rasterization and FAN")
    if not cfg.source_path or not cfg.model_path:
        raise ValueError("Please specify -s data/zh01 and -m output/zh02")
    if not Path(args.phone_csv).is_file():
        raise FileNotFoundError("Phone CSV not found: {}".format(args.phone_csv))
    if not Path(args.weights).is_file():
        raise FileNotFoundError("S1 weights not found: {}".format(args.weights))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "previews").mkdir(exist_ok=True)

    print("[1/7] Validate phone CSV against TG frame IDs", flush=True)
    expected_ids = read_expected_frame_ids(cfg.source_path)
    phone_data = load_phone_csv(
        args.phone_csv, phone_vocab=args.phone_vocab,
        expected_frame_ids=expected_ids, fps=args.phone_fps,
    )
    vocab_size = max(phone_data["vocab"].values()) + 1
    print("[Phone] vocab_size={}".format(vocab_size), flush=True)

    print("[2/7] Load frozen original Fuse checkpoint", flush=True)
    fuse_path = Path(cfg.model_path) / "chkpnt_fuse_latest.pth"
    if not fuse_path.is_file():
        raise FileNotFoundError("Fuse checkpoint not found: {}".format(fuse_path))
    parts = torch.load(str(fuse_path), map_location="cuda")
    if not isinstance(parts, (tuple, list)) or len(parts) != 4:
        raise RuntimeError("Expected TG Fuse checkpoint tuple of four elements")
    from scene import GaussianModel, MotionNetwork, MouthMotionNetwork
    gs_face = GaussianModel(cfg.sh_degree)
    gs_face.restore(parts[0], None)
    net_face = MotionNetwork(args=cfg).to(device).eval()
    net_face.load_state_dict(parts[1], strict=True)
    gs_mouth = GaussianModel(cfg.sh_degree)
    gs_mouth.restore(parts[2], None)
    net_mouth = MouthMotionNetwork(args=cfg).to(device).eval()
    net_mouth.load_state_dict(parts[3], strict=True)
    for module in (net_face, net_mouth):
        module.requires_grad_(False)
    freeze_gaussians(gs_face)
    freeze_gaussians(gs_mouth)

    print("[3/7] Load and validate S1 lip contribution weights", flush=True)
    s1 = np.load(args.weights)
    if "probs" not in s1:
        raise ValueError("S1 npz has no probs array")
    probs = s1["probs"].astype(np.float32)
    n_face = gs_face.get_xyz.shape[0]
    if probs.shape != (n_face, 3):
        raise ValueError("S1 probs shape {} != ({},3)".format(probs.shape, n_face))
    if not np.isfinite(probs).all() or (probs < -1e-5).any():
        raise ValueError("S1 probs contain invalid values")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("S1 probs are not normalized per Gaussian")
    lip_weight = torch.from_numpy(probs[:, 0] + probs[:, 1]).to(device).contiguous()
    print(
        "[S1] Face Gaussian N={} | weight>0.5={} | max={:.4f}".format(
            n_face, int((lip_weight > 0.5).sum().item()),
            float(lip_weight.max().item())
        ), flush=True
    )

    print("[4/7] Build cameras with aligned phoneme windows", flush=True)
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import cameraList_from_camInfos
    train_infos = readCamerasFromTransforms(
        cfg.source_path, "transforms_train.json", cfg.white_background,
        audio_file=cfg.audio, audio_extractor=cfg.audio_extractor,
        preload=False, phone_data=phone_data,
    )
    val_infos = readCamerasFromTransforms(
        cfg.source_path, "transforms_val.json", cfg.white_background,
        audio_file=cfg.audio, audio_extractor=cfg.audio_extractor,
        preload=False, phone_data=phone_data,
    )
    train_views = cameraList_from_camInfos(train_infos, 1.0, cfg)
    val_views = cameraList_from_camInfos(val_infos, 1.0, cfg)
    if not train_views or not val_views:
        raise RuntimeError("TG train/val camera list is empty")
    selected_val = choose_frames(val_views, cfg.source_path, args.val_frames)
    print(
        "[Camera] train={} val={} selected_val={}".format(
            len(train_views), len(val_views), len(selected_val)
        ), flush=True
    )

    print("[5/7] Load frozen FAN and initialize residual network", flush=True)
    fan = load_fan().eval()
    fan.requires_grad_(False)
    model = LipResidualMotionNetwork(
        vocab_size=vocab_size, max_delta=args.max_delta, use_au=args.use_au
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_step = 0
    best_val = float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location=device)
        if int(saved.get("vocab_size", -1)) != vocab_size:
            raise ValueError("Resume checkpoint vocabulary size mismatch")
        if bool(saved.get("use_au", args.use_au)) != bool(args.use_au):
            raise ValueError("Resume checkpoint AU setting mismatch")
        if abs(float(saved.get("max_delta", args.max_delta)) - args.max_delta) > 1e-12:
            raise ValueError("Resume checkpoint max_delta mismatch")
        if saved.get("phone_vocab") and saved["phone_vocab"] != phone_data["vocab"]:
            raise ValueError("Resume checkpoint phone vocabulary mapping mismatch")
        model.load_state_dict(saved["model"], strict=True)
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved.get("iteration", 0))
        best_val = float(saved.get("best_val", float("inf")))
        print("[Resume] iteration={} best_val={:.6f}".format(start_step, best_val), flush=True)

    bg = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    log_path = outdir / "train_metrics.csv"
    if not log_path.exists() or start_step == 0:
        with log_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "step", "frame_id", "total", "geo", "landmark20", "rgb",
                "delta_reg", "delta_abs_mean", "delta_abs_max",
                "network_grad_norm", "delta_grad_norm", "fan_gt_vs_lms_px",
            ])

    print("[6/7] Baseline validation", flush=True)
    initial = evaluate(
        model, val_views, selected_val, cfg.source_path, gs_face, net_face,
        gs_mouth, net_mouth, lip_weight, fan, pipe, bg, args, start_step
    )
    with (outdir / "val_metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(initial, ensure_ascii=False) + "\n")
    print(
        "[VAL SUMMARY] total={:.5f} geo={:.5f} landmark20={:.5f} rgb={:.5f}".format(
            initial["trained_total"], initial["trained_geo"],
            initial["trained_landmark"], initial["trained_rgb"]
        ), flush=True
    )
    if not args.resume:
        best_val = initial["trained_total"]
        save_checkpoint(
            outdir / "lip_residual_best.pth", model, optimizer,
            start_step, best_val, phone_data, args
        )
        print("[SAVE] zero-residual baseline stored as initial best checkpoint", flush=True)

    print("[7/7] Optimize only LipResidualMotionNetwork", flush=True)
    model.train()
    rng = random.Random(args.seed + start_step)
    skipped_total = 0
    skipped_consecutive = 0
    step = start_step
    attempts = 0
    while step < args.iterations:
        attempts += 1
        if attempts > (args.iterations - start_step) * 5 + 100:
            raise RuntimeError("Too many rejected samples; inspect FAN calibration.")
        frame = build_frame(
            rng.choice(train_views), cfg.source_path, gs_face, net_face,
            gs_mouth, net_mouth, fan, pipe, bg,
            args.softargmax_beta, device
        )
        if frame["fan_error"] > args.max_gt_fan_error:
            skipped_total += 1
            skipped_consecutive += 1
            if skipped_total % 10 == 0:
                print(
                    "[TRAIN][SKIP] total={} consecutive={} latest id={} FANerr={:.2f}px".format(
                        skipped_total, skipped_consecutive,
                        frame["frame_id"], frame["fan_error"]
                    ), flush=True
                )
            if skipped_consecutive >= 50:
                raise RuntimeError(
                    "50 consecutive frames failed FAN calibration; inspect S2b before training."
                )
            continue
        skipped_consecutive = 0
        step += 1

        delta = predict_delta(model, frame, gs_face, lip_weight, args.use_au, device)
        delta.retain_grad()
        pred_rgb = render_fuse(frame, gs_face, lip_weight, delta, pipe, bg)
        pred_points = fan_landmarks(
            fan, pred_rgb, frame["face_bbox"], args.softargmax_beta
        )
        losses = compute_losses(
            pred_rgb, frame["gt_rgb"], pred_points, frame["gt_points"],
            frame["mouth_bbox"]
        )
        loss, reg = combine_losses(losses, delta, lip_weight, args)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite total loss at step {}".format(step))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError(
                "No finite gradient reached Gaussian residual at step {}. "
                "Check rasterizer/FAN autograd path.".format(step)
            )
        delta_grad_norm = float(delta.grad.norm().item())
        if delta_grad_norm <= 0:
            raise RuntimeError(
                "Zero gradient reached Gaussian residual at step {}.".format(step)
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise RuntimeError("Non-finite network gradient at step {}".format(step))
        optimizer.step()

        values = {
            "total": float(loss.detach().item()),
            "geo": float(losses["geo"].detach().item()),
            "landmark": float(losses["landmark"].detach().item()),
            "rgb": float(losses["rgb"].detach().item()),
        }
        dmean = float(delta.detach().abs().mean().item())
        dmax = float(delta.detach().abs().max().item())
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                step, frame["frame_id"], values["total"], values["geo"],
                values["landmark"], values["rgb"], float(reg.detach().item()),
                dmean, dmax, float(torch.as_tensor(grad_norm).item()),
                delta_grad_norm, frame["fan_error"],
            ])

        if step % 10 == 0:
            print(
                "[TRAIN] {}/{} id={} total={:.5f} geo={:.5f} lm20={:.5f} "
                "rgb={:.5f} delta(mean/max)={:.3g}/{:.3g} "
                "net_grad={:.3g} delta_grad={:.3g} FANerr={:.2f}px".format(
                    step, args.iterations, frame["frame_id"],
                    values["total"], values["geo"], values["landmark"],
                    values["rgb"], dmean, dmax,
                    float(torch.as_tensor(grad_norm).item()),
                    delta_grad_norm, frame["fan_error"]
                ), flush=True
            )

        if step % args.eval_every == 0 or step == args.iterations:
            val = evaluate(
                model, val_views, selected_val, cfg.source_path, gs_face, net_face,
                gs_mouth, net_mouth, lip_weight, fan, pipe, bg, args, step
            )
            with (outdir / "val_metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(val, ensure_ascii=False) + "\n")
            if val["trained_total"] < best_val:
                best_val = val["trained_total"]
                save_checkpoint(
                    outdir / "lip_residual_best.pth", model, optimizer,
                    step, best_val, phone_data, args
                )
                print("[SAVE] best residual checkpoint val={:.6f}".format(best_val), flush=True)

        if step % args.save_every == 0 or step == args.iterations:
            save_checkpoint(
                outdir / "lip_residual_latest.pth", model, optimizer,
                step, best_val, phone_data, args
            )
            print("[SAVE] latest checkpoint step={}".format(step), flush=True)

    print("[DONE] output={}".format(outdir.resolve()), flush=True)
    print("[DONE] Original TG Fuse checkpoint was not modified.", flush=True)


def parse_args():
    from arguments import ModelParams, PipelineParams
    parser = argparse.ArgumentParser(
        description="Train phoneme-conditioned residual motion on frozen TG Fuse"
    )
    mp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--weights", default="output/lip_stage1_contributions/lip_contribution_weights.npz")
    parser.add_argument("--phone-csv", default="frame_phone.csv")
    parser.add_argument("--phone-vocab", default="")
    parser.add_argument("--phone-fps", type=float, default=25.0)
    parser.add_argument("--out", default="output/lip_residual_e1")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-delta", type=float, default=0.0015)
    parser.add_argument("--lambda-geo", type=float, default=1.0)
    parser.add_argument("--lambda-landmark", type=float, default=0.5)
    parser.add_argument("--lambda-rgb", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=1e-4)
    parser.add_argument("--softargmax-beta", type=float, default=25.0)
    parser.add_argument("--max-gt-fan-error", type=float, default=5.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-frames", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--resume", default="")
    parser.add_argument("--no-au", dest="use_au", action="store_false", default=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1 or args.lr <= 0 or args.max_delta <= 0:
        parser.error("iterations, lr, and max-delta must be positive")
    if min(args.eval_every, args.save_every, args.val_frames) < 1:
        parser.error("eval-every, save-every, and val-frames must be >= 1")
    if min(args.lambda_geo, args.lambda_landmark, args.lambda_rgb, args.lambda_delta) < 0:
        parser.error("Loss weights must be non-negative")
    return args, mp.extract(args), pp.extract(args)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        run_self_test()
    else:
        main()
