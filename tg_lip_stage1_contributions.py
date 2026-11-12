#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TalkingGaussian E0/Fuse: multi-frame *actual alpha-contribution* lip labeling.

No additional CUDA extension, no edits to original TG modules, no training.
Uses differentiation of the ORIGINAL Gaussian rasterizer w.r.t. RGB colors:

  d(sum_{pixels in region} rendered_channel)/d(color_{i,channel})
  == sum_{pixels in region} alpha_{i,pixel} * transmittance_{i,pixel}.

With three channels (upper-lip, lower-lip, other), a single render + backward
calculates per-Gaussian regional contributions. Unlike center projections,
Gaussians behind opaque surfaces receive little/no credit.

Requirements: launch inside user's TalkingGaussian root with original Conda env.
The Fuse checkpoint is used so Gaussian indices match final rendering.
The computed normalized contributions are *semantic weights*, not calibrated
class probabilities, and inaccurate 68-point masks can introduce errors.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch


def lip_masks_from_landmarks(lm, h, w, erosion=0):
    """The conventional 68-point layout; lm[:,0] = x, lm[:,1] = y.

    Upper lip: outer points 48..54, inner upper 64..60.
    Lower lip: inner lower 60,67..64, then outer lower 54..59.
    Exclude rare intersections rather than assigning them to both classes.
    """
    lm = np.asarray(lm, dtype=np.float32)
    if lm.shape != (68, 2) or not np.isfinite(lm).all():
        raise ValueError('Expected finite landmarks [68,2]')
    upper_indices = [48, 49, 50, 51, 52, 53, 54, 64, 63, 62, 61, 60]
    lower_indices = [48, 60, 67, 66, 65, 64, 54, 55, 56, 57, 58, 59]
    u = np.zeros((h, w), np.uint8)
    l = np.zeros((h, w), np.uint8)
    cv2.fillPoly(u, [np.rint(lm[upper_indices]).astype(np.int32)], 1)
    cv2.fillPoly(l, [np.rint(lm[lower_indices]).astype(np.int32)], 1)
    if erosion:
        k = np.ones((3, 3), np.uint8)
        u = cv2.erode(u, k, iterations=erosion)
        l = cv2.erode(l, k, iterations=erosion)
    overlapping = (u > 0) & (l > 0)
    u[overlapping] = 0
    l[overlapping] = 0
    other = ((u == 0) & (l == 0)).astype(np.uint8)
    if u.sum() < 2 or l.sum() < 2:
        raise ValueError('Empty or tiny lip polygon; check 68 landmarks and image resolution')
    return np.stack([u, l, other], axis=0).astype(np.float32)


def color_gradient_contributions(image, colors, masks):
    """Single differentiable call yields Nx3 per-class alpha-T contributions."""
    assert image.shape[0] == 3 and masks.shape == image.shape
    scalar = (image * masks).sum()
    (result,) = torch.autograd.grad(scalar, colors, create_graph=False, retain_graph=False)
    return torch.clamp(result.detach(), min=0)


def to_bgr(im):
    img = (im.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def make_overlay(path, view, mask_np, semantic_img, frame_id, scores):
    """Three columns: original with target masks, predicted soft segmentation, blend."""
    h, w = mask_np.shape[-2:]
    original = view.original_image.float().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    gt = original.copy()
    um, lm = mask_np[0] > 0, mask_np[1] > 0
    gt[um] = 0.65 * gt[um] + 0.35 * np.array([235, 65, 65])
    gt[lm] = 0.65 * gt[lm] + 0.35 * np.array([65, 95, 235])
    gt = np.clip(gt, 0, 255).astype(np.uint8)

    s = semantic_img.detach().cpu().numpy()
    alpha = np.maximum(s.sum(0), 1e-6)
    up = np.clip(s[0] / alpha, 0, 1)
    lo = np.clip(s[1] / alpha, 0, 1)
    confidence = np.clip(s.sum(0), 0, 1)
    col = np.zeros((h, w, 3), np.float32)
    col[..., 0] = 245 * up
    col[..., 2] = 245 * lo
    # Mask background; semantic color exists only when 3DGS rendered visible content
    col *= confidence[..., None]
    col = col.astype(np.uint8)
    blended = np.clip(original.astype(np.float32) * 0.65 + col * 0.75, 0, 255).astype(np.uint8)
    titles = ['GT lip polygons', 'Gaussian contribution map', 'Overlay (GT + predicted)']
    trip = np.concatenate([gt, col, blended], axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(trip, cv2.COLOR_RGB2BGR))
    binary_u = (up > .5) & (confidence > .3)
    binary_l = (lo > .5) & (confidence > .3)
    def iou(a, b):
        return float((a & b).sum() / max(1, (a | b).sum()))
    scores.append(dict(frame_id=int(frame_id), upper_iou_rough=iou(binary_u, um),
                       lower_iou_rough=iou(binary_l, lm),
                       note='IoU depends on polygon quality and heuristic 0.5 threshold'))


def save_ply(path, xyz, probs):
    """ASCII PLY: color encodes soft upper/lower/other contribution ratios."""
    col = np.zeros((len(xyz), 3), np.float32)
    col += probs[:, 0:1] * np.array([235, 65, 65], np.float32)
    col += probs[:, 1:2] * np.array([65, 95, 235], np.float32)
    col += probs[:, 2:3] * np.array([120, 120, 120], np.float32)
    col = np.clip(col, 0, 255).astype(np.uint8)
    with open(str(path), 'w', encoding='ascii') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write('element vertex %d\n' % len(xyz))
        for a in ('x', 'y', 'z'):
            f.write('property float %s\n' % a)
        for a in ('red', 'green', 'blue'):
            f.write('property uchar %s\n' % a)
        f.write('end_header\n')
        for pos, rgb in zip(xyz, col):
            f.write('%.7g %.7g %.7g %d %d %d\n' % (pos[0], pos[1], pos[2], rgb[0], rgb[1], rgb[2]))


def parse_args():
    from arguments import ModelParams, PipelineParams
    p = argparse.ArgumentParser(description='TG alpha-contribution lip Gaussian soft segmentation')
    mp = ModelParams(p)
    pp = PipelineParams(p)
    p.add_argument('--out', type=str, default='output/lip_stage1_contributions')
    p.add_argument('--frames', type=int, default=40)
    p.add_argument('--preview', type=int, default=5)
    p.add_argument('--validation-previews', type=int, default=3,
                   help='Independently visualize trained semantic weights on held-out val frames')
    p.add_argument('--erode', type=int, default=0,
                   help='Erode lip masks this many pixels; 0 recommended initially')
    p.add_argument('--prior-npz', type=str, default='',
                   help='Optional previous center-projection labels for overlap diagnostics only')
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    if not args.source_path or not args.model_path:
        p.error('Specify -s data/zh01 and -m output/zh02')
    if args.eval:
        p.error('Do not use --eval: it aliases training to validation in TG')
    if args.frames < 1 or args.preview < 0 or args.validation_previews < 0 or args.erode < 0:
        p.error('frames>=1, preview>=0, validation-previews>=0, erode>=0 required')
    return args, mp.extract(args), pp.extract(args)


def main():
    args, cfg, pipe = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for original TG rasterization')
    from scene import GaussianModel, MotionNetwork
    from gaussian_renderer import GaussianRasterizationSettings, GaussianRasterizer
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import cameraList_from_camInfos, loadCamOnTheFly

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.model_path) / 'chkpnt_fuse_latest.pth'
    print('[1/4] Load FUSE checkpoint:', str(ckpt_path), flush=True)
    ckpt = torch.load(str(ckpt_path), map_location='cuda')
    if not isinstance(ckpt, (tuple, list)) or len(ckpt) != 4:
        raise RuntimeError('Expected original TG Fuse tuple (faceGS, faceNet, mouthGS, mouthNet)')
    gs = GaussianModel(cfg.sh_degree)
    gs.restore(ckpt[0], None)
    model = MotionNetwork(args=cfg).cuda().eval()
    model.load_state_dict(ckpt[1], strict=True)
    model.requires_grad_(False)
    for k in ('_xyz', '_features_dc', '_features_rest', '_identity', '_rotation', '_scaling', '_opacity'):
        t = getattr(gs, k, None)
        if torch.is_tensor(t):
            t.requires_grad_(False)
    n = gs.get_xyz.shape[0]
    print('      Gaussian N=%d' % n, flush=True)

    print('[2/4] Select training frames, build original TG cameras', flush=True)
    infos = readCamerasFromTransforms(cfg.source_path, 'transforms_train.json',
                                      cfg.white_background, audio_file=cfg.audio,
                                      audio_extractor=cfg.audio_extractor, preload=False)
    views = cameraList_from_camInfos(infos, 1.0, cfg)
    if not views:
        raise RuntimeError('No train views available')
    chosen = np.unique(np.linspace(0, len(views) - 1, min(len(views), args.frames)).astype(np.int64))
    total_counts = torch.zeros((n, 3), dtype=torch.float64, device='cpu')
    frame_ids = []
    preview_indices = set(np.unique(np.linspace(0, len(chosen)-1,
                                                   min(args.preview, len(chosen))).astype(int)).tolist())
    preview_records = []
    bg = torch.zeros(3, dtype=torch.float32, device='cuda')
    stored = []

    print('[3/4] Compute exact alpha*T regional contributions (one color grad per frame)', flush=True)
    for j, idx in enumerate(chosen):
        view = loadCamOnTheFly(views[int(idx)])
        fid = int(view.talking_dict['img_id'])
        lms = np.loadtxt(str(Path(cfg.source_path) / 'ori_imgs' / ('%d.lms' % fid)))
        _, h, w = view.original_image.shape
        masks_np = lip_masks_from_landmarks(lms, h, w, erosion=args.erode)
        # Freeze TG deformation and Gaussian appearance; only render colors need grad.
        with torch.no_grad():
            pred = model(gs.get_xyz, view.talking_dict['auds'].cuda(),
                         view.talking_dict['au_exp'].cuda(), None)
            means3d = (gs.get_xyz + pred['d_xyz']).contiguous()
            scales = gs.scaling_activation(gs._scaling + pred['d_scale']).contiguous()
            rots = gs.rotation_activation(gs._rotation + pred['d_rot']).contiguous()
            opacities = gs.get_opacity.contiguous()
        settings = GaussianRasterizationSettings(
            image_height=h, image_width=w,
            tanfovx=math.tan(view.FoVx * 0.5), tanfovy=math.tan(view.FoVy * 0.5),
            bg=bg, scale_modifier=1.0,
            viewmatrix=view.world_view_transform,
            projmatrix=view.full_proj_transform,
            sh_degree=gs.active_sh_degree,
            campos=view.camera_center,
            prefiltered=False, debug=False)
        raster = GaussianRasterizer(raster_settings=settings)
        colors = torch.ones((n, 3), device='cuda', dtype=torch.float32, requires_grad=True)
        pixels, radii, depth, alpha = raster(
            means3D=means3d, means2D=torch.zeros_like(means3d),
            shs=None, colors_precomp=colors,
            opacities=opacities, scales=scales, rotations=rots,
            cov3D_precomp=None)
        masks = torch.from_numpy(masks_np).to(device='cuda')
        contrib = color_gradient_contributions(pixels, colors, masks)
        if j == 0:
            expected = (pixels.detach() * masks).sum(dim=(1, 2))
            observed = contrib.sum(dim=0)
            relative_error = ((expected - observed).abs() / expected.clamp(min=1.)).max().item()
            print('      Alpha contribution conservation check: max relative error=%.6f' % relative_error, flush=True)
            if relative_error > 0.03:
                raise RuntimeError('Renderer color-gradient conservation failed (%.3g). Do NOT trust soft labels.' % relative_error)
        if not torch.isfinite(contrib).all():
            raise RuntimeError('Non-finite color gradient at frame %d' % fid)
        total_counts += contrib.double().cpu()
        frame_ids.append(fid)
        if j in preview_indices:
            stored.append((fid, int(idx), masks_np))
        print('      frame %d/%d, id=%d, lip pixels U=%d L=%d, contrib U=%.1f L=%.1f' %
              (j+1, len(chosen), fid, masks_np[0].sum(), masks_np[1].sum(),
               contrib[:,0].sum().item(), contrib[:,1].sum().item()), flush=True)
        # Drop all frame-dependent CUDA intermediates before next iteration.
        del pred, means3d, scales, rots, raster, pixels, radii, depth, alpha, colors, masks, contrib

    totals = total_counts.numpy()
    sum_all = totals.sum(axis=1)
    probs = (totals / np.maximum(sum_all[:, None], 1e-8)).astype(np.float32)
    # Unobserved Gaussians receive zero lip score and 1.0 'other', not a spurious class.
    invalid = sum_all <= 1e-5
    probs[invalid] = [0., 0., 1.]
    lip_weight = probs[:,0] + probs[:,1]
    hard_label = np.argmax(probs, axis=1).astype(np.uint8)
    hard_label[invalid] = 2
    summary = {
        'source': 'TG chkpnt_fuse_latest.pth (Face Gaussian, alpha*transmittance contribution)',
        'definition': 'Normalized regional alpha-contribution shares, NOT calibrated semantic probabilities',
        'face_gaussians': int(n),
        'frames_used': int(len(frame_ids)),
        'image_ids': frame_ids,
        'unobserved_gaussians': int(invalid.sum()),
        'soft_lip_weight_gt_05': int((lip_weight >= .5).sum()),
        'soft_lip_weight_gt_08': int((lip_weight >= .8).sum()),
        'upper_majority_count': int(((hard_label == 0) & (lip_weight > .5)).sum()),
        'lower_majority_count': int(((hard_label == 1) & (lip_weight > .5)).sum()),
        'max_lip_weight': float(lip_weight.max()),
        'notes': ['Face-only contribution; Fuse outer lips originate in Face layer',
                  'Mask polygons are approximated from 68 landmarks',
                  'Soft weights are computed over selected train frames, and indexed by Fuse GS order',
                  'Semantic maps are an auxiliary diagnostic, not lip aperture readout'],
    }
    np.savez_compressed(str(out / 'lip_contribution_weights.npz'),
                        probs=probs, upper_weight=probs[:,0], lower_weight=probs[:,1],
                        other_weight=probs[:,2], lip_weight=lip_weight, hard_label=hard_label,
                        contribution=totals.astype(np.float32), observation=sum_all.astype(np.float32),
                        image_ids=np.array(frame_ids, dtype=np.int64))
    save_ply(out / 'face_lip_contributions.ply', gs.get_xyz.detach().cpu().numpy(), probs)

    # Render final segmentation weights WITHOUT additional alpha-gradient computations.
    print('[4/4] Write preview maps and diagnostics', flush=True)
    for fid, idx, mask_np in stored:
        view = loadCamOnTheFly(views[idx])
        _, h, w = view.original_image.shape
        with torch.no_grad():
            pred = model(gs.get_xyz, view.talking_dict['auds'].cuda(),
                         view.talking_dict['au_exp'].cuda(), None)
            means3d = gs.get_xyz + pred['d_xyz']
            scales = gs.scaling_activation(gs._scaling + pred['d_scale'])
            rots = gs.rotation_activation(gs._rotation + pred['d_rot'])
            settings = GaussianRasterizationSettings(
                image_height=h, image_width=w,
                tanfovx=math.tan(view.FoVx * .5), tanfovy=math.tan(view.FoVy * .5),
                bg=bg, scale_modifier=1.0,
                viewmatrix=view.world_view_transform, projmatrix=view.full_proj_transform,
                sh_degree=gs.active_sh_degree, campos=view.camera_center,
                prefiltered=False, debug=False)
            rast = GaussianRasterizer(raster_settings=settings)
            semantic, _, _, _ = rast(
                means3D=means3d, means2D=torch.zeros_like(means3d),
                shs=None, colors_precomp=torch.from_numpy(probs).cuda(),
                opacities=gs.get_opacity,
                scales=scales, rotations=rots, cov3D_precomp=None)
        make_overlay(out / ('preview_%05d.jpg' % fid), view, mask_np, semantic, fid, preview_records)
    summary['rough_preview_iou'] = preview_records

    if args.validation_previews:
        val_infos = readCamerasFromTransforms(cfg.source_path, 'transforms_val.json',
                                             cfg.white_background, audio_file=cfg.audio,
                                             audio_extractor=cfg.audio_extractor, preload=False)
        val_views = cameraList_from_camInfos(val_infos, 1.0, cfg)
        validation_idx = np.unique(np.linspace(0, len(val_views)-1,
                                               min(args.validation_previews, len(val_views))).astype(int))
        validation_records = []
        for idx in validation_idx:
            view = loadCamOnTheFly(val_views[int(idx)])
            fid = int(view.talking_dict['img_id'])
            _, h, w = view.original_image.shape
            lm = np.loadtxt(str(Path(cfg.source_path) / 'ori_imgs' / ('%d.lms' % fid)))
            mask_np = lip_masks_from_landmarks(lm, h, w, erosion=args.erode)
            with torch.no_grad():
                pred = model(gs.get_xyz, view.talking_dict['auds'].cuda(),
                             view.talking_dict['au_exp'].cuda(), None)
                means3d = gs.get_xyz + pred['d_xyz']
                scales = gs.scaling_activation(gs._scaling + pred['d_scale'])
                rots = gs.rotation_activation(gs._rotation + pred['d_rot'])
                settings = GaussianRasterizationSettings(
                    image_height=h, image_width=w,
                    tanfovx=math.tan(view.FoVx * .5), tanfovy=math.tan(view.FoVy * .5),
                    bg=bg, scale_modifier=1.0, viewmatrix=view.world_view_transform,
                    projmatrix=view.full_proj_transform, sh_degree=gs.active_sh_degree,
                    campos=view.camera_center, prefiltered=False, debug=False)
                rast = GaussianRasterizer(raster_settings=settings)
                semantic, _, _, _ = rast(
                    means3D=means3d, means2D=torch.zeros_like(means3d),
                    shs=None, colors_precomp=torch.from_numpy(probs).cuda(),
                    opacities=gs.get_opacity, scales=scales, rotations=rots,
                    cov3D_precomp=None)
            make_overlay(out / ('validation_%05d.jpg' % fid), view, mask_np, semantic, fid, validation_records)
        summary['heldout_validation_iou_rough'] = validation_records

    if args.prior_npz:
        try:
            old = np.load(args.prior_npz, allow_pickle=False)
            if 'labels' in old and len(old['labels']) == n:
                prior = np.isin(old['labels'], [1, 2])
                cur = lip_weight > .5
                summary['old_projection_vs_new_contribution_iou'] = float((prior & cur).sum() / max(1, (prior | cur).sum()))
                summary['old_projection_lips'] = int(prior.sum())
                summary['new_contribution_lips'] = int(cur.sum())
            else:
                summary['prior_warning'] = 'Prior indices not length-matched to Fuse; cannot compare'
        except (OSError, ValueError) as exc:
            summary['prior_warning'] = str(exc)

    with open(str(out / 'summary.json'), 'w', encoding='utf8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('      Saved', str(out / 'lip_contribution_weights.npz'))
    print('      Saved', str(out / 'face_lip_contributions.ply'))
    print('      Lip weights > 0.5:', summary['soft_lip_weight_gt_05'])
    print('      Upper / lower majority:', summary['upper_majority_count'], summary['lower_majority_count'])
    print('      Done; review preview_*.jpg AND summary.json before motion-network training.', flush=True)


if __name__ == '__main__':
    main()