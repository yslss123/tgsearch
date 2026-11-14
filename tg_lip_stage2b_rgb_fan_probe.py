#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TG Lip S2b: Final Fuse RGB + differentiable FAN landmark geometry probe.

Diagnostics only: frozen TG Face, Mouth, Gaussian params and frozen FAN.
No model training and no checkpoint changes.

Evaluation: final TG Face+Mouth RGB vs same-frame original GT RGB.
Geometry: frozen FAN 68-point heatmaps -> differentiable soft-argmax -> mouth H/W.
The .lms GT anchors the *same* crop for GT/render and audits FAN GT accuracy.
Gradients flow through FAN + TG rasterizer to an added temporary Face-Gaussian delta.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def soft_argmax_2d(heatmaps, beta=25.0):
    """Heatmap [B,K,H,W] -> [B,K,2] continuous (x,y) in heatmap pixels."""
    if heatmaps.ndim != 4:
        raise ValueError('Expected FAN heatmaps [B,68,H,W], got %r' % (tuple(heatmaps.shape),))
    b, k, h, w = heatmaps.shape
    if k != 68:
        raise ValueError('Expected 68 FAN keypoint heatmaps, got %d' % k)
    logits = (heatmaps.float().reshape(b, k, -1) * beta)
    logits = logits - logits.amax(dim=-1, keepdim=True)
    prob = torch.softmax(logits, dim=-1)
    xs = torch.arange(w, device=heatmaps.device, dtype=prob.dtype).repeat(h)
    ys = torch.arange(h, device=heatmaps.device, dtype=prob.dtype).repeat_interleave(w)
    return torch.stack([(prob * xs).sum(-1), (prob * ys).sum(-1)], dim=-1)


def mouth_geometry(points):
    """[B,68,2] -> height along perpendicular to mouth-corner axis, width."""
    p48, p54 = points[:, 48], points[:, 54]
    p62, p66 = points[:, 62], points[:, 66]
    mouth_axis = p54 - p48
    width = torch.linalg.norm(mouth_axis, dim=-1).clamp_min(1e-6)
    normal = torch.stack((-mouth_axis[:, 1], mouth_axis[:, 0]), dim=-1) / width[:, None]
    height = torch.abs(((p66 - p62) * normal).sum(dim=-1))
    return height, width


def geometry_loss(ph, pw, gh, gw):
    """Scale-normalized so a 1px error has comparable meaning across crops."""
    scale = gw.detach().clamp_min(8.)
    return (F.smooth_l1_loss(ph / scale, gh / scale, beta=0.04)
            + 0.5 * F.smooth_l1_loss(pw / scale, gw / scale, beta=0.04))


def face_bbox(lm, h, w, pad=0.30):
    """Static GT-determined full-face crop shared by GT and render.
    These pixels are only used to define crop coordinates; NOT as pred geometry.
    """
    pts = np.asarray(lm, np.float32)
    if pts.shape != (68, 2):
        raise ValueError('Expected 68-point GT landmarks')
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    rw, rh = max(8., x1 - x0), max(8., y1 - y0)
    # Landmarks exclude forehead; leave more room above than below.
    x0 = max(0, int(math.floor(x0 - rw * pad)))
    x1 = min(w, int(math.ceil(x1 + rw * pad)))
    y0 = max(0, int(math.floor(y0 - rh * (pad + 0.10))))
    y1 = min(h, int(math.ceil(y1 + rh * (pad - 0.10))))
    if x1 - x0 < 32 or y1 - y0 < 32:
        raise ValueError('Face crop too small')
    return x0, y0, x1, y1


def mouth_bbox(lm, h, w):
    from tg_lip_stage2_geometry_probe import bbox_from_landmarks
    return bbox_from_landmarks(lm, h, w)


def crop_fan(image, bbox):
    """Differentiable GT-fixed crop of [3,H,W] RGB [0,1] to FAN [1,3,256,256]."""
    x0, y0, x1, y1 = bbox
    patch = image[:, y0:y1, x0:x1].unsqueeze(0)
    return F.interpolate(patch, (256, 256), mode='bilinear', align_corners=False)


def fan_landmarks(fan, image, bbox, beta):
    """Direct FAN forward; do NOT use no_grad/detach on predicted RGB branch.

    Maps heatmap pixel-center coordinates back into the original full image.
    """
    out = fan(crop_fan(image, bbox))
    if isinstance(out, (list, tuple)):
        out = out[-1]
    if out.ndim != 4:
        raise ValueError('FAN must output heatmaps [1,68,H,W]; got %r' % (tuple(out.shape),))
    hh, hw = out.shape[-2:]
    pts = soft_argmax_2d(out, beta)
    x0, y0, x1, y1 = bbox
    px = x0 + (pts[..., 0] + .5) * ((x1 - x0) / float(hw))
    py = y0 + (pts[..., 1] + .5) * ((y1 - y0) / float(hh))
    return torch.stack([px, py], dim=-1)


def load_fan():
    try:
        import face_alignment
    except ImportError as exc:
        raise RuntimeError('Missing face_alignment: for Python3.7 use a compatible pinned '
                           'version, e.g. python -m pip install --no-deps face-alignment==1.3.6 '
                           '(install missing dependencies separately).') from exc
    lt = getattr(face_alignment.LandmarksType, 'TWO_D', None)
    if lt is None:
        lt = face_alignment.LandmarksType._2D
    # The face detector may initialize, but is NOT used during this probe.
    fa = face_alignment.FaceAlignment(lt, device='cuda', flip_input=False)
    fan = fa.face_alignment_net.eval()
    for p in fan.parameters():
        p.requires_grad_(False)
    return fan


def raster_settings(view, gaussians, pipe, background):
    from diff_gaussian_rasterization import GaussianRasterizationSettings
    return GaussianRasterizationSettings(
        image_height=int(view.image_height), image_width=int(view.image_width),
        tanfovx=math.tan(view.FoVx * 0.5), tanfovy=math.tan(view.FoVy * 0.5),
        bg=background, scale_modifier=1.0, viewmatrix=view.world_view_transform,
        projmatrix=view.full_proj_transform, sh_degree=gaussians.active_sh_degree,
        campos=view.camera_center, prefiltered=False, debug=pipe.debug)


def render_final_fuse(view, gs, face_motion, lip_weight, delta, mouth_image, pipe, background):
    """Exactly reproduce train_fuse.py Face rendering and Face-on-Mouth alpha fuse.

    Only Face-Gaussian positions get a gated residual. Mouth branch RGB is frozen.
    """
    from diff_gaussian_rasterization import GaussianRasterizer
    rast = GaussianRasterizer(raster_settings=raster_settings(view, gs, pipe, background))
    means3d = gs.get_xyz + face_motion['d_xyz'] + lip_weight[:, None] * delta
    scales = gs.scaling_activation(gs._scaling + face_motion['d_scale'])
    rotations = gs.rotation_activation(gs._rotation + face_motion['d_rot'])
    face_image, radii, depth, alpha = rast(
        means3D=means3d, means2D=torch.zeros_like(means3d),
        shs=gs.get_features, colors_precomp=None, opacities=gs.get_opacity,
        scales=scales, rotations=rotations, cov3D_precomp=None)
    result = (face_image - background[:, None, None] * (1.0 - alpha)
              + mouth_image * (1.0 - alpha))
    return result


def make_mouth_background(view, gs_mouth, motion_mouth, pipe, background):
    from gaussian_renderer import render_motion_mouth
    with torch.no_grad():
        pkg = render_motion_mouth(view, gs_mouth, motion_mouth, pipe, background)
        alpha = pkg['alpha']
        return (pkg['render'] - background[:, None, None] * (1.0 - alpha)
                + view.background.cuda() / 255.0 * (1.0 - alpha)).detach()


def draw_panel(img, points, x0, y0, x1, y1, label, height, width):
    """Mouth ROI visualization, selected FAN landmarks shown as small circles."""
    ar = (img.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rgb = ar[y0:y1, x0:x1].copy()
    if points is not None:
        pts = points.detach().cpu().numpy()[0]
        for idx, color in [(48, (20, 220, 20)), (54, (20, 220, 20)),
                           (62, (235, 230, 0)), (66, (235, 230, 0))]:
            x, y = pts[idx]
            cv2.circle(rgb, (int(round(x-x0)), int(round(y-y0))), 2, color, -1)
    rgb = cv2.resize(rgb, (max(1, (x1-x0)*4), max(1, (y1-y0)*4)),
                     interpolation=cv2.INTER_NEAREST)
    bar = np.zeros((50, rgb.shape[1], 3), np.uint8)
    cv2.putText(bar, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, 'H=%.2f  W=%.2f' % (height, width),
                (5, 39), cv2.FONT_HERSHEY_SIMPLEX, .48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, rgb], axis=0)


def save_preview(path, gt, before, after, gt_pts, before_pts, after_pts, roi, metrics):
    x0, y0, x1, y1 = roi
    ims = [gt, before, after]
    pts = [gt_pts, before_pts, after_pts]
    labels = ['GT RGB', 'TG Fuse baseline', 'TG Fuse + trial delta']
    hs = [metrics['gt_h'], metrics['before_h'], metrics['after_h']]
    ws = [metrics['gt_w'], metrics['before_w'], metrics['after_w']]
    panels = [draw_panel(im, p, x0, y0, x1, y1, label, h, w)
              for im, p, label, h, w in zip(ims, pts, labels, hs, ws)]
    res = np.concatenate(panels, axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))


def self_test():
    yy, xx = torch.meshgrid(torch.arange(64, dtype=torch.float32),
                            torch.arange(64, dtype=torch.float32))
    shift = torch.tensor(0.0, requires_grad=True)
    fields = []
    for k in range(68):
        xc = 20.0 + (k % 7) * 3.0
        yc = 20.0 + (k % 6) * 3.0
        if k == 66:
            yc += shift
        fields.append(torch.exp(-((xx-xc)**2 + (yy-yc)**2) / (2.0 * 1.8**2)))
    hm = torch.stack(fields, 0).unsqueeze(0)
    landmarks = soft_argmax_2d(hm)
    h, w = mouth_geometry(landmarks)
    (g,) = torch.autograd.grad(h, shift)
    if not torch.isfinite(g) or abs(g.item()) < .01:
        raise RuntimeError('FAN soft-argmax geometry gradient failed the CPU test')
    print('[SELF-TEST] soft-argmax height %.3f width %.3f | dh/d(deltaY)=%.5f [PASS]' %
          (h.item(), w.item(), g.item()))


def parse_args():
    from arguments import ModelParams, PipelineParams
    p = argparse.ArgumentParser(description='S2b real Fuse RGB + differentiable FAN 68-landmark probe')
    mp = ModelParams(p)
    pp = PipelineParams(p)
    p.add_argument('--weights', default='output/lip_stage1_contributions/lip_contribution_weights.npz')
    p.add_argument('--out', default='output/lip_stage2b_rgb_fan')
    p.add_argument('--split', choices=['train', 'val'], default='val')
    p.add_argument('--frames', type=int, default=4)
    p.add_argument('--step-size', type=float, default=0.0002,
                   help='World-coordinate one-step diagnostic displacement; NOT training')
    p.add_argument('--softargmax-beta', type=float, default=25.0)
    p.add_argument('--geo-weight', type=float, default=1.0)
    p.add_argument('--rgb-weight', type=float, default=1.0)
    p.add_argument('--max-gt-fan-error', type=float, default=5.0,
                   help='GT FAN vs supplied .lms mouth landmark mean error threshold in original px')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    return args, mp.extract(args), pp.extract(args)


def main():
    args, cfg, pipe = parse_args()
    self_test()
    if args.self_test:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('TG rasterization and pretrained FAN require CUDA')
    if args.frames < 1 or args.step_size <= 0 or args.softargmax_beta <= 0:
        raise ValueError('frames >=1 and positive --step-size / --softargmax-beta required')
    from scene import GaussianModel, MotionNetwork, MouthMotionNetwork
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import cameraList_from_camInfos, loadCamOnTheFly
    from tg_lip_stage2_geometry_probe import choose_frames
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(cfg.model_path) / 'chkpnt_fuse_latest.pth'
    print('[1/4] Load frozen final Fuse (Face + Mouth) and S1 lip weights', flush=True)
    parts = torch.load(str(ckpt), map_location='cuda')
    if len(parts) < 4:
        raise RuntimeError('Expected Fuse checkpoint tuple of four elements')
    gs_face = GaussianModel(cfg.sh_degree)
    gs_face.restore(parts[0], None)
    net_face = MotionNetwork(args=cfg).cuda().eval()
    net_face.load_state_dict(parts[1], strict=True)
    gs_mouth = GaussianModel(cfg.sh_degree)
    gs_mouth.restore(parts[2], None)
    net_mouth = MouthMotionNetwork(args=cfg).cuda().eval()
    net_mouth.load_state_dict(parts[3], strict=True)
    for obj in (net_face, net_mouth):
        obj.requires_grad_(False)
    # GaussianModel is not torch.nn.Module; freeze its tensors explicitly.
    for gs in (gs_face, gs_mouth):
        for name in ('_xyz', '_features_dc', '_features_rest', '_identity',
                     '_scaling', '_rotation', '_opacity'):
            tensor = getattr(gs, name, None)
            if torch.is_tensor(tensor):
                tensor.requires_grad_(False)
    data = np.load(args.weights)
    probs = data['probs'].astype(np.float32)
    if probs.shape != (gs_face.get_xyz.shape[0], 3):
        raise ValueError('S1 weights shape mismatch, got %r for N=%d' %
                         (probs.shape, gs_face.get_xyz.shape[0]))
    lip_weight = torch.from_numpy(probs[:, 0] + probs[:, 1]).cuda().contiguous()
    fan = load_fan()
    bg = torch.tensor([0., 1., 0.], device='cuda')
    split = 'transforms_val.json' if args.split == 'val' else 'transforms_train.json'
    infos = readCamerasFromTransforms(cfg.source_path, split, cfg.white_background,
                                      audio_file=cfg.audio, audio_extractor=cfg.audio_extractor,
                                      preload=False)
    views = cameraList_from_camInfos(infos, 1.0, cfg)
    chosen = choose_frames(views, cfg.source_path, args.frames)
    print('[2/4] %d %s frames; GT and TG evaluated from RGB via the same frozen FAN' %
          (len(chosen), args.split), flush=True)
    rows = []
    for k, (aperture_ratio, index, fid) in enumerate(chosen):
        view = loadCamOnTheFly(views[int(index)])
        lm = np.loadtxt(str(Path(cfg.source_path) / 'ori_imgs' / ('%d.lms' % fid)))
        _, h, w = view.original_image.shape
        fbox = face_bbox(lm, h, w)
        mbox = mouth_bbox(lm, h, w)
        gt_rgb = (view.original_image.cuda().float() / 255.).clamp(0, 1)
        with torch.no_grad():
            motion = net_face(gs_face.get_xyz, view.talking_dict['auds'].cuda(),
                              view.talking_dict['au_exp'].cuda(), None)
            mouth_rgb = make_mouth_background(view, gs_mouth, net_mouth, pipe, bg)
            gt_pts = fan_landmarks(fan, gt_rgb, fbox, args.softargmax_beta)
            gt_h, gt_w = mouth_geometry(gt_pts)
        delta = torch.zeros_like(gs_face.get_xyz, requires_grad=True)
        base_rgb = render_final_fuse(view, gs_face, motion, lip_weight,
                                     delta, mouth_rgb, pipe, bg).clamp(0, 1)
        base_pts = fan_landmarks(fan, base_rgb, fbox, args.softargmax_beta)
        base_h, base_w = mouth_geometry(base_pts)
        loss_geo = geometry_loss(base_h, base_w, gt_h, gt_w)
        x0, y0, x1, y1 = mbox
        # Differentiable final-RGB image comparison, same original pixel locations.
        loss_rgb = torch.sqrt((base_rgb[:, y0:y1, x0:x1]
                               - gt_rgb[:, y0:y1, x0:x1]) ** 2 + 1e-6).mean()
        loss = args.geo_weight * loss_geo + args.rgb_weight * loss_rgb
        (grad,) = torch.autograd.grad(loss, delta)
        finite = bool(torch.isfinite(grad).all().item())
        core = lip_weight > .5
        grad_core = float(grad[core].norm().item()) if core.any() else 0.
        grad_other = float(grad[~core].norm().item()) if (~core).any() else 0.
        if not finite:
            raise RuntimeError('Non-finite gradient on frame %d' % fid)
        direction = grad.detach()
        largest = direction.abs().max()
        if largest.item() > 0:
            direction = direction / largest
        trial_delta = -args.step_size * direction
        with torch.no_grad():
            trial_rgb = render_final_fuse(view, gs_face, motion, lip_weight,
                                          trial_delta, mouth_rgb, pipe, bg).clamp(0, 1)
            trial_pts = fan_landmarks(fan, trial_rgb, fbox, args.softargmax_beta)
            trial_h, trial_w = mouth_geometry(trial_pts)
            trial_geo = geometry_loss(trial_h, trial_w, gt_h, gt_w)
            trial_rgb_loss = torch.sqrt((trial_rgb[:, y0:y1, x0:x1]
                                         - gt_rgb[:, y0:y1, x0:x1]) ** 2 + 1e-6).mean()
            trial_loss = args.geo_weight * trial_geo + args.rgb_weight * trial_rgb_loss
        gt_pts_np = gt_pts.detach().cpu().numpy()[0]
        lip_lm_error = float(np.linalg.norm(gt_pts_np[48:68] - lm[48:68], axis=1).mean())
        row = {
            'frame_id': int(fid), 'gt_landmark_aperture_ratio': float(aperture_ratio),
            'gt_fan_vs_lms_mouth_mean_px': lip_lm_error,
            'fan_gt_acceptable': bool(lip_lm_error <= args.max_gt_fan_error),
            'gt_height_px': float(gt_h.item()), 'baseline_height_px': float(base_h.item()),
            'trial_height_px': float(trial_h.item()),
            'gt_width_px': float(gt_w.item()), 'baseline_width_px': float(base_w.item()),
            'trial_width_px': float(trial_w.item()),
            'baseline_geo_loss': float(loss_geo.item()),
            'trial_geo_loss': float(trial_geo.item()),
            'baseline_rgb_loss': float(loss_rgb.item()),
            'trial_rgb_loss': float(trial_rgb_loss.item()),
            'baseline_total_loss': float(loss.item()),
            'trial_total_loss': float(trial_loss.item()),
            'trial_total_change': float(trial_loss.item() - loss.item()),
            'grad_core_norm': grad_core, 'grad_other_norm': grad_other,
            'grad_finite': finite, 'grad_nonzero': bool(largest.item() > 0),
            'full_face_bbox': list(fbox), 'mouth_bbox': list(mbox),
        }
        rows.append(row)
        save_preview(outdir / ('s2b_frame_%05d.jpg' % fid), gt_rgb, base_rgb.detach(),
                     trial_rgb, gt_pts, base_pts.detach(), trial_pts, mbox,
                     {'gt_h': row['gt_height_px'], 'gt_w': row['gt_width_px'],
                      'before_h': row['baseline_height_px'], 'before_w': row['baseline_width_px'],
                      'after_h': row['trial_height_px'], 'after_w': row['trial_width_px']})
        print(' [frame %d] id=%d FAN-GT-vs-lms=%.2fpx %s | H: GT %.2f / base %.2f / trial %.2f | '
              'W: GT %.2f / base %.2f / trial %.2f | geo %.4f -> %.4f RGB %.4f -> %.4f '
              'total %.4f -> %.4f | grad core %.3g other %.3g' %
              (k+1, fid, lip_lm_error, 'OK' if row['fan_gt_acceptable'] else 'CHECK',
               row['gt_height_px'], row['baseline_height_px'], row['trial_height_px'],
               row['gt_width_px'], row['baseline_width_px'], row['trial_width_px'],
               row['baseline_geo_loss'], row['trial_geo_loss'],
               row['baseline_rgb_loss'], row['trial_rgb_loss'],
               row['baseline_total_loss'], row['trial_total_loss'], grad_core, grad_other), flush=True)
        del base_rgb, base_pts, trial_rgb, trial_pts, loss, grad, delta
    report = {
        'method': 'S2b frozen-TG Face+Mouth Fuse RGB, frozen FAN heatmap soft-argmax, '
                  'gradient wrt temporary gated Face Gaussian displacement',
        'no_training': True, 's1_weights': args.weights,
        'fan_softargmax_beta': args.softargmax_beta,
        'geo_weight': args.geo_weight, 'rgb_weight': args.rgb_weight,
        'step_size_world': args.step_size, 'split': args.split,
        'tested_frames': len(rows),
        'fan_gt_acceptable_frames': sum(1 for row in rows if row['fan_gt_acceptable']),
        'gradient_nonzero_frames': sum(1 for row in rows if row['grad_nonzero']),
        'total_loss_decreased_frames': sum(1 for row in rows if row['trial_total_change'] < 0),
        'frames': rows,
        'caveats': [
            'FAN predictions are proxies and MUST be visually audited against original .lms.',
            'The static same-frame GT face crop is for diagnostic training/evaluation only.',
            'Only Face Gaussians are displaced; Mouth Gaussians are rendered separately and frozen.',
            'One negative-gradient step is a local diagnostic, NOT training or novel-audio evaluation.',
            'A smaller loss does NOT automatically demonstrate improved lip sync on novel audio.',
        ],
    }
    with open(str(outdir / 's2b_report.json'), 'w', encoding='utf-8') as fd:
        json.dump(report, fd, indent=2, ensure_ascii=False)
    print('[4/4] report:', outdir / 's2b_report.json', flush=True)
    print(' FAN calibrated %d/%d | gradients nonzero %d/%d | loss reduced %d/%d' %
          (report['fan_gt_acceptable_frames'], len(rows),
           report['gradient_nonzero_frames'], len(rows),
           report['total_loss_decreased_frames'], len(rows)), flush=True)
    print('No model parameters updated. Do NOT start S3 if FAN GT keypoints disagree with .lms.', flush=True)


if __name__ == '__main__':
    import sys
    if '--self-test' in sys.argv:
        self_test()
    else:
        main()