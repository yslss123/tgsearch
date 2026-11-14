#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2 probe: differentiable rendered lip geometry on original TG Fuse Gaussians.

No TG source edits, no model training. Requires S1 soft contribution weights.
Render upper/lower semantic colors with *the actual TG Gaussian rasterizer*,
then read approximate lip inner-gap and visible-lip width with differentiable
soft extrema. Compare to the same reader on GT 68-landmark polygon masks.

IMPORTANT: semantic-mask edges are proxies for rendered RGB lip contours.
They require calibration before use in any learning objective. This probe
reports gradients w.r.t. an added temporary per-Gaussian displacement and
compares a small gradient-descent perturbation to the unperturbed result.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tg_lip_stage1_contributions import lip_masks_from_landmarks


def soft_edge(profile, direction, temperature=1.8, threshold=0.07):
    """Approximate first/last occupied coordinate in 1D. Differentiable
    where profile > threshold; robustness is checked by support statistic.
    """
    assert direction in ('first', 'last')
    pos = torch.arange(profile.shape[0], device=profile.device, dtype=profile.dtype)
    support = F.relu(profile - threshold)
    signed = (pos - pos.mean()) / temperature
    if direction == 'first':
        signed = -signed
    logits = torch.log(support + 1e-12) + signed
    edge = (torch.softmax(logits, dim=0) * pos).sum()
    return edge, support.sum()


def lip_geometry(sem, central_x=None):
    """Two-channel [2,H,W] semantic-alpha map -> differentiable geometry.

    Pixel y grows downward. The apparent aperture proxy is lower lip's top
    minus upper lip's bottom near the central mouth region, not centroid gap.
    Width proxy is rightmost minus leftmost visible lip semantic support.
    Coordinates are in units of the cropped ROI pixels.
    """
    assert sem.ndim == 3 and sem.shape[0] == 2
    h, w = sem.shape[-2:]
    if central_x is None:
        central_x = (w-1) * 0.5
    x = torch.arange(w, device=sem.device, dtype=sem.dtype)
    central = torch.exp(-0.5*((x - float(central_x)) / max(2.0, w*0.12))**2)
    central = central / central.sum().clamp(min=1e-8)
    up_row = (sem[0] * central[None,:]).sum(dim=1)
    lo_row = (sem[1] * central[None,:]).sum(dim=1)
    upper_bottom, up_support = soft_edge(up_row, 'last')
    lower_top, lo_support = soft_edge(lo_row, 'first')
    lip_col = torch.max(torch.clamp(sem[0] + sem[1], max=1.0), dim=0).values
    left, width_support = soft_edge(lip_col, 'first')
    right, _ = soft_edge(lip_col, 'last')
    return {
        'gap': lower_top-upper_bottom,
        'width': right-left,
        'upper_inner_y': upper_bottom,
        'lower_inner_y': lower_top,
        'left_x': left,
        'right_x': right,
        'upper_support': up_support,
        'lower_support': lo_support,
        'width_support': width_support,
    }


def bbox_from_landmarks(lm, h, w):
    points = np.asarray(lm, np.float32)[48:60]
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    width = max(8., float(xmax-xmin))
    height = max(6., float(ymax-ymin))
    xpad = max(5., width*.35)
    ypad = max(5., height*.70)
    x0 = max(0, int(math.floor(xmin-xpad)))
    x1 = min(w, int(math.ceil(xmax+xpad+1)))
    y0 = max(0, int(math.floor(ymin-ypad)))
    y1 = min(h, int(math.ceil(ymax+ypad+1)))
    if x1-x0 < 8 or y1-y0 < 8:
        raise ValueError('Lip ROI too small: {} {} {} {}'.format(x0,y0,x1,y1))
    return x0,y0,x1,y1


def choose_frames(views, source_path, n):
    """Include near-closed and maximum-open GT frames rather than arbitrary frames."""
    scores = []
    for j, view in enumerate(views):
        fid = int(view.talking_dict['img_id'])
        path = Path(source_path) / 'ori_imgs' / ('%d.lms' % fid)
        if not path.exists():
            continue
        lm = np.loadtxt(str(path))
        if lm.shape != (68,2):
            continue
        aperture = max(0., float(lm[66,1]-lm[62,1]))
        width = max(1., float(np.linalg.norm(lm[54]-lm[48])))
        scores.append((aperture/width,j,fid))
    if not scores:
        raise RuntimeError('No valid .lms on selected split')
    scores.sort(key=lambda x:x[0])
    if n == 1:
        chosen = [scores[-1]]
    else:
        indexes = np.linspace(0,len(scores)-1,min(n,len(scores))).round().astype(int)
        chosen = [scores[i] for i in indexes]
    return sorted(chosen, key=lambda x:x[0])


def mask_loss_fn(pred, gt):
    pred = pred.clamp(0.,1.)
    gt = gt.float()
    num = 2.*(pred*gt).sum(dim=(1,2)) + 1.
    den = pred.sum(dim=(1,2))+gt.sum(dim=(1,2))+1.
    dice = 1.-num/den
    return dice.mean() + 0.2*F.l1_loss(pred, gt)


def geometry_loss_fn(pred, gt):
    return F.smooth_l1_loss(pred['gap'], gt['gap'], beta=1.) + \
           0.5*F.smooth_l1_loss(pred['width'], gt['width'], beta=1.)


def syntax_and_math_test():
    """CPU-only unit sanity: moving the lower semantic lobe changes gap and grad."""
    yy, xx = torch.meshgrid(torch.arange(60,dtype=torch.float32),
                            torch.arange(90,dtype=torch.float32))
    cy_lower = torch.tensor(30., requires_grad=True)
    up = torch.exp(-((yy-22.)/2.8)**2) * torch.exp(-((xx-45.)/14.)**6)
    down = torch.exp(-((yy-cy_lower)/2.8)**2) * torch.exp(-((xx-45.)/14.)**6)
    sem = torch.stack([up,down],dim=0)
    g = lip_geometry(sem)
    (v,) = torch.autograd.grad(g['gap'],cy_lower)
    assert torch.isfinite(v) and abs(float(v)) > .01, 'S2 analytic lip geometry grad did not propagate'
    narrow_gate = torch.exp(-((xx-45.)/8.)**6)
    narrower = torch.stack([up*narrow_gate,down*narrow_gate],dim=0)
    g2 = lip_geometry(narrower)
    print('[SELF-TEST] gap=%.3f width=%.3f d(gap)/d(lowerY)=%.4f narrower_width=%.3f' %
          (g['gap'].item(), g['width'].item(), v.item(), g2['width'].item()))
    assert g2['width'].item() < g['width'].item()-2.0, 'Width reader not sensitive to lip width'
    print('[SELF-TEST] PASS')


def parse_args():
    from arguments import ModelParams, PipelineParams
    p = argparse.ArgumentParser(description='TG S2 differentiable lip geometry smoke probe')
    mp = ModelParams(p)
    pp = PipelineParams(p)
    p.add_argument('--weights',type=str,default='output/lip_stage1_contributions/lip_contribution_weights.npz')
    p.add_argument('--out',type=str,default='output/lip_stage2_probe')
    p.add_argument('--split',choices=['val','train'],default='val')
    p.add_argument('--frames',type=int,default=4)
    p.add_argument('--step-size',type=float,default=0.0001,
                   help='World-coordinate descent probe step; diagnostic, NOT training')
    p.add_argument('--self-test',action='store_true',help='Only run CPU mathematical self-test')
    args = p.parse_args()
    return args, mp.extract(args), pp.extract(args)


def render_semantic(view, gs, pred, rgb_colors, lip_weight, delta, pipe):
    from gaussian_renderer import GaussianRasterizationSettings, GaussianRasterizer
    h, w = view.image_height, view.image_width
    settings = GaussianRasterizationSettings(
        image_height=h,image_width=w,
        tanfovx=math.tan(view.FoVx*.5),tanfovy=math.tan(view.FoVy*.5),
        bg=torch.zeros(3,device='cuda'),scale_modifier=1.0,
        viewmatrix=view.world_view_transform, projmatrix=view.full_proj_transform,
        sh_degree=gs.active_sh_degree,campos=view.camera_center,
        prefiltered=False,debug=False)
    rast = GaussianRasterizer(raster_settings=settings)
    xyz = gs.get_xyz + pred['d_xyz'] + delta*lip_weight[:,None]
    scaling = gs.scaling_activation(gs._scaling + pred['d_scale'])
    rotation = gs.rotation_activation(gs._rotation + pred['d_rot'])
    img, radii, depth, alpha = rast(
        means3D=xyz, means2D=torch.zeros_like(xyz),
        shs=None, colors_precomp=rgb_colors,
        opacities=gs.get_opacity, scales=scaling,
        rotations=rotation, cov3D_precomp=None)
    return img


def colorize_preview(path, view, gt_mask, pred_mask, roi, gt_geom, pred_geom, fid):
    orig = view.original_image.detach().float().cpu().numpy().transpose(1,2,0).astype(np.uint8)
    pred = pred_mask.detach().clamp(0,1).cpu().numpy()
    gt = gt_mask.cpu().numpy()
    x0,y0,x1,y1 = roi
    gt_rgb = orig.copy()
    gt_rgb[gt[0] > .5] = np.clip(gt_rgb[gt[0]>.5]*.55 + np.array([220,55,55])*.45,0,255)
    gt_rgb[gt[1] > .5] = np.clip(gt_rgb[gt[1]>.5]*.55 + np.array([60,70,240])*.45,0,255)
    p_rgb = np.zeros_like(orig)
    p_rgb[:,:,0] = np.clip(pred[0]*255,0,255).astype(np.uint8)
    p_rgb[:,:,2] = np.clip(pred[1]*255,0,255).astype(np.uint8)
    blend = np.clip(orig*.55 + p_rgb*.65,0,255).astype(np.uint8)
    zoom_tiles = [gt_rgb[y0:y1,x0:x1], p_rgb[y0:y1,x0:x1], blend[y0:y1,x0:x1]]
    trip = np.concatenate(zoom_tiles, axis=1)
    factor = 4
    trip = cv2.resize(trip, (trip.shape[1]*factor,trip.shape[0]*factor), interpolation=cv2.INTER_NEAREST)
    bar = np.zeros((65,trip.shape[1],3),np.uint8)
    labels = ['GT lip mask', 'TG Gaussian lip map', 'GT RGB + semantic map']
    panel_width = (x1-x0)*factor
    for i,label in enumerate(labels):
        cv2.putText(bar, label, (i*panel_width+8,24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,(255,255,255),1,cv2.LINE_AA)
    details='id=%d   aperture GT / pred: %.1f / %.1f px   width GT / pred: %.1f / %.1f px' % (
        fid,gt_geom['gap'].item(),pred_geom['gap'].item(),
        gt_geom['width'].item(),pred_geom['width'].item())
    cv2.putText(bar, details,(8,50),cv2.FONT_HERSHEY_SIMPLEX,
                0.42,(255,255,255),1,cv2.LINE_AA)
    trip=np.concatenate([bar,trip],axis=0)
    cv2.imwrite(str(path),cv2.cvtColor(trip,cv2.COLOR_RGB2BGR))


def main():
    args, cfg, pipe = parse_args()
    syntax_and_math_test()
    if args.self_test:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('S2 needs CUDA for original TG Gaussian rasterizer')
    if args.frames < 1 or args.step_size <= 0:
        raise ValueError('frames>=1 and step-size>0 required')
    from scene import GaussianModel, MotionNetwork
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import cameraList_from_camInfos, loadCamOnTheFly
    wpath=Path(args.weights)
    data=np.load(str(wpath))
    if 'probs' not in data:
        raise ValueError('S1 .npz must include [N,3] probs')
    probs=data['probs'].astype(np.float32)
    if probs.ndim != 2 or probs.shape[1] != 3 or not np.isfinite(probs).all():
        raise ValueError('Invalid S1 weights shape/values')
    if (probs < -1e-5).any() or not np.allclose(probs.sum(axis=1),1.,atol=1e-3):
        raise ValueError('S1 weights must be normalized per Gaussian')
    torch.manual_seed(321)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    ckpt=Path(cfg.model_path)/'chkpnt_fuse_latest.pth'
    print('[1/4] Load original Fuse + S1 semantic weights:',ckpt,wpath,flush=True)
    parts=torch.load(str(ckpt),map_location='cuda')
    gs=GaussianModel(cfg.sh_degree)
    gs.restore(parts[0],None)
    model=MotionNetwork(args=cfg).cuda().eval()
    model.load_state_dict(parts[1],strict=True)
    model.requires_grad_(False)
    if gs.get_xyz.shape[0] != probs.shape[0]:
        raise ValueError('S1 weights N=%d != Fuse N=%d' % (probs.shape[0],gs.get_xyz.shape[0]))
    for name in ('_xyz','_features_dc','_features_rest','_identity','_rotation','_scaling','_opacity'):
        obj=getattr(gs,name,None)
        if torch.is_tensor(obj):
            obj.requires_grad_(False)
    rgb=torch.from_numpy(probs).cuda().contiguous()
    lip_weight=rgb[:,:2].sum(dim=1).contiguous()
    split_file='transforms_val.json' if args.split=='val' else 'transforms_train.json'
    infos=readCamerasFromTransforms(cfg.source_path, split_file, cfg.white_background,
                                     audio_file=cfg.audio,audio_extractor=cfg.audio_extractor,preload=False)
    views=cameraList_from_camInfos(infos,1.0,cfg)
    chosen=choose_frames(views,cfg.source_path,args.frames)
    print('[2/4] %d %s frames selected by GT aperture ratios' % (len(chosen),args.split),flush=True)
    rows=[]
    fatal=0
    for k,(ratio,j,fid) in enumerate(chosen):
        view=loadCamOnTheFly(views[int(j)])
        lm=np.loadtxt(str(Path(cfg.source_path)/'ori_imgs'/('%d.lms'%fid)))
        _,h,w=view.original_image.shape
        x0,y0,x1,y1=bbox_from_landmarks(lm,h,w)
        masks_np=lip_masks_from_landmarks(lm,h,w)
        gt_mask=torch.from_numpy(masks_np[:2]).cuda()
        with torch.no_grad():
            pred=model(gs.get_xyz,view.talking_dict['auds'].cuda(),
                       view.talking_dict['au_exp'].cuda(),None)
        delta=torch.zeros_like(gs.get_xyz,requires_grad=True)
        rendered=render_semantic(view,gs,pred,rgb,lip_weight,delta,pipe)
        gt_crop=gt_mask[:,y0:y1,x0:x1]
        pred_crop=rendered[:2,y0:y1,x0:x1]
        target=lip_geometry(gt_crop)
        current=lip_geometry(pred_crop)
        geo_loss=geometry_loss_fn(current,target)
        mask_loss=mask_loss_fn(pred_crop,gt_crop)
        grad_geo=torch.autograd.grad(geo_loss,delta,retain_graph=True)[0]
        grad_mask=torch.autograd.grad(mask_loss,delta)[0]
        valid=torch.isfinite(grad_geo).all().item() and torch.isfinite(grad_mask).all().item()
        core=lip_weight>.5
        core_norm=float(grad_geo[core].norm().item()) if core.any() else 0.
        other_norm=float(grad_geo[~core].norm().item()) if (~core).any() else 0.
        mask_norm=float(grad_mask.norm().item())
        up_support=float(current['upper_support'].item())
        lo_support=float(current['lower_support'].item())
        width_support=float(current['width_support'].item())
        supported=(up_support>.005 and lo_support>.005 and width_support>.005)
        active=bool(valid and core_norm>1e-9 and supported)
        if not active:
            fatal+=1
        direction=grad_geo.detach()
        if direction.abs().max().item()>0:
            direction=direction/direction.abs().max().clamp(min=1e-12)
        with torch.no_grad():
            perturbed=render_semantic(view,gs,pred,rgb,lip_weight,
                                      -args.step_size*direction,pipe)
            pg=lip_geometry(perturbed[:2,y0:y1,x0:x1])
            step_loss=geometry_loss_fn(pg,target).item()
        diff=float(step_loss-geo_loss.item())
        row={
            'frame_id':int(fid),'aperture_gt_landmark_ratio':float(ratio),
            'roi_xyxy':[x0,y0,x1,y1],
            'gt_gap_proxy_px':float(target['gap'].item()),
            'pred_gap_proxy_px':float(current['gap'].item()),
            'gt_width_proxy_px':float(target['width'].item()),
            'pred_width_proxy_px':float(current['width'].item()),
            'geo_loss':float(geo_loss.item()),'mask_loss':float(mask_loss.item()),
            'geo_grad_core_norm':core_norm,'geo_grad_other_norm':other_norm,
            'mask_grad_total_norm':mask_norm,'finite_gradients':bool(valid),
            'semantic_support':[up_support,lo_support,width_support],
            'gradient_probe_loss_change':diff,
            'gradient_probe_reduced_loss':bool(diff < 0),
            'healthy':active,
        }
        rows.append(row)
        print('  [frame %d] id=%d GT gap/width %.2f/%.2f predicted %.2f/%.2f | geom %.4f mask %.4f | dxyz grad core %.3g other %.3g | trial dLoss %+.5f [%s]' %
              (k+1,fid,row['gt_gap_proxy_px'],row['gt_width_proxy_px'],
               row['pred_gap_proxy_px'],row['pred_width_proxy_px'],
               row['geo_loss'],row['mask_loss'],core_norm,other_norm,diff,'OK' if active else 'CHECK'),flush=True)
        colorize_preview(out/('s2_frame_%05d.jpg'%fid),view,gt_mask,rendered[:2],
                         (x0,y0,x1,y1),target,current,fid)
        del rendered,delta,grad_geo,grad_mask
    report={'method':'S2 differentiable semantic-render proxy (not RGB landmark extraction)',
            'weights_file':str(wpath),'split':args.split,'frames':rows,
            'healthy_frames':int(len(rows)-fatal),'tested_frames':len(rows),
            'notes':['No training, no checkpoint saved, no original TG files modified',
                     'Aperture proxy uses inner semantic borders near center and may be biased by occlusion/label errors',
                     'Mask IoU and numerical proxy alignment must be inspected before calling this lip opening loss',
                     'Gradient probe is diagnostic; negative dLoss means small displacement reduced geometry loss']}
    with open(str(out/'s2_report.json'),'w',encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False)
    print('[4/4] Saved',str(out/'s2_report.json'),flush=True)
    print('      Healthy gradient frames: %d/%d' % (len(rows)-fatal,len(rows)),flush=True)
    if fatal:
        print('      WARNING: Some frames have missing/invalid geometry gradients. Do not train S3.',flush=True)
    else:
        print('      Gradient paths exist. Inspect GT-vs-pred proxy alignment before S3.',flush=True)


if __name__=='__main__':
    import sys
    if '--self-test' in sys.argv:
        syntax_and_math_test()
    else:
        main()
