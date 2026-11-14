# TG-Lip S2：可微嘴部几何损失诊断

状态：独立诊断实验，不训练或覆盖原版 TalkingGaussian 权重。

## 输入

- E0/Fuse checkpoint: `output/zh02/chkpnt_fuse_latest.pth`
- S1 soft weights: `output/lip_stage1_contributions/lip_contribution_weights.npz`
- GT landmarks: `data/zh01/ori_imgs/<img_id>.lms`

## 运行

```bash
conda activate talking_gaussian
cd ~/code/mysty/TalkingGaussian-main_backup

python tg_lip_stage2_geometry_probe.py --self-test

python tg_lip_stage2_geometry_probe.py \
  -s data/zh01 -m output/zh02 \
  --weights output/lip_stage1_contributions/lip_contribution_weights.npz \
  --out output/lip_stage2_probe \
  --split val --frames 4
```

如果 40 帧的 S1 权重尚未生成，可暂用 `output/lip_contrib_smoke/lip_contribution_weights.npz` 做调试，但不能据此判定 S2 已通过正式验收。

## 测量及验收

该工具通过原版可微 Gaussian 光栅器把上下唇软权重渲染成二维软掩码，对 GT 68 点上下唇多边形和渲染结果采用同一几何读出，计算：

- 中部上下唇内缘的有符号间距 `gap`（像素）
- 上下唇合并软掩码的左右宽度 `width`（像素）
- soft Dice/L1 掩码误差
- 几何损失与掩码损失对测试位移 `delta_xyz` 的梯度
- 在负梯度方向进行一次小扰动，观察几何损失变化

输出：`output/lip_stage2_probe/s2_report.json`、`s2_frame_XXXXX.jpg`。

通过条件：大部分检测帧的梯度有限、非零；GT 和生成语义软掩码在视觉上对应；gap 与 width 能跟随真实口型合理变化。**这些量是语义软掩码代理值，不等于真实 RGB 图像的唇缘测量。** 先校准这些读数再接入 S3 运动增强网络。