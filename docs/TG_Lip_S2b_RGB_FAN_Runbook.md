# TalkingGaussian S2b：最终 Fuse RGB + 可微 FAN 几何验证

## 当前研究阶段

**诊断，不训练。** 保留原始 TG 及现有 `tg_lip_stage1_contributions.py`、`tg_lip_stage2_geometry_probe.py`；另新增 `tg_lip_stage2b_rgb_fan_probe.py`。

- 冻结原始 Fuse Face/Mouth Gaussians、Face/Mouth MotionNetwork。
- 加载 S1 的每个 Face Gaussian 上下唇权重。
- 原版 Face MotionNetwork 预测初始运动；额外 `delta` 仅按嘴唇权重作用于 Face Gaussians 的 XYZ。
- 使用 TG 原有 Face+Mouth alpha 合成公式得到**最终 RGB 人脸**。
- 同一帧的真实 GT RGB 和生成 RGB，使用**同一个冻结的 FAN**提取 68 个热力图，通过可微 `soft_argmax` 获取关键点。
- 计算 48–54 嘴角宽度、62–66 沿嘴角法线的张口高度，再计算归一化 SmoothL1 几何损失。
- 从最终 RGB 在**同一个 GT 嘴部 ROI**计算 Charbonnier 图像损失；同时与几何损失求和。
- 对临时 Face Gaussian 位移 `delta` 求梯度，沿负梯度做一次极小位移，比较最终 RGB 和关键点指标。
- **不训练网络，不保存新模型**。

## 兼容性

你当前 `talking_gaussian` 环境为 Python 3.7 / PyTorch 1.12.1。**不要安装最新版 `face-alignment`（最新版要求更高版本 Python/PyTorch）。** 先检测：

```bash
conda activate talking_gaussian
python -c "import torch, face_alignment; print('torch', torch.__version__, 'face_alignment', face_alignment.__file__)"
```

如果没有 `face_alignment`，先确认环境中其他依赖已经存在，再考虑：

```bash
python -m pip install --no-deps 'face-alignment==1.3.6'
```

`--no-deps` 是避免 pip 为 FAN 升级/替换已经跑通的旧 TG PyTorch/CUDA。仍需要兼容的 scipy、scikit-image、numba 等依赖，若缺失需要逐项检查。FAN 首次载入可能下载预训练权重，需能访问模型下载服务器。不要修改 torch 和 CUDA 版本。

CUDA 扩展 GCC9 固定配置沿用此前：

```bash
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=/usr/bin/g++-9
export TORCH_EXTENSIONS_DIR="$CONDA_PREFIX/torch_extensions_gcc9"
```

## 执行

将 `tg_lip_stage2b_rgb_fan_probe.py` 放到 TG 仓库根目录。先执行：

```bash
cd ~/code/mysty/TalkingGaussian-main_backup
conda activate talking_gaussian
python tg_lip_stage2b_rgb_fan_probe.py --self-test
```

再运行 4 帧诊断：

```bash
python tg_lip_stage2b_rgb_fan_probe.py \
  -s data/zh01 \
  -m output/zh02 \
  --weights output/lip_stage1_contributions/lip_contribution_weights.npz \
  --out output/lip_stage2b_rgb_fan \
  --split val \
  --frames 4
```

**优先查看输出：**

- `s2b_frame_*.jpg`：**左 GT RGB，中 TG Fuse 原始结果，右 TG Fuse + 负梯度测试位移**。四个彩色小点分别是 FAN 预测的 48/54/62/66 号嘴部关键点；顶部标出 FAN 得到的 H/W（原始图像像素尺度）。
- `s2b_report.json`：GT、原版、试探位移后的高度、宽度、RGB 和几何损失；`grad_core_norm`、`grad_other_norm`。
- `gt_fan_vs_lms_mouth_mean_px`：**FAN 从真实 GT 图像预测的 20 个嘴部点，与已有 .lms 标注的平均像素误差。** 若偏大，需要先调整 FAN 裁剪、温度或更换关键点模型；不能误认为模型在最终生成 RGB 上训练有效。

如果 `GT-FAN-vs-lms` 误差较小、梯度正常，再验证 30 帧：

```bash
python tg_lip_stage2b_rgb_fan_probe.py \
  -s data/zh01 -m output/zh02 \
  --weights output/lip_stage1_contributions/lip_contribution_weights.npz \
  --out output/lip_stage2b_rgb_fan_30 \
  --split val --frames 30
```

打包上传：

```bash
tar -czf lip_stage2b_review.tar.gz -C output lip_stage2b_rgb_fan
```

## 科研与工程边界

1. 这不是当前最终 RGB 嘴型准确性的验收结论；只能说明新损失具有可微性及局部响应。是否准确要对比 GT 原始 `.lms`、叠加点和生成图像。
2. FAN 是冻结的可微检测网络，`no_grad()` **只包 GT 和测试后无需反传的渲染**，绝对不能包 `pred_rgb -> FAN -> loss -> delta` 的主梯度路径。
3. ROI 与裁剪框在当前验证阶段由 GT landmarks 指定，保证两幅图同区域对齐。对新音频推理时不能使用未来帧 GT；这套框仅用于有标注的训练/验证计算。
4. Mouth Gaussian 仍按原版单独渲染，残差**只施加到 Face Gaussians**，避免误以为 Mouth 分支也被训练。
5. 如果几何损失下降但最终肉眼嘴型变差，不要启动 S3；先检查关键点失真、牙齿遮挡和训练目标过拟合。
6. 即使在 GT 同步验证帧改善，也不能证明新中文语音泛化，必须另做未见语音实验。
7. 若 `face_alignment` 权重下载卡住，与前面的 `gridencoder` 编译阻塞不是同一类问题。