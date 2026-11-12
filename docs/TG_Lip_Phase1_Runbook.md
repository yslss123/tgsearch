# TalkingGaussian 嘴部增强：实施步骤与第一阶段运行说明

## 固定技术路线

- **TG E0 不覆盖、不重训**：`output/zh02/chkpnt_fuse_latest.pth` 作为基础。
- **S1（当前）**：多帧 RGB 颜色梯度提取 Gaussian→上唇/下唇/其他区域的真实 alpha 贡献比例，导出软权重。无需新网络，无需新 CUDA 扩展。
- **S2**：验证二维上下唇软掩码能否稳定、可微地映射为口型几何量。尤其需要保证预测量与**渲染可见唇缘**对应，而不是仅让一个预测网络输出匹配数字。
- **S3**：独立的嘴部 Gaussian 动态残差增强分支，接入几何损失，短训 300 步检查困难口型和画面完整性。
- **S4**：逐步解冻 Face MotionNetwork 的部分输出层，进行局部联合微调，初步规划 500—2000 步，按验证曲线调整。
- **S5**：独立中文音素→目标口型状态预测网络，在 S3/S4 成功后接入；不提前融合 DeepSpeech 和音素。

## S1 实现原理（不是之前的 2D 中心投影）

TG 标准 Gaussian rasterizer 对固定几何参数的颜色是线性的。令每个 Gaussian 的预计算颜色为 `[1,1,1]`，对应输出 `R[3,H,W]`，按 GT 68 点关键点绘制上唇、下唇、其他三个互斥掩码 `M`。

对 `sum(R * M)` 进行自动微分，对每个 Gaussian 的输入颜色梯度 `dL / dcolor[i,k]`，正好是它对第 k 类像素的总 `alpha * transmittance` 贡献（受栅格化器具体实现和数值误差影响）。最后将多帧贡献在 Gaussian 维度归一化，得到 `[N,3]` 权重。

程序会在第一帧校验“Gaussian 梯度贡献之和 ≈ 掩码区域渲染 alpha 之和”。如数值不一致，将主动终止。

**注意**：68 点生成的上下唇掩码是近似边界，不是精确人工标注；输出为贡献比例，不是准确率或校准概率。

## 准备

将 `tg_lip_stage1_contributions.py` 放在 `/home/lzh/code/mysty/TalkingGaussian-main_backup/` 根目录，保持原始源码、Conda 环境、所有已训练的 checkpoint 不变。

### A. 两帧烟雾测试

```bash
conda activate talking_gaussian
cd ~/code/mysty/TalkingGaussian-main_backup
python tg_lip_stage1_contributions.py \
  -s data/zh01 \
  -m output/zh02 \
  --out output/lip_contrib_smoke \
  --frames 2 --preview 2 --validation-previews 1 \
  --prior-npz output/lip_gaussian_probe_no_depth/lip_gaussian_labels.npz
```

通过条件：打印的第一帧 `Alpha contribution conservation check: max relative error` 不超过 0.03，最后成功保存 `summary.json` 和 `lip_contribution_weights.npz`，生成上下唇预览图。

### B. 40 帧完整分组

两帧测试通过后：

```bash
python tg_lip_stage1_contributions.py \
  -s data/zh01 \
  -m output/zh02 \
  --out output/lip_stage1_contributions \
  --frames 40 --preview 5 --validation-previews 4 \
  --prior-npz output/lip_gaussian_probe_no_depth/lip_gaussian_labels.npz
```

40 帧是初始采样量，可在结果稳定后扩大；不是精度保证。

### C. 检查结果

```bash
ls -lh output/lip_stage1_contributions
cat output/lip_stage1_contributions/summary.json
```

主要文件：

| 文件 | 含义 |
|---|---|
| `lip_contribution_weights.npz` | `probs [N,3]`, `upper_weight`, `lower_weight`, `lip_weight`, `observation` 等 |
| `face_lip_contributions.ply` | 原始 3D Gaussian 位置按上下唇权重染色 |
| `preview_XXXXX.jpg` | 参与分组的训练帧：GT 掩码、模型软语义图、叠加图 |
| `validation_XXXXX.jpg` | 没用于 S1 贡献统计的验证帧，检查软语义图是否一致 |
| `summary.json` | Gaussian 数量、统计帧、粗略掩码 IoU、旧投影标签对照信息 |

预览颜色：**红=上唇，蓝=下唇**。验证时要特别关注闭嘴帧唇缘是否出现大量红蓝混杂、嘴角是否漏掉、鼻下或脸颊是否误判。

### D. 上传结果供复核

```bash
cd ~/code/mysty/TalkingGaussian-main_backup
tar -czf lip_stage1_review.tar.gz -C output lip_stage1_contributions
```

上传 `lip_stage1_review.tar.gz`。下一步才实施 S2 几何损失和梯度测试。

## 停止条件

- 梯度贡献守恒检查失败；
- 训练/验证帧中红蓝区域大量偏离真实上下唇；
- 绝大多数上唇、下唇的分组权重近零；
- 极端口型上严重错分且无法通过改进二维掩码修复。

满足上述任意一条，不进入 S3/S4 训练，应先修正分组监督、可见性或 Face/Mouth 对应关系。

## 当前可确认的验证边界

已完成：Python 语法检查、模拟 68 点掩码测试、CPU 颜色梯度线性贡献单元测试。**未完成**：在用户 RTX 3090 上的真实 CUDA 栅格化前向/反向，以及真实图像准确性验证。