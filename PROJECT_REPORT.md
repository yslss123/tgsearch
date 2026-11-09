# TalkingGaussian 项目结构报告

> 对象：初次接手该代码库的开发者
> 对应论文：*TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting* (ECCV 2024)
> 代码基线：[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) + [RAD-NeRF](https://github.com/ashawkey/RAD-NeRF) + [AD-NeRF](https://github.com/YudongGuo/AD-NeRF)

---

## 1. 项目在做什么

输入一段**单目、固定机位、25FPS、约 512×512** 的人物说话视频，输出一个**可用任意音频驱动的 3D 说话头**。

核心思路是"**结构持久（structure-persistent）**"：不把整个头部当作一个黑盒运动场，而是把它拆成两个互相独立的 3D Gaussian 集合：

| 分支 | 负责区域 | 运动模型 | 变形能力 |
|---|---|---|---|
| **Face / 上脸** | 脸 + 头发（不含嘴） | `MotionNetwork` | 位移 + 旋转 + 缩放 + 不透明度 |
| **Mouth / 嘴部** | 嘴唇 + 牙齿（含口腔内部） | `MouthMotionNetwork` | 仅位移（且 x/z 方向位移被压缩 5 倍） |

两个分支**分阶段独立训练**，最后由第三个阶段（fusion）学一个像素级 alpha 把它们缝合起来。这样做的动机是：嘴部需要高频、大幅度的非刚性形变，而脸/头需要的是刚性、稳定的姿态保持——用同一个形变场会互相打架，导致脸部"漂移"。分离后，面部结构在说话过程中保持稳定。

### 关键技术点

1. **两套解耦的 Gaussian + Motion Network**（不是一套网络输出两套结果，是两套独立的参数集）。
2. **音频条件**：DeepSpeech / HuBERT / Esperanto-wav2vec2 三种特征可选，默认 DeepSpeech（29 维）。
3. **表情条件**：从 OpenFace 的 AU（Action Unit）取 6 维作为额外条件，并用 AU45（眨眼）做注意力调制。
4. **Hash-grid 位置编码**：对每个 Gaussian 的 xyz 做三个 2D 平面投影（xy/yz/xz）分别哈希编码，避免 3D 哈希网格的参数爆炸。
5. **颜色编码抠图策略**：训练初期用**纯绿背景 `[0,1,0]`** 合成 GT，并在最后阶段把绿色高斯直接剪掉（`bg_prune`）；同时用解析图（parsing）生成 face/hair/mouth 掩码来做 loss 的区域加权。这样模型学到的背景是"可识别的绿色"，方便分离。
6. **课程式采样（curriculum）**：不是随机抽帧，而是按嘴张开程度（AU25）或嘴部大小（mouth bound）**从易到难**逐步扩大采样区间，保证难帧（大张嘴、露牙）被充分训练。
7. **LPIPS patch loss**：在训练后段对随机尺寸的 patch 计算 LPIPS，专门提升嘴部区域的感知质量。

---

## 2. 环境与依赖

- 官方测试环境：Ubuntu 18.04 / CUDA 11.3 / PyTorch 1.12.1 / Python 3.7，见 [environment.yml](environment.yml)
- 需要**编译 CUDA 扩展**：`diff-gaussian-rasterization`（ashawkey fork，额外返回 alpha 和 depth）、`simple-knn`、`gridencoder`
- 需要额外装：`pytorch3d`、`tensorflow-gpu==2.8.0`（DeepSpeech 用）、`mmcv-full==1.7.1`（EasyPortrait 用）、`face_alignment`（评估用）
- **资源需求**：数据默认全量预加载进内存，官方注释给出 **约 N×32GB RAM 对应 N×5k 帧**。读 [scene/dataset_readers.py:98](scene/dataset_readers.py#L98) 的 `preload` 参数可关闭。

> 当前工作目录**不是 git 仓库**，`submodules/` 是空的——如果要真正跑起来，必须先拿到 `diff-gaussian-rasterization` 和 `simple-knn` 两个子模块（[.gitmodules](.gitmodules) 里有地址）。

---

## 3. 目录总览

```
TalkingGaussian-main/
├── arguments/              # 命令行参数定义（三组 ParamGroup）
├── assets/                 # README 用的示意图
├── data/                   # 数据集根目录（空，只有 .gitkeep）
├── data_utils/             # 【预处理】视频 → 训练数据 的全套脚本
│   ├── deepspeech_features/    # DeepSpeech 音频特征（VOCA 代码）
│   ├── easyportrait/           # 牙齿分割（内嵌 mmseg，184 个文件）
│   ├── face_parsing/           # BiSeNet 人脸解析
│   ├── face_tracking/          # 3DMM 拟合 → 相机位姿
│   └── *.py                    # 主入口 process.py + 三种音频特征提取
├── gaussian_renderer/      # 【渲染】三个 render 函数（静态/带运动/嘴部）
├── gridencoder/            # hash-grid 编码的 CUDA 扩展（torch-ngp）
├── lpipsPyTorch/           # LPIPS 官方实现的拷贝（未被引用，见 §8）
├── scene/                  # 【核心】GaussianModel / MotionNetwork / Scene / 数据读取
├── scripts/                # 一键预处理与训练的 shell 脚本
├── submodules/             # 空目录，需 git submodule 初始化
├── utils/                  # 通用工具（相机、损失、SH、图像…）
├── train_face.py           # 阶段 1a：训练上脸分支
├── train_mouth.py          # 阶段 1b：训练嘴部分支
├── train_fuse.py           # 阶段 2：融合两个分支
├── synthesize_fuse.py      # 推理 / 渲染测试视频
├── metrics.py              # 视频级评估（LMD / PSNR / LPIPS）
├── auerror.py              # AU 误差评估（需 OpenFace 输出）
├── encoding.py             # 位置编码工厂（hashgrid / frequency / SH…）
├── environment.yml
└── README.md
```

文件量分布：**内嵌第三方 `data_utils/easyportrait` 占了 184 个文件**，项目自身的有效代码只有约 50 个 Python 文件。

---

## 4. 模块关系与数据流

### 4.1 整体管线

```
                    ┌─────────────────────────────────────────┐
   data/<ID>/<ID>.mp4│  预处理（离线，只做一次）              │
                    └─────────────────────────────────────────┘
                                     │
              data_utils/process.py  │ 9 个 task，逐步产出
                                     ▼
   data/<ID>/ ├── ori_imgs/        抽帧（25FPS）
              ├── aud.wav          抽音频（16k 单声道）
              ├── aud_ds.npy       DeepSpeech 特征 [T,16,29]
              ├── parsing/         BiSeNet 19 类解析图（颜色编码掩码）
              ├── *.lms            face_alignment 68 点关键点
              ├── bc.jpg           重建出的静态背景图
              ├── gt_imgs/         用 bc.jpg 替换背景后的干净图
              ├── torso_imgs/      躯干+脖子（带 alpha 通道，竖向 inpainting）
              ├── au.csv           【手工】OpenFace 输出
              ├── teeth_mask/      【EasyPortrait】牙齿二值掩码 .npy
              ├── track_params.pt  3DMM 拟合出的位姿/焦距
              └── transforms_{train,val}.json  相机 c2w 矩阵 + 帧索引

                    ┌─────────────────────────────────────────┐
                    │  训练（三阶段，顺序不可颠倒）           │
                    └─────────────────────────────────────────┘
                                     │
        train_mouth.py ── 50k iters ─┼─► chkpnt_mouth_latest.pth
        train_face.py  ── 50k iters ─┼─► chkpnt_face_latest.pth
                                     │        （两者互相独立，可并行）
                                     ▼
        train_fuse.py  ── 10k iters ──► chkpnt_fuse_latest.pth
                                     │
                    ┌────────────────┴────────────────────────┐
                    │  推理 / 评估                             │
                    └─────────────────────────────────────────┘
        synthesize_fuse.py ─► output/<name>/test/ours_None/renders/out.mp4
        metrics.py         ─► 与 gt/out.mp4 对比，打印 LMD / PSNR / LPIPS
        auerror.py         ─► 与 OpenFace CSV 对比，打印 AU MSE
```

### 4.2 运行时对象图（训练阶段）

```
train_*.py
  ├─ arguments.ModelParams / OptimizationParams / PipelineParams   ← 参数组
  ├─ Scene(args, gaussians)                     ← scene/__init__.py
  │    ├─ dataset_readers.readNerfSyntheticInfo ← 读 transforms_*.json
  │    │    └─ readCamerasFromTransforms        ← 构造 talking_dict（掩码/音频/AU/嘴框）
  │    ├─ camera_utils.cameraList_from_camInfos ← CameraInfo → Camera
  │    └─ GaussianModel.create_from_pcd         ← 随机初始化点云
  ├─ MotionNetwork / MouthMotionNetwork         ← scene/motion_net.py
  │    └─ encoding.get_encoder('hashgrid')      ← encoding.py → gridencoder/
  ├─ gaussian_renderer.render / render_motion / render_motion_mouth
  │    └─ diff_gaussian_rasterization.GaussianRasterizer ⊗ CUDA 扩展
  └─ utils.loss_utils (l1 + D-SSIM + LPIPS patch) + lpips
```

### 4.3 三个 render 函数的关系

| 函数 | 位置 | 变形 | 用途 |
|---|---|---|---|
| `render()` | [gaussian_renderer/__init__.py:19](gaussian_renderer/__init__.py#L19) | **无**（原始 3DGS） | 每个分支训练的 warm-up 阶段（前 3000 iter），先建静态结构 |
| `render_motion()` | [gaussian_renderer/__init__.py:106](gaussian_renderer/__init__.py#L106) | `d_xyz / d_rot / d_scale` | 上脸分支；`return_attn=True` 时额外渲染注意力图用于正则 |
| `render_motion_mouth()` | [gaussian_renderer/__init__.py:201](gaussian_renderer/__init__.py#L201) | 仅 `d_xyz` | 嘴部分支 |

三者返回值统一为 `{render, viewspace_points, visibility_filter, depth, alpha, radii}`，前两者额外带 `motion`，`render_motion` 还带 `attn`。

> 细节：`MotionNetwork` 会预测 `d_opa`，但 `render_motion` 里不透明度那行被注释掉了（[gaussian_renderer/__init__.py:150](gaussian_renderer/__init__.py#L150)），所以 `d_opa` **只参与正则、不参与渲染**。

---

## 5. 逐目录逐文件说明

### 5.1 根目录

| 文件 | 作用 |
|---|---|
| [README.md](README.md) | 官方说明：安装、数据准备、训练、测试命令 |
| [environment.yml](environment.yml) | conda 环境定义（含两个本地 CUDA 扩展的 pip 安装） |
| [.gitmodules](.gitmodules) | 声明 `submodules/simple-knn` 与 `submodules/diff-gaussian-rasterization` |
| [.gitignore](.gitignore) | 忽略 `*.pth`、`data/*`、`output*`、3DMM 权重等大文件 |
| [train_face.py](train_face.py) | **阶段 1a**：训练上脸分支（50k iter） |
| [train_mouth.py](train_mouth.py) | **阶段 1b**：训练嘴部分支（50k iter） |
| [train_fuse.py](train_fuse.py) | **阶段 2**：融合（10k iter） |
| [synthesize_fuse.py](synthesize_fuse.py) | **推理**：加载 fuse checkpoint，导出 mp4 |
| [metrics.py](metrics.py) | **评估**：LMD / PSNR / LPIPS 三个 Meter 类 + 直接跑视频对比的主程序 |
| [auerror.py](auerror.py) | **评估**：AU 均方误差（需 OpenFace 处理后的 CSV，路径硬编码为 `./OpenFace_2.0_win_x64/processed`） |
| [encoding.py](encoding.py) | 位置编码工厂：`None / frequency / sphere_harmonics / hashgrid / tiledgrid / ash`，返回 `(encoder, output_dim)`。本项目的运动网络只用 `hashgrid` |

### 5.2 `arguments/`

单文件 [arguments/\_\_init\_\_.py](arguments/__init__.py)，是 gaussian-splatting 的**参数注入机制**：

- `ParamGroup` 基类用反射遍历实例属性自动注册 CLI 参数；属性名以 `_` 开头会同时注册 `-s` / `-S` 短选项。
- `ModelParams`：`sh_degree=2`、`source_path`、`model_path`、`images`、`resolution`、`white_background`、`data_device`、`eval`、`audio`、**`init_num=10000`（初始点数）**、**`audio_extractor`（deepspeech/esperanto/hubert）**。
- `PipelineParams`：`convert_SHs_python`、`compute_cov3D_python`、`debug`。
- `OptimizationParams`：50k iter、各参数学习率、`lambda_dssim=0.2`、稠密化区间与阈值。
- `get_combined_args(parser)`：**推理时专用**。先从 `model_path/cfg_args` 读回训练时的完整参数快照，再用命令行里非 None 的项覆盖。这就是为什么 `synthesize_fuse.py` 只要 `-S/-M` 就能复现训练配置。

### 5.3 `scene/` —— 核心目录

| 文件 | 作用 |
|---|---|
| [\_\_init\_\_.py](scene/__init__.py) | `Scene` 类：只支持 `transforms_train.json` 一种数据格式（断言强制）。负责复制 `input.ply` → `cameras.json`、随机打乱相机、按 `resolution_scales` 构建相机列表、初始化或加载 Gaussian 点云 |
| [gaussian_model.py](scene/gaussian_model.py) | **3DGS 数据表示**。在原始实现上多了一个 `_identity` 参数（当前恒为 0，为多身份预留）。含 `capture/restore`（checkpoint）、`save_ply/load_ply`、`densify_and_clone/split/prune`、`prune_points`、`add_densification_stats`。激活函数：scale=softplus、opacity=sigmoid、rotation=normalize |
| [motion_net.py](scene/motion_net.py) | **项目的核心创新**。四个类：`AudioAttNet`（对 8 帧音频窗口做时间维注意力池化）、`AudioNet`（4 层 1D 卷积把 29/44/1024 维音频压到 32/64 维）、`MLP`（无激活的最后一层）、`MotionNetwork`（上脸，输出 11 维）与 `MouthMotionNetwork`（嘴部，只输出 3 维 `d_xyz`）。位置编码用三个 2D hashgrid（xy/yz/xz 平面）拼接；上脸分支的 `eye_att` 由 6 维 AU 驱动 |
| [cameras.py](scene/cameras.py) | `Camera`（持 `talking_dict`、`world_view_transform`、`full_proj_transform`、`background`）与 `MiniCam`（GUI 用） |
| [dataset_readers.py](scene/dataset_readers.py) | **数据合同中心**，见下方 §6。核心产物是每帧的 `talking_dict` |
| [colmap_loader.py](scene/colmap_loader.py) | ⚠️ **死代码**：gaussian-splatting 的 COLMAP 读取器，全项目零引用（本项目只用姿态已拟合好的 json） |

`talking_dict` 的字段（[dataset_readers.py:193-239](scene/dataset_readers.py#L193-L239)）：

| 键 | 含义 |
|---|---|
| `auds` | 音频特征窗口 `[8, 29, 16]`（index±4） |
| `au_exp` | 6 维表情 `[AU01,04,05,06,07,45]` |
| `blink` | AU45 归一化到 `[0,1]`（用于采样） |
| `au25` | `[当前值, 25%, 50%, 75%, max]` 五元组 → 嘴张开程度的课程采样 |
| `face_mask` / `hair_mask` / `mouth_mask` | 由 `parsing` 图 + `teeth_mask` 合成 |
| `lips_rect` / `lhalf_rect` | 唇部外接框（已扩成正方形）/ 下半脸框 |
| `mouth_bound` | `[嘴宽min, 嘴宽max, 当前嘴宽]` |
| `img_id` | 帧号 |

### 5.4 `gaussian_renderer/`

| 文件 | 作用 |
|---|---|
| [\_\_init\_\_.py](gaussian_renderer/__init__.py) | 三个 render 函数（见 §4.3），并在此处 `re-export` 了 `GaussianModel / MotionNetwork / MouthMotionNetwork`，所以外部可以写 `from gaussian_renderer import GaussianModel` |
| [network_gui.py](gaussian_renderer/network_gui.py) | ⚠️ **死代码**：gaussian-splatting 的实时预览 socket 服务端，本项目训练脚本已不再调用（虽然注释里还留着 "Start GUI server"） |

### 5.5 `utils/`

| 文件 | 关键函数 | 说明 |
|---|---|---|
| [loss_utils.py](utils/loss_utils.py) | `l1_loss` `l2_loss` `ssim` `patchify` | `patchify` 用 `F.unfold` 把图切成随机尺寸 patch，供 LPIPS patch loss 使用 |
| [audio_utils.py](utils/audio_utils.py) | `get_audio_features(features, att_mode, index)` | `att_mode=2` 时取 `index-4 ~ index+4` 共 8 帧，越界补零 |
| [camera_utils.py](utils/camera_utils.py) | `loadCam` `cameraList_from_camInfos` `camera_to_JSON` **`loadCamOnTheFly`** | `loadCamOnTheFly` 是 `preload=False` 模式的关键：按需读原图、`bc.jpg`、`torso_imgs`、`teeth_mask`、`parsing` 并重建掩码 |
| [general_utils.py](utils/general_utils.py) | `PILtoTorch` `get_expon_lr_func` `build_rotation` `build_scaling_rotation` `strip_symmetric` `safe_state` | 注意 `PILtoTorch` **被改过：不做 /255 归一化**，图像以 0-255 表示，归一化在各训练脚本里手动做 |
| [graphics_utils.py](utils/graphics_utils.py) | `BasicPointCloud` `getWorld2View2` `getProjectionMatrix` `focal2fov` `fov2focal` | 相机数学基础 |
| [sh_utils.py](utils/sh_utils.py) | `eval_sh` `RGB2SH` `SH2RGB` | 0-3 阶球谐求值 |
| [image_utils.py](utils/image_utils.py) | `psnr` `mse` | 训练日志用 |
| [system_utils.py](utils/system_utils.py) | `mkdir_p` `searchForMaxIteration` | 文件工具 |

### 5.6 `data_utils/` —— 预处理

**顶层：**

| 文件 | 作用 |
|---|---|
| [process.py](data_utils/process.py) | **预处理总入口**，9 个 task 顺序执行，全通过 `os.system` 调子脚本或 ffmpeg：<br>1 抽音频 → 2 抽音频特征 → 3 抽帧 → 4 人脸解析 → 5 重建背景 `bc.jpg` → 6 生成 gt/torso 图 → 7 提取关键点 → 8 3DMM 位姿拟合 → 9 写出 `transforms_*.json`<br>可用 `--task N` 只跑某一步 |
| [hubert.py](data_utils/hubert.py) | 用 `facebook/hubert-large-ls960-ft` 提 1024 维特征，分块处理长音频，输出 `*_hu.npy` |
| [wav2vec.py](data_utils/wav2vec.py) | RAD-NeRF 移植的 Esperanto wav2vec2 ASR 类（`cpierse/wav2vec2-large-xlsr-53-esperanto`），输出 `*_eo.npy`；还带麦克风实时推理的 demo。⚠️ [process.py:21](data_utils/process.py#L21) 里调用的是 `python nerf/asr.py`，**该路径在本仓库不存在**（RAD-NeRF 遗留），用 wav2vec 模式需手动改成调本文件 |
| [wav2mel.py](data_utils/wav2mel.py) / [wav2mel_hparams.py](data_utils/wav2mel_hparams.py) | Mel 频谱提取（RAD-NeRF 遗留），主流程未使用 |

**`deepspeech_features/`**（来自 VOCA/AD-NeRF，本项目默认音频特征来源）：

| 文件 | 作用 |
|---|---|
| [extract_ds_features.py](data_utils/deepspeech_features/extract_ds_features.py) | CLI 入口：`--input <wav>` → 输出同名 `.npy`，内容为 `[T, 16, 29]` 的滑动窗口 |
| [deepspeech_features.py](data_utils/deepspeech_features/deepspeech_features.py) | 核心实现：MFCC(26)+上下文窗口(9) → TensorFlow 加载 `deepspeech-0_1_0-b90017e8.pb` 图 → logits → 插值到视频帧率 → 滑窗 |
| [deepspeech_store.py](data_utils/deepspeech_features/deepspeech_store.py) | 自动下载并校验 DeepSpeech pb 模型（sha1 校验，存到 `~/.tensorflow/models`） |
| [extract_wav.py](data_utils/deepspeech_features/extract_wav.py) | 从视频批量抽 16k 单声道 wav（ffmpeg 封装） |
| [fea_win.py](data_utils/deepspeech_features/fea_win.py) | 独立小工具脚本：把 `french.ds.npy` 重新切窗口，**硬编码文件名，仅作参考** |
| [README.md](data_utils/deepspeech_features/README.md) | 上游说明（提到的 `requirements.txt` 在本仓库并不存在） |

**`face_parsing/`** —— BiSeNet 19 类人脸解析：

| 文件 | 作用 |
|---|---|
| [test.py](data_utils/face_parsing/test.py) | CLI 入口（`--imgpath` / `--respath` / `--modelpath`），推理并 `vis_parsing_maps` 把类别 id **映射成颜色**写 png。颜色约定：**红=脸、绿=脖子、蓝=躯干、黑=头发、`(100,100,100)`=嘴、白=背景**。这套约定被 `dataset_readers` 和 `process.extract_torso_and_gt` 共同依赖 |
| [model.py](data_utils/face_parsing/model.py) | BiSeNet 结构（SpatialPath + ContextPath + FeatureFusionModule + BiSeNetOutput） |
| [resnet.py](data_utils/face_parsing/resnet.py) | 作为 ContextPath backbone 的 ResNet18 |
| [logger.py](data_utils/face_parsing/logger.py) | 训练时的日志工具（本项目只用推理，实际上没用到） |
| ⚠️ `79999_iter.pth` | **权重文件缺失**，由 [scripts/prepare.sh](scripts/prepare.sh) 下载；已列入 `.gitignore` |

**`face_tracking/`** —— 3DMM 拟合，把 68 点关键点拟合成相机位姿：

| 文件 | 作用 |
|---|---|
| [face_tracker.py](data_utils/face_tracking/face_tracker.py) | 主流程，四段式优化：① 遍历 focal ∈ [600,1500) 拟合焦距 ② 全帧粗拟合（位姿+id+exp） ③ 拟合光照与纹理（可微渲染） ④ **逐 batch 精拟合**（带 Laplacian 时序平滑 `loss_lap`）。最终 `torch.save` 出 `track_params.pt`（含 `id/exp/euler/trans/focal`） |
| [facemodel.py](data_utils/face_tracking/facemodel.py) | `Face_3DMM`：从 `3DMM_info.npy` 载入 shape/exp/tex 基底，提供 `get_3dlandmarks`（轮廓点用 argmin/argmax 做可见性选择）、`forward_geo`、`forward_tex` |
| [render_3dmm.py](data_utils/face_tracking/render_3dmm.py) | 基于 pytorch3d 的可微渲染器，把 3DMM mesh + 纹理 + 光照渲成图，用于第 ③ 步的 color loss |
| [util.py](data_utils/face_tracking/util.py) | 损失函数（`cal_lan_loss` 关键点、`cal_col_loss` 颜色、`cal_lap_loss` 拉普拉斯）与投影变换 |
| [geo_transform.py](data_utils/face_tracking/geo_transform.py) | 独立的旋转/投影工具（`euler2rot` 等），与 util.py 功能重叠 |
| [data_loader.py](data_utils/face_tracking/data_loader.py) | 读 `*.lms` 文件到一个张量 |
| [convert_BFM.py](data_utils/face_tracking/convert_BFM.py) | **一次性脚本**：把 Basel Face Model 的 `01_MorphableModel.mat` 与 `exp_info.npy`/`topology_info.npy` 合并成 `3DMM_info.npy` |
| [render_land.py](data_utils/face_tracking/render_land.py) | ⚠️ **不可用**：`import render_util`，而 `render_util` 在本仓库中不存在（应是被编译的 CUDA 扩展，未提供）。全项目零引用 |
| `3DMM/` | 权重目录，**当前只有 `.gitkeep`**。需要 `prepare.sh` 下载 4 个文件 + 手动放入 `01_MorphableModel.mat` 后跑 `convert_BFM.py` |

**`easyportrait/`** —— 牙齿分割（内嵌第三方，184 个文件）：

| 文件 | 作用 |
|---|---|
| [create_teeth_mask.py](data_utils/easyportrait/create_teeth_mask.py) | **唯一需要关心的文件**。读 `data/<ID>/ori_imgs/*.jpg`，用 mmseg 的 `fpn-fp` 配置 + `fpn-fp-512.pth` 推理，保留类别 id **7**（牙齿），存成 `teeth_mask/<id>.npy` 布尔数组 |
| `mmseg/` | 内嵌裁剪版 OpenMMLab 语义分割库（apis / models / datasets / core / ops），**仅供上面这个脚本使用** |
| `local_configs/` | 实验配置：`__base__`（datasets/models/schedules 基础配置）+ `easyportrait_experiments_v2/` 下 14 个模型的配置（fpn-fp 是默认用的那个） |
| ⚠️ `fpn-fp-512.pth` | 权重缺失，需按 README 用 wget 下载 |

> 运行前必须 `export PYTHONPATH=./data_utils/easyportrait`，否则 `import mmseg` 失败。

### 5.7 `gridencoder/`

torch-ngp 的 hash-grid 编码 CUDA 扩展，**通过 [encoding.py](encoding.py) 被 `scene/motion_net.py` 使用**（`get_encoder('hashgrid', ...)`）。

| 文件 | 作用 |
|---|---|
| [grid.py](gridencoder/grid.py) | `GridEncoder` 模块 + 自定义 `torch.autograd.Function` `_grid_encode`（前向/反向手写，支持 `hash` 与 `tiled` 两种 gridtype） |
| [backend.py](gridencoder/backend.py) | 扩展加载与 `assert` 环境检查 |
| [setup.py](gridencoder/setup.py) | 编译脚本（JIT 或 `pip install .`） |
| `src/gridencoder.cu` / `.h` / `bindings.cpp` | CUDA 内核与 pybind 绑定 |

### 5.8 `lpipsPyTorch/`

LPIPS 官方实现的一份拷贝（`LPIPS` 模块 + alex/vgg/squeeze 网络定义 + 工具函数）。
⚠️ **全项目零引用** —— 训练脚本用的是 pip 安装的 `lpips` 包。属于打包时一并带上的冗余目录，可以安全忽略。

### 5.9 `scripts/`

| 文件 | 作用 |
|---|---|
| [train_xx.sh](scripts/train_xx.sh) | **完整流程脚本**：`bash scripts/train_xx.sh data/<ID> output/<name> <GPU_ID>`。内容 = train_mouth → train_face → train_fuse → synthesize_fuse → metrics。文件里还注释掉了"两卡并行"版本（因为 mouth 与 face 互不依赖） |
| [prepare.sh](scripts/prepare.sh) | 下载人脸解析权重 `79999_iter.pth` 与 4 个 3DMM 文件 |

> 这两个是 **bash 脚本**，在 Windows PowerShell 下需要手动逐条执行。

### 5.10 `data/`、`assets/`、`submodules/`

| 路径 | 说明 |
|---|---|
| [data/](data/) | 数据集根，当前只有 `.gitkeep`。约定为 `data/<ID>/<ID>.mp4` |
| [assets/main.png](assets/main.png) | README 顶部的效果图 |
| `submodules/` | **空目录**。需初始化 `diff-gaussian-rasterization`（ashawkey fork：额外输出 alpha 与 depth）和 `simple-knn`（`distCUDA2` 被 `GaussianModel.create_from_pcd` 使用） |

---

## 6. `data/<ID>/` 数据目录合同

这是理解整个 pipeline 的关键——**每个文件的格式和来源**：

| 路径 | 产出者 | 格式 / 说明 |
|---|---|---|
| `<ID>.mp4` | 用户 | 25FPS、全员像、约 512×512、1-5 分钟 |
| `ori_imgs/<id>.jpg` | `process.py` task 3 | 抽帧，编号从 0 开始 |
| `ori_imgs/<id>.lms` | task 7 | 68×2 关键点文本（face_alignment） |
| `aud.wav` | task 1 | 16kHz 单声道 |
| `aud_ds.npy` | task 2 | `[T, 16, 29]`，**注意文件名固定为 `aud_ds`**（另有 `aud_eo.npy` / `aud_hu.npy`，由 `postfix_dict` 决定） |
| `parsing/<id>.png` | task 4 | 19 类解析图，**用 RGB 颜色编码类别**（红脸/绿脖/蓝躯/黑发/灰嘴/白背景） |
| `bc.jpg` | task 5 | 用 KNN 从多帧中重建的静态背景图 |
| `gt_imgs/<id>.jpg` | task 6 | 把 `ori_imgs` 里的背景像素替换成 `bc.jpg` |
| `torso_imgs/<id>.png` | task 6 | 4 通道 RGBA，躯干+脖子，做了竖向 inpainting（每列取顶部像素向上复制并逐行变暗 0.98） |
| `au.csv` | **用户手工** | OpenFace FeatureExtraction 输出（列名形如 `' AU25_r'`，**前面有空格**） |
| `teeth_mask/<id>.npy` | EasyPortrait | 布尔数组 |
| `track_params.pt` | task 8 | dict：`id/exp/euler/trans/focal` |
| `transforms_train.json` / `transforms_val.json` | task 9 | `focal_len` + `frames[{img_id, aud_id, transform_matrix}]`；**训练/验证按 10:11 切分**（前 10/11 训练） |

> `readCamerasFromTransforms` 会检查 `frame['img_id'] > auds.shape[0]` 并 **break**——所以音频长度不足会静默截断数据集。

---

## 7. 训练与推理细节对照

| | train_mouth.py | train_face.py | train_fuse.py |
|---|---|---|---|
| 起始点 | 随机点云 `init_num=10000` | `init_num=2000`（脚本传入） | 加载前两者的 checkpoint |
| 迭代数 | 50000 | 50000 | **10000**（硬编码覆盖） |
| densify | 到 `iterations-1000` | 到 `iterations-1000` | **`densify_until_iter=0`，完全不稠密化** |
| warm-up | 前 3000 iter 用静态 `render()` | 同左 | 无（直接 `render_motion`） |
| 背景色 | 前段绿 `[0,1,0]` → 最后 1000 iter 转黑 | 全程绿 | 绿 → 最后 1000 iter 用真实背景 |
| GT 构造 | `gt * mouth_mask + bg * ~mouth_mask` | `gt * head_mask + bg * ~head_mask` | `gt * head_mask + bg * ~head_mask` |
| 采样策略 | 按 AU25 从高到低逐步扩大区间 | 按 `mouth_bound`（前 3000 iter）+ `blink`（之后） | 纯随机 |
| 正则 | lips 框 alpha 正则 | alpha 正则 + `attn` 正则（嘴框/头发区域）+ 变形量 L1 | alpha 正则 |
| LPIPS | `patchify` patch(16-21)*2，权重 0.5 | 唇框整块 + 全图 patch，权重 0.01 / 0.2 | patch(16-21)*2，权重 0.5 |
| 打磨 | 对绿色高斯降梯度、降不透明度、缩尺寸 | `prune_points` 直接删绿色高斯 | — |
| 输出 | `chkpnt_mouth_{iter,latest}.pth` | `chkpnt_face_{iter,latest}.pth` | `chkpnt_fuse_{iter,latest}.pth` |

**checkpoint 结构不同，注意对应关系：**
- face/mouth：`(gaussians.capture(), motion_net.state_dict(), optimizer.state_dict(), iteration)`
- fuse：`(gaussians, motion_net, gaussians_mouth, motion_net_mouth)` —— **不含 optimizer、不含 iteration**

推理脚本 [synthesize_fuse.py:97](synthesize_fuse.py#L97) 只读 `chkpnt_fuse_latest.pth`。

**推理时的合成公式**（[synthesize_fuse.py:59-68](synthesize_fuse.py#L59-L68)）：

```
mouth_image = mouth_render + real_bg * (1 - alpha_mouth)
image       = face_render  + mouth_image * (1 - alpha_face)
```

即：先用嘴部分支的 alpha 把嘴贴到真实背景上，再用上脸分支的 alpha 让脸盖住嘴——**嘴部区域由上脸 alpha 决定"露多少"**。

**环境变量**：`train_xx.sh` 用 `CUDA_VISIBLE_DEVICES` 指定 GPU；Python 侧 `safe_state()` 会再次 `torch.cuda.set_device("cuda:0")` 并固定随机种子。

---

## 8. 已知坑与遗留问题

按"踩坑概率"排序：

1. **`submodules/` 为空** → `import diff_gaussian_rasterization` / `simple_knn._C` 直接失败。这是本项目最前置的阻塞点，且当前不是 git 仓库，无法用 `git submodule update --init` 一步解决。
2. **`PYTHONPATH`**：跑 EasyPortrait 生成牙齿掩码前必须 `export PYTHONPATH=./data_utils/easyportrait`。
3. **两个权重文件缺失**：`data_utils/face_parsing/79999_iter.pth`、`data_utils/easyportrait/fpn-fp-512.pth`；3DMM 目录还需 4 个 npy/obj + 手动下载的 `01_MorphableModel.mat` 并执行 `convert_BFM.py`。
4. **`process.py` 的 wav2vec 分支指向不存在的路径**（`nerf/asr.py`）。默认 `deepspeech` 不受影响。
5. **`data_utils/face_tracking/render_land.py` 无法导入**（缺 `render_util`），`scene/colmap_loader.py`、`gaussian_renderer/network_gui.py`、`lpipsPyTorch/` 均无引用。以上都不影响主流程，但搜索代码时容易误入。
6. **`au.csv` 的列名带前导空格**（`' AU25_r'`），自己解析时容易漏。且 `au.csv` 需要用户手工用 OpenFace 生成，README 只提了一句。
7. **无音频 / 音频过短**会被静默截断或 break，不报错。
8. **内存**：默认 `preload=True`，每帧都常驻原图 + 4 张掩码 + 背景。5k 帧量级就需要 32GB+。改 `preload=False`（[dataset_readers.py:98](scene/dataset_readers.py#L98)）可换时间省内存。
9. **参数被硬编码覆盖**：`train_fuse.py` 里 `opt.iterations = 10000`、`densify_until_iter = 0` 会覆盖命令行；`train_face/mouth` 里 `densify_until_iter = iterations - 1000` 也是硬写。调参时要知道命令行的 `--iterations` 在 fuse 阶段无效。
10. **`scene/dataset_readers.py:276` 的 `or True`** 使得每次都会重新生成随机点云 ply，`fetchPly` 的复用逻辑实际无效——结果是每次启动训练初始点云都不同（这也意味着**固定种子后仍不完全可复现**）。
11. **Window 环境**：脚本是 bash；`loadCamOnTheFly` 用 `"/"` 拼路径（Python 能兼容）、`metrics.py` 里 `.cuda()` 与 `face_alignment` 为硬编码，需 NVIDIA GPU。
12. **评估路径硬编码**：`auerror.py` 要求当前目录下存在 `./OpenFace_2.2.0_win_x64/processed/`；`metrics.py` 用 `sys.argv` 直接取两个视频路径。

---

## 9. 推荐阅读顺序

### 路线 A：想先跑通（约 30 分钟）

1. [README.md](README.md) —— 安装与命令全貌
2. [environment.yml](environment.yml) —— 依赖与 CUDA 扩展清单
3. [scripts/train_xx.sh](scripts/train_xx.sh) —— 5 行看懂完整流程
4. [scripts/prepare.sh](scripts/prepare.sh) —— 缺哪些权重
5. [data_utils/process.py](data_utils/process.py) —— 数据从哪来（重点看 `__main__` 里的 9 个 task）

### 路线 B：理解模型原理（推荐主线）

按"数据 → 表示 → 网络 → 渲染 → 训练 → 推理"的顺序：

| 序 | 文件 | 读它的目的 |
|---|---|---|
| 1 | [README.md](README.md) | 问题设定与整体框架 |
| 2 | [arguments/__init__.py](arguments/__init__.py) | 有哪些超参、`get_combined_args` 的机制 |
| 3 | [scene/dataset_readers.py](scene/dataset_readers.py) | **数据合同**：`talking_dict` 的每个字段从哪来 |
| 4 | [utils/camera_utils.py](utils/camera_utils.py) | 相机/掩码的按需加载路径 |
| 5 | [utils/graphics_utils.py](utils/graphics_utils.py) + [utils/sh_utils.py](utils/sh_utils.py) | 相机数学与球谐（3DGS 必备背景） |
| 6 | [scene/gaussian_model.py](scene/gaussian_model.py) | 3DGS 的参数与其稠密化/剪枝机制 |
| 7 | [encoding.py](encoding.py) → [gridencoder/grid.py](gridencoder/grid.py) | hash-grid 位置编码怎么接入 |
| 8 | [scene/motion_net.py](scene/motion_net.py) | **核心创新**：两个运动网络的结构差异 |
| 9 | [gaussian_renderer/\_\_init\_\_.py](gaussian_renderer/__init__.py) | 三个 render 函数的差异（哪个分支变形什么） |
| 10 | [train_face.py](train_face.py) | 上脸分支的完整训练策略（warm-up / 采样 / 掩码 / 剪枝） |
| 11 | [train_mouth.py](train_mouth.py) | 与上脸的**差异点**逐条对照读 |
| 12 | [train_fuse.py](train_fuse.py) | 融合阶段怎么把两套结果 alpha 合成 |
| 13 | [synthesize_fuse.py](synthesize_fuse.py) | 推理时的合成公式与产物路径 |
| 14 | [metrics.py](metrics.py) + [auerror.py](auerror.py) | 论文里的指标是怎么算的 |

> 关键提示：**读 10/11 时一定要对照着读**，两个文件结构几乎一致，差异全部在细节（背景色切换、采样依据、正则项、剪枝方式）里，这些差异正是论文的卖点。

### 路线 C：只想改数据预处理

1. [data_utils/process.py](data_utils/process.py) —— 调度顺序
2. [data_utils/deepspeech_features/deepspeech_features.py](data_utils/deepspeech_features/deepspeech_features.py) —— 音频特征
3. [data_utils/face_parsing/test.py](data_utils/face_parsing/test.py) + [model.py](data_utils/face_parsing/model.py) —— 颜色编码掩码的来源
4. [data_utils/face_tracking/face_tracker.py](data_utils/face_tracking/face_tracker.py) —— 位姿拟合（配合 [geo_transform.py](data_utils/face_tracking/geo_transform.py)、[util.py](data_utils/face_tracking/util.py)）
5. [data_utils/face_tracking/convert_BFM.py](data_utils/face_tracking/convert_BFM.py) —— 3DMM 权重格式
6. [data_utils/easyportrait/create_teeth_mask.py](data_utils/easyportrait/create_teeth_mask.py) —— 牙齿掩码（`mmseg/` 可整体当黑盒）

### 路线 D：可用音频驱动推理（最小闭环）

若已有别人训好的 checkpoint，最短路径是：
[arguments/\_\_init\_\_.py](arguments/__init__.py)（看 `get_combined_args`）→ [synthesize_fuse.py](synthesize_fuse.py) → [data_utils/deepspeech_features/extract_ds_features.py](data_utils/deepspeech_features/extract_ds_features.py)（准备自定义音频特征）。

---

## 10. 命令速查

```bash
# ---- 预处理 ----
python data_utils/process.py data/<ID>/<ID>.mp4          # 9 步全跑
python data_utils/process.py data/<ID>/<ID>.mp4 --task 6 # 只跑第 6 步
export PYTHONPATH=./data_utils/easyportrait
python ./data_utils/easyportrait/create_teeth_mask.py ./data/<ID>   # 牙齿掩码

# ---- 音频特征 ----
python data_utils/deepspeech_features/extract_ds_features.py --input data/<ID>/aud.wav  # → aud_ds.npy
python data_utils/hubert.py --wav data/<ID>/aud.wav                                     # → aud_hu.npy

# ---- 训练（三阶段）----
python train_mouth.py -s data/<ID> -m output/<name> --audio_extractor deepspeech
python train_face.py  -s data/<ID> -m output/<name> --init_num 2000 --densify_grad_threshold 0.0005
python train_fuse.py  -s data/<ID> -m output/<name> --opacity_lr 0.001
# 或一键：bash scripts/train_xx.sh data/<ID> output/<name> 0

# ---- 推理 ----
python synthesize_fuse.py -S data/<ID> -M output/<name> --eval
python synthesize_fuse.py -S data/<ID> -M output/<name> --use_train --audio <feat>.npy

# ---- 评估 ----
python metrics.py output/<name>/test/ours_None/renders/out.mp4 output/<name>/test/ours_None/gt/out.mp4
python auerror.py A_generated A_GT   # 需要 ./OpenFace_2.2.0_win_x64/processed/*.csv

# ---- 可视化 ----
tensorboard --logdir output/
```

**产物目录结构**（`output/<name>/`）：

```
cfg_args                     # 训练参数的 Namespace 快照，推理时回读
cameras.json / input.ply     # Scene 初始化时写入
chkpnt_{face,mouth,fuse}_{iter,latest}.pth
point_cloud/iteration_{N}[_face|_mouth]/point_cloud.ply
```
