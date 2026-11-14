#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), ("-" + key[0:1].upper()), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), ("-" + key[0:1].upper()), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 2             # 球谐函数阶数，决定视角相关颜色的表达能力（3 阶即每通道 9 个系数）
        self._source_path = ""         # -s：数据目录，即 data/<ID>（extract 中会被转成绝对路径）
        self._model_path = ""          # -m：模型与日志的输出目录，即 output/<project_name>
        self._images = "images"        # -i：图像子目录名（gaussian-splatting 遗留参数；本分支改从 transforms_*.json 读图，实际未使用）
        self._resolution = -1          # -r：图像下采样分辨率（同样为遗留参数；本分支固定用原始分辨率，实际未使用）
        self._white_background = False # -w：是否按白色背景合成图像；本分支背景由 bc.jpg 与绿幕合成逻辑决定，一般不启用
        self.data_device = "cpu"       # 相机图像/GT 的驻留设备；设为 cpu 可省显存，渲染前再搬到 GPU
        self.eval = False              # 评估模式：只用 transforms_val.json 当训练集，不做真正的训练/测试划分
        self.audio = ""                # 推理时指定的音频特征 .npy 路径；留空则读数据目录下的 aud_ds/aud_hu/aud_eo.npy
        self.init_num = 10_000         # 初始高斯点数量，用于随机生成初始点云 points3d.ply
        self.audio_extractor = "deepspeech"  # 音频特征提取器：deepspeech(29维) / hubert(1024维) / esperanto(44维)
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)

        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False    # 用 Python 而非 CUDA 核计算球谐颜色（仅供调试，速度很慢）
        self.compute_cov3D_python = False  # 用 Python 而非 CUDA 核计算 3D 协方差（仅供调试，速度很慢）
        self.debug = False                 # 把渲染中间变量塞进 render_pkg 返回，便于排查问题
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 50_000                # 总训练迭代数；train_*.py 会基于它推导各阶段的分界点
        self.position_lr_init = 0.00016         # 高斯位置学习率初值
        self.position_lr_final = 0.0000016      # 位置学习率末值（指数衰减的目标值）
        self.position_lr_delay_mult = 0.01      # 位置学习率的 warm-up 系数：起始值为 init × 该系数，避免前期跳变
        self.position_lr_max_steps = 45_000     # 位置学习率衰减到 final 所需的步数
        self.feature_lr = 0.0025                # 颜色（球谐系数）学习率
        self.opacity_lr = 0.05                  # 不透明度学习率
        self.scaling_lr = 0.003                 # 高斯尺度学习率
        self.rotation_lr = 0.001                # 高斯旋转（四元数）学习率
        self.percent_dense = 0.005              # 稠密化分类阈值：尺度超过 该比例×场景范围 的算"大高斯"，用 split 而非 clone
        self.lambda_dssim = 0.2                 # DSSIM 项权重：loss = (1-λ)·L1 + λ·(1-SSIM)
        self.densification_interval = 100       # 每多少步执行一次稠密化（clone / split）
        self.opacity_reset_interval = 3000      # 每多少步把不透明度重置一次，用于清除漂浮/冗余高斯
        self.densify_from_iter = 500            # 从第几步开始允许稠密化


        self.densify_until_iter = 45_000        # 停止稠密化的步数（train_*.py 会覆盖为 iterations - 1000）
        self.densify_grad_threshold = 0.0002    # 位置梯度均值超过该阈值的高斯才参与稠密化
        self.random_background = False          # 训练时随机化背景色以抑制背景过拟合；本分支改用固定绿幕，未启用
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
