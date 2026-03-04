<div align="center">

<h2>CFG-Ctrl: 基于控制理论的 Classifier-Free Guidance</h2>

<p><b>CVPR 2026</b></p>

Hanyang Wang*, Yiyang Liu*, Jiawei Chi, Fangfu Liu, Ran Xue, Yueqi Duan†<br>
Tsinghua University<br>
<sub>* Equal contribution &nbsp;&nbsp; † Corresponding author</sub>

<a href='https://github.com/MaxLiuyy/CFG-Ctrl'><img src='https://img.shields.io/badge/GitHub-Repo-blue'></a>
<a href='https://github.com/MaxLiuyy/CFG-Ctrl/blob/main/LICENSE'><img src='https://img.shields.io/badge/License-Apache_2.0-green.svg'></a>
<a href='https://arxiv.org/abs/2603.03281'><img src='https://img.shields.io/badge/ArXiv-2603.03281-red'></a>

</div>

## 摘要

Classifier-Free Guidance (CFG) 已成为基于流匹配的扩散模型中增强语义对齐的核心方法。本文提出统一框架 **CFG-Ctrl**，将 CFG 重新解释为施加在一阶连续时间生成流上的控制，以条件与无条件预测之差作为误差信号来调整速度场。在此框架下，标准 CFG 等价于固定增益的比例控制器（P-control），而现有变体则对应不同的控制律设计。现有方法依赖线性控制，在大引导尺度下存在不稳定、过冲和语义保真度下降的问题。为此，本文提出 **SMC-CFG（Sliding Mode Control CFG）**，将生成流强制引导至快速收敛的滑模流形。具体地，在语义预测误差上定义指数滑模面，并引入切换控制项实现非线性反馈纠正，同时提供 Lyapunov 稳定性分析以理论保证有限时间收敛。在 Stable Diffusion 3.5、Flux、Qwen-Image 等文生图模型上的实验表明，SMC-CFG 在语义对齐和大范围引导尺度鲁棒性方面均优于标准 CFG。

## 更新日志

- `2026/02/26` 代码发布
- `2026/03/04` ArXiv 论文发布：[2603.03281](https://arxiv.org/abs/2603.03281)

## TODO

- [x] 发布推理代码及 pipeline 实现
- [x] 发布 ArXiv 论文

## 目录

- [方法概述](#方法概述)
- [安装](#安装)
- [快速开始](#快速开始)
- [支持的模型](#支持的模型)
- [参数说明](#参数说明)
- [引用](#引用)
- [致谢](#致谢)

## 方法概述

**CFG-Ctrl** 提供了一个统一的控制论框架来理解和改进 Classifier-Free Guidance：

- **标准 CFG** 等价于固定增益 `w` 的比例控制器（P-control）：
  ```
  v_guided = v_uncond + w * (v_cond - v_uncond)
  ```

- **SMC-CFG** 引入非线性滑模控制器，将引导误差强制收敛至指数滑模面：
  ```
  s_t = (e_t - e_{t-1}) + lambda * e_{t-1}    （滑模面）
  u_sw = -K * sign(s_t)                         （切换控制）
  ```
  其中 `e_t = v_cond - v_uncond` 为引导误差信号，`lambda` 控制指数衰减率，`K` 为切换增益。

该设计有效避免了标准 CFG 在大引导尺度下的不稳定和过冲现象。

## 安装

### 环境要求

- Python >= 3.10（推荐 3.10）
- 支持 CUDA 的 GPU

### 方式一：Conda（推荐）

```bash
git clone https://github.com/MaxLiuyy/CFG-Ctrl.git
cd CFG-Ctrl
conda env create -f environment.yml
conda activate cfg-ctrl
```

### 方式二：pip

```bash
conda create -n cfg-ctrl python=3.10 -y
conda activate cfg-ctrl
git clone https://github.com/MaxLiuyy/CFG-Ctrl.git
cd CFG-Ctrl
pip install -r requirements.txt
```

## 快速开始

所有示例需在 `CFG-Ctrl/` 目录下运行。

### 文生图（FLUX）

```bash
python examples/flux_cfg_ctrl_example.py \
  --prompt "A cinematic portrait of a cat astronaut" \
  --cfg-scale 3 \
  --steps 30 \
  --smc-cfg-enable \
  --smc-cfg-lambda 5.0 \
  --smc-cfg-k 0.2 \
  --no-cfg-warmup-steps 2
```

### 文生图（Qwen-Image）

```bash
python examples/qwen_cfg_ctrl_example.py \
  --prompt "A ginger cat sitting by a rainy window, cinematic lighting" \
  --cfg-scale 4 \
  --steps 30 \
  --smc-cfg-enable \
  --smc-cfg-lambda 5.0 \
  --smc-cfg-k 0.2
```

### 文生图（SD3）

```bash
python examples/sd3_cfg_ctrl_example.py \
  --prompt "A futuristic city at sunrise, cinematic, ultra detailed" \
  --cfg-scale 7.5 \
  --steps 30 \
  --smc-cfg-enable \
  --smc-cfg-lambda 5.0 \
  --smc-cfg-k 0.2
```

### 文生视频（Wan Video）

```bash
python examples/wan_video_cfg_ctrl_example.py \
  --prompt "A golden retriever running on a beach at sunset" \
  --cfg-scale 5 \
  --steps 50 \
  --smc-cfg-enable \
  --smc-cfg-lambda 5.0 \
  --smc-cfg-k 0.2
```

图生视频模式添加 `--input-image path/to/image.png`。

## 支持的模型

| 模型 | Pipeline 类 | 任务 |
|------|------------|------|
| [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | `FluxImagePipelineCFGCtrl` | 文生图 |
| [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | `QwenImagePipelineCFGCtrl` | 文生图 |
| SD3 / SD3.5 | `SD3ImagePipelineCFGCtrl` | 文生图 |
| [Wan2.1](https://huggingface.co/Wan-AI) / Wan2.2 | `WanVideoPipelineCFGCtrl` | 文生视频 / 图生视频 |

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `smc_cfg_enable` | bool | False | 启用 SMC-CFG 稳定化 |
| `smc_cfg_lambda` | float | 0.05 | 滑模面的指数衰减系数 |
| `smc_cfg_k` / `smc_cfg_K` | float | 0.3 | 切换增益 |
| `no_cfg_warmup_steps` | int | 0 | 前 N 步不使用 CFG |

**推荐参数**：
- FLUX: `smc_cfg_lambda=5.0, smc_cfg_k=0.2, cfg_scale=2~3`
- Qwen-Image: `smc_cfg_lambda=5.0, smc_cfg_k=0.2, cfg_scale=4`
- SD3/SD3.5: `smc_cfg_lambda=5.0, smc_cfg_k=0.2, cfg_scale=7.5`
- Wan Video: `smc_cfg_lambda=5.0, smc_cfg_k=0.2, cfg_scale=5`

## 引用

```bibtex
@inproceedings{cfg-ctrl,
  title={CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance},
  author={Wang, Hanyang and Liu, Yiyang and Chi, Jiawei and Liu, Fangfu and Xue, Ran and Duan, Yueqi},
  booktitle={CVPR},
  year={2026},
  note={arXiv:2603.03281},
  url={https://arxiv.org/abs/2603.03281}
}
```

## 致谢

本项目基于 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 构建。
