<div align="center">

<h1>CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance</h1>

<h3 align="center">CVPR 2026</h3>

Hanyang Wang*, Yiyang Liu*, Jiawei Chi, Fangfu Liu, Ran Xue, Yueqi Duan†<br>
Tsinghua University<br>
<sub>* Equal contribution &nbsp;&nbsp; † Corresponding author</sub>

<a href='https://hanyang-21.github.io/CFG-Ctrl'><img src='https://img.shields.io/badge/Project-Website-green.svg'></a>
<a href='https://arxiv.org/abs/XXXX.XXXXX'><img src='https://img.shields.io/badge/ArXiv-XXXX.XXXXX-red'></a>
<a href='https://github.com/hanyang-21/CFG-Ctrl/blob/main/LICENSE'><img src='https://img.shields.io/badge/License-Apache_2.0-green.svg'></a>


</div>

## Abstract

Classifier-Free Guidance (CFG) has emerged as a central approach for enhancing semantic alignment in flow-based diffusion models. In this paper, we explore a unified framework called **CFG-Ctrl**, which reinterprets CFG as a control applied to the first-order continuous-time generative flow, using the conditional-unconditional discrepancy as an error signal to adjust the velocity field. From this perspective, we summarize vanilla CFG as a proportional controller (P-control) with fixed gain, and typical follow-up variants develop extended control-law designs derived from it. However, existing methods mainly rely on linear control, inherently leading to instability, overshooting, and degraded semantic fidelity especially on large guidance scales. To address this, we introduce Sliding Mode Control CFG (**SMC-CFG**), which enforces the generative flow toward a rapidly convergent sliding manifold. Specifically, we define an exponential sliding mode surface over the semantic prediction error and introduce a switching control term to establish nonlinear feedback-guided correction. Moreover, we provide a Lyapunov stability analysis to theoretically support finite-time convergence. Experiments across text-to-image generation models including Stable Diffusion 3.5, Flux, and Qwen-Image demonstrate that SMC-CFG outperforms standard CFG in semantic alignment and enhances robustness across a wide range of guidance scales.

## Updates

- `2026/02/26` Code released.

## TODO List

- [x] Release inference code and pipeline implementations
- [ ] Release ArXiv paper

## Table of Contents

- [Abstract](#abstract)
- [Updates](#updates)
- [TODO List](#todo-list)
- [Table of Contents](#table-of-contents)
- [Method Overview](#method-overview)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Option A: Conda (recommended)](#option-a-conda-recommended)
  - [Option B: pip](#option-b-pip)
- [Quick Start](#quick-start)
  - [Text-to-Image (FLUX)](#text-to-image-flux)
  - [Text-to-Image (Qwen-Image)](#text-to-image-qwen-image)
  - [Text-to-Image (SD3)](#text-to-image-sd3)
  - [Text-to-Video (Wan Video)](#text-to-video-wan-video)
- [Supported Models](#supported-models)
- [Parameters](#parameters)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Method Overview

**CFG-Ctrl** provides a unified control-theoretic framework for Classifier-Free Guidance:

- **Vanilla CFG** is equivalent to a proportional controller (P-control) with fixed gain `w`:
  ```
  v_guided = v_uncond + w * (v_cond - v_uncond)
  ```

- **SMC-CFG** introduces a nonlinear sliding mode controller that enforces the guidance error to converge along an exponential sliding surface:
  ```
  s_t = (e_t - e_{t-1}) + lambda * e_{t-1}    (sliding surface)
  u_sw = -K * sign(s_t)                         (switching control)
  ```
  where `e_t = v_cond - v_uncond` is the guidance error signal, `lambda` controls the exponential decay rate, and `K` is the switching gain.

This design prevents the instability and overshooting observed with standard CFG at large guidance scales.

## Installation

### Prerequisites

- Python >= 3.10 (recommended: 3.10)
- CUDA-compatible GPU

### Option A: Conda (recommended)

```bash
git clone https://github.com/MaxLiuyy/CFG-Ctrl.git
cd CFG-Ctrl
conda env create -f environment.yml
conda activate cfg-ctrl
```

### Option B: pip

```bash
conda create -n cfg-ctrl python=3.10 -y
conda activate cfg-ctrl
git clone https://github.com/MaxLiuyy/CFG-Ctrl.git
cd CFG-Ctrl
pip install -r requirements.txt
```

## Quick Start

All examples should be run from the `CFG-Ctrl/` directory.

### Text-to-Image (FLUX)

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

### Text-to-Image (Qwen-Image)

```bash
python examples/qwen_cfg_ctrl_example.py \
  --prompt "A ginger cat sitting by a rainy window, cinematic lighting" \
  --cfg-scale 4 \
  --steps 30 \
  --smc-cfg-enable \
  --smc-cfg-lambda 5.0 \
  --smc-cfg-k 0.2
```

### Text-to-Image (SD3)

```bash
python examples/sd3_cfg_ctrl_example.py \
  --prompt "A futuristic city at sunrise, cinematic, ultra detailed" \
  --cfg-scale 7.5 \
  --steps 30 \
  --smc-cfg-enable \
  --smc-cfg-lambda 5.0 \
  --smc-cfg-k 0.2
```

### Text-to-Video (Wan Video)

```bash
python examples/wan_video_cfg_ctrl_example.py \
  --prompt "A golden retriever running on a beach at sunset" \
  --cfg-scale 5 \
  --steps 50 \
  --smc-cfg-enable \
  --smc-cfg-lambda 5.0 \
  --smc-cfg-k 0.2
```

For Image-to-Video mode, add `--input-image path/to/image.png`.

## Supported Models

| Model | Pipeline Class | Task |
|-------|---------------|------|
| [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | `FluxImagePipelineCFGCtrl` | Text-to-Image |
| [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | `QwenImagePipelineCFGCtrl` | Text-to-Image |
| SD3 / SD3.5 | `SD3ImagePipelineCFGCtrl` | Text-to-Image |
| [Wan2.1](https://huggingface.co/Wan-AI) / Wan2.2 | `WanVideoPipelineCFGCtrl` | Text-to-Video / Image-to-Video |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smc_cfg_enable` | bool | False | Enable SMC-CFG stabilization |
| `smc_cfg_lambda` | float | 0.05 | Exponential decay coefficient for sliding surface |
| `smc_cfg_k` / `smc_cfg_K` | float | 0.3 | Switching gain |
| `no_cfg_warmup_steps` | int | 0 | Number of initial steps without CFG |

**Recommended settings**:
- FLUX: `smc_cfg_lambda=5.0, smc_cfg_k=0.2, cfg_scale=2~3`
- Qwen-Image: `smc_cfg_lambda=5.0, smc_cfg_k=0.2, cfg_scale=4`
- SD3/SD3.5: `smc_cfg_lambda=5.0, smc_cfg_k=0.2, cfg_scale=7.5`
- Wan Video: `smc_cfg_lambda=5.0, smc_cfg_k=0.2, cfg_scale=5`

## Citation

```bibtex
@inproceedings{cfg-ctrl,
  title={CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance},
  author={Wang, Hanyang and Liu, Yiyang and Chi, Jiawei and Liu, Fangfu and Xue, Ran and Duan, Yueqi},
  booktitle={CVPR},
  year={2026}
}
```

## Acknowledgements

This project is built on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).
