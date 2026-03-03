import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline import SD3ImagePipelineCFGCtrl  # noqa: E402
from diffsynth import ModelManager, download_models  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="SD3 (diffsynth v1.1.9) + CFG-Ctrl params demo")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg-scale", type=float, default=7.5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--out", type=str, default="sd3_cfg_ctrl.png")
    p.add_argument("--smc-cfg-enable", action="store_true")
    p.add_argument("--smc-cfg-lambda", type=float, default=5.0)
    p.add_argument("--smc-cfg-k", type=float, default=0.2)
    p.add_argument("--no-cfg-warmup-steps", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    # 与 upstream 示例一致：如无模型可先自动下载（按需注释）
    download_models(["StableDiffusion3_without_T5"])
    model_manager = ModelManager(
        torch_dtype=torch.float16,
        device=args.device,
        file_path_list=["models/stable_diffusion_3/sd3_medium_incl_clips.safetensors"],
    )
    pipe = SD3ImagePipelineCFGCtrl.from_model_manager(model_manager)

    torch.manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        smc_cfg_enable=args.smc_cfg_enable,
        smc_cfg_lambda=args.smc_cfg_lambda,
        smc_cfg_k=args.smc_cfg_k,
        no_cfg_warmup_steps=args.no_cfg_warmup_steps,
    )
    image.save(args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

