import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline import FluxImagePipelineCFGCtrl  # noqa: E402
from diffsynth.pipelines.flux_image_new import ModelConfig  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="FLUX (diffsynth v1.1.9) + CFG-Ctrl params demo")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--out", type=str, default="flux_cfg_ctrl.png")
    p.add_argument("--smc-cfg-enable", action="store_true")
    p.add_argument("--smc-cfg-lambda", type=float, default=5.0)
    p.add_argument("--smc-cfg-k", type=float, default=0.2)
    p.add_argument("--no-cfg-warmup-steps", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    pipe = FluxImagePipelineCFGCtrl.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", offload_device="cpu"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", offload_device="cpu"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/", offload_device="cpu"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management()

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.steps,
        smc_cfg_enable=args.smc_cfg_enable,
        smc_cfg_lambda=args.smc_cfg_lambda,
        smc_cfg_k=args.smc_cfg_k,
        no_cfg_warmup_steps=args.no_cfg_warmup_steps,
    )
    image.save(args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

