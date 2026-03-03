import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline import WanVideoPipelineCFGCtrl  # noqa: E402
from diffsynth.pipelines.wan_video_new import ModelConfig  # noqa: E402
from diffsynth import save_video  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Wan Video (diffsynth v1.1.9) + CFG-Ctrl params demo")
    # Prompt
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative-prompt", type=str, default="")
    # Input image (for I2V mode)
    p.add_argument("--input-image", type=str, default=None, help="Path to input image for I2V mode")
    # Generation
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg-scale", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=81)
    # Output
    p.add_argument("--out", type=str, default="wan_cfg_ctrl.mp4")
    p.add_argument("--fps", type=int, default=24)
    # Model
    p.add_argument("--model-id", type=str, default="Wan-AI/Wan2.1-T2V-14B",
                   help="HuggingFace model id (e.g. Wan-AI/Wan2.1-T2V-14B, Wan-AI/Wan2.2-TI2V-5B)")
    # CFG-Ctrl
    p.add_argument("--smc-cfg-enable", action="store_true")
    p.add_argument("--smc-cfg-lambda", type=float, default=5.0)
    p.add_argument("--smc-cfg-k", type=float, default=0.2)
    p.add_argument("--no-cfg-warmup-steps", type=int, default=0)
    # Device
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    pipe = WanVideoPipelineCFGCtrl.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management()

    input_image = None
    if args.input_image:
        from PIL import Image
        input_image = Image.open(args.input_image).convert("RGB")

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=input_image,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        smc_cfg_enable=args.smc_cfg_enable,
        smc_cfg_lambda=args.smc_cfg_lambda,
        smc_cfg_k=args.smc_cfg_k,
        no_cfg_warmup_steps=args.no_cfg_warmup_steps,
    )
    save_video(video, args.out, fps=args.fps)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
