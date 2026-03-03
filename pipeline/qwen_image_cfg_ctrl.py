from __future__ import annotations

from typing import Union

import torch
from PIL import Image
from tqdm import tqdm

from diffsynth.pipelines.qwen_image import (
    QwenImagePipeline as _QwenImagePipeline,
    ModelConfig,
    ControlNetInput,
)

from .common_cfg_ctrl import CFGCtrlMixin, CFGCtrlParams, CFGCtrlState


class QwenImagePipelineCFGCtrl(_QwenImagePipeline, CFGCtrlMixin):
    @staticmethod
    def from_pretrained(*args, **kwargs) -> "QwenImagePipelineCFGCtrl":
        pipe = _QwenImagePipeline.from_pretrained(*args, **kwargs)
        pipe.__class__ = QwenImagePipelineCFGCtrl
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 4.0,
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        inpaint_mask: Image.Image = None,
        inpaint_blur_size: int = None,
        inpaint_blur_sigma: float = None,
        height: int = 1328,
        width: int = 1328,
        seed: int = None,
        rand_device: str = "cpu",
        num_inference_steps: int = 30,
        exponential_shift_mu: float = None,
        blockwise_controlnet_inputs: list[ControlNetInput] = None,
        eligen_entity_prompts: list[str] = None,
        eligen_entity_masks: list[Image.Image] = None,
        eligen_enable_on_negative: bool = False,
        edit_image: Image.Image = None,
        edit_image_auto_resize: bool = True,
        edit_rope_interpolation: bool = False,
        context_image: Image.Image = None,
        enable_fp8_attention: bool = False,
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        progress_bar_cmd=tqdm,
        # CFG-Ctrl 扩展参数
        smc_cfg_enable: bool = False,
        smc_cfg_lambda: float = 0.05,
        smc_cfg_k: float = None,
        smc_cfg_K: float = None,
        no_cfg_warmup_steps: int = 0,
    ):
        params = CFGCtrlParams.build(
            smc_cfg_enable=smc_cfg_enable,
            smc_cfg_lambda=smc_cfg_lambda,
            smc_cfg_k=smc_cfg_k,
            smc_cfg_K=smc_cfg_K,
            no_cfg_warmup_steps=no_cfg_warmup_steps,
        )
        if not params.enabled:
            return super().__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                cfg_scale=cfg_scale,
                input_image=input_image,
                denoising_strength=denoising_strength,
                inpaint_mask=inpaint_mask,
                inpaint_blur_size=inpaint_blur_size,
                inpaint_blur_sigma=inpaint_blur_sigma,
                height=height,
                width=width,
                seed=seed,
                rand_device=rand_device,
                num_inference_steps=num_inference_steps,
                exponential_shift_mu=exponential_shift_mu,
                blockwise_controlnet_inputs=blockwise_controlnet_inputs,
                eligen_entity_prompts=eligen_entity_prompts,
                eligen_entity_masks=eligen_entity_masks,
                eligen_enable_on_negative=eligen_enable_on_negative,
                edit_image=edit_image,
                edit_image_auto_resize=edit_image_auto_resize,
                edit_rope_interpolation=edit_rope_interpolation,
                context_image=context_image,
                enable_fp8_attention=enable_fp8_attention,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                progress_bar_cmd=progress_bar_cmd,
            )

        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            dynamic_shift_len=(height // 16) * (width // 16),
            exponential_shift_mu=exponential_shift_mu,
        )

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "inpaint_mask": inpaint_mask, "inpaint_blur_size": inpaint_blur_size, "inpaint_blur_sigma": inpaint_blur_sigma,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "enable_fp8_attention": enable_fp8_attention,
            "num_inference_steps": num_inference_steps,
            "blockwise_controlnet_inputs": blockwise_controlnet_inputs,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "eligen_entity_prompts": eligen_entity_prompts, "eligen_entity_masks": eligen_entity_masks, "eligen_enable_on_negative": eligen_enable_on_negative,
            "edit_image": edit_image, "edit_image_auto_resize": edit_image_auto_resize, "edit_rope_interpolation": edit_rope_interpolation,
            "context_image": context_image,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        ctrl_state = CFGCtrlState()
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep, progress_id=progress_id)
            warmup_no_cfg = params.no_cfg_warmup_steps > 0 and progress_id < params.no_cfg_warmup_steps
            compute_cfg_branch = (cfg_scale != 1.0) or warmup_no_cfg
            if compute_cfg_branch:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep, progress_id=progress_id)
                noise_pred = self._cfg_ctrl_apply(
                    noise_pred_posi=noise_pred_posi,
                    noise_pred_nega=noise_pred_nega,
                    cfg_scale=cfg_scale,
                    progress_id=progress_id,
                    params=params,
                    state=ctrl_state,
                )
            else:
                noise_pred = noise_pred_posi

            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)

        self.load_models_to_device(["vae"])
        image = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])
        return image

