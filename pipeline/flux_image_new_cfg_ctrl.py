from __future__ import annotations

from typing import Union

import torch
from PIL import Image
from tqdm import tqdm

from diffsynth.pipelines.flux_image_new import (
    FluxImagePipeline as _FluxImagePipeline,
    ControlNetInput,
    ModelConfig,
)

from .common_cfg_ctrl import CFGCtrlMixin, CFGCtrlParams, CFGCtrlState


class FluxImagePipelineCFGCtrl(_FluxImagePipeline, CFGCtrlMixin):
    """基于 diffsynth.flux_image_new 的轻量重写版本。

    仅重写 `__call__` 中的 CFG 合成逻辑，新增 SMC/Warmup 参数。
    其余流程（unit_runner / model_fn / scheduler / decode）保持 upstream 行为。
    """

    @staticmethod
    def from_pretrained(*args, **kwargs) -> "FluxImagePipelineCFGCtrl":
        pipe = _FluxImagePipeline.from_pretrained(*args, **kwargs)
        pipe.__class__ = FluxImagePipelineCFGCtrl
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        embedded_guidance: float = 3.5,
        t5_sequence_length: int = 512,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Scheduler
        sigma_shift: float = None,
        # Steps
        num_inference_steps: int = 30,
        # local prompts
        multidiffusion_prompts=(),
        multidiffusion_masks=(),
        multidiffusion_scales=(),
        # Kontext
        kontext_images: Union[list[Image.Image], Image.Image] = None,
        # ControlNet
        controlnet_inputs: list[ControlNetInput] = None,
        # IP-Adapter
        ipadapter_images: Union[list[Image.Image], Image.Image] = None,
        ipadapter_scale: float = 1.0,
        # EliGen
        eligen_entity_prompts: list[str] = None,
        eligen_entity_masks: list[Image.Image] = None,
        eligen_enable_on_negative: bool = False,
        eligen_enable_inpaint: bool = False,
        # InfiniteYou
        infinityou_id_image: Image.Image = None,
        infinityou_guidance: float = 1.0,
        # Flex
        flex_inpaint_image: Image.Image = None,
        flex_inpaint_mask: Image.Image = None,
        flex_control_image: Image.Image = None,
        flex_control_strength: float = 0.5,
        flex_control_stop: float = 0.5,
        # Value Controller
        value_controller_inputs: Union[list[float], float] = None,
        # Step1x
        step1x_reference_image: Image.Image = None,
        # NexusGen
        nexus_gen_reference_image: Image.Image = None,
        # LoRA Encoder
        lora_encoder_inputs: Union[list[ModelConfig], ModelConfig, str] = None,
        lora_encoder_scale: float = 1.0,
        # TeaCache
        tea_cache_l1_thresh: float = None,
        # Tile
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        # Progress bar
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
                embedded_guidance=embedded_guidance,
                t5_sequence_length=t5_sequence_length,
                input_image=input_image,
                denoising_strength=denoising_strength,
                height=height,
                width=width,
                seed=seed,
                rand_device=rand_device,
                sigma_shift=sigma_shift,
                num_inference_steps=num_inference_steps,
                multidiffusion_prompts=multidiffusion_prompts,
                multidiffusion_masks=multidiffusion_masks,
                multidiffusion_scales=multidiffusion_scales,
                kontext_images=kontext_images,
                controlnet_inputs=controlnet_inputs,
                ipadapter_images=ipadapter_images,
                ipadapter_scale=ipadapter_scale,
                eligen_entity_prompts=eligen_entity_prompts,
                eligen_entity_masks=eligen_entity_masks,
                eligen_enable_on_negative=eligen_enable_on_negative,
                eligen_enable_inpaint=eligen_enable_inpaint,
                infinityou_id_image=infinityou_id_image,
                infinityou_guidance=infinityou_guidance,
                flex_inpaint_image=flex_inpaint_image,
                flex_inpaint_mask=flex_inpaint_mask,
                flex_control_image=flex_control_image,
                flex_control_strength=flex_control_strength,
                flex_control_stop=flex_control_stop,
                value_controller_inputs=value_controller_inputs,
                step1x_reference_image=step1x_reference_image,
                nexus_gen_reference_image=nexus_gen_reference_image,
                lora_encoder_inputs=lora_encoder_inputs,
                lora_encoder_scale=lora_encoder_scale,
                tea_cache_l1_thresh=tea_cache_l1_thresh,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                progress_bar_cmd=progress_bar_cmd,
            )

        # 以下代码基于 upstream `flux_image_new.py::__call__`，仅修改 CFG 合成逻辑。
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale, "embedded_guidance": embedded_guidance, "t5_sequence_length": t5_sequence_length,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "sigma_shift": sigma_shift, "num_inference_steps": num_inference_steps,
            "multidiffusion_prompts": multidiffusion_prompts, "multidiffusion_masks": multidiffusion_masks, "multidiffusion_scales": multidiffusion_scales,
            "kontext_images": kontext_images,
            "controlnet_inputs": controlnet_inputs,
            "ipadapter_images": ipadapter_images, "ipadapter_scale": ipadapter_scale,
            "eligen_entity_prompts": eligen_entity_prompts, "eligen_entity_masks": eligen_entity_masks, "eligen_enable_on_negative": eligen_enable_on_negative, "eligen_enable_inpaint": eligen_enable_inpaint,
            "infinityou_id_image": infinityou_id_image, "infinityou_guidance": infinityou_guidance,
            "flex_inpaint_image": flex_inpaint_image, "flex_inpaint_mask": flex_inpaint_mask, "flex_control_image": flex_control_image, "flex_control_strength": flex_control_strength, "flex_control_stop": flex_control_stop,
            "value_controller_inputs": value_controller_inputs,
            "step1x_reference_image": step1x_reference_image,
            "nexus_gen_reference_image": nexus_gen_reference_image,
            "lora_encoder_inputs": lora_encoder_inputs, "lora_encoder_scale": lora_encoder_scale,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "progress_bar_cmd": progress_bar_cmd,
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

            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])

        self.load_models_to_device(["vae_decoder"])
        image = self.vae_decoder(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])
        return image

