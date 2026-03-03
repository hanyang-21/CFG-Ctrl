from __future__ import annotations

import torch
from tqdm import tqdm

from diffsynth.pipelines.sd3_image import SD3ImagePipeline as _SD3ImagePipeline

from .common_cfg_ctrl import CFGCtrlMixin, CFGCtrlParams, CFGCtrlState


class SD3ImagePipelineCFGCtrl(_SD3ImagePipeline, CFGCtrlMixin):
    @staticmethod
    def from_model_manager(*args, **kwargs) -> "SD3ImagePipelineCFGCtrl":
        pipe = _SD3ImagePipeline.from_model_manager(*args, **kwargs)
        pipe.__class__ = SD3ImagePipelineCFGCtrl
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        local_prompts=[],
        masks=[],
        mask_scales=[],
        negative_prompt="",
        cfg_scale=7.5,
        input_image=None,
        denoising_strength=1.0,
        height=1024,
        width=1024,
        num_inference_steps=20,
        t5_sequence_length=77,
        tiled=False,
        tile_size=128,
        tile_stride=64,
        seed=None,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
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
                local_prompts=local_prompts,
                masks=masks,
                mask_scales=mask_scales,
                negative_prompt=negative_prompt,
                cfg_scale=cfg_scale,
                input_image=input_image,
                denoising_strength=denoising_strength,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                t5_sequence_length=t5_sequence_length,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                seed=seed,
                progress_bar_cmd=progress_bar_cmd,
                progress_bar_st=progress_bar_st,
            )

        height, width = self.check_resize_height_width(height, width)
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        if input_image is not None:
            self.load_models_to_device(["vae_encoder"])
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.torch_dtype)
            latents = self.encode_image(image, **tiler_kwargs)
            noise = self.generate_noise((1, 16, height // 8, width // 8), seed=seed, device=self.device, dtype=self.torch_dtype)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = self.generate_noise((1, 16, height // 8, width // 8), seed=seed, device=self.device, dtype=self.torch_dtype)

        self.load_models_to_device(["text_encoder_1", "text_encoder_2", "text_encoder_3"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True, t5_sequence_length=t5_sequence_length)
        prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False, t5_sequence_length=t5_sequence_length)
        prompt_emb_locals = [self.encode_prompt(prompt_local, t5_sequence_length=t5_sequence_length) for prompt_local in local_prompts]

        self.load_models_to_device(["dit"])
        ctrl_state = CFGCtrlState()
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            inference_callback = lambda prompt_emb_posi: self.dit(
                latents, timestep=timestep, **prompt_emb_posi, **tiler_kwargs,
            )
            noise_pred_posi = self.control_noise_via_local_prompts(
                prompt_emb_posi, prompt_emb_locals, masks, mask_scales, inference_callback
            )

            warmup_no_cfg = params.no_cfg_warmup_steps > 0 and progress_id < params.no_cfg_warmup_steps
            compute_cfg_branch = (cfg_scale != 1.0) or warmup_no_cfg
            if compute_cfg_branch:
                noise_pred_nega = self.dit(latents, timestep=timestep, **prompt_emb_nega, **tiler_kwargs)
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

            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))

        self.load_models_to_device(["vae_decoder"])
        image = self.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        self.load_models_to_device([])
        return image

