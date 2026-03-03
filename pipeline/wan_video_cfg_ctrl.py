from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing_extensions import Literal

from diffsynth.pipelines.wan_video_new import WanVideoPipeline as _WanVideoPipeline

from .common_cfg_ctrl import CFGCtrlMixin, CFGCtrlParams, CFGCtrlState


class WanVideoPipelineCFGCtrl(_WanVideoPipeline, CFGCtrlMixin):
    """基于 diffsynth.wan_video_new 的轻量重写版本。

    仅重写 `__call__` 中的 CFG 合成逻辑，新增 SMC/Warmup 参数。
    其余流程（unit_runner / model_fn / scheduler / decode）保持 upstream 行为。
    """

    @staticmethod
    def from_pretrained(*args, **kwargs) -> "WanVideoPipelineCFGCtrl":
        pipe = _WanVideoPipeline.from_pretrained(*args, **kwargs)
        pipe.__class__ = WanVideoPipelineCFGCtrl
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1 / 54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
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
                input_image=input_image,
                end_image=end_image,
                input_video=input_video,
                denoising_strength=denoising_strength,
                input_audio=input_audio,
                audio_embeds=audio_embeds,
                audio_sample_rate=audio_sample_rate,
                s2v_pose_video=s2v_pose_video,
                s2v_pose_latents=s2v_pose_latents,
                motion_video=motion_video,
                control_video=control_video,
                reference_image=reference_image,
                camera_control_direction=camera_control_direction,
                camera_control_speed=camera_control_speed,
                camera_control_origin=camera_control_origin,
                vace_video=vace_video,
                vace_video_mask=vace_video_mask,
                vace_reference_image=vace_reference_image,
                vace_scale=vace_scale,
                seed=seed,
                rand_device=rand_device,
                height=height,
                width=width,
                num_frames=num_frames,
                cfg_scale=cfg_scale,
                cfg_merge=cfg_merge,
                no_cfg_warmup_steps=0,
                switch_DiT_boundary=switch_DiT_boundary,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                motion_bucket_id=motion_bucket_id,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                sliding_window_size=sliding_window_size,
                sliding_window_stride=sliding_window_stride,
                tea_cache_l1_thresh=tea_cache_l1_thresh,
                tea_cache_model_id=tea_cache_model_id,
                progress_bar_cmd=progress_bar_cmd,
            )

        # 以下代码基于 upstream `wan_video_new.py::__call__`，仅修改 CFG 合成逻辑。
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        ctrl_state = CFGCtrlState()
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2

            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            warmup_no_cfg = params.no_cfg_warmup_steps > 0 and progress_id < params.no_cfg_warmup_steps
            compute_cfg_branch = (cfg_scale != 1.0) or warmup_no_cfg
            if compute_cfg_branch:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
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

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        # VACE
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(["vae"])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video
