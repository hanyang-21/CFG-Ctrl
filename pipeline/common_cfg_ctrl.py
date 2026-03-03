from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CFGCtrlParams:
    smc_cfg_enable: bool = False
    smc_cfg_lambda: float = 0.05
    smc_cfg_K: float = 0.3
    no_cfg_warmup_steps: int = 0

    @classmethod
    def build(
        cls,
        *,
        smc_cfg_enable: bool = False,
        smc_cfg_lambda: float = 0.05,
        smc_cfg_k: Optional[float] = None,
        smc_cfg_K: Optional[float] = None,
        no_cfg_warmup_steps: int = 0,
    ) -> "CFGCtrlParams":
        if smc_cfg_K is None:
            smc_cfg_K = 0.3 if smc_cfg_k is None else smc_cfg_k
        return cls(
            smc_cfg_enable=bool(smc_cfg_enable),
            smc_cfg_lambda=float(smc_cfg_lambda),
            smc_cfg_K=float(smc_cfg_K),
            no_cfg_warmup_steps=int(no_cfg_warmup_steps),
        )

    @property
    def enabled(self) -> bool:
        return (
            self.smc_cfg_enable
            or self.no_cfg_warmup_steps > 0
        )


@dataclass
class CFGCtrlState:
    prev_guidance_eps: Optional[torch.Tensor] = None


class CFGCtrlMixin:
    def _cfg_ctrl_apply(
        self,
        *,
        noise_pred_posi: torch.Tensor,
        noise_pred_nega: torch.Tensor,
        cfg_scale: float,
        progress_id: int,
        params: CFGCtrlParams,
        state: CFGCtrlState,
    ) -> torch.Tensor:
        warmup_no_cfg = params.no_cfg_warmup_steps > 0 and progress_id < params.no_cfg_warmup_steps
        guidance_eps = noise_pred_posi - noise_pred_nega

        if params.smc_cfg_enable and not warmup_no_cfg:
            if state.prev_guidance_eps is None:
                state.prev_guidance_eps = guidance_eps.detach()
            s = (guidance_eps - state.prev_guidance_eps) + params.smc_cfg_lambda * state.prev_guidance_eps
            u_sw = -params.smc_cfg_K * torch.sign(s)
            guidance_eps = guidance_eps + u_sw
            state.prev_guidance_eps = guidance_eps.detach()
            return noise_pred_nega + cfg_scale * guidance_eps

        if warmup_no_cfg:
            # 与 CFG-Ctrl-ToComplete 一致：warmup 阶段直接用 conditional 预测。
            return noise_pred_posi

        return noise_pred_nega + cfg_scale * guidance_eps

