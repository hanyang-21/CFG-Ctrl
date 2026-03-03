# ---------------------------------------------------------------------------
# Compatibility patches for diffsynth 1.1.9 + newer transformers (>=4.49).
# diffsynth references internal transformers symbols that were moved/renamed.
# We monkey-patch them back so imports succeed without modifying DiffSynth.
# ---------------------------------------------------------------------------
import transformers.modeling_utils as _mu

# 1) PretrainedConfig: moved from modeling_utils to configuration_utils
if not hasattr(_mu, "PretrainedConfig"):
    from transformers import PretrainedConfig as _PC
    _mu.PretrainedConfig = _PC

# 2) Qwen2RMSNorm: renamed in newer transformers; diffsynth's qwen_image
#    pipeline imports it for enable_vram_management()
try:
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as _qvl
    if not hasattr(_qvl, "Qwen2RMSNorm"):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as _RMS
        _qvl.Qwen2RMSNorm = _RMS
except ImportError:
    pass

from .flux_image_new_cfg_ctrl import FluxImagePipelineCFGCtrl
from .qwen_image_cfg_ctrl import QwenImagePipelineCFGCtrl
from .sd3_image_cfg_ctrl import SD3ImagePipelineCFGCtrl
from .wan_video_cfg_ctrl import WanVideoPipelineCFGCtrl

__all__ = [
    "FluxImagePipelineCFGCtrl",
    "QwenImagePipelineCFGCtrl",
    "SD3ImagePipelineCFGCtrl",
    "WanVideoPipelineCFGCtrl",
]

