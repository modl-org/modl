from .train_adapter import run_train
from .gen_adapter import run_generate, run_generate_with_pipeline
from .caption_adapter import run_caption
from .resize_adapter import run_resize
from .tag_adapter import run_tag
from .score_adapter import run_score
from .detect_adapter import run_detect
from .compare_adapter import run_compare
from .segment_adapter import run_segment
from .face_restore_adapter import run_face_restore
from .upscale_adapter import run_upscale
from .remove_bg_adapter import run_remove_bg

# Config building (used by train_adapter, available for testing)
from .config_builder import spec_to_aitoolkit_config  # noqa: F401
from .arch_config import ARCH_CONFIGS, MODEL_REGISTRY  # noqa: F401

__all__ = [
    "run_train", "run_generate", "run_caption", "run_resize", "run_tag",
    "run_score", "run_detect", "run_compare",
    "run_segment", "run_face_restore", "run_upscale", "run_remove_bg",
]
