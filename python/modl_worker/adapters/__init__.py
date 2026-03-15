from .train_adapter import run_train
from .gen_adapter import run_generate, run_generate_with_pipeline
from .pipeline_loader import load_pipeline  # noqa: F401
from .edit_adapter import run_edit, run_edit_with_pipeline
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
from .face_crop_adapter import run_face_crop
from .ground_adapter import run_ground
from .describe_adapter import run_describe
from .vl_tag_adapter import run_vl_tag
from .preprocess_adapter import run_preprocess

# Config building (used by train_adapter, available for testing)
from .config_builder import spec_to_aitoolkit_config  # noqa: F401
from .arch_config import ARCH_CONFIGS, MODEL_REGISTRY  # noqa: F401

__all__ = [
    "run_train", "run_generate", "run_edit", "run_caption", "run_resize",
    "run_tag", "run_score", "run_detect", "run_compare",
    "run_segment", "run_face_restore", "run_upscale", "run_remove_bg",
    "run_face_crop", "run_ground", "run_describe", "run_vl_tag",
    "run_preprocess",
]
