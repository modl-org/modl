"""Architecture configuration tables and model resolution helpers.

This is the single source of truth for per-model-family settings used by
the config builder, train adapter, and gen adapter.  Each entry in
ARCH_CONFIGS drives both training YAML generation and inference pipeline
selection without ad-hoc if/elif chains.

Fields in each ARCH_CONFIGS entry:
    pipeline_class     – diffusers pipeline class name for generation
    model_flags        – merged into the ai-toolkit "model" block
    noise_scheduler    – scheduler type for the "train" block
    dtype              – training precision
    train_text_encoder – whether to train the text encoder
    resolutions        – resolution buckets for the dataset
    default_resolution – fallback when user doesn't specify
    sample             – sampler, steps, guidance, neg for sample/generate defaults
    extra_train        – extra keys merged into "train" block
"""

import os
import sqlite3
from pathlib import Path

# -----------------------------------------------------------------------
# Qwen-Image quantization defaults
# -----------------------------------------------------------------------
# Style on 24GB: 3-bit + ARA (Accuracy Recovery Adapter) — proven by Ostris
#   to produce good results on RTX 4090 (~23GB used).
# Character/object on 32GB: uint6 (6-bit) — needs ~30GB VRAM at 1024px.
# Character on 24GB: NOT currently recommended. Would need int4 (severe
#   quality degradation per Ostris) + resolution drop to 512-768.
#   Ostris: "It currently won't run on 24 gigs, I'm still working on that."
# Users can override via MODL_QWEN_QTYPE env var.
QWEN_32GB_DEFAULT_QTYPE = "uint6"
QWEN_24GB_STYLE_QTYPE = "uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors"

# -----------------------------------------------------------------------
# Architecture config table
# -----------------------------------------------------------------------

ARCH_CONFIGS: dict[str, dict] = {
    "flux": {
        "pipeline_class": "FluxPipeline",
        "model_flags": {"is_flux": True, "quantize": True},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 20, "guidance": 4.0, "neg": ""},
    },
    "flux_schnell": {
        "pipeline_class": "FluxPipeline",
        "model_flags": {
            "is_flux": True,
            "quantize": True,
            "assistant_lora_path": "ostris/FLUX.1-schnell-training-adapter",
        },
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 4, "guidance": 1.0, "neg": ""},
    },
    "zimage_turbo": {
        "pipeline_class": "ZImagePipeline",
        "gen_components": {
            "text_encoder": "z-image-text-encoder",
            "vae": "z-image-vae",
        },
        "model_flags": {
            "arch": "zimage",
            # quantize/low_vram set dynamically by config_builder based on VRAM
            # Ostris: "if you have 24 gigs or more, set this to none" — no quantize
            # Without quantize: ~17GB VRAM, much faster iteration
            "assistant_lora_path": "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v1.safetensors",
        },
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "extra_train": {
            "timestep_type": "weighted",
            # linear_timesteps2 (high-noise bias) set per lora_type in config_builder
            "cache_text_embeddings": True,
        },
        "sample": {"sampler": "flowmatch", "steps": 8, "guidance": 1.0, "neg": ""},
    },
    "zimage": {
        "pipeline_class": "ZImagePipeline",
        "gen_components": {
            "text_encoder": "z-image-text-encoder",
            "vae": "z-image-vae",
        },
        "model_flags": {
            "arch": "zimage",
            "quantize": True,
            "quantize_te": True,
            "low_vram": True,
        },
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "extra_train": {"timestep_type": "weighted"},
        "sample": {"sampler": "flowmatch", "steps": 30, "guidance": 4.0, "neg": ""},
    },
    "chroma": {
        "pipeline_class": "FluxPipeline",
        "gen_components": {
            "text_encoder": "z-image-text-encoder",
        },
        "model_flags": {"arch": "chroma", "quantize": True},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 25, "guidance": 4.0, "neg": ""},
    },
    "qwen_image": {
        "pipeline_class": "FluxPipeline",
        "model_flags": {
            "arch": "qwen_image",
            "quantize": True,
            "quantize_te": True,
            "qtype_te": "qfloat8",
            "low_vram": True,
        },
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "extra_train": {
            "cache_text_embeddings": True,
            "timestep_type": "sigmoid",
        },
        "sample": {"sampler": "flowmatch", "steps": 25, "guidance": 3.0, "neg": ""},
    },
    "sdxl": {
        "pipeline_class": "StableDiffusionXLPipeline",
        "model_flags": {"arch": "sdxl"},
        "noise_scheduler": "ddpm",
        "dtype": "bf16",
        "train_text_encoder": True,
        "resolutions": [768, 1024],
        "default_resolution": 1024,
        "extra_train": {"max_denoising_steps": 1000},
        "sample": {"sampler": "euler", "steps": 30, "guidance": 7.5, "neg": "blurry, low quality, deformed"},
    },
    "sd15": {
        "pipeline_class": "StableDiffusionPipeline",
        "model_flags": {},
        "noise_scheduler": "ddpm",
        "dtype": "fp16",
        "train_text_encoder": True,
        "resolutions": [512],
        "default_resolution": 512,
        "extra_train": {"max_denoising_steps": 1000},
        "sample": {"sampler": "euler", "steps": 30, "guidance": 7.5, "neg": "blurry, low quality, deformed"},
    },
}

# -----------------------------------------------------------------------
# Model registry: modl model IDs → (arch_key, HuggingFace hub ID)
# -----------------------------------------------------------------------

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "flux-dev":       ("flux",          "black-forest-labs/FLUX.1-dev"),
    "flux-schnell":   ("flux_schnell",  "black-forest-labs/FLUX.1-schnell"),
    "z-image-turbo":  ("zimage_turbo",  "Tongyi-MAI/Z-Image-Turbo"),
    "z-image":        ("zimage",        "Tongyi-MAI/Z-Image"),
    "chroma":         ("chroma",        "lodestones/Chroma"),
    "qwen-image":     ("qwen_image",    "Qwen/Qwen-Image"),
    "qwen_image":     ("qwen_image",    "Qwen/Qwen-Image"),
    "sdxl-base-1.0":  ("sdxl",          "stabilityai/stable-diffusion-xl-base-1.0"),
    "sdxl-turbo":     ("sdxl",          "stabilityai/sdxl-turbo"),
    "sd-1.5":         ("sd15",          "stable-diffusion-v1-5/stable-diffusion-v1-5"),
}


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def detect_arch(base_model_id: str) -> str:
    """Detect architecture key from a base model ID.

    First checks MODEL_REGISTRY for an exact match, then falls back to
    substring heuristics.  Returns a key into ARCH_CONFIGS.
    """
    entry = MODEL_REGISTRY.get(base_model_id)
    if entry:
        return entry[0]

    bid = base_model_id.lower()
    if "qwen-image" in bid or "qwen_image" in bid:
        return "qwen_image"
    if "z-image-turbo" in bid or "z_image_turbo" in bid:
        return "zimage_turbo"
    if "z-image" in bid or "z_image" in bid or "zimage" in bid:
        return "zimage"
    if "chroma" in bid:
        return "chroma"
    if "flux" in bid and "schnell" in bid:
        return "flux_schnell"
    if "flux" in bid:
        return "flux"
    if "sdxl" in bid or "xl" in bid:
        return "sdxl"
    if "sd-1.5" in bid or "sd15" in bid or "1.5" in bid:
        return "sd15"
    return "sdxl"  # safe default


def resolve_model_path(base_model_id: str) -> str:
    """Resolve a modl model ID to a HuggingFace hub path."""
    entry = MODEL_REGISTRY.get(base_model_id)
    if entry:
        return entry[1]
    return base_model_id


def resolve_pipeline_class(base_model_id: str) -> str:
    """Return the diffusers pipeline class name for a model ID."""
    arch = detect_arch(base_model_id)
    config = ARCH_CONFIGS.get(arch, ARCH_CONFIGS["sdxl"])
    return config["pipeline_class"]


def resolve_gen_defaults(base_model_id: str) -> dict:
    """Return default generation params (steps, guidance) for a model ID.

    Values come from the ``sample`` block in ARCH_CONFIGS, which is also
    used for training preview generation — one source of truth.
    """
    arch = detect_arch(base_model_id)
    config = ARCH_CONFIGS.get(arch, ARCH_CONFIGS["sdxl"])
    sample = config.get("sample", {})
    return {
        "steps": sample.get("steps", 28),
        "guidance": sample.get("guidance", 3.5),
    }


def _get_installed_path(model_id: str) -> str | None:
    """Look up a model's store path from the modl state DB."""
    db_path = Path.home() / ".modl" / "state.db"
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT store_path FROM installed WHERE id = ?", (model_id,)
        ).fetchone()
        conn.close()
        if row and Path(row[0]).exists():
            return row[0]
    except Exception:
        pass
    return None


def resolve_gen_components(base_model_id: str) -> dict[str, str]:
    """Resolve component paths (text_encoder, vae) for generation.

    Returns a dict like {"text_encoder": "/path/to/qwen3.safetensors", "vae": ...}
    for models that require separate component loading (z-image, chroma, etc.).
    Returns empty dict if no components needed or not found.
    """
    arch = detect_arch(base_model_id)
    config = ARCH_CONFIGS.get(arch, {})
    gen_components = config.get("gen_components", {})
    if not gen_components:
        return {}

    resolved = {}
    for component_type, model_id in gen_components.items():
        path = _get_installed_path(model_id)
        if path:
            resolved[component_type] = path
    return resolved


def resolve_qwen_qtype(lora_type: str) -> str:
    """Pick the right Qwen-Image quantization type based on lora_type.

    Style LoRAs default to 3-bit + ARA (fits 24GB cards, ~23GB used).
    Character/object LoRAs default to uint6 (needs 32GB-class GPU, ~30GB used).

    Character on 24GB is NOT recommended — uint4 causes severe quality
    degradation and resolution must be dropped significantly.
    """
    if lora_type == "style":
        default = QWEN_24GB_STYLE_QTYPE
    else:
        default = QWEN_32GB_DEFAULT_QTYPE

    qtype = os.getenv("MODL_QWEN_QTYPE", default).strip()
    if qtype == "int6":
        qtype = "uint6"  # ai-toolkit uses uint* naming
    if not qtype:
        qtype = default
    return qtype
