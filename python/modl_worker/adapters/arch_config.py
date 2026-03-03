"""Architecture configuration tables and model resolution helpers.

This is the single source of truth for per-model-family settings used by
the config builder and train adapter.  Each entry in ARCH_CONFIGS drives
the ai-toolkit YAML generation without ad-hoc if/elif chains.

Fields in each ARCH_CONFIGS entry:
    model_flags        – merged into the ai-toolkit "model" block
    noise_scheduler    – scheduler type for the "train" block
    dtype              – training precision
    train_text_encoder – whether to train the text encoder
    resolutions        – resolution buckets for the dataset
    default_resolution – fallback when user doesn't specify
    sample             – sampler, steps, guidance, neg for sample block
    extra_train        – extra keys merged into "train" block
"""

import os

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
        "model_flags": {"is_flux": True, "quantize": True},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 20, "guidance": 4.0, "neg": ""},
    },
    "flux_schnell": {
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
        "model_flags": {
            "arch": "zimage",
            "quantize": True,
            "quantize_te": True,
            "low_vram": True,
            "assistant_lora_path": "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors",
        },
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "extra_train": {"timestep_type": "weighted"},
        "sample": {"sampler": "flowmatch", "steps": 8, "guidance": 1.0, "neg": ""},
    },
    "zimage": {
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
        "model_flags": {"arch": "chroma", "quantize": True},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 25, "guidance": 4.0, "neg": ""},
    },
    "qwen_image": {
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
