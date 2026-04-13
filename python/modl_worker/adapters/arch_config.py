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

Optional sub-dicts (only present when values differ from defaults):
    inference          – generation/edit-time overrides:
        guidance_param              – kwarg name for guidance (default: "guidance_scale")
        force_negative_prompt       – negative prompt always passed (default: None)
        supports_negative_prompt    – whether neg prompt is accepted (default: False)
        default_negative_prompt     – fallback neg prompt when user omits one
        editing_mode                – "standard" or "native" (Klein-style image param)
        scheduler_shift             – fixed scheduler shift value (e.g. 3.1)
    training           – training-time overrides:
        max_learning_rate           – LR ceiling (clamped if user exceeds it)
        differential_guidance       – {"scale": float} for style training
        sample_frequency_multiplier – sample N times during training (default: 10)
        network_alpha_mode          – "rank" to set alpha=rank for style LoRAs
        noise_offset                – {"style": float, "default": float}
        use_ema                     – whether to use EMA (default: True)
        caption_dropout             – forced caption dropout rate
        required_tokenizer_repo     – HF repo to pre-cache for training
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
        "img2img_class": "FluxImg2ImgPipeline",
        "inpaint_class": "FluxInpaintPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "FluxTransformer2DModel",
                "config_dir": "flux-dev-transformer",
            },
            "text_encoder": {
                "model_id": "clip-l",
                "model_class": "CLIPTextModel",
                "config_dir": "clip-l",
            },
            "tokenizer": {
                "model_class": "CLIPTokenizer",
                "config_dir": "clip-tokenizer",
            },
            "text_encoder_2": {
                "model_id": ["t5-xxl-fp8", "t5-xxl-fp16"],
                "model_class": "T5EncoderModel",
                "config_dir": "t5-xxl",
            },
            "tokenizer_2": {
                "model_class": "T5TokenizerFast",
                "config_dir": "t5-tokenizer",
            },
            "vae": {
                "model_id": "flux-vae",
                "model_class": "AutoencoderKL",
                "config_dir": "flux-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "flux-dev-scheduler",
            },
        },
        "model_flags": {"is_flux": True, "quantize": True, "low_vram": True},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 20, "guidance": 4.0, "neg": ""},
    },
    "flux_schnell": {
        "pipeline_class": "FluxPipeline",
        "img2img_class": "FluxImg2ImgPipeline",
        "inpaint_class": "FluxInpaintPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "FluxTransformer2DModel",
                "config_dir": "flux-schnell-transformer",
            },
            "text_encoder": {
                "model_id": "clip-l",
                "model_class": "CLIPTextModel",
                "config_dir": "clip-l",
            },
            "tokenizer": {
                "model_class": "CLIPTokenizer",
                "config_dir": "clip-tokenizer",
            },
            "text_encoder_2": {
                "model_id": ["t5-xxl-fp8", "t5-xxl-fp16"],
                "model_class": "T5EncoderModel",
                "config_dir": "t5-xxl",
            },
            "tokenizer_2": {
                "model_class": "T5TokenizerFast",
                "config_dir": "t5-tokenizer",
            },
            "vae": {
                "model_id": "flux-vae",
                "model_class": "AutoencoderKL",
                "config_dir": "flux-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "flux-schnell-scheduler",
            },
        },
        "model_flags": {
            "is_flux": True,
            "quantize": True,
            "low_vram": True,
            "assistant_lora_path": "ostris/FLUX.1-schnell-training-adapter",
        },
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 4, "guidance": 1.0, "neg": ""},
    },
    "flux2": {
        "pipeline_class": "Flux2Pipeline",
        "gen_components": {
            "transformer": {
                "model_class": "Flux2Transformer2DModel",
                "config_dir": "flux2-dev-transformer",
            },
            "text_encoder": {
                "model_id": "flux2-mistral-text-encoder",
                "model_class": "Mistral3ForConditionalGeneration",
                "config_dir": "flux2-text-encoder",
                "quantize_nf4": True,
                "hf_dir": True,
            },
            "tokenizer": {
                "model_class": "AutoProcessor",
                "config_dir": "flux2-processor",
            },
            "vae": {
                "model_id": "flux2-vae",
                "model_class": "AutoencoderKLFlux2",
                "config_dir": "flux2-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "flux2-dev-scheduler",
            },
        },
        "model_flags": {"is_flux": True, "quantize": True},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 28, "guidance": 4.0, "neg": ""},
    },
    "flux2_klein": {
        "pipeline_class": "Flux2KleinPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "Flux2Transformer2DModel",
                "config_dir": "flux2-klein-4b-transformer",
            },
            "text_encoder": {
                "model_id": "flux2-qwen3-4b-text-encoder",
                "model_class": "Qwen3ForCausalLM",
                "config_dir": "qwen3-4b",
            },
            "tokenizer": {
                "model_class": "AutoTokenizer",
                "config_dir": "qwen3-tokenizer",
            },
            "vae": {
                "model_id": "flux2-vae",
                "model_class": "AutoencoderKLFlux2",
                "config_dir": "flux2-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "flux2-klein-scheduler",
            },
        },
        "pipeline_kwargs": {"is_distilled": True},
        "model_flags": {"arch": "flux2_klein_4b"},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 4, "guidance": 1.0, "neg": ""},
        "inference": {"editing_mode": "native"},
        "training": {"max_learning_rate": 5e-5, "required_tokenizer_repo": "Qwen/Qwen3-4B", "cache_text_embeddings": True},
    },
    "flux2_klein_base": {
        "pipeline_class": "Flux2KleinPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "Flux2Transformer2DModel",
                "config_dir": "flux2-klein-4b-transformer",
            },
            "text_encoder": {
                "model_id": "flux2-qwen3-4b-text-encoder",
                "model_class": "Qwen3ForCausalLM",
                "config_dir": "qwen3-4b",
            },
            "tokenizer": {
                "model_class": "AutoTokenizer",
                "config_dir": "qwen3-tokenizer",
            },
            "vae": {
                "model_id": "flux2-vae",
                "model_class": "AutoencoderKLFlux2",
                "config_dir": "flux2-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "flux2-klein-scheduler",
            },
        },
        "model_flags": {"arch": "flux2_klein_4b"},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 50, "guidance": 4.0, "neg": ""},
        "inference": {"editing_mode": "native"},
        "training": {"max_learning_rate": 5e-5, "required_tokenizer_repo": "Qwen/Qwen3-4B", "cache_text_embeddings": True},
    },
    "flux2_klein_9b": {
        "pipeline_class": "Flux2KleinPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "Flux2Transformer2DModel",
                "config_dir": "flux2-klein-9b-transformer",
            },
            "text_encoder": {
                "model_id": "flux2-qwen3-8b-text-encoder",
                "model_class": "Qwen3ForCausalLM",
                "config_dir": "qwen3-8b",
            },
            "tokenizer": {
                "model_class": "AutoTokenizer",
                "config_dir": "qwen3-tokenizer",
            },
            "vae": {
                "model_id": "flux2-vae",
                "model_class": "AutoencoderKLFlux2",
                "config_dir": "flux2-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "flux2-klein-scheduler",
            },
        },
        "pipeline_kwargs": {"is_distilled": True},
        "model_flags": {"arch": "flux2_klein_9b"},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 4, "guidance": 1.0, "neg": ""},
        "inference": {"editing_mode": "native"},
        "training": {"max_learning_rate": 1e-4, "required_tokenizer_repo": "Qwen/Qwen3-8B", "cache_text_embeddings": True},
    },
    "flux2_klein_base_9b": {
        "pipeline_class": "Flux2KleinPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "Flux2Transformer2DModel",
                "config_dir": "flux2-klein-9b-transformer",
            },
            "text_encoder": {
                "model_id": "flux2-qwen3-8b-text-encoder",
                "model_class": "Qwen3ForCausalLM",
                "config_dir": "qwen3-8b",
            },
            "tokenizer": {
                "model_class": "AutoTokenizer",
                "config_dir": "qwen3-tokenizer",
            },
            "vae": {
                "model_id": "flux2-vae",
                "model_class": "AutoencoderKLFlux2",
                "config_dir": "flux2-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "flux2-klein-scheduler",
            },
        },
        "model_flags": {"arch": "flux2_klein_9b"},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 50, "guidance": 4.0, "neg": ""},
        "inference": {"editing_mode": "native"},
        "training": {"max_learning_rate": 1e-4, "required_tokenizer_repo": "Qwen/Qwen3-8B", "cache_text_embeddings": True},
    },
    "zimage_turbo": {
        "pipeline_class": "ZImagePipeline",
        "img2img_class": "ZImageImg2ImgPipeline",
        "inpaint_class": "ZImageInpaintPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "ZImageTransformer2DModel",
                "config_dir": "zimage-turbo-transformer",
            },
            "text_encoder": {
                "model_id": "z-image-text-encoder",
                "model_class": "Qwen3ForCausalLM",
                "config_dir": "qwen3-4b",
            },
            "tokenizer": {
                "model_class": "AutoTokenizer",
                "config_dir": "qwen3-tokenizer",
            },
            "vae": {
                "model_id": "z-image-vae",
                "model_class": "AutoencoderKL",
                "config_dir": "flux-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "zimage-turbo-scheduler",
            },
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
        "sample": {"sampler": "flowmatch", "steps": 8, "guidance": 0.0, "neg": ""},
        "training": {"max_learning_rate": 1e-4, "differential_guidance": {"scale": 3.0}, "sample_frequency_multiplier": 5, "network_alpha_mode": "rank", "cache_text_embeddings": True},
    },
    "zimage": {
        "pipeline_class": "ZImagePipeline",
        "img2img_class": "ZImageImg2ImgPipeline",
        "inpaint_class": "ZImageInpaintPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "ZImageTransformer2DModel",
                "config_dir": "zimage-transformer",
            },
            "text_encoder": {
                "model_id": "z-image-text-encoder",
                "model_class": "Qwen3ForCausalLM",
                "config_dir": "qwen3-4b",
            },
            "tokenizer": {
                "model_class": "AutoTokenizer",
                "config_dir": "qwen3-tokenizer",
            },
            "vae": {
                "model_id": "z-image-vae",
                "model_class": "AutoencoderKL",
                "config_dir": "flux-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "zimage-scheduler",
            },
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
        "training": {"max_learning_rate": 1e-4, "differential_guidance": {"scale": 3.0}, "cache_text_embeddings": True},
    },
    "chroma": {
        "pipeline_class": "ChromaPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "ChromaTransformer2DModel",
                "config_dir": "chroma-transformer",
            },
            "text_encoder": {
                "model_id": ["t5-xxl-fp8", "t5-xxl-fp16"],
                "model_class": "T5EncoderModel",
                "config_dir": "t5-xxl",
            },
            "tokenizer": {
                "model_class": "T5TokenizerFast",
                "config_dir": "t5-tokenizer",
            },
            "vae": {
                "model_id": "flux-vae",
                "model_class": "AutoencoderKL",
                "config_dir": "flux-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "chroma-scheduler",
            },
        },
        "model_flags": {"arch": "chroma", "quantize": True},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 25, "guidance": 4.0, "neg": ""},
        "inference": {"supports_negative_prompt": True, "default_negative_prompt": "low quality, ugly, unfinished, out of focus, deformed, disfigured, blurry"},
    },
    "qwen_image": {
        "pipeline_class": "QwenImagePipeline",
        "gen_components": {
            "transformer": {
                "model_class": "QwenImageTransformer2DModel",
                "config_dir": "qwen-image-transformer",
            },
            "text_encoder": {
                "model_id": "qwen-image-clip",
                "model_class": "Qwen2_5_VLForConditionalGeneration",
                "config_dir": "qwen-image-text-encoder",
            },
            "tokenizer": {
                "model_class": "AutoTokenizer",
                "config_dir": "qwen-image-tokenizer",
            },
            "vae": {
                "model_id": "qwen-image-vae",
                "model_class": "AutoencoderKLQwenImage",
                "config_dir": "qwen-image-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "qwen-image-scheduler",
            },
        },
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
        "inference": {"guidance_param": "true_cfg_scale", "force_negative_prompt": " "},
        "training": {"caption_dropout": 0.0, "cache_text_embeddings": True},
    },
    "qwen_image_edit": {
        "pipeline_class": "QwenImageEditPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "QwenImageTransformer2DModel",
                "config_dir": "qwen-image-edit-transformer",
            },
            "text_encoder": {
                "model_id": "qwen-image-clip",
                "model_class": "Qwen2_5_VLForConditionalGeneration",
                "config_dir": "qwen-image-text-encoder",
            },
            "tokenizer": {
                "model_class": "AutoTokenizer",
                "config_dir": "qwen-image-tokenizer",
            },
            "processor": {
                "model_class": "Qwen2VLProcessor",
                "config_dir": "qwen-image-processor",
            },
            "vae": {
                "model_id": "qwen-image-vae",
                "model_class": "AutoencoderKLQwenImage",
                "config_dir": "qwen-image-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "qwen-image-edit-scheduler",
            },
        },
        "model_flags": {
            "arch": "qwen_image_edit",
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
        "sample": {"sampler": "flowmatch", "steps": 50, "guidance": 4.0, "neg": ""},
        "inference": {"guidance_param": "true_cfg_scale", "force_negative_prompt": " "},
    },
    # 2511 "Plus" revision — uses QwenImageEditPlusPipeline (different from original)
    "qwen_image_edit_2511": {
        "pipeline_class": "QwenImageEditPlusPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "QwenImageTransformer2DModel",
                "config_dir": "qwen-image-edit-transformer",
            },
            "text_encoder": {
                "model_id": "qwen-image-clip",
                "model_class": "Qwen2_5_VLForConditionalGeneration",
                "config_dir": "qwen-image-text-encoder",
            },
            "tokenizer": {
                "model_class": "AutoTokenizer",
                "config_dir": "qwen-image-tokenizer",
            },
            "processor": {
                "model_class": "Qwen2VLProcessor",
                "config_dir": "qwen-image-processor",
            },
            "vae": {
                "model_id": "qwen-image-vae",
                "model_class": "AutoencoderKLQwenImage",
                "config_dir": "qwen-image-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "qwen-image-edit-scheduler",
            },
        },
        "model_flags": {
            "arch": "qwen_image_edit_2511",
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
        "sample": {"sampler": "flowmatch", "steps": 40, "guidance": 4.0, "neg": ""},
        "inference": {"guidance_param": "true_cfg_scale", "force_negative_prompt": " ", "scheduler_shift": 3.1},
    },
    "flux_fill": {
        "pipeline_class": "FluxFillPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "FluxTransformer2DModel",
                "config_dir": "flux-fill-transformer",
            },
            "text_encoder": {
                "model_id": "clip-l",
                "model_class": "CLIPTextModel",
                "config_dir": "clip-l",
            },
            "tokenizer": {
                "model_class": "CLIPTokenizer",
                "config_dir": "clip-tokenizer",
            },
            "text_encoder_2": {
                "model_id": ["t5-xxl-fp8", "t5-xxl-fp16"],
                "model_class": "T5EncoderModel",
                "config_dir": "t5-xxl",
            },
            "tokenizer_2": {
                "model_class": "T5TokenizerFast",
                "config_dir": "t5-tokenizer",
            },
            "vae": {
                "model_id": "flux-vae",
                "model_class": "AutoencoderKL",
                "config_dir": "flux-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "flux-dev-scheduler",
            },
        },
        "model_flags": {"is_flux": True, "quantize": True, "low_vram": True},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 50, "guidance": 30.0, "neg": ""},
        "inference": {"skip_strength_in_inpaint": True},
    },
    "flux_fill_onereward": {
        "pipeline_class": "FluxFillPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "FluxTransformer2DModel",
                "config_dir": "flux-fill-transformer",
            },
            "text_encoder": {
                "model_id": "clip-l",
                "model_class": "CLIPTextModel",
                "config_dir": "clip-l",
            },
            "tokenizer": {
                "model_class": "CLIPTokenizer",
                "config_dir": "clip-tokenizer",
            },
            "text_encoder_2": {
                "model_id": ["t5-xxl-fp8", "t5-xxl-fp16"],
                "model_class": "T5EncoderModel",
                "config_dir": "t5-xxl",
            },
            "tokenizer_2": {
                "model_class": "T5TokenizerFast",
                "config_dir": "t5-tokenizer",
            },
            "vae": {
                "model_id": "flux-vae",
                "model_class": "AutoencoderKL",
                "config_dir": "flux-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "flux-dev-scheduler",
            },
        },
        "model_flags": {"is_flux": True, "quantize": True, "low_vram": True},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 50, "guidance": 30.0, "neg": ""},
        "inference": {"skip_strength_in_inpaint": True},
    },
    # -------------------------------------------------------------------
    # LTX Video 2.3 (Lightricks, 2025)
    # -------------------------------------------------------------------
    "ltx2_video": {
        "pipeline_class": "LTX2Pipeline",
        "img2vid_class": "LTX2ImageToVideoPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "LTX2VideoTransformer3DModel",
                "config_dir": "ltx2-transformer",
            },
            "text_encoder": {
                "model_id": "ltx2-gemma3-12b",
                "model_class": "Gemma3ForConditionalGeneration",
                "config_dir": "ltx2-text-encoder",
            },
            "tokenizer": {
                "model_class": "AutoTokenizer",
                "config_dir": "ltx2-tokenizer",
            },
            "vae": {
                "model_id": "ltx2-vae",
                "model_class": "AutoencoderKLLTX2Video",
                "config_dir": "ltx2-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "ltx2-scheduler",
            },
        },
        "model_flags": {"is_video": True},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768],
        "default_resolution": 768,
        "sample": {"sampler": "flowmatch", "steps": 30, "guidance": 3.0, "neg": ""},
    },
    "ltx2_video_distilled": {
        "pipeline_class": "LTX2Pipeline",
        "img2vid_class": "LTX2ImageToVideoPipeline",
        "gen_components": {
            "transformer": {
                "model_class": "LTX2VideoTransformer3DModel",
                "config_dir": "ltx23-transformer",
            },
            "text_encoder": {
                "model_id": "ltx2-gemma3-12b",
                "model_class": "Gemma3ForConditionalGeneration",
                "config_dir": "ltx2-text-encoder",
            },
            "tokenizer": {
                "model_class": "AutoTokenizer",
                "config_dir": "ltx2-tokenizer",
            },
            "vae": {
                "model_id": "ltx2-vae",
                "model_class": "AutoencoderKLLTX2Video",
                "config_dir": "ltx2-vae",
            },
            "scheduler": {
                "model_class": "FlowMatchEulerDiscreteScheduler",
                "config_dir": "ltx2-scheduler",
            },
        },
        "model_flags": {"is_video": True, "is_distilled": True},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768],
        "default_resolution": 768,
        "sample": {"sampler": "flowmatch", "steps": 8, "guidance": 1.0, "neg": ""},
    },
    "sdxl": {
        "pipeline_class": "StableDiffusionXLPipeline",
        "img2img_class": "StableDiffusionXLImg2ImgPipeline",
        "inpaint_class": "StableDiffusionXLInpaintPipeline",
        "model_flags": {"arch": "sdxl"},
        "noise_scheduler": "ddpm",
        "dtype": "bf16",
        "train_text_encoder": True,
        "resolutions": [768, 1024],
        "default_resolution": 1024,
        "extra_train": {"max_denoising_steps": 1000},
        "sample": {"sampler": "euler", "steps": 30, "guidance": 7.5, "neg": "blurry, low quality, deformed"},
        "inference": {"supports_negative_prompt": True},
        "training": {"noise_offset": {"style": 0.0357, "default": 0.0}},
    },
    "sd15": {
        "pipeline_class": "StableDiffusionPipeline",
        "img2img_class": "StableDiffusionImg2ImgPipeline",
        "inpaint_class": "StableDiffusionInpaintPipeline",
        "model_flags": {},
        "noise_scheduler": "ddpm",
        "dtype": "fp16",
        "train_text_encoder": True,
        "resolutions": [512],
        "default_resolution": 512,
        "extra_train": {"max_denoising_steps": 1000},
        "sample": {"sampler": "euler", "steps": 30, "guidance": 7.5, "neg": "blurry, low quality, deformed"},
        "inference": {"supports_negative_prompt": True},
        "training": {"use_ema": False},
    },
}

# -----------------------------------------------------------------------
# Model registry: modl model IDs → (arch_key, HuggingFace hub ID)
# -----------------------------------------------------------------------

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "flux2-dev":      ("flux2",         "black-forest-labs/FLUX.2-dev"),
    "flux2-klein-4b": ("flux2_klein",   "black-forest-labs/FLUX.2-klein-4b-fp8"),
    "flux2-klein-9b": ("flux2_klein_9b", "black-forest-labs/FLUX.2-klein-9b-fp8"),
    "flux2-klein-base-4b": ("flux2_klein_base", "black-forest-labs/FLUX.2-klein-base-4B"),
    "flux2-klein-base-9b": ("flux2_klein_base_9b", "black-forest-labs/FLUX.2-klein-base-9B"),
    "flux-dev":       ("flux",          "black-forest-labs/FLUX.1-dev"),
    "flux-schnell":   ("flux_schnell",  "black-forest-labs/FLUX.1-schnell"),
    "z-image-turbo":  ("zimage_turbo",  "Tongyi-MAI/Z-Image-Turbo"),
    "z-image":        ("zimage",        "Tongyi-MAI/Z-Image"),
    "chroma":         ("chroma",        "lodestones/Chroma"),
    "qwen-image":     ("qwen_image",    "Qwen/Qwen-Image-2512"),
    "qwen_image":     ("qwen_image",    "Qwen/Qwen-Image-2512"),
    "qwen-image-edit":       ("qwen_image_edit",      "Qwen/Qwen-Image-Edit"),
    "qwen_image_edit":       ("qwen_image_edit",      "Qwen/Qwen-Image-Edit"),
    "qwen-image-edit-2511":  ("qwen_image_edit_2511", "Qwen/Qwen-Image-Edit-2511"),
    "qwen_image_edit_2511":  ("qwen_image_edit_2511", "Qwen/Qwen-Image-Edit-2511"),
    "flux-fill-dev":            ("flux_fill",            "black-forest-labs/FLUX.1-Fill-dev"),
    "flux-fill-dev-onereward":  ("flux_fill_onereward",  "yichengup/flux.1-fill-dev-OneReward"),
    "ltx-2.3-22b":         ("ltx2_video_distilled", "dg845/LTX-2.3-Distilled-Diffusers"),
    "ltx-video-2-3":       ("ltx2_video_distilled", "dg845/LTX-2.3-Distilled-Diffusers"),
    "ltx-video-distilled": ("ltx2_video_distilled", "dg845/LTX-2.3-Distilled-Diffusers"),
    "ltx-video-dev":       ("ltx2_video",           "Lightricks/LTX-2"),
    "sdxl-base-1.0":  ("sdxl",          "stabilityai/stable-diffusion-xl-base-1.0"),
    "sdxl-turbo":     ("sdxl",          "stabilityai/sdxl-turbo"),
    "sd-1.5":         ("sd15",          "stable-diffusion-v1-5/stable-diffusion-v1-5"),
}


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def detect_arch(base_model_id: str, arch_key: str | None = None) -> str:
    """Detect architecture key from a base model ID.

    If *arch_key* is provided (e.g. from a job spec), it is used directly
    when it matches a known ARCH_CONFIGS entry — no inference needed.
    Otherwise checks MODEL_REGISTRY for an exact match, then falls back
    to substring heuristics.  Returns a key into ARCH_CONFIGS.
    """
    if arch_key and arch_key in ARCH_CONFIGS:
        return arch_key

    entry = MODEL_REGISTRY.get(base_model_id)
    if entry:
        return entry[0]

    bid = base_model_id.lower()
    if "ltx" in bid:
        if "distill" in bid:
            return "ltx2_video_distilled"
        return "ltx2_video"
    if "fill" in bid and "onereward" in bid:
        return "flux_fill_onereward"
    if "fill" in bid:
        return "flux_fill"
    if ("qwen" in bid and "edit" in bid) or "qwen_image_edit" in bid:
        if "2511" in bid:
            return "qwen_image_edit_2511"
        return "qwen_image_edit"
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
    if "klein" in bid and ("flux2" in bid or "flux.2" in bid or "flux-2" in bid):
        is_base = "base" in bid
        is_9b = "9b" in bid
        if is_base and is_9b:
            return "flux2_klein_base_9b"
        if is_9b:
            return "flux2_klein_9b"
        if is_base:
            return "flux2_klein_base"
        return "flux2_klein"
    if "flux" in bid and ("2" in bid or "flux.2" in bid or "flux2" in bid):
        return "flux2"
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


def resolve_pipeline_class_for_mode(base_model_id: str, mode: str = "txt2img") -> str:
    """Return the diffusers pipeline class name for a model ID and generation mode.

    Args:
        base_model_id: modl model ID (e.g. "flux-dev")
        mode: "txt2img", "img2img", or "inpaint"
    """
    arch = detect_arch(base_model_id)
    config = ARCH_CONFIGS.get(arch, ARCH_CONFIGS["sdxl"])
    if mode == "img2img":
        return config.get("img2img_class", config["pipeline_class"])
    elif mode == "inpaint":
        return config.get("inpaint_class", config["pipeline_class"])
    elif mode == "img2vid":
        return config.get("img2vid_class", config["pipeline_class"])
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
    for models that require separate component loading.
    Returns empty dict if no components needed or not found.

    Handles both old simple format ({"text_encoder": "clip-l"}) and
    new richer format ({"text_encoder": {"model_id": "clip-l", ...}}).
    """
    arch = detect_arch(base_model_id)
    config = ARCH_CONFIGS.get(arch, {})
    gen_components = config.get("gen_components", {})
    if not gen_components:
        return {}

    resolved = {}
    for component_type, spec in gen_components.items():
        # New richer format: spec is a dict with model_id, model_class, config_dir
        if isinstance(spec, dict):
            model_ids = spec.get("model_id")
            if model_ids is None:
                continue  # transformer or config-only component
            if isinstance(model_ids, str):
                model_ids = [model_ids]
        else:
            # Old simple format: spec is a string or list of strings
            model_ids = [spec] if isinstance(spec, str) else spec

        for model_id in model_ids:
            path = _get_installed_path(model_id)
            if path:
                resolved[component_type] = path
                break
    return resolved


def resolve_gen_assembly(base_model_id: str) -> dict[str, dict] | None:
    """Return the full assembly spec for a model, or None if not available.

    Returns the gen_components dict with model_class, config_dir, and resolved
    model paths for each component. Used by assemble_pipeline() in gen_adapter.
    """
    arch = detect_arch(base_model_id)
    config = ARCH_CONFIGS.get(arch, {})
    gen_components = config.get("gen_components", {})
    if not gen_components:
        return None

    # Only works with the new richer format (dicts with model_class)
    first_val = next(iter(gen_components.values()))
    if not isinstance(first_val, dict):
        return None

    assembly = {}
    for component_type, spec in gen_components.items():
        entry = dict(spec)  # copy
        model_ids = spec.get("model_id")
        if model_ids is not None:
            if isinstance(model_ids, str):
                model_ids = [model_ids]
            for mid in model_ids:
                path = _get_installed_path(mid)
                if path:
                    entry["resolved_path"] = path
                    break
        assembly[component_type] = entry
    return assembly


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
