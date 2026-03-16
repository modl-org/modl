//! Model family definitions — capabilities, param counts, defaults, and UI metadata.
//!
//! This is the Rust-side source of truth for model capabilities and generation
//! defaults. Used by:
//!   - CLI validation (reject unsupported modes early)
//!   - UI API (`GET /api/model-families`) for capability badges and filtering
//!   - Default steps/guidance resolution (replaces string matching)
//!
//! The Python `arch_config.py` remains the source of truth for pipeline classes
//! and training flags — this module covers the user-facing / validation side.

use serde::Serialize;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ModelFamily {
    pub id: &'static str,
    pub name: &'static str,
    pub vendor: &'static str,
    pub year: u16,
    pub models: &'static [ModelInfo],
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    /// Matches the key in Python `ARCH_CONFIGS`
    pub arch_key: &'static str,

    // -- Parameters --
    pub transformer_b: f32,
    pub text_encoder_name: &'static str,
    pub text_encoder_b: f32,
    pub total_b: f32,

    // -- VRAM --
    pub vram_bf16_gb: u32,
    pub vram_fp8_gb: u32,

    // -- Capabilities --
    pub capabilities: Capabilities,

    // -- Generation defaults --
    pub default_steps: u32,
    pub default_guidance: f32,
    pub default_resolution: u32,

    // -- UI quick-glance (1-5 scale) --
    /// Relative output quality (1=dated, 5=SOTA)
    pub quality: u8,
    /// Relative speed (1=slow, 5=very fast)
    pub speed: u8,
    /// Can render legible text in images
    pub text_rendering: bool,

    // -- Description --
    pub description: &'static str,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Capabilities {
    pub txt2img: bool,
    pub img2img: bool,
    pub inpaint: bool,
    pub edit: bool,
    pub lora: bool,
    pub training: bool,
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

pub static FAMILIES: &[ModelFamily] = &[
    // -----------------------------------------------------------------------
    // Flux 1 (Black Forest Labs, 2024)
    // -----------------------------------------------------------------------
    ModelFamily {
        id: "flux1",
        name: "Flux 1",
        vendor: "Black Forest Labs",
        year: 2024,
        models: &[
            ModelInfo {
                id: "flux-dev",
                name: "Flux Dev",
                arch_key: "flux",
                transformer_b: 12.0,
                text_encoder_name: "CLIP-L + T5-XXL",
                text_encoder_b: 5.0,
                total_b: 17.0,
                vram_bf16_gb: 34,
                vram_fp8_gb: 20,
                capabilities: Capabilities {
                    txt2img: true,
                    img2img: true,
                    inpaint: true,
                    edit: false,
                    lora: true,
                    training: true,
                },
                default_steps: 28,
                default_guidance: 3.5,
                default_resolution: 1024,
                quality: 4,
                speed: 2,
                text_rendering: false,
                description: "High quality, strong prompt following, 28 steps",
            },
            ModelInfo {
                id: "flux-schnell",
                name: "Flux Schnell",
                arch_key: "flux_schnell",
                transformer_b: 12.0,
                text_encoder_name: "CLIP-L + T5-XXL",
                text_encoder_b: 5.0,
                total_b: 17.0,
                vram_bf16_gb: 34,
                vram_fp8_gb: 20,
                capabilities: Capabilities {
                    txt2img: true,
                    img2img: true,
                    inpaint: true,
                    edit: false,
                    lora: true,
                    training: true,
                },
                default_steps: 4,
                default_guidance: 1.0,
                default_resolution: 1024,
                quality: 3,
                speed: 5,
                text_rendering: false,
                description: "Distilled, 4 steps, fast iteration",
            },
            ModelInfo {
                id: "chroma",
                name: "Chroma",
                arch_key: "chroma",
                transformer_b: 8.9,
                text_encoder_name: "T5-XXL (no CLIP)",
                text_encoder_b: 4.8,
                total_b: 13.7,
                vram_bf16_gb: 28,
                vram_fp8_gb: 16,
                capabilities: Capabilities {
                    txt2img: true,
                    img2img: true,
                    inpaint: true,
                    edit: false,
                    lora: true,
                    training: true,
                },
                default_steps: 40,
                default_guidance: 5.0,
                default_resolution: 1024,
                quality: 4,
                speed: 2,
                text_rendering: false,
                description: "Apache 2.0 Flux fork, 8.9B, negative prompts, uncensored",
            },
        ],
    },
    // -----------------------------------------------------------------------
    // Flux Fill (Black Forest Labs, 2024) — dedicated inpainting
    // -----------------------------------------------------------------------
    ModelFamily {
        id: "flux_fill",
        name: "Flux Fill",
        vendor: "Black Forest Labs",
        year: 2024,
        models: &[
            ModelInfo {
                id: "flux-fill-dev",
                name: "Flux Fill Dev",
                arch_key: "flux_fill",
                transformer_b: 12.0,
                text_encoder_name: "CLIP-L + T5-XXL",
                text_encoder_b: 5.0,
                total_b: 17.0,
                vram_bf16_gb: 34,
                vram_fp8_gb: 20,
                capabilities: Capabilities {
                    txt2img: false,
                    img2img: false,
                    inpaint: true,
                    edit: false,
                    lora: true,
                    training: false,
                },
                default_steps: 50,
                default_guidance: 30.0,
                default_resolution: 1024,
                quality: 5,
                speed: 2,
                text_rendering: false,
                description: "Dedicated inpainting model, 384-ch input, no boundary artifacts",
            },
            ModelInfo {
                id: "flux-fill-dev-onereward",
                name: "Flux Fill Dev OneReward",
                arch_key: "flux_fill_onereward",
                transformer_b: 12.0,
                text_encoder_name: "CLIP-L + T5-XXL",
                text_encoder_b: 5.0,
                total_b: 17.0,
                vram_bf16_gb: 34,
                vram_fp8_gb: 20,
                capabilities: Capabilities {
                    txt2img: false,
                    img2img: false,
                    inpaint: true,
                    edit: false,
                    lora: true,
                    training: false,
                },
                default_steps: 50,
                default_guidance: 30.0,
                default_resolution: 1024,
                quality: 5,
                speed: 2,
                text_rendering: false,
                description: "RLHF-tuned Fill, outperforms Flux Fill Pro, best inpainting",
            },
        ],
    },
    // -----------------------------------------------------------------------
    // Flux 2 (Black Forest Labs, 2025)
    // -----------------------------------------------------------------------
    ModelFamily {
        id: "flux2",
        name: "Flux 2",
        vendor: "Black Forest Labs",
        year: 2025,
        models: &[
            ModelInfo {
                id: "flux2-dev",
                name: "Flux 2 Dev",
                arch_key: "flux2",
                transformer_b: 19.0,
                text_encoder_name: "Mistral 3 Small",
                text_encoder_b: 27.0,
                total_b: 46.0,
                vram_bf16_gb: 60,
                vram_fp8_gb: 35,
                capabilities: Capabilities {
                    txt2img: true,
                    img2img: false,
                    inpaint: false,
                    edit: false,
                    lora: true,
                    training: true,
                },
                default_steps: 28,
                default_guidance: 4.0,
                default_resolution: 1024,
                quality: 5,
                speed: 1,
                text_rendering: false,
                description: "Best quality, 46B total params, needs 80GB+ GPU",
            },
            ModelInfo {
                id: "flux2-klein-4b",
                name: "Flux 2 Klein 4B",
                arch_key: "flux2_klein",
                transformer_b: 4.0,
                text_encoder_name: "Qwen3 4B",
                text_encoder_b: 4.7,
                total_b: 9.0,
                vram_bf16_gb: 14,
                vram_fp8_gb: 10,
                capabilities: Capabilities {
                    txt2img: true,
                    img2img: false,
                    inpaint: false,
                    edit: true,
                    lora: true,
                    training: true,
                },
                default_steps: 4,
                default_guidance: 1.0,
                default_resolution: 1024,
                quality: 3,
                speed: 5,
                text_rendering: false,
                description: "4B distilled, fits on consumer GPUs, 4 steps",
            },
            ModelInfo {
                id: "flux2-klein-9b",
                name: "Flux 2 Klein 9B",
                arch_key: "flux2_klein_9b",
                transformer_b: 9.0,
                text_encoder_name: "Qwen3 8B",
                text_encoder_b: 9.0,
                total_b: 18.0,
                vram_bf16_gb: 24,
                vram_fp8_gb: 16,
                capabilities: Capabilities {
                    txt2img: true,
                    img2img: false,
                    inpaint: false,
                    edit: true,
                    lora: true,
                    training: true,
                },
                default_steps: 4,
                default_guidance: 1.0,
                default_resolution: 1024,
                quality: 4,
                speed: 4,
                text_rendering: false,
                description: "9B distilled, good quality/size balance, 4 steps",
            },
        ],
    },
    // -----------------------------------------------------------------------
    // Z-Image (Tongyi-MAI / Alibaba, 2025)
    // -----------------------------------------------------------------------
    ModelFamily {
        id: "zimage",
        name: "Z-Image",
        vendor: "Tongyi-MAI",
        year: 2025,
        models: &[
            ModelInfo {
                id: "z-image",
                name: "Z-Image",
                arch_key: "zimage",
                transformer_b: 6.0,
                text_encoder_name: "Qwen3 4B",
                text_encoder_b: 4.7,
                total_b: 11.0,
                vram_bf16_gb: 20,
                vram_fp8_gb: 14,
                capabilities: Capabilities {
                    txt2img: true,
                    img2img: true,
                    inpaint: true,
                    edit: false,
                    lora: true,
                    training: true,
                },
                default_steps: 20,
                default_guidance: 4.0,
                default_resolution: 1024,
                quality: 4,
                speed: 2,
                text_rendering: false,
                description: "Strong aesthetics, 6B transformer, 20 steps",
            },
            ModelInfo {
                id: "z-image-turbo",
                name: "Z-Image Turbo",
                arch_key: "zimage_turbo",
                transformer_b: 6.0,
                text_encoder_name: "Qwen3 4B",
                text_encoder_b: 4.7,
                total_b: 11.0,
                vram_bf16_gb: 20,
                vram_fp8_gb: 14,
                capabilities: Capabilities {
                    txt2img: true,
                    img2img: true,
                    inpaint: true,
                    edit: false,
                    lora: true,
                    training: true,
                },
                default_steps: 8,
                default_guidance: 1.0,
                default_resolution: 1024,
                quality: 3,
                speed: 4,
                text_rendering: false,
                description: "Distilled Z-Image, 8 steps, good speed/quality",
            },
        ],
    },
    // -----------------------------------------------------------------------
    // Qwen Image (Qwen / Alibaba, 2025)
    // -----------------------------------------------------------------------
    ModelFamily {
        id: "qwen_image",
        name: "Qwen Image",
        vendor: "Qwen",
        year: 2025,
        models: &[
            ModelInfo {
                id: "qwen-image",
                name: "Qwen Image",
                arch_key: "qwen_image",
                transformer_b: 20.0,
                text_encoder_name: "Qwen2.5-VL 7B",
                text_encoder_b: 7.0,
                total_b: 27.0,
                vram_bf16_gb: 50,
                vram_fp8_gb: 30,
                capabilities: Capabilities {
                    txt2img: true,
                    img2img: false,
                    inpaint: false,
                    edit: false,
                    lora: true,
                    training: true,
                },
                default_steps: 25,
                default_guidance: 3.0,
                default_resolution: 1024,
                quality: 5,
                speed: 2,
                text_rendering: true,
                description: "Best text rendering, Chinese/English, 20B transformer",
            },
            ModelInfo {
                id: "qwen-image-edit",
                name: "Qwen Image Edit",
                arch_key: "qwen_image_edit",
                transformer_b: 20.0,
                text_encoder_name: "Qwen2.5-VL 7B",
                text_encoder_b: 7.0,
                total_b: 27.0,
                vram_bf16_gb: 50,
                vram_fp8_gb: 30,
                capabilities: Capabilities {
                    txt2img: false,
                    img2img: false,
                    inpaint: false,
                    edit: true,
                    lora: true,
                    training: false,
                },
                default_steps: 50,
                default_guidance: 4.0,
                default_resolution: 1024,
                quality: 5,
                speed: 1,
                text_rendering: true,
                description: "Instruction-based editing, text editing, style transfer",
            },
        ],
    },
    // -----------------------------------------------------------------------
    // Legacy SD (Stability AI)
    // -----------------------------------------------------------------------
    ModelFamily {
        id: "legacy_sd",
        name: "Stable Diffusion",
        vendor: "Stability AI",
        year: 2023,
        models: &[
            ModelInfo {
                id: "sdxl",
                name: "SDXL",
                arch_key: "sdxl",
                transformer_b: 2.6,
                text_encoder_name: "CLIP-L + OpenCLIP-G",
                text_encoder_b: 1.0,
                total_b: 3.7,
                vram_bf16_gb: 7,
                vram_fp8_gb: 5,
                capabilities: Capabilities {
                    txt2img: true,
                    img2img: true,
                    inpaint: true,
                    edit: false,
                    lora: true,
                    training: true,
                },
                default_steps: 30,
                default_guidance: 7.5,
                default_resolution: 1024,
                quality: 3,
                speed: 2,
                text_rendering: false,
                description: "Mature ecosystem, huge LoRA library, needs negative prompts",
            },
            ModelInfo {
                id: "sd-1.5",
                name: "SD 1.5",
                arch_key: "sd15",
                transformer_b: 0.9,
                text_encoder_name: "CLIP-L",
                text_encoder_b: 0.2,
                total_b: 1.1,
                vram_bf16_gb: 4,
                vram_fp8_gb: 3,
                capabilities: Capabilities {
                    txt2img: true,
                    img2img: true,
                    inpaint: true,
                    edit: false,
                    lora: true,
                    training: true,
                },
                default_steps: 30,
                default_guidance: 7.5,
                default_resolution: 512,
                quality: 2,
                speed: 4,
                text_rendering: false,
                description: "Lightweight, runs on any GPU, dated quality",
            },
        ],
    },
];

// ---------------------------------------------------------------------------
// Lightning (distillation) LoRA configs
// ---------------------------------------------------------------------------

/// Configuration for `--fast` mode: a distillation LoRA that drastically
/// reduces the number of inference steps while preserving quality.
pub struct LightningConfig {
    /// Base model ID this config applies to
    pub base_model_id: &'static str,
    /// Registry ID of the Lightning LoRA
    pub lora_registry_id: &'static str,
    /// Recommended variant for `modl pull`
    pub lora_variant: &'static str,
    /// Override steps
    pub steps: u32,
    /// Override guidance
    pub guidance: f32,
}

pub static LIGHTNING_CONFIGS: &[LightningConfig] = &[
    LightningConfig {
        base_model_id: "qwen-image-edit",
        lora_registry_id: "qwen-image-edit-lightning",
        lora_variant: "edit-8step-v1-bf16",
        steps: 8,
        guidance: 1.0,
    },
    // qwen-image (standalone gen): needs Qwen-Image-2512 model + safetensors variant
    // qwen-image-2512-lightning exists in registry but model version mismatch blocks it
    // Future: sdxl-lightning, flux-lightning, etc.
];

/// Look up the Lightning config for a model (fuzzy-matched).
pub fn lightning_config(model_id: &str) -> Option<&'static LightningConfig> {
    let resolved = resolve_model(model_id)?;
    LIGHTNING_CONFIGS
        .iter()
        .find(|c| c.base_model_id == resolved.id)
}

// ---------------------------------------------------------------------------
// Lookup helpers
// ---------------------------------------------------------------------------

/// Find model info by model ID (e.g. "flux-schnell", "qwen-image-edit").
pub fn find_model(model_id: &str) -> Option<&'static ModelInfo> {
    let lower = model_id.to_lowercase();
    FAMILIES
        .iter()
        .flat_map(|f| f.models.iter())
        .find(|m| m.id == lower)
}

/// Find model info by arch key (e.g. "flux_schnell", "qwen_image").
#[allow(dead_code)]
pub fn find_by_arch_key(arch_key: &str) -> Option<&'static ModelInfo> {
    FAMILIES
        .iter()
        .flat_map(|f| f.models.iter())
        .find(|m| m.arch_key == arch_key)
}

/// Find the family a model belongs to.
#[allow(dead_code)]
pub fn find_family(model_id: &str) -> Option<&'static ModelFamily> {
    let lower = model_id.to_lowercase();
    FAMILIES
        .iter()
        .find(|f| f.models.iter().any(|m| m.id == lower))
}

/// Fuzzy-match a model ID to the best ModelInfo.
///
/// Handles installed model names that don't exactly match our IDs
/// (e.g. "FLUX.1-schnell-fp8" → "flux-schnell").
pub fn resolve_model(model_id: &str) -> Option<&'static ModelInfo> {
    // Exact match first
    if let Some(m) = find_model(model_id) {
        return Some(m);
    }

    let lower = model_id.to_lowercase();

    // Qwen Image Edit before Qwen Image (more specific first)
    if lower.contains("qwen") && lower.contains("edit") {
        return find_model("qwen-image-edit");
    }
    if lower.contains("qwen") && lower.contains("image") {
        return find_model("qwen-image");
    }
    if lower.contains("z-image-turbo") || lower.contains("z_image_turbo") {
        return find_model("z-image-turbo");
    }
    if lower.contains("z-image") || lower.contains("z_image") || lower.contains("zimage") {
        return find_model("z-image");
    }
    if lower.contains("klein") && lower.contains("9b") {
        return find_model("flux2-klein-9b");
    }
    if lower.contains("klein") {
        return find_model("flux2-klein-4b");
    }
    if lower.contains("chroma") {
        return find_model("chroma");
    }
    if lower.contains("fill") && lower.contains("onereward") {
        return find_model("flux-fill-dev-onereward");
    }
    if lower.contains("fill") {
        return find_model("flux-fill-dev");
    }
    if lower.contains("schnell") {
        return find_model("flux-schnell");
    }
    if lower.contains("flux2") || lower.contains("flux.2") || lower.contains("flux-2") {
        return find_model("flux2-dev");
    }
    if lower.contains("flux") {
        return find_model("flux-dev");
    }
    if lower.contains("sdxl") || lower.contains("xl") {
        return find_model("sdxl");
    }
    if lower.contains("sd-1.5") || lower.contains("sd15") || lower.contains("1.5") {
        return find_model("sd-1.5");
    }

    None
}

/// Get default steps and guidance for a model.
pub fn model_defaults(model_id: &str) -> (u32, f32) {
    match resolve_model(model_id) {
        Some(m) => (m.default_steps, m.default_guidance),
        None => (28, 3.5),
    }
}

/// Validate that a model supports the requested generation mode.
/// Returns `Ok(())` or an error with a helpful message.
pub fn validate_mode(model_id: &str, mode: &str) -> Result<(), String> {
    let info = match resolve_model(model_id) {
        Some(m) => m,
        None => return Ok(()), // unknown model, let the worker handle it
    };

    let supported = match mode {
        "txt2img" => info.capabilities.txt2img,
        "img2img" => info.capabilities.img2img,
        "inpaint" => info.capabilities.inpaint,
        "edit" => info.capabilities.edit,
        _ => return Ok(()),
    };

    if supported {
        return Ok(());
    }

    // Build list of models that DO support this mode
    let alternatives: Vec<&str> = FAMILIES
        .iter()
        .flat_map(|f| f.models.iter())
        .filter(|m| match mode {
            "img2img" => m.capabilities.img2img,
            "inpaint" => m.capabilities.inpaint,
            "edit" => m.capabilities.edit,
            _ => false,
        })
        .map(|m| m.id)
        .collect();

    Err(format!(
        "{} does not support {mode}. Models with {mode}: {}",
        info.name,
        if alternatives.is_empty() {
            "none".to_string()
        } else {
            alternatives.join(", ")
        }
    ))
}

// ---------------------------------------------------------------------------
// ControlNet support metadata
// ---------------------------------------------------------------------------

/// Describes ControlNet availability for a base model.
#[allow(dead_code)]
pub struct ControlNetSupport {
    pub base_model_id: &'static str,
    /// Registry manifest ID for the ControlNet weights
    pub manifest_id: &'static str,
    /// Control types this ControlNet supports
    pub supported_types: &'static [&'static str],
    /// Default conditioning strength
    pub default_strength: f32,
    /// Default control end fraction
    pub default_end: f32,
    /// Recommended minimum steps when using ControlNet
    pub recommended_min_steps: u32,
}

pub static CONTROLNET_SUPPORT: &[ControlNetSupport] = &[
    ControlNetSupport {
        base_model_id: "z-image-turbo",
        manifest_id: "z-image-turbo-controlnet-union",
        supported_types: &["canny", "hed", "depth", "pose", "mlsd", "scribble", "gray"],
        default_strength: 0.6,
        default_end: 0.8,
        recommended_min_steps: 8,
    },
    ControlNetSupport {
        base_model_id: "flux-dev",
        manifest_id: "flux-dev-controlnet-union",
        supported_types: &["canny", "depth", "pose", "softedge", "gray"],
        default_strength: 0.7,
        default_end: 0.8,
        recommended_min_steps: 28,
    },
    ControlNetSupport {
        base_model_id: "qwen-image",
        manifest_id: "qwen-image-controlnet-union",
        supported_types: &["canny", "depth", "pose", "softedge"],
        default_strength: 0.8,
        default_end: 1.0,
        recommended_min_steps: 25,
    },
    ControlNetSupport {
        base_model_id: "sdxl",
        manifest_id: "sdxl-controlnet-union",
        supported_types: &[
            "canny", "depth", "pose", "softedge", "tile", "scribble", "hed", "mlsd", "normal",
        ],
        default_strength: 0.75,
        default_end: 1.0,
        recommended_min_steps: 28,
    },
    ControlNetSupport {
        base_model_id: "flux-schnell",
        manifest_id: "flux-dev-controlnet-union",
        supported_types: &["canny", "depth", "pose", "softedge", "gray"],
        default_strength: 0.7,
        default_end: 0.8,
        recommended_min_steps: 4,
    },
    ControlNetSupport {
        base_model_id: "z-image",
        manifest_id: "z-image-turbo-controlnet-union",
        supported_types: &["canny", "hed", "depth", "pose", "mlsd", "scribble", "gray"],
        default_strength: 0.6,
        default_end: 0.8,
        recommended_min_steps: 20,
    },
    // flux2-dev: ControlNet exists (alibaba-pai) but requires VideoX-Fun
    // pipeline not yet in diffusers. Defer until Flux2ControlNetPipeline lands.
];

/// Find ControlNet support for a base model.
pub fn controlnet_support(model_id: &str) -> Option<&'static ControlNetSupport> {
    let resolved = resolve_model(model_id)?;
    CONTROLNET_SUPPORT
        .iter()
        .find(|c| c.base_model_id == resolved.id)
}

/// Validate that a model supports ControlNet with a specific control type.
/// Returns Ok(()) or an error message with suggestions.
pub fn validate_controlnet(model_id: &str, control_type: &str) -> Result<(), String> {
    let resolved = match resolve_model(model_id) {
        Some(m) => m,
        None => return Ok(()), // unknown model, let worker handle it
    };

    let support = match CONTROLNET_SUPPORT
        .iter()
        .find(|c| c.base_model_id == resolved.id)
    {
        Some(s) => s,
        None => {
            // Find models that DO have ControlNet
            let alternatives: Vec<&str> =
                CONTROLNET_SUPPORT.iter().map(|c| c.base_model_id).collect();
            return Err(format!(
                "{} does not have ControlNet support. Models with ControlNet: {}",
                resolved.name,
                alternatives.join(", ")
            ));
        }
    };

    if !support.supported_types.contains(&control_type) {
        return Err(format!(
            "{} ControlNet does not support '{}'. Supported types: {}",
            resolved.name,
            control_type,
            support.supported_types.join(", ")
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Style reference / IP-Adapter support metadata
// ---------------------------------------------------------------------------

/// Describes style-ref availability for a base model.
#[allow(dead_code)]
pub struct StyleRefSupport {
    pub base_model_id: &'static str,
    /// "native" (built-in, e.g. Klein multi-ref) or "ip-adapter"
    pub mechanism: &'static str,
    /// Registry manifest ID for the IP-Adapter weights (None for native)
    pub manifest_id: Option<&'static str>,
    pub default_strength: f32,
}

pub static STYLE_REF_SUPPORT: &[StyleRefSupport] = &[
    StyleRefSupport {
        base_model_id: "flux2-klein-4b",
        mechanism: "native",
        manifest_id: None,
        default_strength: 0.6,
    },
    StyleRefSupport {
        base_model_id: "flux2-klein-9b",
        mechanism: "native",
        manifest_id: None,
        default_strength: 0.6,
    },
    StyleRefSupport {
        base_model_id: "sdxl",
        mechanism: "ip-adapter",
        manifest_id: Some("sdxl-ip-adapter"),
        default_strength: 0.6,
    },
    StyleRefSupport {
        base_model_id: "flux-dev",
        mechanism: "ip-adapter",
        manifest_id: Some("flux-dev-ip-adapter"),
        default_strength: 0.5,
    },
];

/// Find style-ref support for a base model.
#[allow(dead_code)]
pub fn style_ref_support(model_id: &str) -> Option<&'static StyleRefSupport> {
    let resolved = resolve_model(model_id)?;
    STYLE_REF_SUPPORT
        .iter()
        .find(|s| s.base_model_id == resolved.id)
}

/// Validate that a model supports --style-ref.
/// Returns Ok(()) or an error message with suggestions.
pub fn validate_style_ref(model_id: &str) -> Result<(), String> {
    let resolved = match resolve_model(model_id) {
        Some(m) => m,
        None => return Ok(()),
    };

    if STYLE_REF_SUPPORT
        .iter()
        .any(|s| s.base_model_id == resolved.id)
    {
        return Ok(());
    }

    let alternatives: Vec<&str> = STYLE_REF_SUPPORT.iter().map(|s| s.base_model_id).collect();
    Err(format!(
        "{} does not support --style-ref (no IP-Adapter available).\n\n\
         Alternatives:\n\
         • Train a style LoRA: modl train --dataset <folder> --base {} --lora-type style\n\
         • Use a model with style-ref support: {}",
        resolved.name,
        resolved.id,
        alternatives.join(", ")
    ))
}

/// List all models that support a given capability.
#[allow(dead_code)]
pub fn models_with_capability(capability: &str) -> Vec<&'static ModelInfo> {
    FAMILIES
        .iter()
        .flat_map(|f| f.models.iter())
        .filter(|m| match capability {
            "txt2img" => m.capabilities.txt2img,
            "img2img" => m.capabilities.img2img,
            "inpaint" => m.capabilities.inpaint,
            "edit" => m.capabilities.edit,
            "lora" => m.capabilities.lora,
            "training" => m.capabilities.training,
            "text_rendering" => m.text_rendering,
            _ => false,
        })
        .collect()
}
