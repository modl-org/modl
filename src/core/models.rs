//! TOML-driven model metadata — the single source of truth.
//!
//! Parses `models.toml` (embedded at compile time) into typed structs.
//! Consumed by `model_family` (public API) and validated against
//! Python's `arch_config.py` via `scripts/validate-arch-sync.py`.

use serde::Deserialize;
use std::collections::HashMap;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// TOML schema types (deserialization)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct TomlRoot {
    families: HashMap<String, TomlFamily>,
}

#[derive(Debug, Deserialize)]
struct TomlFamily {
    name: String,
    vendor: String,
    year: u16,
    models: HashMap<String, TomlModel>,
}

#[derive(Debug, Deserialize)]
struct TomlModel {
    name: String,
    arch_key: String,
    description: String,
    transformer_b: f32,
    text_encoder: String,
    text_encoder_b: f32,
    total_b: f32,
    vram_bf16: u32,
    vram_fp8: u32,
    capabilities: Vec<String>,
    quality: u8,
    speed: u8,
    #[serde(default)]
    text_rendering: bool,
    defaults: TomlDefaults,
    #[serde(default)]
    video_defaults: Option<TomlVideoDefaults>,
    #[serde(default)]
    lightning: Option<TomlLightning>,
    #[serde(default)]
    controlnet: Option<TomlControlNet>,
    #[serde(default)]
    style_ref: Option<TomlStyleRef>,
}

#[derive(Debug, Deserialize)]
struct TomlDefaults {
    steps: u32,
    guidance: f32,
    resolution: u32,
}

#[derive(Debug, Deserialize)]
struct TomlVideoDefaults {
    frames: u32,
    fps: u32,
}

#[derive(Debug, Deserialize)]
struct TomlLightning {
    lora_registry_id: String,
    variant_4step: String,
    variant_8step: String,
    guidance: f32,
    scheduler_overrides: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct TomlControlNet {
    manifest_id: String,
    types: Vec<String>,
    default_strength: f32,
    default_end: f32,
    min_steps: u32,
}

#[derive(Debug, Deserialize)]
struct TomlStyleRef {
    mechanism: String,
    manifest_id: String,
    default_strength: f32,
}

// ---------------------------------------------------------------------------
// Runtime types (owned, long-lived in LazyLock)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelFamily {
    pub id: String,
    pub name: String,
    pub vendor: String,
    pub year: u16,
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub arch_key: String,
    pub transformer_b: f32,
    pub text_encoder_name: String,
    pub text_encoder_b: f32,
    pub total_b: f32,
    pub vram_bf16_gb: u32,
    pub vram_fp8_gb: u32,
    pub capabilities: Capabilities,
    pub default_steps: u32,
    pub default_guidance: f32,
    pub default_resolution: u32,
    pub quality: u8,
    pub speed: u8,
    pub text_rendering: bool,
    pub default_frames: u32,
    pub default_fps: u32,
    pub description: String,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Capabilities {
    pub txt2img: bool,
    pub img2img: bool,
    pub inpaint: bool,
    pub edit: bool,
    pub lora: bool,
    pub training: bool,
    pub lanpaint_inpaint: bool,
    pub txt2vid: bool,
    pub img2vid: bool,
}

impl Capabilities {
    fn from_list(caps: &[String]) -> Self {
        Self {
            txt2img: caps.iter().any(|c| c == "txt2img"),
            img2img: caps.iter().any(|c| c == "img2img"),
            inpaint: caps.iter().any(|c| c == "inpaint"),
            edit: caps.iter().any(|c| c == "edit"),
            lora: caps.iter().any(|c| c == "lora"),
            training: caps.iter().any(|c| c == "training"),
            lanpaint_inpaint: caps.iter().any(|c| c == "lanpaint_inpaint"),
            txt2vid: caps.iter().any(|c| c == "txt2vid"),
            img2vid: caps.iter().any(|c| c == "img2vid"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LightningConfig {
    pub base_model_id: String,
    pub lora_registry_id: String,
    pub variant_4step: String,
    pub variant_8step: String,
    pub guidance: f32,
    pub scheduler_overrides: Vec<(String, String)>,
}

impl LightningConfig {
    pub fn resolve(&self, fast_steps: u32) -> (&str, u32) {
        if fast_steps <= 4 {
            (&self.variant_4step, 4)
        } else {
            (&self.variant_8step, 8)
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ControlNetSupport {
    pub base_model_id: String,
    pub manifest_id: String,
    pub supported_types: Vec<String>,
    pub default_strength: f32,
    pub default_end: f32,
    pub recommended_min_steps: u32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct StyleRefSupport {
    pub base_model_id: String,
    pub mechanism: String,
    pub manifest_id: Option<String>,
    pub default_strength: f32,
}

// ---------------------------------------------------------------------------
// Parsed data (static lifetime via LazyLock)
// ---------------------------------------------------------------------------

struct ParsedModels {
    families: Vec<ModelFamily>,
    lightning: Vec<LightningConfig>,
    controlnet: Vec<ControlNetSupport>,
    style_ref: Vec<StyleRefSupport>,
    /// model_id → index into flattened models list (for fast lookup)
    id_index: HashMap<String, (usize, usize)>, // (family_idx, model_idx)
    /// Fuzzy match patterns: substring → model_id
    fuzzy_patterns: Vec<(Vec<String>, String)>,
}

static PARSED: LazyLock<ParsedModels> = LazyLock::new(|| {
    let toml_str = include_str!("../../models.toml");
    let root: TomlRoot = toml::from_str(toml_str).expect("Failed to parse models.toml");
    build_parsed(root)
});

fn build_parsed(root: TomlRoot) -> ParsedModels {
    let mut families = Vec::new();
    let mut lightning = Vec::new();
    let mut controlnet = Vec::new();
    let mut style_ref = Vec::new();
    let mut id_index = HashMap::new();

    // Sort families by year then name for stable ordering
    let mut sorted_families: Vec<_> = root.families.into_iter().collect();
    sorted_families.sort_by(|a, b| a.0.cmp(&b.0));

    for (fam_id, fam) in sorted_families {
        let mut models = Vec::new();

        // Sort models within family for stable ordering
        let mut sorted_models: Vec<_> = fam.models.into_iter().collect();
        sorted_models.sort_by(|a, b| a.0.cmp(&b.0));

        for (model_id, m) in sorted_models {
            let model_idx = models.len();
            let family_idx = families.len();
            id_index.insert(model_id.clone(), (family_idx, model_idx));

            // Extract lightning config
            if let Some(l) = m.lightning {
                lightning.push(LightningConfig {
                    base_model_id: model_id.clone(),
                    lora_registry_id: l.lora_registry_id,
                    variant_4step: l.variant_4step,
                    variant_8step: l.variant_8step,
                    guidance: l.guidance,
                    scheduler_overrides: l.scheduler_overrides.into_iter().collect(),
                });
            }

            // Extract controlnet support
            if let Some(cn) = m.controlnet {
                controlnet.push(ControlNetSupport {
                    base_model_id: model_id.clone(),
                    manifest_id: cn.manifest_id,
                    supported_types: cn.types,
                    default_strength: cn.default_strength,
                    default_end: cn.default_end,
                    recommended_min_steps: cn.min_steps,
                });
            }

            // Extract style ref support
            if let Some(sr) = m.style_ref {
                style_ref.push(StyleRefSupport {
                    base_model_id: model_id.clone(),
                    mechanism: sr.mechanism,
                    manifest_id: Some(sr.manifest_id),
                    default_strength: sr.default_strength,
                });
            }

            let (default_frames, default_fps) = match &m.video_defaults {
                Some(v) => (v.frames, v.fps),
                None => (0, 0),
            };

            models.push(ModelInfo {
                id: model_id,
                name: m.name,
                arch_key: m.arch_key,
                transformer_b: m.transformer_b,
                text_encoder_name: m.text_encoder,
                text_encoder_b: m.text_encoder_b,
                total_b: m.total_b,
                vram_bf16_gb: m.vram_bf16,
                vram_fp8_gb: m.vram_fp8,
                capabilities: Capabilities::from_list(&m.capabilities),
                default_steps: m.defaults.steps,
                default_guidance: m.defaults.guidance,
                default_resolution: m.defaults.resolution,
                quality: m.quality,
                speed: m.speed,
                text_rendering: m.text_rendering,
                default_frames,
                default_fps,
                description: m.description,
            });
        }

        families.push(ModelFamily {
            id: fam_id,
            name: fam.name,
            vendor: fam.vendor,
            year: fam.year,
            models,
        });
    }

    // Build fuzzy match patterns from model IDs
    let fuzzy_patterns = build_fuzzy_patterns(&families);

    ParsedModels {
        families,
        lightning,
        controlnet,
        style_ref,
        id_index,
        fuzzy_patterns,
    }
}

/// Build fuzzy match patterns for resolve_model().
///
/// Each pattern is a list of substrings that must ALL match (case-insensitive)
/// in the input, mapped to the target model ID. More specific patterns come first.
fn build_fuzzy_patterns(families: &[ModelFamily]) -> Vec<(Vec<String>, String)> {
    // These are ordered from most specific to least specific.
    // resolve_model() returns the first match, so order matters.
    let patterns: &[(&[&str], &str)] = &[
        // Qwen (most specific first)
        (&["qwen", "edit", "2511"], "qwen-image-edit-2511"),
        (&["qwen", "edit"], "qwen-image-edit"),
        (&["qwen", "image"], "qwen-image"),
        // Z-Image (turbo before base)
        (&["z-image-turbo"], "z-image-turbo"),
        (&["z_image_turbo"], "z-image-turbo"),
        (&["z-image"], "z-image"),
        (&["z_image"], "z-image"),
        (&["zimage"], "z-image"),
        // Klein (9b before 4b)
        (&["klein", "9b"], "flux2-klein-9b"),
        (&["klein"], "flux2-klein-4b"),
        // Others
        (&["chroma"], "chroma"),
        (&["fill", "onereward"], "flux-fill-dev-onereward"),
        (&["fill"], "flux-fill-dev"),
        (&["schnell"], "flux-schnell"),
        (&["flux2"], "flux2-dev"),
        (&["flux.2"], "flux2-dev"),
        (&["flux-2"], "flux2-dev"),
        (&["flux"], "flux-dev"),
        // LTX Video
        (&["ltx", "2.3"], "ltx-video-2-3"),
        (&["ltx", "2-3"], "ltx-video-2-3"),
        (&["ltx", "video"], "ltx-video-2-3"),
        (&["ltx", "dev"], "ltx-video-2-3"),
        (&["ltx2"], "ltx-video-2-3"),
        (&["ltx"], "ltx-video-2-3"),
        // Stable Diffusion
        (&["sdxl"], "sdxl"),
        (&["xl"], "sdxl"),
        (&["sd-1.5"], "sd-1.5"),
        (&["sd15"], "sd-1.5"),
        (&["1.5"], "sd-1.5"),
    ];

    // Only include patterns whose target actually exists
    let all_ids: Vec<&str> = families
        .iter()
        .flat_map(|f| f.models.iter())
        .map(|m| m.id.as_str())
        .collect();

    patterns
        .iter()
        .filter(|(_, target)| all_ids.contains(target))
        .map(|(subs, target)| {
            (
                subs.iter().map(|s| s.to_string()).collect(),
                target.to_string(),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Public accessors
// ---------------------------------------------------------------------------

pub fn families() -> &'static [ModelFamily] {
    &PARSED.families
}

pub fn find_model(model_id: &str) -> Option<&'static ModelInfo> {
    let lower = model_id.to_lowercase();
    let (fi, mi) = PARSED.id_index.get(&lower)?;
    Some(&PARSED.families[*fi].models[*mi])
}

#[allow(dead_code)]
pub fn find_by_arch_key(arch_key: &str) -> Option<&'static ModelInfo> {
    PARSED
        .families
        .iter()
        .flat_map(|f| f.models.iter())
        .find(|m| m.arch_key == arch_key)
}

#[allow(dead_code)]
pub fn find_family(model_id: &str) -> Option<&'static ModelFamily> {
    let lower = model_id.to_lowercase();
    let (fi, _) = PARSED.id_index.get(&lower)?;
    Some(&PARSED.families[*fi])
}

pub fn resolve_model(model_id: &str) -> Option<&'static ModelInfo> {
    // Exact match first
    if let Some(m) = find_model(model_id) {
        return Some(m);
    }

    // Fuzzy match via substring patterns
    let lower = model_id.to_lowercase();
    for (subs, target) in &PARSED.fuzzy_patterns {
        if subs.iter().all(|s| lower.contains(s.as_str())) {
            return find_model(target);
        }
    }

    None
}

pub fn lightning_configs() -> &'static [LightningConfig] {
    &PARSED.lightning
}

pub fn lightning_config(model_id: &str) -> Option<&'static LightningConfig> {
    let resolved = resolve_model(model_id)?;
    PARSED
        .lightning
        .iter()
        .find(|c| c.base_model_id == resolved.id)
}

pub fn controlnet_support_list() -> &'static [ControlNetSupport] {
    &PARSED.controlnet
}

pub fn controlnet_support(model_id: &str) -> Option<&'static ControlNetSupport> {
    let resolved = resolve_model(model_id)?;
    PARSED
        .controlnet
        .iter()
        .find(|c| c.base_model_id == resolved.id)
}

pub fn style_ref_support_list() -> &'static [StyleRefSupport] {
    &PARSED.style_ref
}

#[allow(dead_code)]
pub fn style_ref_support(model_id: &str) -> Option<&'static StyleRefSupport> {
    let resolved = resolve_model(model_id)?;
    PARSED
        .style_ref
        .iter()
        .find(|s| s.base_model_id == resolved.id)
}

pub fn model_defaults(model_id: &str) -> (u32, f32) {
    match resolve_model(model_id) {
        Some(m) => (m.default_steps, m.default_guidance),
        None => (28, 3.5),
    }
}

#[allow(dead_code)]
pub fn models_with_capability(capability: &str) -> Vec<&'static ModelInfo> {
    PARSED
        .families
        .iter()
        .flat_map(|f| f.models.iter())
        .filter(|m| match capability {
            "txt2img" => m.capabilities.txt2img,
            "img2img" => m.capabilities.img2img,
            "inpaint" => m.capabilities.inpaint || m.capabilities.lanpaint_inpaint,
            "edit" => m.capabilities.edit,
            "lora" => m.capabilities.lora,
            "training" => m.capabilities.training,
            "text_rendering" => m.text_rendering,
            "txt2vid" => m.capabilities.txt2vid,
            "img2vid" => m.capabilities.img2vid,
            _ => false,
        })
        .collect()
}

pub fn validate_mode(model_id: &str, mode: &str) -> Result<(), String> {
    let info = match resolve_model(model_id) {
        Some(m) => m,
        None => return Ok(()),
    };

    let supported = match mode {
        "txt2img" => info.capabilities.txt2img,
        "img2img" => info.capabilities.img2img,
        "inpaint" => info.capabilities.inpaint || info.capabilities.lanpaint_inpaint,
        "edit" => info.capabilities.edit,
        "txt2vid" => info.capabilities.txt2vid,
        "img2vid" => info.capabilities.img2vid,
        _ => return Ok(()),
    };

    if supported {
        return Ok(());
    }

    let alternatives: Vec<&str> = PARSED
        .families
        .iter()
        .flat_map(|f| f.models.iter())
        .filter(|m| match mode {
            "img2img" => m.capabilities.img2img,
            "inpaint" => m.capabilities.inpaint || m.capabilities.lanpaint_inpaint,
            "edit" => m.capabilities.edit,
            "txt2vid" => m.capabilities.txt2vid,
            "img2vid" => m.capabilities.img2vid,
            _ => false,
        })
        .map(|m| m.id.as_str())
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

pub fn validate_controlnet(model_id: &str, control_type: &str) -> Result<(), String> {
    let resolved = match resolve_model(model_id) {
        Some(m) => m,
        None => return Ok(()),
    };

    let support = match PARSED
        .controlnet
        .iter()
        .find(|c| c.base_model_id == resolved.id)
    {
        Some(s) => s,
        None => {
            let alternatives: Vec<&str> = PARSED
                .controlnet
                .iter()
                .map(|c| c.base_model_id.as_str())
                .collect();
            return Err(format!(
                "{} does not have ControlNet support. Models with ControlNet: {}",
                resolved.name,
                alternatives.join(", ")
            ));
        }
    };

    if !support.supported_types.iter().any(|t| t == control_type) {
        return Err(format!(
            "{} ControlNet does not support '{}'. Supported types: {}",
            resolved.name,
            control_type,
            support.supported_types.join(", ")
        ));
    }

    Ok(())
}

pub fn validate_style_ref(model_id: &str) -> Result<(), String> {
    let resolved = match resolve_model(model_id) {
        Some(m) => m,
        None => return Ok(()),
    };

    if PARSED
        .style_ref
        .iter()
        .any(|s| s.base_model_id == resolved.id)
    {
        return Ok(());
    }

    let alternatives: Vec<&str> = PARSED
        .style_ref
        .iter()
        .map(|s| s.base_model_id.as_str())
        .collect();
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

/// Returns (default_frames, default_fps) for a video model, or None if not a video model.
pub fn video_defaults(model_id: &str) -> Option<(u32, u32)> {
    let info = resolve_model(model_id)?;
    if info.capabilities.txt2vid || info.capabilities.img2vid {
        Some((info.default_frames, info.default_fps))
    } else {
        None
    }
}

/// Returns true if the model supports txt2vid or img2vid.
pub fn is_video_model(model_id: &str) -> bool {
    resolve_model(model_id)
        .map(|m| m.capabilities.txt2vid || m.capabilities.img2vid)
        .unwrap_or(false)
}
