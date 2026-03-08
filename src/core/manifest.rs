use serde::{Deserialize, Serialize};

/// Top-level manifest for a registry item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub asset_type: AssetType,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub homepage: Option<String>,
    #[serde(default)]
    pub description: Option<String>,

    // Checkpoint-style: multiple variants
    #[serde(default)]
    pub variants: Vec<Variant>,

    // LoRA/simple-style: single file
    #[serde(default)]
    pub file: Option<FileInfo>,

    #[serde(default)]
    pub requires: Vec<Dependency>,

    #[serde(default)]
    pub auth: Option<AuthRequirement>,

    #[serde(default)]
    pub defaults: Option<ModelDefaults>,

    // LoRA-specific
    #[serde(default)]
    pub base_models: Vec<String>,
    #[serde(default)]
    pub trigger_words: Vec<String>,
    #[serde(default)]
    pub recommended_weight: Option<f32>,
    #[serde(default)]
    pub weight_range: Option<(f32, f32)>,

    // ControlNet-specific
    #[serde(default)]
    pub preprocessor: Option<String>,

    // Upscaler-specific
    #[serde(default)]
    pub scale_factor: Option<u32>,

    // IP-Adapter-specific
    #[serde(default)]
    pub clip_vision_model: Option<String>,

    // Cloud availability
    #[serde(default)]
    pub cloud_available: bool,
    #[serde(default)]
    pub cloud_training: Option<CloudTrainingInfo>,
    #[serde(default)]
    pub cloud_inference: Option<CloudInferenceInfo>,

    // Category (for LoRAs and recipes)
    #[serde(default)]
    pub category: Option<String>,

    // Community LoRA training details
    #[serde(default)]
    pub training_details: Option<TrainingDetails>,
    #[serde(default)]
    pub sample_images: Vec<SampleImage>,

    // Recipe config (for recipe type)
    #[serde(default)]
    pub recipe: Option<RecipeConfig>,

    // Vision-language: HuggingFace repo ID (multi-file transformer checkpoints)
    #[serde(default)]
    pub huggingface_repo: Option<String>,

    // Publisher (distinct from author, for community contributions)
    #[serde(default)]
    pub publisher: Option<String>,

    #[serde(default)]
    pub preview_images: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub rating: Option<f32>,
    #[serde(default)]
    pub downloads: Option<u64>,
    #[serde(default)]
    pub added: Option<String>,
    #[serde(default)]
    pub updated: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum AssetType {
    Checkpoint,
    #[value(name = "diffusion_model")]
    DiffusionModel,
    Lora,
    Vae,
    #[value(name = "text_encoder", alias = "textencoder")]
    TextEncoder,
    Controlnet,
    Upscaler,
    Embedding,
    Ipadapter,
    Segmentation,
    #[value(name = "vision_language")]
    VisionLanguage,
    Analysis,
    Recipe,
}

impl std::fmt::Display for AssetType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Checkpoint => write!(f, "checkpoint"),
            Self::DiffusionModel => write!(f, "diffusion_model"),
            Self::Lora => write!(f, "lora"),
            Self::Vae => write!(f, "vae"),
            Self::TextEncoder => write!(f, "text_encoder"),
            Self::Controlnet => write!(f, "controlnet"),
            Self::Upscaler => write!(f, "upscaler"),
            Self::Embedding => write!(f, "embedding"),
            Self::Ipadapter => write!(f, "ipadapter"),
            Self::Segmentation => write!(f, "segmentation"),
            Self::VisionLanguage => write!(f, "vision_language"),
            Self::Analysis => write!(f, "analysis"),
            Self::Recipe => write!(f, "recipe"),
        }
    }
}

impl std::str::FromStr for AssetType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "checkpoint" => Ok(Self::Checkpoint),
            "diffusion_model" => Ok(Self::DiffusionModel),
            "lora" => Ok(Self::Lora),
            "vae" => Ok(Self::Vae),
            "text_encoder" => Ok(Self::TextEncoder),
            "controlnet" => Ok(Self::Controlnet),
            "upscaler" => Ok(Self::Upscaler),
            "embedding" => Ok(Self::Embedding),
            "ipadapter" => Ok(Self::Ipadapter),
            "segmentation" => Ok(Self::Segmentation),
            "vision_language" => Ok(Self::VisionLanguage),
            "analysis" => Ok(Self::Analysis),
            "recipe" => Ok(Self::Recipe),
            _ => anyhow::bail!("Unknown asset type: {}", s),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    pub id: String,
    pub file: String,
    pub url: String,
    pub sha256: String,
    pub size: u64,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub precision: Option<String>,
    #[serde(default)]
    pub vram_required: Option<u64>,
    #[serde(default)]
    pub vram_recommended: Option<u64>,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub url: String,
    pub sha256: String,
    pub size: u64,
    #[serde(default)]
    pub format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub id: String,
    #[serde(rename = "type")]
    pub dep_type: AssetType,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub optional_variant: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthRequirement {
    pub provider: String,
    #[serde(default)]
    pub terms_url: Option<String>,
    #[serde(default = "default_false")]
    pub gated: bool,
}

fn default_false() -> bool {
    false
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefaults {
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub cfg: Option<f32>,
    #[serde(default)]
    pub sampler: Option<String>,
    #[serde(default)]
    pub scheduler: Option<String>,
    #[serde(default)]
    pub resolution: Option<String>,
    #[serde(default)]
    pub clip_skip: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudTrainingInfo {
    #[serde(default)]
    pub gpu: Option<String>,
    #[serde(default)]
    pub estimated_minutes: Option<u32>,
    #[serde(default)]
    pub supported_ranks: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudInferenceInfo {
    #[serde(default)]
    pub gpu: Option<String>,
    #[serde(default)]
    pub supports_lora: bool,
    #[serde(default)]
    pub cold_start_seconds: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDetails {
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub rank: Option<u32>,
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default)]
    pub optimizer: Option<String>,
    #[serde(default)]
    pub dataset_size: Option<u32>,
    #[serde(default)]
    pub training_time_minutes: Option<u32>,
    #[serde(default)]
    pub cloud_trained: bool,
    #[serde(default)]
    pub recipe_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleImage {
    pub url: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub cfg: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeTraining {
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default)]
    pub rank: Option<u32>,
    #[serde(default)]
    pub optimizer: Option<String>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub resolution: Option<String>,
    #[serde(default)]
    pub dataset_min_images: Option<u32>,
    #[serde(default)]
    pub dataset_max_images: Option<u32>,
    #[serde(default)]
    pub caption_strategy: Option<String>,
    #[serde(default)]
    pub trigger_word_template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeConfig {
    pub base_model: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub training: Option<RecipeTraining>,
    #[serde(default)]
    pub generation: Option<ModelDefaults>,
    #[serde(default)]
    pub recommended_lora_weight: Option<f32>,
    #[serde(default)]
    pub tips: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_checkpoint_manifest() {
        let yaml = r#"
id: flux-dev
name: "FLUX.1 Dev"
type: checkpoint
architecture: flux
author: black-forest-labs
license: flux-1-dev-non-commercial
description: "High-quality text-to-image model."

variants:
  - id: fp16
    file: flux1-dev.safetensors
    url: https://example.com/flux1-dev.safetensors
    sha256: "abcdef1234567890"
    size: 23800000000
    format: safetensors
    precision: fp16
    vram_required: 24576

requires:
  - id: flux-vae
    type: vae
    reason: "Required VAE"

auth:
  provider: huggingface
  gated: true

tags: [flux, text-to-image]
"#;
        let manifest: Manifest = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(manifest.id, "flux-dev");
        assert_eq!(manifest.asset_type, AssetType::Checkpoint);
        assert_eq!(manifest.variants.len(), 1);
        assert_eq!(manifest.requires.len(), 1);
        assert!(manifest.auth.unwrap().gated);
    }

    #[test]
    fn test_parse_lora_manifest() {
        let yaml = r#"
id: realistic-skin-v3
name: "Realistic Skin Texture v3"
type: lora
author: test-user

base_models: [flux-dev, flux-schnell]

file:
  url: https://example.com/model.safetensors
  sha256: "abcdef"
  size: 186000000
  format: safetensors

trigger_words: ["realistic skin texture"]
recommended_weight: 0.7
tags: [portrait, skin]
"#;
        let manifest: Manifest = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(manifest.id, "realistic-skin-v3");
        assert_eq!(manifest.asset_type, AssetType::Lora);
        assert_eq!(manifest.base_models, vec!["flux-dev", "flux-schnell"]);
        assert!(manifest.file.is_some());
    }
}
