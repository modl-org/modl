use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Top-level job spec — the serialization boundary between CLI and executor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainJobSpec {
    pub dataset: DatasetRef,
    pub model: ModelRef,
    pub output: OutputRef,
    pub params: TrainingParams,
    pub runtime: RuntimeRef,
    pub target: ExecutionTarget,
    #[serde(default)]
    pub labels: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateJobSpec {
    pub prompt: String,
    pub model: ModelRef,
    #[serde(default)]
    pub lora: Option<LoraRef>,
    pub output: GenerateOutputRef,
    pub params: GenerateParams,
    pub runtime: RuntimeRef,
    pub target: ExecutionTarget,
    #[serde(default)]
    pub labels: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionJobSpec {
    pub dataset_path: String,
    #[serde(default = "default_caption_model")]
    pub model: String,
    #[serde(default)]
    pub overwrite: bool,
    /// Local path to a pre-downloaded VL model (from registry).
    /// When set, the Python adapter loads from this path instead of
    /// downloading from HuggingFace Hub.
    #[serde(default)]
    pub model_path: Option<String>,
    /// Style LoRA mode: describe content only, omit art style/medium/technique.
    /// When true, Florence-2 uses <CAPTION> (shorter, factual) instead of
    /// <DETAILED_CAPTION>, and BLIP-2 gets a prompt instructing content-only
    /// descriptions.
    #[serde(default)]
    pub style: bool,
}

fn default_caption_model() -> String {
    "florence-2".to_string()
}

fn default_tag_model() -> String {
    "florence-2".to_string()
}

fn default_resize_resolution() -> u32 {
    1024
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagJobSpec {
    pub dataset_path: String,
    #[serde(default = "default_tag_model")]
    pub model: String,
    #[serde(default)]
    pub overwrite: bool,
    /// Local path to a pre-downloaded VL model (from registry).
    #[serde(default)]
    pub model_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeJobSpec {
    pub dataset_path: String,
    #[serde(default = "default_resize_resolution")]
    pub resolution: u32,
    /// "cover" (crop to fill, default), "contain" (fit inside, pad), or "squish" (stretch)
    #[serde(default = "default_resize_method")]
    pub method: String,
}

fn default_resize_method() -> String {
    "contain".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceCropJobSpec {
    pub dataset_path: String,
    #[serde(default = "default_resize_resolution")]
    pub resolution: u32,
    #[serde(default = "default_face_crop_padding")]
    pub padding: f32,
    #[serde(default)]
    pub trigger_word: String,
    #[serde(default)]
    pub class_word: String,
}

fn default_face_crop_padding() -> f32 {
    1.8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreJobSpec {
    pub image_paths: Vec<String>,
    #[serde(default = "default_score_model")]
    pub model: String,
    #[serde(default)]
    pub clip_model_path: Option<String>,
    #[serde(default)]
    pub predictor_path: Option<String>,
}

fn default_score_model() -> String {
    "laion-aesthetic-v2".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectJobSpec {
    pub image_paths: Vec<String>,
    #[serde(default = "default_detect_type")]
    pub detect_type: String,
    #[serde(default = "default_detect_model")]
    pub model: String,
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default)]
    pub return_embeddings: bool,
}

fn default_detect_type() -> String {
    "face".to_string()
}

fn default_detect_model() -> String {
    "insightface-buffalo-l".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundJobSpec {
    pub image_paths: Vec<String>,
    pub query: String,
    #[serde(default = "default_ground_model")]
    pub model: String,
    #[serde(default)]
    pub threshold: Option<f64>,
}

fn default_ground_model() -> String {
    "qwen25-vl-3b".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeJobSpec {
    pub image_paths: Vec<String>,
    #[serde(default = "default_describe_model")]
    pub model: String,
    #[serde(default = "default_describe_detail")]
    pub detail: String,
}

fn default_describe_model() -> String {
    "qwen25-vl-3b".to_string()
}
fn default_describe_detail() -> String {
    "detailed".to_string()
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VlTagJobSpec {
    pub image_paths: Vec<String>,
    #[serde(default = "default_vl_tag_model")]
    pub model: String,
    #[serde(default)]
    pub max_tags: Option<usize>,
}

#[allow(dead_code)]
fn default_vl_tag_model() -> String {
    "qwen25-vl-3b".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareJobSpec {
    pub image_paths: Vec<String>,
    #[serde(default)]
    pub reference_path: Option<String>,
    #[serde(default = "default_compare_model")]
    pub model: String,
    #[serde(default)]
    pub clip_model_path: Option<String>,
}

fn default_compare_model() -> String {
    "clip-vit-large-patch14".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentJobSpec {
    pub image_path: String,
    pub output_mask_path: String,
    #[serde(default = "default_segment_method")]
    pub method: String,
    #[serde(default)]
    pub bbox: Option<[f32; 4]>,
    #[serde(default)]
    pub point: Option<[f32; 2]>,
    #[serde(default = "default_segment_model")]
    pub model: String,
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default = "default_expand_px")]
    pub expand_px: u32,
}

fn default_segment_method() -> String {
    "bbox".to_string()
}

fn default_segment_model() -> String {
    "sam-vit-base".to_string()
}

fn default_expand_px() -> u32 {
    10
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceRestoreJobSpec {
    pub image_paths: Vec<String>,
    pub output_dir: String,
    #[serde(default = "default_face_restore_model")]
    pub model: String,
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default = "default_fidelity")]
    pub fidelity: f32,
}

#[allow(dead_code)]
fn default_face_restore_model() -> String {
    "codeformer".to_string()
}

#[allow(dead_code)]
fn default_fidelity() -> f32 {
    0.7
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpscaleJobSpec {
    pub image_paths: Vec<String>,
    pub output_dir: String,
    #[serde(default = "default_upscale_scale")]
    pub scale: u32,
    #[serde(default)]
    pub model_path: Option<String>,
}

fn default_upscale_scale() -> u32 {
    4
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessJobSpec {
    pub image_paths: Vec<String>,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_dir: Option<String>,
    /// Canny low threshold
    #[serde(default = "default_canny_low")]
    pub canny_low: u32,
    /// Canny high threshold
    #[serde(default = "default_canny_high")]
    pub canny_high: u32,
    /// Depth model variant: small, base, large
    #[serde(default = "default_depth_model")]
    pub depth_model: String,
    /// Scribble binary threshold
    #[serde(default = "default_scribble_threshold")]
    pub scribble_threshold: u32,
    /// Include hands in pose detection
    #[serde(default = "default_true")]
    pub include_hands: bool,
    /// Include face landmarks in pose detection
    #[serde(default = "default_true")]
    pub include_face: bool,
}

impl Default for PreprocessJobSpec {
    fn default() -> Self {
        Self {
            image_paths: Vec::new(),
            method: "canny".to_string(),
            output_dir: None,
            canny_low: default_canny_low(),
            canny_high: default_canny_high(),
            depth_model: default_depth_model(),
            scribble_threshold: default_scribble_threshold(),
            include_hands: true,
            include_face: true,
        }
    }
}

fn default_canny_low() -> u32 {
    100
}
fn default_canny_high() -> u32 {
    200
}
fn default_depth_model() -> String {
    "small".to_string()
}
fn default_scribble_threshold() -> u32 {
    128
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoveBgJobSpec {
    pub image_paths: Vec<String>,
    pub output_dir: String,
    #[serde(default)]
    pub model_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditJobSpec {
    pub prompt: String,
    pub model: ModelRef,
    #[serde(default)]
    pub lora: Option<LoraRef>,
    pub output: GenerateOutputRef,
    pub params: EditParams,
    pub runtime: RuntimeRef,
    pub target: ExecutionTarget,
    #[serde(default)]
    pub labels: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditParams {
    /// Source image paths (1 or more, already resolved to local files)
    pub image_paths: Vec<String>,
    pub steps: u32,
    pub guidance: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    pub count: u32,
    /// Optional output dimensions (for outpainting — larger than source image)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraRef {
    pub name: String,
    pub path: String,
    #[serde(default = "default_lora_weight")]
    pub weight: f32,
}

fn default_lora_weight() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOutputRef {
    pub output_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateParams {
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    pub count: u32,
    /// Source image path for img2img / inpainting
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_image: Option<String>,
    /// Mask image path (white = regenerate region) for inpainting
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,
    /// Denoising strength for img2img (0.0 = no change, 1.0 = full denoise)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
    /// ControlNet inputs
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub controlnet: Vec<ControlNetInput>,
    /// Style reference inputs (IP-Adapter or native multi-ref)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub style_ref: Vec<StyleRefInput>,
    /// Inpainting method: "standard" (diffusers/Flux Fill) or "lanpaint" (training-free)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inpaint_method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleRefInput {
    /// Path to the reference image
    pub image: String,
    /// How strongly the reference influences output (0.0-1.0)
    #[serde(default = "default_style_strength")]
    pub strength: f32,
    /// Style type: style, face, content (SDXL IP-Adapter only)
    #[serde(default = "default_style_type")]
    pub style_type: String,
}

fn default_style_strength() -> f32 {
    0.6
}
fn default_style_type() -> String {
    "style".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlNetInput {
    /// Path to the control image
    pub image: String,
    /// Control type: canny, depth, pose, softedge, scribble, hed, mlsd, gray, normal
    pub control_type: String,
    /// Conditioning strength (0.0-1.5, default 0.75)
    #[serde(default = "default_cn_strength")]
    pub strength: f32,
    /// Stop applying control at this fraction of total steps (0.0-1.0)
    #[serde(default = "default_cn_end")]
    pub control_end: f32,
}

fn default_cn_strength() -> f32 {
    0.75
}
fn default_cn_end() -> f32 {
    0.8
}

// ---------------------------------------------------------------------------
// Ref types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRef {
    pub name: String,
    pub path: String,
    pub image_count: u32,
    pub caption_coverage: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef {
    pub base_model_id: String,
    #[serde(default)]
    pub base_model_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputRef {
    pub lora_name: String,
    pub destination_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    pub preset: Preset,
    pub lora_type: LoraType,
    pub trigger_word: String,
    pub steps: u32,
    pub rank: u32,
    pub learning_rate: f64,
    pub optimizer: Optimizer,
    pub resolution: u32,
    #[serde(default)]
    pub seed: Option<u64>,
    pub quantize: bool,
    /// Batch size per training step (higher = faster but more VRAM)
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    /// Dataset repetitions per epoch (higher = more exposure per image)
    #[serde(default = "default_num_repeats")]
    pub num_repeats: u32,
    /// Probability of dropping captions (higher = learn visual style over text)
    #[serde(default = "default_caption_dropout")]
    pub caption_dropout_rate: f64,
    /// Resume training from a checkpoint .safetensors file
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_from: Option<String>,
    /// Class word for character/object LoRAs (e.g. "man", "woman", "dog").
    /// Used in sample prompts alongside trigger word for better convergence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_word: Option<String>,
}

fn default_batch_size() -> u32 {
    0 // 0 = let adapter choose per lora_type
}
fn default_num_repeats() -> u32 {
    0 // 0 = let adapter choose per lora_type
}
fn default_caption_dropout() -> f64 {
    -1.0 // negative = let adapter choose per lora_type
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum LoraType {
    Style,
    Character,
    Object,
}

impl std::fmt::Display for LoraType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Style => write!(f, "style"),
            Self::Character => write!(f, "character"),
            Self::Object => write!(f, "object"),
        }
    }
}

impl std::str::FromStr for LoraType {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "style" => Ok(Self::Style),
            "character" | "char" => Ok(Self::Character),
            "object" | "obj" => Ok(Self::Object),
            _ => anyhow::bail!("Unknown lora type: {s}. Expected style, character, or object."),
        }
    }
}

/// Validated optimizer choices supported by ai-toolkit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Optimizer {
    /// AdamW 8-bit (bitsandbytes) — best default, low VRAM
    #[value(alias = "adamw8bit")]
    Adamw8bit,
    /// Prodigy — adaptive LR, no manual LR tuning needed
    Prodigy,
    /// AdamW (full precision) — uses more VRAM
    Adamw,
    /// Adafactor — memory-efficient, good for large models
    Adafactor,
    /// SGD with momentum — simple, sometimes useful for fine-tuning
    Sgd,
}

impl std::fmt::Display for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Adamw8bit => write!(f, "adamw8bit"),
            Self::Prodigy => write!(f, "prodigy"),
            Self::Adamw => write!(f, "adamw"),
            Self::Adafactor => write!(f, "adafactor"),
            Self::Sgd => write!(f, "sgd"),
        }
    }
}

impl std::str::FromStr for Optimizer {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "adamw8bit" | "adamw_8bit" => Ok(Self::Adamw8bit),
            "prodigy" => Ok(Self::Prodigy),
            "adamw" => Ok(Self::Adamw),
            "adafactor" => Ok(Self::Adafactor),
            "sgd" => Ok(Self::Sgd),
            _ => anyhow::bail!(
                "Unknown optimizer: {s}. Options: adamw8bit, prodigy, adamw, adafactor, sgd"
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeRef {
    pub profile: String,
    #[serde(default)]
    pub python_version: Option<String>,
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionTarget {
    Local,
    Cloud,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Preset {
    Quick,
    Standard,
    Advanced,
}

impl std::fmt::Display for Preset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Quick => write!(f, "quick"),
            Self::Standard => write!(f, "standard"),
            Self::Advanced => write!(f, "advanced"),
        }
    }
}

impl std::str::FromStr for Preset {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "quick" => Ok(Self::Quick),
            "standard" => Ok(Self::Standard),
            "advanced" => Ok(Self::Advanced),
            _ => anyhow::bail!("Unknown preset: {s}. Expected quick, standard, or advanced."),
        }
    }
}

// ---------------------------------------------------------------------------
// Job events — the protocol envelope
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobEvent {
    pub schema_version: String,
    pub job_id: String,
    pub sequence: u64,
    pub timestamp: String,
    pub source: String,
    pub event: EventPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventPayload {
    JobAccepted {
        #[serde(default)]
        worker_pid: Option<u32>,
    },
    JobStarted {
        #[serde(default)]
        config: Option<String>,
        #[serde(default)]
        command: Option<Vec<String>>,
    },
    Progress {
        stage: String,
        step: u32,
        total_steps: u32,
        #[serde(default)]
        loss: Option<f64>,
        #[serde(default)]
        eta_seconds: Option<f64>,
    },
    Artifact {
        path: String,
        #[serde(default)]
        sha256: Option<String>,
        #[serde(default)]
        size_bytes: Option<u64>,
    },
    Log {
        level: String,
        message: String,
    },
    Warning {
        code: String,
        message: String,
    },
    Completed {
        #[serde(default)]
        message: Option<String>,
    },
    Error {
        code: String,
        message: String,
        #[serde(default)]
        recoverable: bool,
        #[serde(default)]
        details: Option<serde_json::Value>,
    },
    Cancelled,
    Heartbeat,
    /// Structured result data from analysis commands (score, detect, compare).
    Result {
        result_type: String,
        data: serde_json::Value,
    },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_job_spec_roundtrip() {
        let spec = TrainJobSpec {
            dataset: DatasetRef {
                name: "headshots".into(),
                path: "/tmp/headshots".into(),
                image_count: 15,
                caption_coverage: 0.8,
            },
            model: ModelRef {
                base_model_id: "flux-schnell".into(),
                base_model_path: None,
            },
            output: OutputRef {
                lora_name: "headshots-v1".into(),
                destination_dir: "/tmp/output".into(),
            },
            params: TrainingParams {
                preset: Preset::Standard,
                lora_type: LoraType::Character,
                trigger_word: "OHWX".into(),
                steps: 3000,
                rank: 16,
                learning_rate: 5e-5,
                optimizer: Optimizer::Adamw8bit,
                resolution: 1024,
                seed: Some(42),
                quantize: true,
                batch_size: 0,
                num_repeats: 0,
                caption_dropout_rate: -1.0,
                resume_from: None,
                class_word: None,
            },
            runtime: RuntimeRef {
                profile: "trainer-cu124".into(),
                python_version: Some("3.11.11".into()),
            },
            target: ExecutionTarget::Local,
            labels: std::collections::HashMap::new(),
        };

        let yaml = serde_yaml::to_string(&spec).unwrap();
        let back: TrainJobSpec = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(back.params.steps, 3000);
        assert_eq!(back.params.preset, Preset::Standard);
        assert_eq!(back.target, ExecutionTarget::Local);
    }

    #[test]
    fn test_preset_from_str() {
        assert_eq!("quick".parse::<Preset>().unwrap(), Preset::Quick);
        assert_eq!("Standard".parse::<Preset>().unwrap(), Preset::Standard);
        assert!("unknown".parse::<Preset>().is_err());
    }

    #[test]
    fn test_caption_job_spec_roundtrip() {
        let spec = CaptionJobSpec {
            dataset_path: "/tmp/my-dataset".into(),
            model: "florence-2".into(),
            overwrite: false,
            model_path: None,
            style: false,
        };
        let yaml = serde_yaml::to_string(&spec).unwrap();
        let back: CaptionJobSpec = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(back.dataset_path, "/tmp/my-dataset");
        assert_eq!(back.model, "florence-2");
        assert!(!back.overwrite);
        assert!(!back.style);
    }

    #[test]
    fn test_caption_job_spec_style_mode() {
        let spec = CaptionJobSpec {
            dataset_path: "/tmp/style-dataset".into(),
            model: "blip".into(),
            overwrite: true,
            model_path: None,
            style: true,
        };
        let yaml = serde_yaml::to_string(&spec).unwrap();
        let back: CaptionJobSpec = serde_yaml::from_str(&yaml).unwrap();
        assert!(back.style);
        assert_eq!(back.model, "blip");
    }

    #[test]
    fn test_caption_job_spec_defaults() {
        let yaml = "dataset_path: /data/photos\n";
        let spec: CaptionJobSpec = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(spec.model, "florence-2");
        assert!(!spec.overwrite);
    }

    #[test]
    fn test_event_payload_serde() {
        let event = JobEvent {
            schema_version: "v1".into(),
            job_id: "job-123".into(),
            sequence: 1,
            timestamp: "2026-01-01T00:00:00Z".into(),
            source: "modl_worker".into(),
            event: EventPayload::Progress {
                stage: "train".into(),
                step: 100,
                total_steps: 2000,
                loss: Some(0.0345),
                eta_seconds: Some(120.5),
            },
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: JobEvent = serde_json::from_str(&json).unwrap();
        if let EventPayload::Progress { step, loss, .. } = back.event {
            assert_eq!(step, 100);
            assert!((loss.unwrap() - 0.0345).abs() < 1e-6);
        } else {
            panic!("Expected Progress event");
        }
    }
}
