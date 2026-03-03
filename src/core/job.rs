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

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareJobSpec {
    pub source_dir: String,
    pub dataset_name: String,
    #[serde(default = "default_resize_resolution")]
    pub resolution: u32,
    #[serde(default = "default_tag_model")]
    pub tag_model: String,
    #[serde(default = "default_caption_model")]
    pub caption_model: String,
    #[serde(default)]
    pub skip_resize: bool,
    #[serde(default)]
    pub skip_tag: bool,
    #[serde(default)]
    pub skip_caption: bool,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
pub enum JobStatus {
    Queued,
    Accepted,
    Running,
    Completed,
    Error,
    Cancelled,
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Accepted => write!(f, "accepted"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Error => write!(f, "error"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::str::FromStr for JobStatus {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "queued" => Ok(Self::Queued),
            "accepted" => Ok(Self::Accepted),
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "error" => Ok(Self::Error),
            "cancelled" => Ok(Self::Cancelled),
            _ => anyhow::bail!("Unknown job status: {s}"),
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
    fn test_job_status_display_roundtrip() {
        for status in [JobStatus::Queued, JobStatus::Running, JobStatus::Completed] {
            let s = status.to_string();
            let back: JobStatus = s.parse().unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn test_caption_job_spec_roundtrip() {
        let spec = CaptionJobSpec {
            dataset_path: "/tmp/my-dataset".into(),
            model: "florence-2".into(),
            overwrite: false,
            model_path: None,
        };
        let yaml = serde_yaml::to_string(&spec).unwrap();
        let back: CaptionJobSpec = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(back.dataset_path, "/tmp/my-dataset");
        assert_eq!(back.model, "florence-2");
        assert!(!back.overwrite);
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
