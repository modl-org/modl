use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_yaml::Value;

use crate::core::job::TrainJobSpec;

pub const RUN_MANIFEST_FILE: &str = "modl-run.yaml";
const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub schema_version: u32,
    pub generated_at: String,
    pub source: String,
    pub run_name: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    pub run_dir: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    #[serde(default)]
    pub model: RunModelMetadata,
    #[serde(default)]
    pub dataset: RunDatasetMetadata,
    #[serde(default)]
    pub training: RunTrainingMetadata,
    #[serde(default)]
    pub sample_prompts: Vec<String>,
    #[serde(default)]
    pub checkpoints: Vec<RunCheckpointMetadata>,
    #[serde(default)]
    pub sample_groups: Vec<RunSampleGroupMetadata>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunModelMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name_or_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model_path: Option<String>,
    #[serde(default)]
    pub trigger_words: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_type: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunDatasetMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caption_coverage: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hub_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_caption: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunTrainingMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_every: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_every: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learning_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub resolution: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_from: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunCheckpointMetadata {
    pub file_name: String,
    pub relative_path: String,
    pub size_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<u32>,
    pub is_final: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSampleGroupMetadata {
    pub step: u32,
    #[serde(default)]
    pub images: Vec<RunSampleImageMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSampleImageMetadata {
    pub file_name: String,
    pub relative_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

pub fn run_inner_dir_for_name(run_name: &str) -> PathBuf {
    let root = crate::core::paths::modl_root();
    let primary = root.join("training_output").join(run_name).join(run_name);
    if primary.exists() {
        return primary;
    }
    let legacy = root
        .join(".modl")
        .join("training_output")
        .join(run_name)
        .join(run_name);
    if legacy.exists() {
        return legacy;
    }
    primary
}

pub fn manifest_path(run_inner_dir: &Path) -> PathBuf {
    run_inner_dir.join(RUN_MANIFEST_FILE)
}

pub fn load_manifest(run_inner_dir: &Path) -> Result<Option<RunManifest>> {
    let path = manifest_path(run_inner_dir);
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let parsed: RunManifest =
        serde_yaml::from_str(&content).with_context(|| format!("Invalid {}", path.display()))?;
    Ok(Some(parsed))
}

pub fn refresh_manifest_for_spec(
    spec: &TrainJobSpec,
    job_id: Option<&str>,
    status: &str,
) -> Result<RunManifest> {
    let run_inner = PathBuf::from(&spec.output.destination_dir).join(&spec.output.lora_name);
    refresh_manifest(&run_inner, Some(spec), job_id, status)
}

pub fn refresh_manifest_for_run_name(run_name: &str, status: &str) -> Result<RunManifest> {
    let run_inner = run_inner_dir_for_name(run_name);
    refresh_manifest(&run_inner, None, None, status)
}

fn refresh_manifest(
    run_inner_dir: &Path,
    spec: Option<&TrainJobSpec>,
    job_id: Option<&str>,
    status: &str,
) -> Result<RunManifest> {
    std::fs::create_dir_all(run_inner_dir)
        .with_context(|| format!("Failed to create run dir {}", run_inner_dir.display()))?;

    let run_name = spec
        .map(|s| s.output.lora_name.clone())
        .or_else(|| run_name_from_inner_dir(run_inner_dir))
        .context("Could not infer run name for manifest")?;

    let mut manifest = base_manifest_from_spec(spec, &run_name, status, job_id);

    let config_path = run_inner_dir.join("config.yaml");
    if config_path.exists() {
        manifest.config_path = Some("config.yaml".to_string());
        apply_config_to_manifest(&mut manifest, &config_path)?;
    }

    manifest.checkpoints = scan_checkpoints(run_inner_dir, &run_name)?;
    manifest.sample_groups = scan_samples(run_inner_dir, &manifest.sample_prompts)?;
    manifest.generated_at = Utc::now().to_rfc3339();

    let serialized =
        serde_yaml::to_string(&manifest).context("Failed to serialize run manifest")?;
    let out_path = manifest_path(run_inner_dir);
    std::fs::write(&out_path, serialized)
        .with_context(|| format!("Failed to write {}", out_path.display()))?;

    Ok(manifest)
}

fn base_manifest_from_spec(
    spec: Option<&TrainJobSpec>,
    run_name: &str,
    status: &str,
    job_id: Option<&str>,
) -> RunManifest {
    let mut model = RunModelMetadata::default();
    let mut dataset = RunDatasetMetadata::default();
    let mut training = RunTrainingMetadata::default();

    if let Some(spec) = spec {
        model.base_model_id = Some(spec.model.base_model_id.clone());
        model.base_model_path = spec.model.base_model_path.clone();
        model.trigger_words = vec![spec.params.trigger_word.clone()];
        model.lora_type = Some(spec.params.lora_type.to_string());

        dataset.name = Some(spec.dataset.name.clone());
        dataset.local_path = Some(spec.dataset.path.clone());
        dataset.image_count = Some(spec.dataset.image_count);
        dataset.caption_coverage = Some(spec.dataset.caption_coverage);

        training.preset = Some(spec.params.preset.to_string());
        training.steps = Some(spec.params.steps);
        training.optimizer = Some(spec.params.optimizer.to_string());
        training.learning_rate = Some(spec.params.learning_rate);
        training.batch_size = Some(spec.params.batch_size);
        training.resolution = vec![spec.params.resolution, spec.params.resolution];
        training.seed = spec.params.seed;
        training.resume_from = spec.params.resume_from.clone();
    }

    RunManifest {
        schema_version: SCHEMA_VERSION,
        generated_at: Utc::now().to_rfc3339(),
        source: "modl-cli".to_string(),
        run_name: run_name.to_string(),
        status: status.to_string(),
        job_id: job_id.map(str::to_string),
        run_dir: format!("training_output/{run_name}/{run_name}"),
        config_path: None,
        model,
        dataset,
        training,
        sample_prompts: Vec::new(),
        checkpoints: Vec::new(),
        sample_groups: Vec::new(),
    }
}

fn run_name_from_inner_dir(run_inner_dir: &Path) -> Option<String> {
    let run_inner = run_inner_dir.file_name()?.to_str()?.to_string();
    let run_outer = run_inner_dir.parent()?.file_name()?.to_str()?.to_string();
    if run_inner == run_outer {
        Some(run_inner)
    } else {
        Some(run_outer)
    }
}

fn apply_config_to_manifest(manifest: &mut RunManifest, config_path: &Path) -> Result<()> {
    let content = std::fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    let root: Value = serde_yaml::from_str(&content)
        .with_context(|| format!("Failed to parse {}", config_path.display()))?;

    let Some(process) = root
        .get("config")
        .and_then(|c| c.get("process"))
        .and_then(Value::as_sequence)
        .and_then(|seq| seq.first())
    else {
        return Ok(());
    };

    if manifest.model.model_name_or_path.is_none()
        && let Some(v) = process
            .get("model")
            .and_then(|m| m.get("name_or_path"))
            .and_then(Value::as_str)
    {
        manifest.model.model_name_or_path = Some(v.to_string());
    }

    if manifest.model.trigger_words.is_empty()
        && let Some(v) = process.get("trigger_word").and_then(Value::as_str)
        && !v.trim().is_empty()
    {
        manifest.model.trigger_words = vec![v.to_string()];
    }

    if manifest.dataset.local_path.is_none()
        && let Some(v) = process
            .get("datasets")
            .and_then(Value::as_sequence)
            .and_then(|seq| seq.first())
            .and_then(|ds| ds.get("folder_path"))
            .and_then(Value::as_str)
    {
        manifest.dataset.local_path = Some(v.to_string());
    }

    if manifest.dataset.default_caption.is_none()
        && let Some(v) = process
            .get("datasets")
            .and_then(Value::as_sequence)
            .and_then(|seq| seq.first())
            .and_then(|ds| ds.get("default_caption"))
            .and_then(Value::as_str)
        && !v.trim().is_empty()
    {
        manifest.dataset.default_caption = Some(v.to_string());
    }

    if manifest.dataset.name.is_none()
        && let Some(ref p) = manifest.dataset.local_path
        && let Some(name) = Path::new(p).file_name().and_then(|n| n.to_str())
        && !name.is_empty()
    {
        manifest.dataset.name = Some(name.to_string());
    }

    if let Some(v) = process
        .get("train")
        .and_then(|t| t.get("steps"))
        .and_then(Value::as_u64)
    {
        manifest.training.steps = Some(v as u32);
    }

    if let Some(v) = process
        .get("save")
        .and_then(|s| s.get("save_every"))
        .and_then(Value::as_u64)
    {
        manifest.training.save_every = Some(v as u32);
    }

    if let Some(v) = process
        .get("sample")
        .and_then(|s| s.get("sample_every"))
        .and_then(Value::as_u64)
    {
        manifest.training.sample_every = Some(v as u32);
    }

    if let Some(v) = process
        .get("train")
        .and_then(|t| t.get("optimizer"))
        .and_then(Value::as_str)
    {
        manifest.training.optimizer = Some(v.to_string());
    }

    if let Some(v) = process
        .get("train")
        .and_then(|t| t.get("lr"))
        .and_then(Value::as_f64)
    {
        manifest.training.learning_rate = Some(v);
    }

    if let Some(v) = process
        .get("train")
        .and_then(|t| t.get("batch_size"))
        .and_then(Value::as_u64)
    {
        manifest.training.batch_size = Some(v as u32);
    }

    if manifest.training.resolution.is_empty()
        && let Some(values) = process
            .get("datasets")
            .and_then(Value::as_sequence)
            .and_then(|seq| seq.first())
            .and_then(|ds| ds.get("resolution"))
            .and_then(yaml_seq_u32)
    {
        manifest.training.resolution = values;
    }

    if let Some(v) = process
        .get("train")
        .and_then(|t| t.get("seed"))
        .and_then(Value::as_u64)
    {
        manifest.training.seed = Some(v);
    }

    if let Some(v) = process
        .get("network")
        .and_then(|n| n.get("pretrained_lora_path"))
        .and_then(Value::as_str)
    {
        manifest.training.resume_from = Some(v.to_string());
    }

    if let Some(v) = process
        .get("train")
        .and_then(|t| t.get("content_or_style"))
        .and_then(Value::as_str)
    {
        manifest.model.lora_type = Some(v.to_string());
    }

    if let Some(prompts) = process
        .get("sample")
        .and_then(|s| s.get("prompts"))
        .and_then(Value::as_sequence)
    {
        manifest.sample_prompts = prompts
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
    }

    Ok(())
}

fn yaml_seq_u32(value: &Value) -> Option<Vec<u32>> {
    let seq = value.as_sequence()?;
    let mut out = Vec::new();
    for v in seq {
        let Some(num) = v.as_u64() else {
            continue;
        };
        out.push(num as u32);
    }
    if out.is_empty() { None } else { Some(out) }
}

fn scan_checkpoints(run_inner_dir: &Path, run_name: &str) -> Result<Vec<RunCheckpointMetadata>> {
    let mut checkpoints = Vec::new();
    let final_name = format!("{run_name}.safetensors");

    for entry in std::fs::read_dir(run_inner_dir)
        .with_context(|| format!("Failed to read {}", run_inner_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if !path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
        {
            continue;
        }

        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model.safetensors")
            .to_string();
        let step = step_from_checkpoint_name(&file_name);
        let is_final = file_name == final_name;
        let size_bytes = std::fs::metadata(&path)
            .with_context(|| format!("Failed to stat {}", path.display()))?
            .len();

        checkpoints.push(RunCheckpointMetadata {
            file_name: file_name.clone(),
            relative_path: file_name,
            size_bytes,
            step,
            is_final,
            modified_at: modified_timestamp(&path),
        });
    }

    checkpoints.sort_by(|a, b| match (a.step, b.step) {
        (Some(sa), Some(sb)) => sa.cmp(&sb),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.file_name.cmp(&b.file_name),
    });

    Ok(checkpoints)
}

fn scan_samples(run_inner_dir: &Path, prompts: &[String]) -> Result<Vec<RunSampleGroupMetadata>> {
    let samples_dir = run_inner_dir.join("samples");
    if !samples_dir.exists() {
        return Ok(Vec::new());
    }

    let mut groups: BTreeMap<u32, Vec<RunSampleImageMetadata>> = BTreeMap::new();

    for entry in std::fs::read_dir(&samples_dir)
        .with_context(|| format!("Failed to read {}", samples_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if !path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("png"))
        {
            continue;
        }

        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("sample.png")
            .to_string();

        let Some((step, prompt_idx)) = parse_sample_step_and_index(&file_name) else {
            continue;
        };

        let prompt = prompts.get(prompt_idx).cloned();
        groups
            .entry(step)
            .or_default()
            .push(RunSampleImageMetadata {
                file_name: file_name.clone(),
                relative_path: format!("samples/{file_name}"),
                prompt_index: Some(prompt_idx),
                prompt,
            });
    }

    let mut out = Vec::new();
    for (step, mut images) in groups {
        images.sort_by(|a, b| a.file_name.cmp(&b.file_name));
        out.push(RunSampleGroupMetadata { step, images });
    }
    Ok(out)
}

fn parse_sample_step_and_index(file_name: &str) -> Option<(u32, usize)> {
    let stem = Path::new(file_name).file_stem()?.to_str()?;
    let (_, rhs) = stem.split_once("__")?;
    let (step_str, idx_str) = rhs.rsplit_once('_')?;
    if !step_str.chars().all(|c| c.is_ascii_digit()) || !idx_str.chars().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    Some((
        step_str.parse::<u32>().ok()?,
        idx_str.parse::<usize>().ok()?,
    ))
}

fn step_from_checkpoint_name(file_name: &str) -> Option<u32> {
    let stem = Path::new(file_name).file_stem()?.to_str()?;
    let (_, suffix) = stem.rsplit_once('_')?;
    if !suffix.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    suffix.parse::<u32>().ok()
}

fn modified_timestamp(path: &Path) -> Option<String> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    let dt: DateTime<Utc> = modified.into();
    Some(dt.to_rfc3339())
}
