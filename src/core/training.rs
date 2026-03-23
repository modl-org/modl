use anyhow::{Result, bail};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::env;
use std::path::PathBuf;

use super::db::Database;
use super::paths;
use super::training_status;

/// Resolve the path to the Python worker package root.
///
/// Search order:
///   1. `MODL_WORKER_PYTHON_ROOT` env var (explicit override)
///   2. `<binary_dir>/python` (next to the installed binary)
///   3. `<binary_dir>/../python` (symlink → repo: target/release/../python)
///   4. `CARGO_MANIFEST_DIR/python` (dev builds)
pub fn resolve_worker_python_root() -> Result<PathBuf> {
    if let Ok(custom) = env::var("MODL_WORKER_PYTHON_ROOT") {
        let path = PathBuf::from(custom);
        if path.exists() {
            return Ok(path);
        }
        bail!(
            "MODL_WORKER_PYTHON_ROOT points to missing path: {}",
            path.display()
        );
    }

    // Resolve relative to the actual binary (following symlinks)
    if let Ok(exe) = env::current_exe() {
        let exe = exe.canonicalize().unwrap_or(exe);
        if let Some(bin_dir) = exe.parent() {
            // <bin>/python (installed layout)
            let candidate = bin_dir.join("python");
            if candidate.exists() {
                return Ok(candidate);
            }
            // <bin>/../python (symlink into repo: target/release/../python)
            if let Some(parent) = bin_dir.parent() {
                let candidate = parent.join("python");
                if candidate.exists() {
                    return Ok(candidate);
                }
            }
        }
    }

    // Compile-time fallback (dev builds only)
    let default_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    if default_path.exists() {
        Ok(default_path)
    } else {
        bail!("Worker python package not found. Set MODL_WORKER_PYTHON_ROOT to a valid path.")
    }
}

// ---------------------------------------------------------------------------
// Training run management
// ---------------------------------------------------------------------------

/// List training run names (directories under training_output/).
pub fn list_training_runs() -> Result<Vec<String>> {
    let output_dir = paths::modl_root().join("training_output");
    let mut runs = Vec::new();

    if output_dir.exists() {
        for entry in std::fs::read_dir(&output_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                runs.push(entry.file_name().to_string_lossy().to_string());
            }
        }
    }
    runs.sort();
    Ok(runs)
}

/// Delete a training run: output directory, log file, LoRA symlink, and DB records.
///
/// Refuses to delete if training is currently running.
pub fn delete_training_run(name: &str) -> Result<()> {
    // Safety: refuse to delete if training is currently running
    let running = training_status::get_status(name)
        .map(|s| s.is_running)
        .unwrap_or(false);
    if running {
        bail!("Cannot delete '{name}': training is still running. Pause it first.");
    }

    let root = paths::modl_root();
    let run_dir = root.join("training_output").join(name);
    let log_file = root.join("training_output").join(format!("{name}.log"));

    if !run_dir.exists() && !log_file.exists() {
        bail!("Training run '{name}' not found");
    }

    // Remove training output directory (samples, checkpoints, config)
    if run_dir.exists() {
        std::fs::remove_dir_all(&run_dir)?;
    }
    // Remove log file
    if log_file.exists() {
        std::fs::remove_file(&log_file)?;
    }

    // Remove LoRA symlink from loras/ directory
    let lora_link = root.join("loras").join(format!("{name}.safetensors"));
    if lora_link.symlink_metadata().is_ok() {
        std::fs::remove_file(&lora_link)?;
    }

    // Clean up DB job records
    if let Ok(db) = Database::open() {
        let _ = db.delete_jobs_by_lora_name(name);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Training run scanning (moved from ui/routes/training.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct TrainingRun {
    pub name: String,
    pub config: Option<serde_json::Value>,
    pub samples: Vec<SampleGroup>,
    pub lora_path: Option<String>,
    pub lora_size: Option<u64>,
    pub lora_promoted: bool,
    pub lineage: Option<TrainingLineage>,
    pub total_steps: Option<u64>,
    pub sample_every: Option<u64>,
    pub checkpoints: Vec<CheckpointInfo>,
}

#[derive(Debug, Serialize)]
pub struct CheckpointInfo {
    pub step: u64,
    pub path: String,
    pub size_bytes: u64,
    pub promoted: bool,
}

#[derive(Debug, Serialize)]
pub struct TrainingLineage {
    pub dataset_name: Option<String>,
    pub dataset_image_count: Option<u32>,
    pub base_model: Option<String>,
    pub jobs: Vec<JobSummary>,
}

#[derive(Debug, Serialize)]
pub struct JobSummary {
    pub job_id: String,
    pub status: String,
    pub steps: Option<u64>,
    pub created_at: String,
    pub resumed_from: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SampleGroup {
    pub step: u64,
    pub images: Vec<String>,
}

/// Parse step number from sample filename like `1772410330707__000000000_0.jpg`
pub fn parse_step_from_filename(filename: &str) -> Option<u64> {
    let parts: Vec<&str> = filename.split("__").collect();
    if parts.len() == 2 {
        let rest = parts[1];
        let step_str = rest.split('_').next()?;
        step_str.parse::<u64>().ok()
    } else {
        None
    }
}

/// Given a list of sample image paths for a single step, infer the expected
/// number of prompts.
fn infer_prompt_count(images: &[String]) -> usize {
    let mut max_idx: usize = 0;
    let mut count = 0usize;
    for img in images {
        if let Some(fname) = img.rsplit('/').next()
            && let Some(stem) = fname
                .strip_suffix(".jpg")
                .or_else(|| fname.strip_suffix(".png"))
            && let Some(idx_str) = stem.rsplit('_').next()
            && let Ok(idx) = idx_str.parse::<usize>()
        {
            if idx >= max_idx {
                max_idx = idx;
            }
            count += 1;
        }
    }
    if count == 0 {
        images.len()
    } else {
        max_idx + 1
    }
}

/// Scan a training output directory for sample images, checkpoints, and lineage.
pub fn scan_training_run(name: &str) -> Result<TrainingRun> {
    let root = paths::modl_root();
    let run_dir = root.join("training_output").join(name).join(name);

    // Parse config
    let config_path = run_dir.join("config.yaml");
    let config = if config_path.exists() {
        let yaml_str = std::fs::read_to_string(&config_path)?;
        let yaml_val: serde_yaml::Value = serde_yaml::from_str(&yaml_str)?;
        let json_val = serde_json::to_value(yaml_val)?;
        Some(json_val)
    } else {
        None
    };

    // Scan samples directory
    let samples_dir = run_dir.join("samples");
    let mut step_map: HashMap<u64, Vec<String>> = HashMap::new();

    if samples_dir.exists() {
        let mut entries: Vec<_> = std::fs::read_dir(&samples_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|ext| ext == "jpg" || ext == "png")
            })
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let fname = entry.file_name().to_string_lossy().to_string();
            if let Some(step) = parse_step_from_filename(&fname) {
                let rel = format!("training_output/{name}/{name}/samples/{fname}");
                step_map.entry(step).or_default().push(rel);
            }
        }
    }

    let mut samples: Vec<SampleGroup> = step_map
        .into_iter()
        .map(|(step, mut images)| {
            images.sort();
            let expected = infer_prompt_count(&images);
            if images.len() > expected && expected > 0 {
                if step == 0 {
                    images.truncate(expected);
                } else {
                    images = images.split_off(images.len() - expected);
                }
            }
            SampleGroup { step, images }
        })
        .collect();
    samples.sort_by_key(|s| s.step);

    // Check for final LoRA
    let final_lora_name = format!("{name}.safetensors");
    let lora_path = run_dir.join(&final_lora_name);
    let (lora_p, lora_s) = if lora_path.exists() {
        let meta = std::fs::metadata(&lora_path)?;
        (
            Some(lora_path.to_string_lossy().to_string()),
            Some(meta.len()),
        )
    } else {
        (None, None)
    };

    // Scan for intermediate checkpoint files
    let installed_ids: HashSet<String> = Database::open()
        .ok()
        .and_then(|db| db.list_installed(Some("lora")).ok())
        .map(|models| models.into_iter().map(|m| m.id).collect())
        .unwrap_or_default();

    let mut checkpoints: Vec<CheckpointInfo> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&run_dir) {
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().to_string();
            if !fname.ends_with(".safetensors") || fname == final_lora_name {
                continue;
            }
            if let Some(stem) = fname.strip_suffix(".safetensors")
                && let Some(step_str) = stem.rsplit('_').next()
                && let Ok(step) = step_str.parse::<u64>()
            {
                let path = run_dir.join(&fname).to_string_lossy().to_string();
                let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
                let ckpt_name = format!("{}_{:09}", name, step);
                let promoted = installed_ids
                    .iter()
                    .any(|id: &String| id.starts_with("lib:") && id.contains(&ckpt_name));
                checkpoints.push(CheckpointInfo {
                    step,
                    path,
                    size_bytes,
                    promoted,
                });
            }
        }
    }
    checkpoints.sort_by_key(|c| c.step);

    // Check if the final LoRA has been promoted to the library
    let lora_promoted = if lora_path.exists() {
        installed_ids
            .iter()
            .any(|id: &String| id.starts_with("lib:") && id.contains(name))
    } else {
        false
    };

    let lineage = build_lineage(name);

    let total_steps = lineage
        .as_ref()
        .and_then(|l| l.jobs.first())
        .and_then(|j| j.steps);

    let sample_every = {
        let mut steps: Vec<u64> = samples.iter().map(|s| s.step).collect();
        steps.sort();
        steps.into_iter().find(|&s| s > 0)
    };

    Ok(TrainingRun {
        name: name.to_string(),
        config,
        samples,
        lora_path: lora_p,
        lora_size: lora_s,
        lora_promoted,
        lineage,
        total_steps,
        sample_every,
        checkpoints,
    })
}

/// Build training lineage by querying the jobs DB for matching runs.
pub fn build_lineage(lora_name: &str) -> Option<TrainingLineage> {
    let db = Database::open().ok()?;
    let jobs = db.find_jobs_by_lora_name(lora_name).ok()?;

    if jobs.is_empty() {
        return None;
    }

    let has_db_running = jobs.iter().any(|j| j.status == "running");
    let is_actually_running = if has_db_running {
        training_status::get_status(lora_name)
            .map(|s| s.is_running)
            .unwrap_or(false)
    } else {
        false
    };

    let lora_exists = paths::modl_root()
        .join("training_output")
        .join(lora_name)
        .join(lora_name)
        .join(format!("{lora_name}.safetensors"))
        .exists();

    let first_spec: serde_json::Value = serde_json::from_str(&jobs[0].spec_json).ok()?;

    let dataset_name = first_spec
        .pointer("/dataset/name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let dataset_image_count = first_spec
        .pointer("/dataset/image_count")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let base_model = first_spec
        .pointer("/model/base_model_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let job_summaries: Vec<JobSummary> = jobs
        .iter()
        .map(|j| {
            let spec: serde_json::Value = serde_json::from_str(&j.spec_json).unwrap_or_default();
            let steps = spec.pointer("/params/steps").and_then(|v| v.as_u64());
            let resumed_from = spec
                .pointer("/params/resume_from")
                .and_then(|v| v.as_str())
                .map(|s| s.rsplit('/').next().unwrap_or(s).to_string());
            let status = if j.status == "running" && !is_actually_running {
                if lora_exists {
                    "completed".to_string()
                } else {
                    "interrupted".to_string()
                }
            } else {
                j.status.clone()
            };
            JobSummary {
                job_id: j.job_id.clone(),
                status,
                steps,
                created_at: j.created_at.clone(),
                resumed_from,
            }
        })
        .collect();

    Some(TrainingLineage {
        dataset_name,
        dataset_image_count,
        base_model,
        jobs: job_summaries,
    })
}
