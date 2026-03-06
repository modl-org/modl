use axum::{Json, extract::Path, http::StatusCode, response::IntoResponse};
use serde::Serialize;
use std::collections::HashMap;

use crate::core::db::Database;
use crate::core::training_status;

use super::super::server::modl_root;

#[derive(Serialize)]
pub struct TrainingRun {
    name: String,
    config: Option<serde_json::Value>,
    samples: Vec<SampleGroup>,
    lora_path: Option<String>,
    lora_size: Option<u64>,
    lineage: Option<TrainingLineage>,
}

#[derive(Serialize)]
struct TrainingLineage {
    dataset_name: Option<String>,
    dataset_image_count: Option<u32>,
    base_model: Option<String>,
    jobs: Vec<JobSummary>,
}

#[derive(Serialize)]
struct JobSummary {
    job_id: String,
    status: String,
    steps: Option<u64>,
    created_at: String,
    resumed_from: Option<String>,
}

#[derive(Serialize)]
struct SampleGroup {
    step: u64,
    images: Vec<String>,
}

/// Parse step number from sample filename like `1772410330707__000000000_0.jpg`
fn parse_step_from_filename(filename: &str) -> Option<u64> {
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

/// Scan a training output directory for sample images grouped by step
fn scan_training_run(name: &str) -> anyhow::Result<TrainingRun> {
    let run_dir = modl_root().join("training_output").join(name).join(name);

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
    let lora_path = run_dir.join(format!("{name}.safetensors"));
    let (lora_p, lora_s) = if lora_path.exists() {
        let meta = std::fs::metadata(&lora_path)?;
        (
            Some(lora_path.to_string_lossy().to_string()),
            Some(meta.len()),
        )
    } else {
        (None, None)
    };

    let lineage = build_lineage(name);

    Ok(TrainingRun {
        name: name.to_string(),
        config,
        samples,
        lora_path: lora_p,
        lora_size: lora_s,
        lineage,
    })
}

/// Build training lineage by querying the jobs DB for matching runs.
fn build_lineage(lora_name: &str) -> Option<TrainingLineage> {
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
                "interrupted".to_string()
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

pub async fn api_list_runs() -> impl IntoResponse {
    let runs = tokio::task::spawn_blocking(|| {
        let output_dir = modl_root().join("training_output");
        let mut runs = Vec::new();

        if output_dir.exists()
            && let Ok(entries) = std::fs::read_dir(&output_dir)
        {
            for entry in entries.flatten() {
                if entry.file_type().is_ok_and(|t| t.is_dir()) {
                    let name = entry.file_name().to_string_lossy().to_string();
                    runs.push(name);
                }
            }
        }
        runs.sort();
        runs
    })
    .await
    .unwrap_or_default();
    Json(runs)
}

pub async fn api_get_run(Path(name): Path<String>) -> impl IntoResponse {
    match tokio::task::spawn_blocking(move || scan_training_run(&name)).await {
        Ok(Ok(run)) => Json(run).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error scanning run: {e}"),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task failed: {e}"),
        )
            .into_response(),
    }
}

pub async fn api_training_status() -> impl IntoResponse {
    match tokio::task::spawn_blocking(|| training_status::get_all_status(false)).await {
        Ok(Ok(runs)) => Json(runs).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error getting training status: {e}"),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task failed: {e}"),
        )
            .into_response(),
    }
}

pub async fn api_training_status_single(Path(name): Path<String>) -> impl IntoResponse {
    match tokio::task::spawn_blocking(move || training_status::get_status(&name)).await {
        Ok(Ok(run)) => Json(run).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error getting training status: {e}"),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task failed: {e}"),
        )
            .into_response(),
    }
}
