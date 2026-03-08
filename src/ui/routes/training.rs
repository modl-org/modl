use axum::{Json, extract::Path, http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};
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
    /// Original total steps from the job spec (covers all phases).
    total_steps: Option<u64>,
    /// Sample interval derived from actual samples on disk (stable across phases).
    sample_every: Option<u64>,
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

    // total_steps from the job spec covers all phases (the config.yaml on
    // disk only reflects the current phase).
    let total_steps = lineage
        .as_ref()
        .and_then(|l| l.jobs.first())
        .and_then(|j| j.steps);

    // Derive sample_every from actual samples on disk — stable across phase
    // transitions (unlike config.yaml which gets overwritten per phase).
    // The first non-zero step is always the sample interval.
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
        lineage,
        total_steps,
        sample_every,
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
        crate::core::training::list_training_runs().unwrap_or_default()
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

// ---------------------------------------------------------------------------
// Loss history (parsed from log file)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct LossPoint {
    step: u64,
    loss: f64,
}

pub async fn api_loss_history(Path(name): Path<String>) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || {
        let log_path = modl_root()
            .join("training_output")
            .join(format!("{name}.log"));
        if !log_path.exists() {
            return Vec::new();
        }

        let content = match std::fs::read_to_string(&log_path) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        // Parse step/loss pairs from tqdm lines
        // Format: <step>/<total> [..., loss: <value>]
        let mut points: Vec<LossPoint> = Vec::new();
        let mut seen_steps: std::collections::HashSet<u64> = std::collections::HashSet::new();

        for segment in content.split(['\r', '\n']) {
            let line = segment.trim();
            if line.is_empty() || !line.contains("loss:") {
                continue;
            }

            // Extract step
            let step = extract_step(line);
            let loss = extract_loss(line);

            if let (Some(step), Some(loss)) = (step, loss) {
                if seen_steps.insert(step) {
                    points.push(LossPoint { step, loss });
                } else {
                    // Update to latest loss for this step
                    if let Some(p) = points.iter_mut().rev().find(|p| p.step == step) {
                        p.loss = loss;
                    }
                }
            }
        }

        // Downsample to ~200 points max for performance
        if points.len() > 200 {
            let step_size = points.len() / 200;
            points = points
                .into_iter()
                .enumerate()
                .filter(|(i, _)| i % step_size == 0)
                .map(|(_, p)| p)
                .collect();
        }

        points
    })
    .await
    .unwrap_or_default();

    Json(result)
}

fn extract_step(line: &str) -> Option<u64> {
    // Find <digits>/<digits> pattern
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut best: Option<u64> = None;

    while i < len {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < len && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i < len && bytes[i] == b'/' {
                let step_str = &line[start..i];
                i += 1;
                let total_start = i;
                while i < len && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i > total_start {
                    let total_str = &line[total_start..i];
                    if let (Ok(step), Ok(total)) =
                        (step_str.parse::<u64>(), total_str.parse::<u64>())
                        && total > 10
                    {
                        best = Some(step);
                    }
                }
            }
        }
        i += 1;
    }

    best
}

fn extract_loss(line: &str) -> Option<f64> {
    let idx = line.find("loss:")?;
    let rest = &line[idx + 5..];
    rest.split_whitespace()
        .next()
        .map(|s| s.trim_end_matches(']').trim_end_matches(','))
        .and_then(|s| s.parse::<f64>().ok())
}

// ---------------------------------------------------------------------------
// Cancel training
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct CancelRequest {
    name: String,
}

pub async fn api_cancel_training(Json(req): Json<CancelRequest>) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || cancel_training_process(&req.name)).await;

    match result {
        Ok(Ok(killed)) => (
            StatusCode::OK,
            Json(serde_json::json!({ "cancelled": true, "pids_killed": killed })),
        )
            .into_response(),
        Ok(Err(msg)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": msg })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Task failed: {e}") })),
        )
            .into_response(),
    }
}

/// Find and kill the training process for a given run name.
fn cancel_training_process(name: &str) -> Result<usize, String> {
    let output = std::process::Command::new("ps")
        .args(["ax", "-o", "pid=,args="])
        .output()
        .map_err(|e| format!("Failed to run ps: {e}"))?;

    if !output.status.success() {
        return Err("ps command failed".to_string());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut killed = 0usize;

    for line in stdout.lines() {
        let trimmed = line.trim();
        if !trimmed.contains("run.py") {
            continue;
        }

        // Extract PID (first field) and config path (last .yaml arg)
        let mut parts = trimmed.splitn(2, char::is_whitespace);
        let pid_str = match parts.next() {
            Some(p) => p.trim(),
            None => continue,
        };
        let args = match parts.next() {
            Some(a) => a.trim(),
            None => continue,
        };

        let config_path = args
            .split_whitespace()
            .last()
            .filter(|a| a.ends_with(".yaml") || a.ends_with(".yml"));

        if let Some(path) = config_path
            && let Ok(contents) = std::fs::read_to_string(path)
        {
            let matches = contents.lines().any(|l| {
                let t = l.trim();
                t.strip_prefix("name:")
                    .map(|n| n.trim().trim_matches('"').trim_matches('\'') == name)
                    .unwrap_or(false)
                    || t.strip_prefix("training_folder:")
                        .and_then(|f| {
                            f.trim()
                                .trim_matches('"')
                                .trim_matches('\'')
                                .rsplit('/')
                                .next()
                        })
                        .is_some_and(|n| n == name)
            });

            if matches && let Ok(pid) = pid_str.parse::<u32>() {
                // Kill the process group to ensure child processes are also terminated
                let _ = std::process::Command::new("kill")
                    .args(["-TERM", &format!("{pid}")])
                    .status();
                killed += 1;
            }
        }
    }

    if killed == 0 {
        return Err(format!("No running training process found for '{name}'"));
    }

    // Update DB status to interrupted
    if let Ok(db) = Database::open() {
        let _ = db.update_job_status_by_lora_name(name, "running", "interrupted");
    }

    Ok(killed)
}

// ---------------------------------------------------------------------------
// Delete training run
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct DeleteRunRequest {
    name: String,
}

pub async fn api_delete_run(Json(req): Json<DeleteRunRequest>) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || delete_training_run(&req.name)).await;

    match result {
        Ok(Ok(())) => {
            (StatusCode::OK, Json(serde_json::json!({ "deleted": true }))).into_response()
        }
        Ok(Err(msg)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": msg })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Task failed: {e}") })),
        )
            .into_response(),
    }
}

fn delete_training_run(name: &str) -> Result<(), String> {
    crate::core::training::delete_training_run(name).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Resume training
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ResumeRequest {
    name: String,
    checkpoint: String,
}

pub async fn api_resume_training(Json(req): Json<ResumeRequest>) -> impl IntoResponse {
    // Find the modl binary path
    let modl_bin = std::env::current_exe().unwrap_or_else(|_| "modl".into());

    let result = tokio::task::spawn_blocking(move || {
        // Spawn `modl train --resume <checkpoint> --name <name>` as a detached process
        let child = std::process::Command::new(&modl_bin)
            .args(["train", "--resume", &req.checkpoint, "--name", &req.name])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();

        match child {
            Ok(_) => Ok(req.name),
            Err(e) => Err(format!("Failed to start training: {e}")),
        }
    })
    .await;

    match result {
        Ok(Ok(name)) => {
            (StatusCode::OK, Json(serde_json::json!({ "started": name }))).into_response()
        }
        Ok(Err(msg)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": msg })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Task failed: {e}") })),
        )
            .into_response(),
    }
}
