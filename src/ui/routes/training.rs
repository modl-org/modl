use axum::{Json, extract::Path, http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};

use crate::core::db::Database;
use crate::core::training;
use crate::core::training_status;

use crate::core::paths::modl_root;

#[derive(Serialize)]
pub struct RunSummary {
    name: String,
    status: String,
    base_model: Option<String>,
    trigger_word: Option<String>,
    created_at: Option<String>,
    has_lora: bool,
    total_steps: Option<u64>,
}

pub async fn api_list_runs() -> impl IntoResponse {
    let summaries = tokio::task::spawn_blocking(|| {
        let names = crate::core::training::list_training_runs().unwrap_or_default();
        let db = Database::open().ok();
        let active = training_status::get_all_status(false).unwrap_or_default();
        let active_map: std::collections::HashMap<String, _> =
            active.into_iter().map(|s| (s.name.clone(), s)).collect();

        names
            .into_iter()
            .map(|name| {
                let is_running = active_map.get(&name).map(|s| s.is_running).unwrap_or(false);

                let lora_exists = modl_root()
                    .join("training_output")
                    .join(&name)
                    .join(&name)
                    .join(format!("{name}.safetensors"))
                    .exists();

                let (status, base_model, trigger_word, created_at, total_steps) =
                    if let Some(ref db) = db
                        && let Ok(jobs) = db.find_jobs_by_lora_name(&name)
                        && !jobs.is_empty()
                    {
                        let latest = &jobs[jobs.len() - 1];
                        let status = if is_running {
                            "running".to_string()
                        } else if latest.status == "running" {
                            // DB says running but process is gone — check if
                            // the final LoRA exists to distinguish completed
                            // from genuinely interrupted.
                            if lora_exists {
                                "completed".to_string()
                            } else {
                                "interrupted".to_string()
                            }
                        } else {
                            latest.status.clone()
                        };

                        let spec: serde_json::Value =
                            serde_json::from_str(&jobs[0].spec_json).unwrap_or_default();
                        let base = spec
                            .pointer("/model/base_model_id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let trigger = spec
                            .pointer("/params/trigger_word")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let steps = spec.pointer("/params/steps").and_then(|v| v.as_u64());
                        let created = Some(jobs[0].created_at.clone());

                        (status, base, trigger, created, steps)
                    } else {
                        // No jobs in DB — infer from disk
                        let status = if lora_exists {
                            "completed".to_string()
                        } else {
                            "unknown".to_string()
                        };
                        (status, None, None, None, None)
                    };

                RunSummary {
                    name,
                    status,
                    base_model,
                    trigger_word,
                    created_at,
                    has_lora: lora_exists,
                    total_steps,
                }
            })
            .collect::<Vec<_>>()
    })
    .await
    .unwrap_or_default();
    Json(summaries)
}

pub async fn api_get_run(Path(name): Path<String>) -> impl IntoResponse {
    match tokio::task::spawn_blocking(move || training::scan_training_run(&name)).await {
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
        let log_path = crate::core::paths::modl_root().join("training.log");
        let stderr_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map(std::process::Stdio::from)
            .unwrap_or_else(|_| std::process::Stdio::null());

        let child = std::process::Command::new(&modl_bin)
            .args(["train", "--resume", &req.checkpoint, "--name", &req.name])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(stderr_file)
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

// ---------------------------------------------------------------------------
// Start training (new run)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct StartTrainingRequest {
    dataset: String,
    base_model: String,
    name: String,
    trigger_word: String,
    lora_type: String,
    preset: Option<String>,
    steps: Option<u64>,
    rank: Option<u32>,
    lr: Option<f64>,
    optimizer: Option<String>,
    seed: Option<u64>,
    class_word: Option<String>,
}

impl StartTrainingRequest {
    fn validate(&self) -> Result<(), String> {
        if let Some(steps) = self.steps
            && (steps == 0 || steps > 100_000)
        {
            return Err(format!("steps must be between 1 and 100000, got {steps}"));
        }
        if let Some(rank) = self.rank
            && (rank == 0 || rank > 256)
        {
            return Err(format!("rank must be between 1 and 256, got {rank}"));
        }
        if let Some(lr) = self.lr
            && (lr <= 0.0 || lr > 1.0)
        {
            return Err(format!("lr must be between 0 and 1, got {lr}"));
        }
        if self.name.is_empty() {
            return Err("name is required".to_string());
        }
        if self.dataset.is_empty() {
            return Err("dataset is required".to_string());
        }
        Ok(())
    }
}

pub async fn api_start_training(Json(req): Json<StartTrainingRequest>) -> impl IntoResponse {
    if let Err(msg) = req.validate() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": msg })),
        )
            .into_response();
    }

    let modl_bin = std::env::current_exe().unwrap_or_else(|_| "modl".into());

    let result = tokio::task::spawn_blocking(move || {
        let mut args = vec![
            "train".to_string(),
            "--dataset".to_string(),
            req.dataset,
            "--base".to_string(),
            req.base_model,
            "--name".to_string(),
            req.name.clone(),
            "--trigger".to_string(),
            req.trigger_word,
            "--lora-type".to_string(),
            req.lora_type,
        ];

        if let Some(preset) = req.preset {
            args.push("--preset".to_string());
            args.push(preset);
        }
        if let Some(steps) = req.steps {
            args.push("--steps".to_string());
            args.push(steps.to_string());
        }
        if let Some(rank) = req.rank {
            args.push("--rank".to_string());
            args.push(rank.to_string());
        }
        if let Some(lr) = req.lr {
            args.push("--lr".to_string());
            args.push(lr.to_string());
        }
        if let Some(optimizer) = req.optimizer {
            args.push("--optimizer".to_string());
            args.push(optimizer);
        }
        if let Some(seed) = req.seed {
            args.push("--seed".to_string());
            args.push(seed.to_string());
        }
        if let Some(class_word) = req.class_word {
            args.push("--class-word".to_string());
            args.push(class_word);
        }

        let log_path = crate::core::paths::modl_root().join("training.log");
        let stderr_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map(std::process::Stdio::from)
            .unwrap_or_else(|_| std::process::Stdio::null());

        let child = std::process::Command::new(&modl_bin)
            .args(&args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(stderr_file)
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

// ---------------------------------------------------------------------------
// Training Queue
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct TrainingQueueItemResponse {
    id: i64,
    position: i64,
    name: String,
    spec: serde_json::Value,
    status: String,
    created_at: String,
}

/// GET /api/train/queue — list pending queue items
pub async fn api_list_training_queue() -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(|| {
        let db = Database::open()?;
        db.list_training_queue()
    })
    .await;

    match result {
        Ok(Ok(items)) => {
            let response: Vec<TrainingQueueItemResponse> = items
                .into_iter()
                .map(|item| TrainingQueueItemResponse {
                    id: item.id,
                    position: item.position,
                    name: item.name,
                    spec: serde_json::from_str(&item.spec_json).unwrap_or_default(),
                    status: item.status,
                    created_at: item.created_at,
                })
                .collect();
            Json(response).into_response()
        }
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "Failed to list training queue" })),
        )
            .into_response(),
    }
}

/// POST /api/train/queue — add item to queue
pub async fn api_add_to_training_queue(Json(req): Json<StartTrainingRequest>) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || {
        let spec_json = serde_json::to_string(&serde_json::json!({
            "dataset": req.dataset,
            "base_model": req.base_model,
            "name": req.name,
            "trigger_word": req.trigger_word,
            "lora_type": req.lora_type,
            "preset": req.preset,
            "steps": req.steps,
            "rank": req.rank,
            "lr": req.lr,
            "optimizer": req.optimizer,
            "seed": req.seed,
            "class_word": req.class_word,
        }))
        .unwrap_or_default();

        let db = Database::open()?;
        let id = db.add_to_training_queue(&req.name, &spec_json)?;
        Ok::<_, anyhow::Error>((id, req.name))
    })
    .await;

    match result {
        Ok(Ok((id, name))) => (
            StatusCode::CREATED,
            Json(serde_json::json!({ "id": id, "name": name })),
        )
            .into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "Failed to add to queue" })),
        )
            .into_response(),
    }
}

/// DELETE /api/train/queue/{id} — remove item from queue
pub async fn api_remove_from_training_queue(Path(id): Path<i64>) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || {
        let db = Database::open()?;
        db.remove_from_training_queue(id)
    })
    .await;

    match result {
        Ok(Ok(())) => StatusCode::NO_CONTENT.into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "Failed to remove from queue" })),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
pub struct UpdatePositionRequest {
    position: i64,
}

/// PUT /api/train/queue/{id}/position — reorder queue item
pub async fn api_reorder_training_queue(
    Path(id): Path<i64>,
    Json(req): Json<UpdatePositionRequest>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || {
        let db = Database::open()?;
        db.update_training_queue_position(id, req.position)
    })
    .await;

    match result {
        Ok(Ok(())) => StatusCode::NO_CONTENT.into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "Failed to reorder queue" })),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// LoRA Library
// ---------------------------------------------------------------------------

/// GET /api/library/loras — list all promoted LoRAs
pub async fn api_list_library_loras() -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(|| {
        let db = Database::open()?;
        db.list_library_loras()
    })
    .await;

    match result {
        Ok(Ok(loras)) => Json(serde_json::to_value(loras).unwrap_or_default()).into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "Failed to list library LoRAs" })),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
pub struct PromoteLoraRequest {
    name: String,
    trigger_word: Option<String>,
    base_model: Option<String>,
    lora_path: String,
    thumbnail: Option<String>,
    step: Option<u64>,
    training_run: Option<String>,
    config_json: Option<String>,
    tags: Option<String>,
}

/// POST /api/library/loras — promote a LoRA to the library
pub async fn api_promote_lora(Json(req): Json<PromoteLoraRequest>) -> impl IntoResponse {
    use crate::core::artifacts::{PromoteLoraParams, promote_lora};

    let result = tokio::task::spawn_blocking(move || {
        promote_lora(PromoteLoraParams {
            name: req.name,
            trigger_word: req.trigger_word,
            base_model: req.base_model,
            lora_path: req.lora_path,
            thumbnail: req.thumbnail,
            step: req.step,
            training_run: req.training_run,
            config_json: req.config_json,
            tags: req.tags,
        })
    })
    .await;

    match result {
        Ok(Ok(id)) => (StatusCode::CREATED, Json(serde_json::json!({ "id": id }))).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Failed to promote LoRA: {e}") })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Task failed: {e}") })),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
pub struct UpdateLibraryLoraRequest {
    name: String,
    tags: Option<String>,
    notes: Option<String>,
}

/// PUT /api/library/loras/{id}
pub async fn api_update_library_lora(
    Path(id): Path<String>,
    Json(req): Json<UpdateLibraryLoraRequest>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || {
        let db = Database::open()?;
        db.update_library_lora(&id, &req.name, req.tags.as_deref(), req.notes.as_deref())
    })
    .await;

    match result {
        Ok(Ok(())) => StatusCode::NO_CONTENT.into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "Failed to update library LoRA" })),
        )
            .into_response(),
    }
}

/// DELETE /api/library/loras/{id}
pub async fn api_delete_library_lora(Path(id): Path<String>) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || {
        let db = Database::open()?;
        // Clean up installed + artifacts entries so the LoRA disappears
        // from the generate dropdown. Store file is kept (content-addressed,
        // cheap, may be referenced elsewhere).
        let _ = db.remove_installed(&id);
        let _ = db.delete_artifact(&id);
        db.delete_library_lora(&id)
    })
    .await;

    match result {
        Ok(Ok(())) => StatusCode::NO_CONTENT.into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "Failed to delete library LoRA" })),
        )
            .into_response(),
    }
}
