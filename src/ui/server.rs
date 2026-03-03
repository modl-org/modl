use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{Path, Query},
    http::{StatusCode, header},
    response::{Html, IntoResponse},
    routing::get,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use tokio::net::TcpListener;

use crate::core::dataset;
use crate::core::db::Database;
use crate::core::training_status;

// ---------------------------------------------------------------------------
// Data types returned by the API
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct TrainingRun {
    name: String,
    config: Option<serde_json::Value>,
    samples: Vec<SampleGroup>,
    lora_path: Option<String>,
    lora_size: Option<u64>,
    lineage: Option<TrainingLineage>,
}

/// Provenance: links a training run back to its job(s), dataset, and artifacts.
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
    images: Vec<String>, // relative paths served by /files/
}

#[derive(Serialize)]
struct DatasetOverview {
    name: String,
    image_count: u32,
    captioned_count: u32,
    coverage: f32,
    images: Vec<DatasetImage>,
}

#[derive(Serialize)]
struct DatasetImage {
    filename: String,
    caption: Option<String>,
    image_url: String, // /files/datasets/...
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

pub async fn start(port: u16, open_browser: bool) -> Result<()> {
    // Kill any existing server on this port so `modl preview` is always re-entrant
    kill_existing_on_port(port);

    let app = Router::new()
        .route("/", get(index_page))
        .route("/api/runs", get(api_list_runs))
        .route("/api/runs/{name}", get(api_get_run))
        .route("/api/status", get(api_training_status))
        .route("/api/status/{name}", get(api_training_status_single))
        .route("/api/datasets", get(api_list_datasets))
        .route("/api/datasets/{name}", get(api_get_dataset))
        .route("/files/{*path}", get(serve_file));

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(addr)
        .await
        .context("Failed to bind to port")?;

    let url = format!("http://127.0.0.1:{port}");
    eprintln!("  Training preview UI running at {url}");
    eprintln!("  Press Ctrl+C to stop\n");

    if open_browser {
        let _ = open::that(&url);
    }

    axum::serve(listener, app).await.context("Server error")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Kill any existing process **listening** on the given port (best-effort).
///
/// IMPORTANT: We use `-sTCP:LISTEN` so we only kill the server process, NOT
/// processes that merely have a connection to the port (e.g. VS Code Remote
/// SSH port-forwarding).  Without this filter, `lsof -ti :PORT` also returns
/// the sshd/vscode-server PID and killing it drops the SSH session.
fn kill_existing_on_port(port: u16) {
    // Try to connect — if it succeeds, something is already listening
    let addr: std::net::SocketAddr = ([127, 0, 0, 1], port).into();
    if std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(200)).is_ok() {
        // Use lsof to find PIDs in LISTEN state only
        if let Ok(output) = std::process::Command::new("lsof")
            .args(["-ti", &format!(":{port}"), "-sTCP:LISTEN"])
            .output()
        {
            let pids = String::from_utf8_lossy(&output.stdout);
            let my_pid = std::process::id().to_string();
            for pid_str in pids.split_whitespace() {
                if pid_str != my_pid {
                    let _ = std::process::Command::new("kill").arg(pid_str).status();
                    eprintln!("  Killed existing server (PID {pid_str}) on port {port}");
                }
            }
            // Brief pause to let the port free up
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
}

fn modl_root() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
}

/// Parse step number from sample filename like `1772410330707__000000000_0.jpg`
fn parse_step_from_filename(filename: &str) -> Option<u64> {
    // Pattern: <timestamp>__<step>_<index>.jpg
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
/// number of prompts.  When a step has duplicates (e.g. from a resumed run)
/// we look at the unique prompt-index suffixes (`_0`, `_1`, …) to determine
/// how many distinct prompts there are per sampling batch.
fn infer_prompt_count(images: &[String]) -> usize {
    // Extract the `_<index>` suffix from each filename and find the max unique
    // index seen.  The filename pattern is: <ts>__<step>_<idx>.jpg
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
fn scan_training_run(name: &str) -> Result<TrainingRun> {
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
            // When a run is resumed, the same step (especially step 0) may have
            // duplicate samples from both the original and resumed runs.
            let expected = infer_prompt_count(&images);
            if images.len() > expected && expected > 0 {
                if step == 0 {
                    // Step 0: keep the *first* batch (original run) — the
                    // resumed run's step 0 is visually identical to the last
                    // checkpoint and adds no information.
                    images.truncate(expected);
                } else {
                    // Other steps: keep the *last* batch (most recent run).
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

    // Query DB for job lineage
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

    // Extract dataset + model info from the first job's spec
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
                .map(|s| {
                    // Show just the filename, not the full path
                    s.rsplit('/').next().unwrap_or(s).to_string()
                });
            JobSummary {
                job_id: j.job_id.clone(),
                status: j.status.clone(),
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

// ---------------------------------------------------------------------------
// API routes
// ---------------------------------------------------------------------------

async fn api_list_runs() -> impl IntoResponse {
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
    Json(runs)
}

async fn api_get_run(Path(name): Path<String>) -> impl IntoResponse {
    match scan_training_run(&name) {
        Ok(run) => Json(serde_json::to_value(run).unwrap()).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error scanning run: {e}"),
        )
            .into_response(),
    }
}

async fn api_training_status() -> impl IntoResponse {
    match training_status::get_all_status(false) {
        Ok(runs) => Json(serde_json::to_value(runs).unwrap()).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error getting training status: {e}"),
        )
            .into_response(),
    }
}

async fn api_training_status_single(Path(name): Path<String>) -> impl IntoResponse {
    match training_status::get_status(&name) {
        Ok(run) => Json(serde_json::to_value(run).unwrap()).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error getting training status: {e}"),
        )
            .into_response(),
    }
}

async fn api_list_datasets() -> impl IntoResponse {
    match dataset::list() {
        Ok(datasets) => {
            let names: Vec<String> = datasets.iter().map(|d| d.name.clone()).collect();
            Json(names).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error listing datasets: {e}"),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct DatasetQuery {
    #[serde(default = "default_page_size")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

fn default_page_size() -> usize {
    50
}

async fn api_get_dataset(
    Path(name): Path<String>,
    Query(q): Query<DatasetQuery>,
) -> impl IntoResponse {
    let ds_path = dataset::resolve_path(&name);

    match dataset::scan(&ds_path) {
        Ok(info) => {
            let mut images: Vec<DatasetImage> = Vec::new();
            let page = info.images.iter().skip(q.offset).take(q.limit);

            for entry in page {
                let fname = entry
                    .path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string();
                let caption = entry
                    .caption_path
                    .as_ref()
                    .and_then(|p| std::fs::read_to_string(p).ok())
                    .map(|s| s.trim().to_string());
                let image_url = format!("datasets/{name}/{fname}");
                images.push(DatasetImage {
                    filename: fname,
                    caption,
                    image_url,
                });
            }

            let overview = DatasetOverview {
                name: info.name,
                image_count: info.image_count,
                captioned_count: info.captioned_count,
                coverage: info.caption_coverage,
                images,
            };

            Json(serde_json::to_value(overview).unwrap()).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error scanning dataset: {e}"),
        )
            .into_response(),
    }
}

/// Serve files from ~/.modl/ (images, samples, etc.)
async fn serve_file(Path(path): Path<String>) -> impl IntoResponse {
    let full_path = modl_root().join(&path);

    // Security: ensure resolved path is still under modl_root
    let canonical = match full_path.canonicalize() {
        Ok(p) => p,
        Err(_) => return (StatusCode::NOT_FOUND, "Not found").into_response(),
    };
    let root_canonical = match modl_root().canonicalize() {
        Ok(p) => p,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "Config error").into_response(),
    };
    if !canonical.starts_with(&root_canonical) {
        return (StatusCode::FORBIDDEN, "Forbidden").into_response();
    }

    match tokio::fs::read(&canonical).await {
        Ok(bytes) => {
            let content_type = match canonical.extension().and_then(|e| e.to_str()).unwrap_or("") {
                "jpg" | "jpeg" => "image/jpeg",
                "png" => "image/png",
                "webp" => "image/webp",
                "yaml" | "yml" => "text/plain; charset=utf-8",
                "json" => "application/json",
                "safetensors" => "application/octet-stream",
                _ => "application/octet-stream",
            };
            ([(header::CONTENT_TYPE, content_type)], bytes).into_response()
        }
        Err(_) => (StatusCode::NOT_FOUND, "Not found").into_response(),
    }
}

// ---------------------------------------------------------------------------
// Main HTML page (self-contained, no external deps except system fonts)
// ---------------------------------------------------------------------------

async fn index_page() -> Html<String> {
    Html(include_str!("index.html").to_string())
}
