use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::{StatusCode, header},
    response::{
        Html, IntoResponse, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use futures_util::stream::{self, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::{Path as FsPath, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::broadcast;

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
struct GeneratedOutput {
    date: String,
    images: Vec<GeneratedImage>,
}

#[derive(Serialize)]
struct GeneratedImage {
    /// Relative path usable as /files/<path>
    path: String,
    /// Filename without directory
    filename: String,
    /// mtime as unix timestamp (seconds)
    modified: u64,
    /// Artifact ID in DB, if tracked
    artifact_id: Option<String>,
    /// Job ID that produced the image, if tracked
    job_id: Option<String>,
    /// Prompt used to generate the image, if available
    prompt: Option<String>,
    /// Base model ID used for generation, if available
    base_model_id: Option<String>,
    /// LoRA name used, if any
    lora_name: Option<String>,
    /// LoRA strength used, if any
    lora_strength: Option<f64>,
    /// Per-image seed, if available
    seed: Option<u64>,
    /// Inference steps, if available
    steps: Option<u32>,
    /// Guidance scale, if available
    guidance: Option<f64>,
    /// Output width, if available
    width: Option<u32>,
    /// Output height, if available
    height: Option<u32>,
    /// Stored artifact size, if tracked
    size_bytes: Option<u64>,
    /// Marker embedded by generator
    generated_with: Option<String>,
}

#[derive(Clone)]
struct OutputArtifactInfo {
    artifact_id: String,
    job_id: Option<String>,
    size_bytes: u64,
    metadata: Option<String>,
}

#[derive(Default)]
struct OutputMetaSummary {
    prompt: Option<String>,
    base_model_id: Option<String>,
    lora_name: Option<String>,
    lora_strength: Option<f64>,
    seed: Option<u64>,
    steps: Option<u32>,
    guidance: Option<f64>,
    width: Option<u32>,
    height: Option<u32>,
    generated_with: Option<String>,
}

fn parse_output_meta(metadata: Option<&str>) -> OutputMetaSummary {
    let Some(raw) = metadata else {
        return OutputMetaSummary::default();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) else {
        return OutputMetaSummary::default();
    };

    OutputMetaSummary {
        prompt: v
            .get("prompt")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
        base_model_id: v
            .get("base_model_id")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
        lora_name: v
            .get("lora_name")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
        lora_strength: v.get("lora_strength").and_then(|x| x.as_f64()),
        seed: v.get("seed").and_then(|x| x.as_u64()),
        steps: v.get("steps").and_then(|x| x.as_u64()).map(|n| n as u32),
        guidance: v.get("guidance").and_then(|x| x.as_f64()),
        width: v.get("width").and_then(|x| x.as_u64()).map(|n| n as u32),
        height: v.get("height").and_then(|x| x.as_u64()).map(|n| n as u32),
        generated_with: v
            .get("generated_with")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
    }
}

fn parse_generate_job_spec_meta(spec_json: &str) -> Option<OutputMetaSummary> {
    let spec: serde_json::Value = serde_json::from_str(spec_json).ok()?;
    Some(OutputMetaSummary {
        prompt: spec
            .get("prompt")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
        base_model_id: spec
            .pointer("/model/base_model_id")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
        lora_name: spec
            .pointer("/lora/name")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
        lora_strength: spec.pointer("/lora/weight").and_then(|x| x.as_f64()),
        seed: spec.pointer("/params/seed").and_then(|x| x.as_u64()),
        steps: spec
            .pointer("/params/steps")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
        guidance: spec.pointer("/params/guidance").and_then(|x| x.as_f64()),
        width: spec
            .pointer("/params/width")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
        height: spec
            .pointer("/params/height")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
        generated_with: Some("modl.run".to_string()),
    })
}

fn load_output_artifact_index() -> HashMap<String, OutputArtifactInfo> {
    let mut by_path: HashMap<String, OutputArtifactInfo> = HashMap::new();
    let Ok(db) = Database::open() else {
        return by_path;
    };
    let Ok(artifacts) = db.list_artifacts(None) else {
        return by_path;
    };

    for artifact in artifacts {
        if artifact.kind != "image" || artifact.path.is_empty() {
            continue;
        }
        if by_path.contains_key(&artifact.path) {
            continue;
        }

        // Fallback for older rows that didn't store per-image metadata.
        let metadata =
            if artifact.metadata.is_none() || artifact.metadata.as_deref() == Some("null") {
                if let Some(job_id) = &artifact.job_id {
                    if let Ok(Some(job)) = db.get_job(job_id) {
                        parse_generate_job_spec_meta(&job.spec_json).map(|m| {
                            serde_json::json!({
                                "generated_with": m.generated_with,
                                "prompt": m.prompt,
                                "base_model_id": m.base_model_id,
                                "lora_name": m.lora_name,
                                "lora_strength": m.lora_strength,
                                "seed": m.seed,
                                "steps": m.steps,
                                "guidance": m.guidance,
                                "width": m.width,
                                "height": m.height,
                            })
                            .to_string()
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                artifact.metadata.clone()
            };

        by_path.insert(
            artifact.path.clone(),
            OutputArtifactInfo {
                artifact_id: artifact.artifact_id,
                job_id: artifact.job_id,
                size_bytes: artifact.size_bytes,
                metadata,
            },
        );
    }

    by_path
}

/// Scan ~/.modl/outputs/ for generated images, grouped by date.
fn scan_outputs_dir() -> Vec<GeneratedOutput> {
    let outputs_root = modl_root().join("outputs");
    let mut result: Vec<GeneratedOutput> = Vec::new();
    let artifacts_by_path = load_output_artifact_index();

    let Ok(dates) = std::fs::read_dir(&outputs_root) else {
        return result;
    };

    let mut date_entries: Vec<_> = dates.filter_map(|e| e.ok()).collect();
    date_entries.sort_by_key(|e| std::cmp::Reverse(e.file_name()));

    for date_entry in date_entries {
        let date_path = date_entry.path();
        if !date_path.is_dir() {
            continue;
        }
        let date_str = date_entry.file_name().to_string_lossy().to_string();

        let Ok(files) = std::fs::read_dir(&date_path) else {
            continue;
        };

        let mut images: Vec<GeneratedImage> = files
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                name.ends_with(".png") || name.ends_with(".jpg") || name.ends_with(".webp")
            })
            .map(|e| {
                let filename = e.file_name().to_string_lossy().to_string();
                let rel = format!("outputs/{}/{}", date_str, filename);
                let abs = date_path.join(&filename).to_string_lossy().to_string();
                let modified = e
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let artifact = artifacts_by_path.get(&abs);
                let meta = parse_output_meta(artifact.and_then(|a| a.metadata.as_deref()));
                GeneratedImage {
                    path: rel,
                    filename,
                    modified,
                    artifact_id: artifact.map(|a| a.artifact_id.clone()),
                    job_id: artifact.and_then(|a| a.job_id.clone()),
                    prompt: meta.prompt,
                    base_model_id: meta.base_model_id,
                    lora_name: meta.lora_name,
                    lora_strength: meta.lora_strength,
                    seed: meta.seed,
                    steps: meta.steps,
                    guidance: meta.guidance,
                    width: meta.width,
                    height: meta.height,
                    size_bytes: artifact.and_then(|a| (a.size_bytes > 0).then_some(a.size_bytes)),
                    generated_with: meta.generated_with,
                }
            })
            .collect();

        images.sort_by_key(|i| std::cmp::Reverse(i.modified));

        if !images.is_empty() {
            result.push(GeneratedOutput {
                date: date_str,
                images,
            });
        }
    }

    result
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

#[derive(Serialize)]
struct GpuStatus {
    name: Option<String>,
    vram_total_mb: Option<u64>,
    vram_free_mb: Option<u64>,
    training_active: bool,
}

#[derive(Serialize)]
struct InstalledModel {
    id: String,
    name: String,
    model_type: String,
    variant: Option<String>,
    size_bytes: u64,
}

#[derive(Clone)]
struct UiState {
    generate_events: broadcast::Sender<String>,
    generate_running: Arc<AtomicBool>,
}

#[derive(Deserialize)]
struct GenerateLoraRequest {
    id: String,
    strength: f32,
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    #[serde(default)]
    negative_prompt: Option<String>,
    model_id: String,
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    #[serde(default)]
    seed: Option<u64>,
    num_images: u32,
    #[serde(default)]
    loras: Vec<GenerateLoraRequest>,
}

#[derive(Serialize)]
struct GenerateAcceptedResponse {
    status: String,
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

pub async fn start(port: u16, open_browser: bool) -> Result<()> {
    // Kill any existing server on this port so `modl serve` is always re-entrant
    kill_existing_on_port(port);
    let (generate_events, _) = broadcast::channel(256);
    let state = UiState {
        generate_events,
        generate_running: Arc::new(AtomicBool::new(false)),
    };

    let app = Router::new()
        .route("/", get(index_page))
        .route("/api/gpu", get(api_gpu_status))
        .route("/api/models", get(api_list_models))
        .route("/api/generate", post(api_generate))
        .route("/api/generate/stream", get(api_generate_stream))
        .route("/api/runs", get(api_list_runs))
        .route("/api/runs/{name}", get(api_get_run))
        .route("/api/status", get(api_training_status))
        .route("/api/status/{name}", get(api_training_status_single))
        .route("/api/datasets", get(api_list_datasets))
        .route("/api/datasets/{name}", get(api_get_dataset))
        .route(
            "/api/outputs",
            get(api_list_outputs).delete(api_delete_output),
        )
        .route("/assets/{*path}", get(serve_ui_asset))
        .route("/files/{*path}", get(serve_file))
        .with_state(state);

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

/// Drop guard that resets `generate_running` to `false` when it goes out of scope,
/// even if the spawned task panics.
struct RunGuard(Arc<AtomicBool>);

impl Drop for RunGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

async fn api_generate(
    State(state): State<UiState>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    if state.generate_running.swap(true, Ordering::SeqCst) {
        return (StatusCode::CONFLICT, "A generate job is already running").into_response();
    }

    let sender = state.generate_events.clone();
    let running = state.generate_running.clone();

    tokio::spawn(async move {
        let _guard = RunGuard(running);
        let _ = sender.send("queued".to_string());

        if req
            .negative_prompt
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty())
        {
            let _ = sender
                .send("warning: negative prompt is currently ignored by preview UI".to_string());
        }

        let size = format!("{}x{}", req.width, req.height);
        let lora_id = req.loras.first().map(|l| l.id.clone());
        let lora_strength = req.loras.first().map(|l| l.strength).unwrap_or(1.0);
        let _ = sender.send("starting generation".to_string());

        let run_result = crate::cli::generate::run(
            &req.prompt,
            Some(&req.model_id),
            lora_id.as_deref(),
            lora_strength,
            req.seed,
            &size,
            Some(req.steps),
            Some(req.guidance),
            req.num_images,
            false,
            None,
            true,
        )
        .await;

        match run_result {
            Ok(()) => {
                let _ = sender.send("completed".to_string());
            }
            Err(err) => {
                let _ = sender.send(format!("error: {err}"));
            }
        }

        // `_guard` drops here, resetting `generate_running` to false
    });

    (
        StatusCode::ACCEPTED,
        Json(GenerateAcceptedResponse {
            status: "queued".to_string(),
        }),
    )
        .into_response()
}

async fn api_generate_stream(
    State(state): State<UiState>,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let running_now = state.generate_running.load(Ordering::SeqCst);
    let initial = if running_now { "running" } else { "idle" }.to_string();

    let first =
        stream::once(async move { Ok::<Event, Infallible>(Event::default().data(initial)) });

    let updates = stream::unfold(state.generate_events.subscribe(), |mut rx| async move {
        match rx.recv().await {
            Ok(msg) => Some((Ok(Event::default().data(msg)), rx)),
            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                let msg = format!("warning: skipped {skipped} progress events");
                Some((Ok(Event::default().data(msg)), rx))
            }
            Err(broadcast::error::RecvError::Closed) => None,
        }
    });

    Sse::new(first.chain(updates)).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(5))
            .text("keepalive"),
    )
}

async fn api_gpu_status() -> impl IntoResponse {
    let training_active = training_status::get_all_status(true)
        .map(|runs| runs.iter().any(|r| r.is_running))
        .unwrap_or(false);

    let (name, vram_total_mb, vram_free_mb) = if let Ok(nvml) = nvml_wrapper::Nvml::init() {
        if let Ok(device) = nvml.device_by_index(0) {
            let name = device.name().ok();
            let mem = device.memory_info().ok();
            (
                name,
                mem.as_ref().map(|m| m.total / (1024 * 1024)),
                mem.as_ref().map(|m| m.free / (1024 * 1024)),
            )
        } else {
            (None, None, None)
        }
    } else {
        (None, None, None)
    };

    Json(GpuStatus {
        name,
        vram_total_mb,
        vram_free_mb,
        training_active,
    })
}

async fn api_list_models() -> impl IntoResponse {
    let db = match Database::open() {
        Ok(db) => db,
        Err(_) => return Json(Vec::<InstalledModel>::new()).into_response(),
    };

    let Ok(models) = db.list_installed(None) else {
        return Json(Vec::<InstalledModel>::new()).into_response();
    };

    let result: Vec<InstalledModel> = models
        .iter()
        .filter(|m| matches!(m.asset_type.as_str(), "checkpoint" | "lora"))
        .map(|m| InstalledModel {
            id: m.id.clone(),
            name: m.name.clone(),
            model_type: m.asset_type.clone(),
            variant: m.variant.clone(),
            size_bytes: m.size,
        })
        .collect();

    Json(result).into_response()
}

async fn api_get_run(Path(name): Path<String>) -> impl IntoResponse {
    match scan_training_run(&name) {
        Ok(run) => Json(run).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error scanning run: {e}"),
        )
            .into_response(),
    }
}

async fn api_training_status() -> impl IntoResponse {
    match training_status::get_all_status(false) {
        Ok(runs) => Json(runs).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error getting training status: {e}"),
        )
            .into_response(),
    }
}

async fn api_training_status_single(Path(name): Path<String>) -> impl IntoResponse {
    match training_status::get_status(&name) {
        Ok(run) => Json(run).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error getting training status: {e}"),
        )
            .into_response(),
    }
}

async fn api_list_outputs() -> impl IntoResponse {
    Json(scan_outputs_dir())
}

#[derive(Deserialize)]
struct DeleteOutputRequest {
    #[serde(default)]
    artifact_id: Option<String>,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Serialize)]
struct DeleteOutputResponse {
    deleted_file: bool,
    deleted_records: usize,
}

async fn api_delete_output(Json(req): Json<DeleteOutputRequest>) -> impl IntoResponse {
    let db = match Database::open() {
        Ok(db) => db,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to open database: {e}"),
            )
                .into_response();
        }
    };

    let mut target_file: Option<PathBuf> = None;
    let mut deleted_records = 0usize;

    if let Some(artifact_id) = req.artifact_id.as_deref()
        && !artifact_id.trim().is_empty()
    {
        match db.get_artifact_exact(artifact_id) {
            Ok(Some(artifact)) => {
                target_file = Some(PathBuf::from(&artifact.path));
                if let Err(e) = db.delete_artifact(artifact_id) {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to delete artifact row: {e}"),
                    )
                        .into_response();
                }
                deleted_records += 1;
            }
            Ok(None) => {}
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to query artifact: {e}"),
                )
                    .into_response();
            }
        }
    }

    if target_file.is_none()
        && let Some(rel_path) = req.path.as_deref()
    {
        if !rel_path.starts_with("outputs/") {
            return (StatusCode::BAD_REQUEST, "Path must be under outputs/").into_response();
        }
        target_file = Some(modl_root().join(rel_path));
    }

    let Some(target_file) = target_file else {
        return (StatusCode::BAD_REQUEST, "Missing artifact_id or path").into_response();
    };

    let outputs_root = modl_root().join("outputs");
    if !is_within_outputs_root(&target_file, &outputs_root) {
        return (StatusCode::FORBIDDEN, "Forbidden").into_response();
    }

    let deleted_file = if target_file.exists() {
        match std::fs::remove_file(&target_file) {
            Ok(()) => true,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to delete file: {e}"),
                )
                    .into_response();
            }
        }
    } else {
        false
    };

    let target_str = target_file.to_string_lossy().to_string();
    match db.delete_artifacts_by_path(&target_str) {
        Ok(n) => deleted_records += n,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to delete artifact rows by path: {e}"),
            )
                .into_response();
        }
    }

    (
        StatusCode::OK,
        Json(DeleteOutputResponse {
            deleted_file,
            deleted_records,
        }),
    )
        .into_response()
}

fn is_within_outputs_root(path: &FsPath, outputs_root: &FsPath) -> bool {
    if path.exists() {
        let Ok(path_canon) = path.canonicalize() else {
            return false;
        };
        let Ok(root_canon) = outputs_root.canonicalize() else {
            return false;
        };
        path_canon.starts_with(root_canon)
    } else {
        path.starts_with(outputs_root)
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

            Json(overview).into_response()
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

/// Serve bundled UI assets embedded at compile time.
async fn serve_ui_asset(Path(path): Path<String>) -> impl IntoResponse {
    match path.as_str() {
        "app.js" => (
            [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
            include_str!("dist/assets/app.js"),
        )
            .into_response(),
        "index.css" => (
            [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
            include_str!("dist/assets/index.css"),
        )
            .into_response(),
        _ => (StatusCode::NOT_FOUND, "Not found").into_response(),
    }
}

// ---------------------------------------------------------------------------
// Main HTML page (self-contained, no external deps except system fonts)
// ---------------------------------------------------------------------------

async fn index_page() -> Html<String> {
    Html(include_str!("dist/index.html").to_string())
}
