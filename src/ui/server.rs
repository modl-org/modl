use anyhow::{Context, Result};
use axum::Router;
use axum::routing::{delete, get, post};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::broadcast;

use super::routes::{
    analysis, civitai, datasets, files, generate, models, outputs, studio, training,
};

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct UiState {
    pub generate_events: broadcast::Sender<String>,
    pub generate_inner: Arc<tokio::sync::Mutex<GenerateInner>>,
    /// Per-session SSE event senders for studio.
    pub studio_events: Arc<tokio::sync::Mutex<HashMap<String, broadcast::Sender<String>>>>,
}

/// Internal generation state protected by a Mutex.
pub struct GenerateInner {
    pub running: bool,
    pub queue: VecDeque<generate::QueuedJob>,
    /// Summary of the currently-executing job (for queue panel display).
    pub current_summary: Option<generate::QueuedJobSummary>,
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
    let addr: std::net::SocketAddr = ([127, 0, 0, 1], port).into();
    if std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(200)).is_ok()
        && let Ok(output) = std::process::Command::new("lsof")
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
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

pub async fn start(port: u16, open_browser: bool) -> Result<()> {
    kill_existing_on_port(port);

    let (generate_events, _) = broadcast::channel(256);
    let state = UiState {
        generate_events,
        generate_inner: Arc::new(tokio::sync::Mutex::new(GenerateInner {
            running: false,
            queue: VecDeque::new(),
            current_summary: None,
        })),
        studio_events: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
    };

    // Background task: auto-advance training queue
    tokio::spawn(training_queue_loop());

    let app = Router::new()
        // Static / index
        .route("/", get(files::index_page))
        .route("/assets/{*path}", get(files::serve_ui_asset))
        .route("/files/{*path}", get(files::serve_file))
        // Models & GPU
        .route("/api/gpu", get(models::api_gpu_status))
        .route("/api/models", get(models::api_list_models))
        .route("/api/model-families", get(models::api_model_families))
        .route("/api/models/{id}", delete(models::api_delete_model))
        .route("/api/registry/search", get(models::api_search_registry))
        .route("/api/models/install", post(models::api_install_model))
        // CivitAI LoRA browsing
        .route("/api/civitai/loras", get(civitai::api_search_loras))
        .route("/api/civitai/install", post(civitai::api_install_lora))
        // File upload (img2img init images, masks)
        .route("/api/upload", post(files::api_upload))
        // Generation
        .route("/api/generate", post(generate::api_generate))
        .route("/api/generate/stream", get(generate::api_generate_stream))
        .route(
            "/api/generate/queue",
            get(generate::api_queue_status).delete(generate::api_clear_queue),
        )
        .route(
            "/api/generate/queue/{index}",
            delete(generate::api_cancel_queue_item),
        )
        .route("/api/enhance", post(generate::api_enhance_prompt))
        // Training
        .route("/api/runs", get(training::api_list_runs))
        .route("/api/runs/{name}", get(training::api_get_run))
        .route("/api/runs/{name}/loss", get(training::api_loss_history))
        .route("/api/runs/resume", post(training::api_resume_training))
        .route("/api/runs/start", post(training::api_start_training))
        .route("/api/runs/cancel", post(training::api_cancel_training))
        .route("/api/runs/delete", post(training::api_delete_run))
        // Training queue
        .route(
            "/api/train/queue",
            get(training::api_list_training_queue).post(training::api_add_to_training_queue),
        )
        .route(
            "/api/train/queue/{id}",
            delete(training::api_remove_from_training_queue),
        )
        .route(
            "/api/train/queue/{id}/position",
            axum::routing::put(training::api_reorder_training_queue),
        )
        .route("/api/status", get(training::api_training_status))
        .route(
            "/api/status/{name}",
            get(training::api_training_status_single),
        )
        // LoRA Library
        .route(
            "/api/library/loras",
            get(training::api_list_library_loras).post(training::api_promote_lora),
        )
        .route(
            "/api/library/loras/{id}",
            axum::routing::put(training::api_update_library_lora)
                .delete(training::api_delete_library_lora),
        )
        // Datasets
        .route("/api/datasets", get(datasets::api_list_datasets))
        .route("/api/datasets/{name}", get(datasets::api_get_dataset))
        // Outputs
        .route(
            "/api/outputs",
            get(outputs::api_list_outputs).delete(outputs::api_delete_output),
        )
        .route(
            "/api/outputs/batch-delete",
            post(outputs::api_batch_delete_outputs),
        )
        .route("/api/outputs/favorite", post(outputs::api_toggle_favorite))
        // Edit (shares generate queue + SSE stream)
        .route("/api/edit", post(generate::api_edit))
        // Analysis (upscale, remove-bg)
        .route("/api/analysis/upscale", post(analysis::api_upscale))
        .route("/api/analysis/remove-bg", post(analysis::api_remove_bg))
        // Studio
        .route(
            "/api/studio/sessions",
            get(studio::api_studio_list_sessions).post(studio::api_studio_create_session),
        )
        .route(
            "/api/studio/sessions/{id}",
            get(studio::api_studio_get_session).delete(studio::api_studio_delete_session),
        )
        .route(
            "/api/studio/sessions/{id}/images",
            post(studio::api_studio_upload_images),
        )
        .route(
            "/api/studio/sessions/{id}/start",
            post(studio::api_studio_start_session),
        )
        .route(
            "/api/studio/sessions/{id}/stream",
            get(studio::api_studio_stream),
        )
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = TcpListener::bind(addr)
        .await
        .context("Failed to bind to port")?;

    let url = format!("http://127.0.0.1:{port}");
    eprintln!("  Training preview UI running at {url}");
    eprintln!("  (also listening on 0.0.0.0:{port} for remote access)");
    eprintln!("  Press Ctrl+C to stop\n");

    if open_browser {
        let _ = open::that(&url);
    }

    axum::serve(listener, app).await.context("Server error")?;

    Ok(())
}

/// Background loop: every 30 seconds, check if any training is running.
/// If not, pop the next item from the training queue and start it.
async fn training_queue_loop() {
    use crate::core::db::Database;
    use crate::core::training_status;

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;

        // Check if any training is currently running
        let is_training = tokio::task::spawn_blocking(|| {
            training_status::get_all_status(false)
                .map(|runs| runs.iter().any(|r| r.is_running))
                .unwrap_or(false)
        })
        .await
        .unwrap_or(true); // Assume busy on error

        if is_training {
            continue;
        }

        // Pop next queue item and start it
        let item = tokio::task::spawn_blocking(|| {
            Database::open()
                .ok()
                .and_then(|db| db.pop_training_queue().ok().flatten())
        })
        .await
        .ok()
        .flatten();

        if let Some(item) = item {
            let spec: serde_json::Value = serde_json::from_str(&item.spec_json).unwrap_or_default();

            let modl_bin = std::env::current_exe().unwrap_or_else(|_| "modl".into());

            let _ = tokio::task::spawn_blocking(move || {
                let mut args = vec![
                    "train".to_string(),
                    "--dataset".to_string(),
                    spec["dataset"].as_str().unwrap_or("").to_string(),
                    "--base".to_string(),
                    spec["base_model"].as_str().unwrap_or("").to_string(),
                    "--name".to_string(),
                    spec["name"].as_str().unwrap_or(&item.name).to_string(),
                    "--trigger".to_string(),
                    spec["trigger_word"].as_str().unwrap_or("OHWX").to_string(),
                    "--lora-type".to_string(),
                    spec["lora_type"]
                        .as_str()
                        .unwrap_or("character")
                        .to_string(),
                ];

                if let Some(preset) = spec["preset"].as_str() {
                    args.push("--preset".to_string());
                    args.push(preset.to_string());
                }
                if let Some(steps) = spec["steps"].as_u64() {
                    args.push("--steps".to_string());
                    args.push(steps.to_string());
                }
                if let Some(rank) = spec["rank"].as_u64() {
                    args.push("--rank".to_string());
                    args.push(rank.to_string());
                }
                if let Some(lr) = spec["lr"].as_f64() {
                    args.push("--lr".to_string());
                    args.push(lr.to_string());
                }

                let log_path = crate::core::paths::modl_root().join("training.log");
                let stderr_file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&log_path)
                    .map(std::process::Stdio::from)
                    .unwrap_or_else(|_| std::process::Stdio::null());

                let _ = std::process::Command::new(&modl_bin)
                    .args(&args)
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(stderr_file)
                    .spawn();
            })
            .await;
        }
    }
}
