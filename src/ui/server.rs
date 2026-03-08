use anyhow::{Context, Result};
use axum::Router;
use axum::routing::{delete, get, post};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::broadcast;

use super::routes::{datasets, files, generate, models, outputs, studio, training};

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
    pub queue: VecDeque<generate::GenerateRequest>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn modl_root() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
}

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
        })),
        studio_events: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
    };

    let app = Router::new()
        // Static / index
        .route("/", get(files::index_page))
        .route("/assets/{*path}", get(files::serve_ui_asset))
        .route("/files/{*path}", get(files::serve_file))
        // Models & GPU
        .route("/api/gpu", get(models::api_gpu_status))
        .route("/api/models", get(models::api_list_models))
        .route("/api/models/{id}", delete(models::api_delete_model))
        .route("/api/registry/search", get(models::api_search_registry))
        .route("/api/models/install", post(models::api_install_model))
        // Generation
        .route("/api/generate", post(generate::api_generate))
        .route("/api/generate/stream", get(generate::api_generate_stream))
        .route(
            "/api/generate/queue",
            get(generate::api_queue_status).delete(generate::api_clear_queue),
        )
        .route("/api/enhance", post(generate::api_enhance_prompt))
        // Training
        .route("/api/runs", get(training::api_list_runs))
        .route("/api/runs/{name}", get(training::api_get_run))
        .route("/api/runs/{name}/loss", get(training::api_loss_history))
        .route("/api/runs/resume", post(training::api_resume_training))
        .route("/api/runs/cancel", post(training::api_cancel_training))
        .route("/api/runs/delete", post(training::api_delete_run))
        .route("/api/status", get(training::api_training_status))
        .route(
            "/api/status/{name}",
            get(training::api_training_status_single),
        )
        // Datasets
        .route("/api/datasets", get(datasets::api_list_datasets))
        .route("/api/datasets/{name}", get(datasets::api_get_dataset))
        // Outputs
        .route(
            "/api/outputs",
            get(outputs::api_list_outputs).delete(outputs::api_delete_output),
        )
        .route("/api/outputs/favorite", post(outputs::api_toggle_favorite))
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
