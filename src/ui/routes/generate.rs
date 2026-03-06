use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive},
    },
};
use futures_util::stream::{self, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::time::Duration;
use tokio::sync::broadcast;

use crate::core::db::Database;

use super::super::server::UiState;

#[derive(Deserialize, Clone)]
pub struct GenerateLoraRequest {
    pub id: String,
    pub strength: f32,
}

#[derive(Deserialize, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    pub model_id: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    pub num_images: u32,
    #[serde(default)]
    pub loras: Vec<GenerateLoraRequest>,
}

#[derive(Serialize)]
struct GenerateAcceptedResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    queue_length: Option<u32>,
}

#[derive(Deserialize)]
pub struct EnhanceApiRequest {
    prompt: String,
    #[serde(default)]
    model_hint: Option<String>,
    #[serde(default = "default_intensity")]
    intensity: String,
}

fn default_intensity() -> String {
    "moderate".to_string()
}

#[derive(Serialize)]
struct EnhanceApiResponse {
    original: String,
    enhanced: String,
    backend: String,
}

/// Run a single generation request, sending progress to the broadcast channel.
async fn run_single_generate(sender: &broadcast::Sender<String>, req: GenerateRequest) {
    eprintln!(
        "[generate] job started: model={} prompt={:?} size={}x{} steps={} seed={:?}",
        req.model_id,
        &req.prompt[..req.prompt.len().min(80)],
        req.width,
        req.height,
        req.steps,
        req.seed
    );

    if req
        .negative_prompt
        .as_deref()
        .is_some_and(|s| !s.trim().is_empty())
    {
        let _ =
            sender.send("warning: negative prompt is currently ignored by preview UI".to_string());
    }

    let size = format!("{}x{}", req.width, req.height);
    let lora_id = req.loras.first().map(|l| l.id.clone());
    let lora_strength = req.loras.first().map(|l| l.strength).unwrap_or(1.0);

    if let Some(ref id) = lora_id {
        eprintln!("[generate]   lora={id} strength={lora_strength}");
    }

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
        false, // no_worker: use persistent worker if available
        true,
    )
    .await;

    match run_result {
        Ok(()) => {
            eprintln!("[generate] job completed successfully");
            let _ = sender.send("completed".to_string());
        }
        Err(err) => {
            eprintln!("[generate] job failed: {err:#}");
            let _ = sender.send(format!("error: {err}"));
        }
    }
}

/// Background loop: process the initial request, then drain the queue.
async fn generate_loop(
    sender: broadcast::Sender<String>,
    inner: std::sync::Arc<tokio::sync::Mutex<super::super::server::GenerateInner>>,
    first_req: GenerateRequest,
) {
    run_single_generate(&sender, first_req).await;

    loop {
        let next = {
            let mut state = inner.lock().await;
            match state.queue.pop_front() {
                Some(req) => {
                    let remaining = state.queue.len();
                    drop(state);
                    let _ = sender.send(format!("queue:{remaining}"));
                    req
                }
                None => {
                    state.running = false;
                    drop(state);
                    let _ = sender.send("queue:empty".to_string());
                    break;
                }
            }
        };
        run_single_generate(&sender, next).await;
    }
}

pub async fn api_generate(
    State(state): State<UiState>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    // Preflight: validate model + runtime before accepting
    if let Err(err) = crate::core::preflight::for_generation(&req.model_id) {
        let msg = format!("{err:#}");
        eprintln!("[generate] preflight failed: {msg}");
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": msg })),
        )
            .into_response();
    }
    // Also validate LoRA references
    if let Some(first_lora) = req.loras.first() {
        let db = match Database::open() {
            Ok(db) => db,
            Err(e) => {
                let msg = format!("Database error: {e:#}");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": msg })),
                )
                    .into_response();
            }
        };
        let installed = db.list_installed(None).unwrap_or_default();
        let found = installed
            .iter()
            .any(|m| (m.id == first_lora.id || m.name == first_lora.id) && m.asset_type == "lora");
        if !found {
            let msg = format!("LoRA not found: {}", first_lora.id);
            eprintln!("[generate] preflight failed: {msg}");
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": msg })),
            )
                .into_response();
        }
    }

    let mut inner = state.generate_inner.lock().await;

    if inner.running {
        // Already generating — enqueue
        inner.queue.push_back(req);
        let pos = inner.queue.len();
        drop(inner);
        let _ = state.generate_events.send(format!("queue:{pos}"));
        eprintln!("[generate] enqueued (position {pos})");
        return (
            StatusCode::ACCEPTED,
            Json(GenerateAcceptedResponse {
                status: "queued".to_string(),
                queue_length: Some(pos as u32),
            }),
        )
            .into_response();
    }

    inner.running = true;
    drop(inner);

    let sender = state.generate_events.clone();
    let gen_inner = state.generate_inner.clone();
    let _ = sender.send("queued".to_string());

    tokio::spawn(async move {
        generate_loop(sender, gen_inner, req).await;
    });

    (
        StatusCode::ACCEPTED,
        Json(GenerateAcceptedResponse {
            status: "queued".to_string(),
            queue_length: Some(0),
        }),
    )
        .into_response()
}

pub async fn api_queue_status(State(state): State<UiState>) -> impl IntoResponse {
    let inner = state.generate_inner.lock().await;
    Json(serde_json::json!({
        "running": inner.running,
        "queue_length": inner.queue.len(),
    }))
}

pub async fn api_clear_queue(State(state): State<UiState>) -> impl IntoResponse {
    let mut inner = state.generate_inner.lock().await;
    let cleared = inner.queue.len();
    inner.queue.clear();
    drop(inner);
    let _ = state.generate_events.send("queue:0".to_string());
    eprintln!("[generate] queue cleared ({cleared} items)");
    Json(serde_json::json!({ "cleared": cleared }))
}

pub async fn api_generate_stream(
    State(state): State<UiState>,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let inner = state.generate_inner.lock().await;
    let running_now = inner.running;
    let queue_len = inner.queue.len();
    drop(inner);
    let initial = if running_now {
        if queue_len > 0 {
            format!("running:queue:{queue_len}")
        } else {
            "running".to_string()
        }
    } else {
        "idle".to_string()
    };

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

pub async fn api_enhance_prompt(Json(req): Json<EnhanceApiRequest>) -> impl IntoResponse {
    use crate::core::enhance::{self, EnhanceIntensity};

    let intensity: EnhanceIntensity = match req.intensity.parse() {
        Ok(i) => i,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, e.to_string()).into_response();
        }
    };

    match enhance::enhance_prompt(&req.prompt, req.model_hint.as_deref(), intensity) {
        Ok(result) => Json(EnhanceApiResponse {
            original: result.original,
            enhanced: result.enhanced,
            backend: result.backend,
        })
        .into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}
