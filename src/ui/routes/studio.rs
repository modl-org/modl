use axum::{
    Json,
    extract::{Multipart, Path, State},
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

use super::super::server::{UiState, modl_root};

#[derive(Deserialize)]
pub struct CreateSessionRequest {
    intent: String,
}

#[derive(Serialize)]
struct StudioSessionResponse {
    id: String,
    intent: String,
    status: String,
    input_images: Vec<String>,
    output_images: Vec<String>,
    events: Vec<serde_json::Value>,
    created_at: String,
    completed_at: Option<String>,
}

pub async fn api_studio_create_session(Json(req): Json<CreateSessionRequest>) -> impl IntoResponse {
    let session_id = uuid::Uuid::new_v4().to_string();

    match tokio::task::spawn_blocking({
        let session_id = session_id.clone();
        let intent = req.intent.clone();
        move || {
            let db = Database::open()?;
            db.create_studio_session(&session_id, &intent)?;

            let session_dir = modl_root().join("studio").join(&session_id);
            std::fs::create_dir_all(session_dir.join("inputs"))?;
            std::fs::create_dir_all(session_dir.join("outputs"))?;

            Ok::<_, anyhow::Error>(())
        }
    })
    .await
    {
        Ok(Ok(())) => (
            StatusCode::CREATED,
            Json(serde_json::json!({
                "id": session_id,
                "status": "pending",
            })),
        )
            .into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create session: {e}"),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task failed: {e}"),
        )
            .into_response(),
    }
}

pub async fn api_studio_list_sessions() -> impl IntoResponse {
    match tokio::task::spawn_blocking(|| {
        let db = Database::open()?;
        let sessions = db.list_studio_sessions()?;
        let mut results = Vec::new();
        for s in sessions {
            let input_images = db
                .get_session_images(&s.id, Some("input"))
                .unwrap_or_default();
            let output_images = db
                .get_session_images(&s.id, Some("output"))
                .unwrap_or_default();
            results.push(StudioSessionResponse {
                id: s.id,
                intent: s.intent,
                status: s.status,
                input_images,
                output_images,
                events: vec![],
                created_at: s.created_at,
                completed_at: s.completed_at,
            });
        }
        Ok::<_, anyhow::Error>(results)
    })
    .await
    {
        Ok(Ok(sessions)) => Json(sessions).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error listing sessions: {e}"),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task failed: {e}"),
        )
            .into_response(),
    }
}

pub async fn api_studio_get_session(Path(id): Path<String>) -> impl IntoResponse {
    match tokio::task::spawn_blocking(move || {
        let db = Database::open()?;
        let session = db
            .get_studio_session(&id)?
            .ok_or_else(|| anyhow::anyhow!("Session not found: {id}"))?;
        let input_images = db
            .get_session_images(&id, Some("input"))
            .unwrap_or_default();
        let output_images = db
            .get_session_images(&id, Some("output"))
            .unwrap_or_default();
        let events: Vec<serde_json::Value> = db
            .get_session_events(&id)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|e| serde_json::from_str(&e.event_json).ok())
            .collect();

        Ok::<_, anyhow::Error>(StudioSessionResponse {
            id: session.id,
            intent: session.intent,
            status: session.status,
            input_images,
            output_images,
            events,
            created_at: session.created_at,
            completed_at: session.completed_at,
        })
    })
    .await
    {
        Ok(Ok(session)) => Json(session).into_response(),
        Ok(Err(e)) => {
            let msg = e.to_string();
            let status = if msg.contains("not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, msg).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task failed: {e}"),
        )
            .into_response(),
    }
}

pub async fn api_studio_delete_session(Path(id): Path<String>) -> impl IntoResponse {
    match tokio::task::spawn_blocking(move || {
        let db = Database::open()?;
        db.delete_studio_session(&id)?;
        let session_dir = modl_root().join("studio").join(&id);
        if session_dir.exists() {
            let _ = std::fs::remove_dir_all(&session_dir);
        }
        Ok::<_, anyhow::Error>(())
    })
    .await
    {
        Ok(Ok(())) => Json(serde_json::json!({ "deleted": true })).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to delete session: {e}"),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task failed: {e}"),
        )
            .into_response(),
    }
}

pub async fn api_studio_upload_images(
    Path(id): Path<String>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let inputs_dir = modl_root().join("studio").join(&id).join("inputs");
    if !inputs_dir.exists() {
        return (StatusCode::NOT_FOUND, "Session not found").into_response();
    }

    let mut uploaded = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let filename = field
            .file_name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("image_{}.jpg", uploaded.len()));

        let data = match field.bytes().await {
            Ok(d) => d,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to read upload: {e}"),
                )
                    .into_response();
            }
        };

        let dest = inputs_dir.join(&filename);
        if let Err(e) = tokio::fs::write(&dest, &data).await {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to save file: {e}"),
            )
                .into_response();
        }

        let rel_path = format!("studio/{id}/inputs/{filename}");
        uploaded.push(rel_path.clone());

        let id_clone = id.clone();
        let _ = tokio::task::spawn_blocking(move || {
            if let Ok(db) = Database::open() {
                let _ = db.insert_session_image(&id_clone, &rel_path, "input");
            }
        })
        .await;
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "uploaded": uploaded.len(),
            "images": uploaded,
        })),
    )
        .into_response()
}

pub async fn api_studio_start_session(
    State(state): State<UiState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let session_data = match tokio::task::spawn_blocking({
        let id = id.clone();
        move || {
            let db = Database::open()?;
            let session = db
                .get_studio_session(&id)?
                .ok_or_else(|| anyhow::anyhow!("Session not found"))?;
            let input_images = db.get_session_images(&id, Some("input"))?;
            db.update_studio_session_status(&id, "running")?;
            Ok::<_, anyhow::Error>((session, input_images))
        }
    })
    .await
    {
        Ok(Ok(data)) => data,
        Ok(Err(e)) => {
            return (StatusCode::NOT_FOUND, format!("{e}")).into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Task failed: {e}"),
            )
                .into_response();
        }
    };

    let (session_record, input_images) = session_data;

    let (event_tx, _) = broadcast::channel::<String>(256);
    {
        let mut events = state.studio_events.lock().await;
        events.insert(id.clone(), event_tx.clone());
    }

    let root = modl_root();
    let image_paths: Vec<std::path::PathBuf> = input_images
        .iter()
        .map(|p| root.join(p))
        .filter(|p| p.exists())
        .collect();

    let mut agent_session = crate::core::agent::AgentSession {
        id: id.clone(),
        intent: session_record.intent.clone(),
        images: image_paths,
        events: vec![],
        status: crate::core::agent::SessionStatus::Running,
    };

    let llm = match crate::core::llm::resolve_backend(false) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to resolve LLM backend: {e}"),
            )
                .into_response();
        }
    };

    let studio_events = state.studio_events.clone();
    let session_id = id.clone();
    tokio::spawn(async move {
        let result =
            crate::core::agent::run_session(&mut agent_session, llm.as_ref(), event_tx.clone())
                .await;

        let final_status = match result {
            Ok(()) => "completed",
            Err(ref e) => {
                eprintln!("[studio] Session {session_id} failed: {e}");
                "failed"
            }
        };

        if let Ok(db) = Database::open() {
            let _ = db.update_studio_session_status(&session_id, final_status);

            for (i, event) in agent_session.events.iter().enumerate() {
                let json = serde_json::to_string(event).unwrap_or_default();
                let _ = db.insert_session_event(&session_id, i as u32, &json);
            }
        }

        let mut events = studio_events.lock().await;
        events.remove(&session_id);
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "status": "running",
            "session_id": id,
        })),
    )
        .into_response()
}

pub async fn api_studio_stream(
    State(state): State<UiState>,
    Path(id): Path<String>,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let rx = {
        let events = state.studio_events.lock().await;
        events.get(&id).map(|tx| tx.subscribe())
    };

    let initial_event = if rx.is_some() {
        "connected"
    } else {
        "session_not_active"
    };

    let first =
        stream::once(async move { Ok::<Event, Infallible>(Event::default().data(initial_event)) });

    let combined: std::pin::Pin<
        Box<dyn Stream<Item = std::result::Result<Event, Infallible>> + Send>,
    > = if let Some(rx) = rx {
        let updates = stream::unfold(rx, |mut rx| async move {
            match rx.recv().await {
                Ok(msg) => Some((Ok(Event::default().data(msg)), rx)),
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    let msg =
                        format!("{{\"type\":\"error\",\"message\":\"skipped {skipped} events\"}}");
                    Some((Ok(Event::default().data(msg)), rx))
                }
                Err(broadcast::error::RecvError::Closed) => None,
            }
        });
        Box::pin(first.chain(updates))
    } else {
        Box::pin(first)
    };

    Sse::new(combined).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(5))
            .text("keepalive"),
    )
}
