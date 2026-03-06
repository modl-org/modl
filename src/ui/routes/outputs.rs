use axum::{Json, http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};

use crate::core::outputs as output_service;

pub async fn api_list_outputs() -> impl IntoResponse {
    let outputs = tokio::task::spawn_blocking(output_service::list_outputs)
        .await
        .unwrap_or_default();
    Json(outputs)
}

#[derive(Deserialize)]
pub struct DeleteOutputRequest {
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

pub async fn api_delete_output(Json(req): Json<DeleteOutputRequest>) -> impl IntoResponse {
    match tokio::task::spawn_blocking(move || {
        output_service::delete_output(req.artifact_id.as_deref(), req.path.as_deref())
    })
    .await
    {
        Ok(Ok(result)) => (
            StatusCode::OK,
            Json(DeleteOutputResponse {
                deleted_file: result.deleted_file,
                deleted_records: result.deleted_records,
            }),
        )
            .into_response(),
        Ok(Err(e)) => {
            let msg = e.to_string();
            let status = if msg.contains("must be under") || msg.contains("Missing") {
                StatusCode::BAD_REQUEST
            } else if msg.contains("within the outputs") {
                StatusCode::FORBIDDEN
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

#[derive(Deserialize)]
pub struct ToggleFavoriteRequest {
    path: String,
}

#[derive(Serialize)]
struct ToggleFavoriteResponse {
    favorited: bool,
}

pub async fn api_toggle_favorite(Json(req): Json<ToggleFavoriteRequest>) -> impl IntoResponse {
    match tokio::task::spawn_blocking(move || output_service::toggle_favorite(&req.path)).await {
        Ok(Ok(result)) => Json(ToggleFavoriteResponse {
            favorited: result.favorited,
        })
        .into_response(),
        Ok(Err(e)) => {
            let msg = e.to_string();
            let status = if msg.contains("must be under") {
                StatusCode::BAD_REQUEST
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
