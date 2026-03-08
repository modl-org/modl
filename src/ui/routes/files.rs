use axum::{
    Json,
    extract::{Multipart, Path},
    http::{StatusCode, header},
    response::{Html, IntoResponse},
};

use super::super::server::modl_root;

/// Serve files from ~/.modl/ (images, samples, etc.)
pub async fn serve_file(Path(path): Path<String>) -> impl IntoResponse {
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
pub async fn serve_ui_asset(Path(path): Path<String>) -> impl IntoResponse {
    match path.as_str() {
        "app.js" => (
            [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
            include_str!("../dist/assets/app.js"),
        )
            .into_response(),
        "index.css" => (
            [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
            include_str!("../dist/assets/index.css"),
        )
            .into_response(),
        _ => (StatusCode::NOT_FOUND, "Not found").into_response(),
    }
}

pub async fn index_page() -> Html<String> {
    Html(include_str!("../dist/index.html").to_string())
}

/// Accept a file upload (multipart), save to ~/.modl/tmp/, return the server path.
/// Used by the UI to upload init images for img2img / inpainting masks.
pub async fn api_upload(mut multipart: Multipart) -> impl IntoResponse {
    let tmp_dir = modl_root().join("tmp");
    if let Err(e) = tokio::fs::create_dir_all(&tmp_dir).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Failed to create tmp dir: {e}") })),
        )
            .into_response();
    }

    let field = match multipart.next_field().await {
        Ok(Some(f)) => f,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "No file in upload" })),
            )
                .into_response();
        }
    };

    let original_name = field.file_name().unwrap_or("upload.png").to_string();

    // Sanitize filename: keep only the extension
    let ext = std::path::Path::new(&original_name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("png");

    // Validate it's an image extension
    if !matches!(ext, "png" | "jpg" | "jpeg" | "webp" | "bmp" | "tiff") {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("Unsupported file type: .{ext}") })),
        )
            .into_response();
    }

    let bytes = match field.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("Failed to read upload: {e}") })),
            )
                .into_response();
        }
    };

    // Size guard: 50MB max
    if bytes.len() > 50 * 1024 * 1024 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "File too large (max 50MB)" })),
        )
            .into_response();
    }

    let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S%.3f");
    let filename = format!("{timestamp}.{ext}");
    let dest = tmp_dir.join(&filename);

    if let Err(e) = tokio::fs::write(&dest, &bytes).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Failed to write file: {e}") })),
        )
            .into_response();
    }

    let server_path = dest.to_string_lossy().to_string();
    (
        StatusCode::OK,
        Json(serde_json::json!({ "path": server_path })),
    )
        .into_response()
}
