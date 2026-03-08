use axum::{
    extract::Path,
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
