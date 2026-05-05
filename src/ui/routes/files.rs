use axum::{
    Json,
    extract::{Multipart, Path, Query},
    http::{StatusCode, header},
    response::{Html, IntoResponse},
};
use include_dir::{Dir, include_dir};
use serde::Deserialize;

use crate::core::paths::modl_root;

static DIST_ASSETS: Dir = include_dir!("$CARGO_MANIFEST_DIR/src/ui/dist/assets");

/// Serve files from ~/.modl/ (images, samples, etc.)
pub async fn serve_file(
    Path(path): Path<String>,
    Query(params): Query<ThumbParams>,
) -> impl IntoResponse {
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

    // If ?w= is requested, serve a cached thumbnail
    if let Some(w) = params.w {
        let w = w.clamp(32, 1920);
        return serve_thumbnail(&canonical, w).await;
    }

    match tokio::fs::read(&canonical).await {
        Ok(bytes) => {
            let ext = canonical.extension().and_then(|e| e.to_str()).unwrap_or("");
            let content_type = match ext {
                "jpg" | "jpeg" => "image/jpeg",
                "png" => "image/png",
                "webp" => "image/webp",
                "yaml" | "yml" => "text/plain; charset=utf-8",
                "json" => "application/json",
                "safetensors" => "application/octet-stream",
                _ => "application/octet-stream",
            };
            let cache = match ext {
                "jpg" | "jpeg" | "png" | "webp" => "public, max-age=86400",
                _ => "no-cache",
            };
            (
                [
                    (header::CONTENT_TYPE, content_type),
                    (header::CACHE_CONTROL, cache),
                ],
                bytes,
            )
                .into_response()
        }
        Err(_) => (StatusCode::NOT_FOUND, "Not found").into_response(),
    }
}

#[derive(Deserialize)]
pub struct ThumbParams {
    /// Desired thumbnail width in pixels (height scales proportionally).
    w: Option<u32>,
}

/// Generate (or serve cached) a WebP thumbnail at the given width.
async fn serve_thumbnail(source: &std::path::Path, width: u32) -> axum::response::Response {
    let cache_dir = modl_root().join("cache").join("thumbs");
    // Build cache key from source path hash + width
    let hash_input = format!("{}:{}", source.to_string_lossy(), width);
    let hash = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(hash_input.as_bytes());
        format!("{:x}", h.finalize())
    };
    let cache_path = cache_dir.join(format!("{}.webp", &hash[..16]));

    // Check if source is newer than cached thumb
    let cache_valid = if cache_path.exists() {
        match (
            source.metadata().and_then(|m| m.modified()),
            cache_path.metadata().and_then(|m| m.modified()),
        ) {
            (Ok(src_mtime), Ok(cache_mtime)) => cache_mtime >= src_mtime,
            _ => false,
        }
    } else {
        false
    };

    if cache_valid && let Ok(bytes) = tokio::fs::read(&cache_path).await {
        return (
            [
                (header::CONTENT_TYPE, "image/webp"),
                (header::CACHE_CONTROL, "public, max-age=86400"),
            ],
            bytes,
        )
            .into_response();
    }

    // Generate thumbnail in a blocking task
    let source_owned = source.to_path_buf();
    let cache_path_owned = cache_path.clone();
    let result = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
        let img = image::open(&source_owned).map_err(|e| format!("Failed to open image: {e}"))?;
        let thumb = img.thumbnail(width, width);

        // Ensure cache dir exists
        if let Some(parent) = cache_path_owned.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        // Encode as WebP (lossy, ~quality 80 equivalent — smaller + sharper than JPEG)
        let mut buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        thumb
            .write_to(&mut cursor, image::ImageFormat::WebP)
            .map_err(|e| format!("Failed to encode thumbnail: {e}"))?;

        // Write cache file (best effort)
        let _ = std::fs::write(&cache_path_owned, &buf);

        Ok(buf)
    })
    .await;

    match result {
        Ok(Ok(bytes)) => (
            [
                (header::CONTENT_TYPE, "image/webp"),
                (header::CACHE_CONTROL, "public, max-age=86400"),
            ],
            bytes,
        )
            .into_response(),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task failed: {e}"),
        )
            .into_response(),
    }
}

/// Serve bundled UI assets embedded at compile time.
pub async fn serve_ui_asset(Path(path): Path<String>) -> impl IntoResponse {
    let content_type = match path.rsplit('.').next().unwrap_or("") {
        "js" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        _ => "application/octet-stream",
    };

    match DIST_ASSETS.get_file(&path) {
        Some(file) => ([(header::CONTENT_TYPE, content_type)], file.contents()).into_response(),
        None => (StatusCode::NOT_FOUND, "Not found").into_response(),
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
