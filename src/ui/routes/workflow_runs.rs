use axum::{
    body::Body,
    extract::Path,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use std::io::Write;
use tokio::task;

use crate::core::db::Database;

/// GET /api/workflow-runs/:run_id/export.zip
///
/// Collects all artifacts for a workflow run and streams them as a ZIP archive.
/// Works with any HTTP client: curl, browser, `modl outputs export --server`.
pub async fn api_export_run_zip(Path(run_id): Path<String>) -> Response {
    let result = task::spawn_blocking(move || build_zip(&run_id)).await;

    match result {
        Ok(Ok((bytes, filename))) => {
            let headers = [
                (header::CONTENT_TYPE, "application/zip".to_string()),
                (
                    header::CONTENT_DISPOSITION,
                    format!("attachment; filename=\"{filename}\""),
                ),
            ];
            (headers, Body::from(bytes)).into_response()
        }
        Ok(Err(ZipError::NotFound(id))) => (
            StatusCode::NOT_FOUND,
            format!("No workflow run found: '{id}'"),
        )
            .into_response(),
        Ok(Err(ZipError::NoArtifacts(id))) => (
            StatusCode::ACCEPTED,
            format!("Run '{id}' exists but has no artifacts yet — try again later"),
        )
            .into_response(),
        Ok(Err(ZipError::Other(e))) => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "ZIP generation failed").into_response(),
    }
}

enum ZipError {
    NotFound(String),
    NoArtifacts(String),
    Other(String),
}

fn build_zip(run_id: &str) -> Result<(Vec<u8>, String), ZipError> {
    let db = Database::open().map_err(|e| ZipError::Other(e.to_string()))?;
    let jobs = db
        .list_jobs_by_run_id(run_id)
        .map_err(|e| ZipError::Other(e.to_string()))?;

    if jobs.is_empty() {
        return Err(ZipError::NotFound(run_id.to_string()));
    }

    let mut artifact_paths: Vec<String> = Vec::new();
    for job in &jobs {
        let arts = db.list_artifacts(Some(&job.job_id)).unwrap_or_default();
        for a in arts {
            if std::path::Path::new(&a.path).is_file() {
                artifact_paths.push(a.path);
            }
        }
    }

    if artifact_paths.is_empty() {
        return Err(ZipError::NoArtifacts(run_id.to_string()));
    }

    // Build ZIP in a tempfile; the finished archive is read into memory before
    // sending. Acceptable for typical run sizes; revisit if >1 GB runs appear.
    let tmp = tempfile::NamedTempFile::new().map_err(|e| ZipError::Other(e.to_string()))?;
    let mut zip = zip::ZipWriter::new(tmp);
    let options = zip::write::FileOptions::<()>::default()
        .compression_method(zip::CompressionMethod::Deflated);

    for (idx, path) in artifact_paths.iter().enumerate() {
        let p = std::path::Path::new(path);
        let base_name = p
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "file".to_string());
        // Prefix with index to avoid collisions when multiple steps produce
        // identically-named files.
        let entry_name = format!("{idx:03}_{base_name}");

        if let Ok(data) = std::fs::read(p) {
            let _ = zip.start_file(&entry_name, options);
            let _ = zip.write_all(&data);
        }
    }

    let tmp_done = zip.finish().map_err(|e| ZipError::Other(e.to_string()))?;
    let bytes = std::fs::read(tmp_done.path()).map_err(|e| ZipError::Other(e.to_string()))?;
    let filename = format!("{run_id}.zip");
    Ok((bytes, filename))
}
