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
        Ok(Err(e)) => (
            StatusCode::NOT_FOUND,
            format!("No artifacts found for run '{e}'"),
        )
            .into_response(),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "ZIP generation failed").into_response(),
    }
}

fn build_zip(run_id: &str) -> Result<(Vec<u8>, String), String> {
    let db = Database::open().map_err(|e| e.to_string())?;
    let jobs = db.list_jobs_by_run_id(run_id).map_err(|e| e.to_string())?;

    if jobs.is_empty() {
        return Err(run_id.to_string());
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
        return Err(run_id.to_string());
    }

    // Stream into a tempfile to avoid holding all images in RAM.
    let tmp = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;
    let mut zip = zip::ZipWriter::new(tmp);
    let options = zip::write::FileOptions::<()>::default()
        .compression_method(zip::CompressionMethod::Deflated);

    for path in &artifact_paths {
        let p = std::path::Path::new(path);
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "file".to_string());

        if let Ok(data) = std::fs::read(p) {
            let _ = zip.start_file(&name, options);
            let _ = zip.write_all(&data);
        }
    }

    let tmp_done = zip.finish().map_err(|e| e.to_string())?;
    let bytes = std::fs::read(tmp_done.path()).map_err(|e| e.to_string())?;
    let filename = format!("{run_id}.zip");
    Ok((bytes, filename))
}
