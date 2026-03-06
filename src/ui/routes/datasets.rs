use axum::{
    Json,
    extract::{Path, Query},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};

use crate::core::dataset;

#[derive(Serialize)]
struct DatasetOverview {
    name: String,
    image_count: u32,
    captioned_count: u32,
    coverage: f32,
    images: Vec<DatasetImage>,
}

#[derive(Serialize)]
struct DatasetImage {
    filename: String,
    caption: Option<String>,
    image_url: String,
}

#[derive(Deserialize)]
pub struct DatasetQuery {
    #[serde(default = "default_page_size")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

fn default_page_size() -> usize {
    50
}

pub async fn api_list_datasets() -> impl IntoResponse {
    match tokio::task::spawn_blocking(dataset::list).await {
        Ok(Ok(datasets)) => {
            let names: Vec<String> = datasets.iter().map(|d| d.name.clone()).collect();
            Json(names).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error listing datasets: {e}"),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task failed: {e}"),
        )
            .into_response(),
    }
}

pub async fn api_get_dataset(
    Path(name): Path<String>,
    Query(q): Query<DatasetQuery>,
) -> impl IntoResponse {
    match tokio::task::spawn_blocking(move || {
        let ds_path = dataset::resolve_path(&name);

        dataset::scan(&ds_path).map(|info| {
            let mut images: Vec<DatasetImage> = Vec::new();
            let page = info.images.iter().skip(q.offset).take(q.limit);

            for entry in page {
                let fname = entry
                    .path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string();
                let caption = entry
                    .caption_path
                    .as_ref()
                    .and_then(|p| std::fs::read_to_string(p).ok())
                    .map(|s| s.trim().to_string());
                let image_url = format!("datasets/{name}/{fname}");
                images.push(DatasetImage {
                    filename: fname,
                    caption,
                    image_url,
                });
            }

            DatasetOverview {
                name: info.name,
                image_count: info.image_count,
                captioned_count: info.captioned_count,
                coverage: info.caption_coverage,
                images,
            }
        })
    })
    .await
    {
        Ok(Ok(overview)) => Json(overview).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error scanning dataset: {e}"),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task failed: {e}"),
        )
            .into_response(),
    }
}
