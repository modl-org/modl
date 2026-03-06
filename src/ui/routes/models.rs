use axum::{Json, response::IntoResponse};
use serde::Serialize;

use crate::core::db::Database;
use crate::core::training_status;

use super::super::server::modl_root;

#[derive(Serialize)]
struct InstalledModel {
    id: String,
    name: String,
    model_type: String,
    variant: Option<String>,
    size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    trigger_word: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    base_model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_image_url: Option<String>,
}

#[derive(Serialize)]
struct GpuStatus {
    name: Option<String>,
    vram_total_mb: Option<u64>,
    vram_free_mb: Option<u64>,
    training_active: bool,
}

pub async fn api_gpu_status() -> impl IntoResponse {
    let status = tokio::task::spawn_blocking(|| {
        let training_active = training_status::get_all_status(true)
            .map(|runs| runs.iter().any(|r| r.is_running))
            .unwrap_or(false);

        let (name, vram_total_mb, vram_free_mb) = if let Ok(nvml) = nvml_wrapper::Nvml::init() {
            if let Ok(device) = nvml.device_by_index(0) {
                let name = device.name().ok();
                let mem = device.memory_info().ok();
                (
                    name,
                    mem.as_ref().map(|m| m.total / (1024 * 1024)),
                    mem.as_ref().map(|m| m.free / (1024 * 1024)),
                )
            } else {
                (None, None, None)
            }
        } else {
            (None, None, None)
        };

        GpuStatus {
            name,
            vram_total_mb,
            vram_free_mb,
            training_active,
        }
    })
    .await
    .unwrap_or(GpuStatus {
        name: None,
        vram_total_mb: None,
        vram_free_mb: None,
        training_active: false,
    });
    Json(status)
}

pub async fn api_list_models() -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(|| {
        let db = match Database::open() {
            Ok(db) => db,
            Err(_) => return Vec::new(),
        };

        let Ok(models) = db.list_installed(None) else {
            return Vec::new();
        };

        models
            .iter()
            .filter(|m| {
                matches!(
                    m.asset_type.as_str(),
                    "checkpoint" | "diffusion_model" | "lora"
                )
            })
            .map(|m| {
                let mut model = InstalledModel {
                    id: m.id.clone(),
                    name: m.name.clone(),
                    model_type: m.asset_type.clone(),
                    variant: m.variant.clone(),
                    size_bytes: m.size,
                    trigger_word: None,
                    base_model_id: None,
                    sample_image_url: None,
                };

                // Enrich LoRAs with artifact metadata + sample image
                if m.asset_type == "lora" {
                    if let Ok(Some(artifact)) = db.find_artifact(&m.id)
                        && let Some(ref meta_str) = artifact.metadata
                        && let Ok(meta) = serde_json::from_str::<serde_json::Value>(meta_str)
                    {
                        model.trigger_word = meta
                            .get("trigger_word")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        model.base_model_id = meta
                            .get("base_model")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                    }

                    let lora_name = if m.id.starts_with("train:") {
                        m.id.split(':').nth(1).map(|s| s.to_string())
                    } else {
                        None
                    };
                    if let Some(name) = &lora_name {
                        let samples_dir = modl_root()
                            .join("training_output")
                            .join(name)
                            .join(name)
                            .join("samples");
                        if samples_dir.exists()
                            && let Ok(entries) = std::fs::read_dir(&samples_dir)
                        {
                            let mut images: Vec<String> = entries
                                .filter_map(|e| e.ok())
                                .filter(|e| {
                                    e.path()
                                        .extension()
                                        .is_some_and(|ext| ext == "jpg" || ext == "png")
                                })
                                .map(|e| e.file_name().to_string_lossy().to_string())
                                .collect();
                            images.sort();
                            if let Some(last) = images.last() {
                                model.sample_image_url =
                                    Some(format!("training_output/{name}/{name}/samples/{last}"));
                            }
                        }
                    }
                }

                model
            })
            .collect()
    })
    .await
    .unwrap_or_default();
    Json(result)
}
