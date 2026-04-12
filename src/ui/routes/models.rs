use axum::{Json, extract::Path, extract::Query, http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::auth::AuthStore;
use crate::core::config::Config;
use crate::core::db::Database;
use crate::core::install;
use crate::core::manifest::AssetType;
use crate::core::model_family;
use crate::core::registry::RegistryIndex;
use crate::core::store::Store;
use crate::core::training_status;

use crate::core::paths::modl_root;

#[derive(Serialize)]
pub struct InstalledModel {
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
    /// IDs of models that depend on this one (e.g. checkpoint requires this VAE)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    depended_on_by: Vec<String>,
    /// IDs of models this one requires
    #[serde(skip_serializing_if = "Vec::is_empty")]
    depends_on: Vec<DependencyRef>,
}

#[derive(Serialize)]
pub struct DependencyRef {
    id: String,
    #[serde(rename = "type")]
    dep_type: String,
    installed: bool,
}

/// Models needed for specific features but not tied to a single checkpoint.
#[derive(Serialize)]
pub struct FeatureDep {
    feature: String,
    description: String,
    model_type: String,
    installed: bool,
    install_hint: Option<String>,
}

#[derive(Serialize)]
struct ModelsResponse {
    models: Vec<InstalledModel>,
    total_size_bytes: u64,
    feature_deps: Vec<FeatureDep>,
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
            Err(_) => {
                return ModelsResponse {
                    models: Vec::new(),
                    total_size_bytes: 0,
                    feature_deps: Vec::new(),
                };
            }
        };

        let Ok(all_models) = db.list_installed(None) else {
            return ModelsResponse {
                models: Vec::new(),
                total_size_bytes: 0,
                feature_deps: Vec::new(),
            };
        };

        // Load registry for dependency info (best-effort)
        let registry = RegistryIndex::load().ok();

        // Build reverse dependency map: model_id -> list of models that require it
        let mut depended_on_by: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        // Build forward dependency map: model_id -> list of deps
        let mut depends_on_map: std::collections::HashMap<String, Vec<DependencyRef>> =
            std::collections::HashMap::new();

        let installed_ids: std::collections::HashSet<String> =
            all_models.iter().map(|m| m.id.clone()).collect();

        if let Some(ref idx) = registry {
            for m in &all_models {
                if let Some(manifest) = idx.find(&m.id) {
                    let mut deps = Vec::new();
                    for req in &manifest.requires {
                        depended_on_by
                            .entry(req.id.clone())
                            .or_default()
                            .push(m.id.clone());
                        deps.push(DependencyRef {
                            id: req.id.clone(),
                            dep_type: req.dep_type.to_string(),
                            installed: installed_ids.contains(&req.id),
                        });
                    }
                    if !deps.is_empty() {
                        depends_on_map.insert(m.id.clone(), deps);
                    }
                }
            }
        }

        let total_size_bytes: u64 = all_models.iter().map(|m| m.size).sum();

        let models: Vec<InstalledModel> = all_models
            .iter()
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
                    depended_on_by: depended_on_by.remove(&m.id).unwrap_or_default(),
                    depends_on: depends_on_map.remove(&m.id).unwrap_or_default(),
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
            .collect();

        // Feature dependencies: LLM/VL models needed for specific capabilities
        let has_vl = all_models.iter().any(|m| {
            m.id.contains("qwen") && m.asset_type == "text_encoder" || m.id.contains("vl")
        });
        let has_llm = all_models.iter().any(|m| m.asset_type == "llm");

        let mut feature_deps = Vec::new();
        if !has_vl {
            feature_deps.push(FeatureDep {
                feature: "Auto-captioning".to_string(),
                description: "Caption dataset images automatically (dataset caption, Studio)"
                    .to_string(),
                model_type: "Vision-Language model".to_string(),
                installed: false,
                install_hint: Some("modl pull qwen-vl".to_string()),
            });
        }
        if !has_llm {
            feature_deps.push(FeatureDep {
                feature: "Prompt enhance".to_string(),
                description: "AI-powered prompt expansion (enhance command, Studio)".to_string(),
                model_type: "LLM".to_string(),
                installed: false,
                install_hint: Some("Configure cloud LLM or Ollama".to_string()),
            });
        }

        ModelsResponse {
            models,
            total_size_bytes,
            feature_deps,
        }
    })
    .await
    .unwrap_or(ModelsResponse {
        models: Vec::new(),
        total_size_bytes: 0,
        feature_deps: Vec::new(),
    });
    Json(result)
}

// ---------------------------------------------------------------------------
// Registry search
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct SearchQuery {
    q: String,
    #[serde(rename = "type")]
    type_filter: Option<String>,
    #[serde(default)]
    all: Option<bool>,
}

#[derive(Serialize)]
struct SearchResult {
    id: String,
    name: String,
    model_type: String,
    author: Option<String>,
    description: Option<String>,
    size_bytes: u64,
    variants: Vec<SearchVariant>,
    installed: bool,
    requires_auth: bool,
}

#[derive(Serialize)]
struct SearchVariant {
    id: String,
    size_bytes: u64,
    precision: Option<String>,
}

pub async fn api_search_registry(Query(params): Query<SearchQuery>) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || {
        let index = match RegistryIndex::load() {
            Ok(idx) => idx,
            Err(_) => return Vec::new(),
        };
        let db = Database::open().ok();
        let installed_ids: HashSet<String> = db
            .as_ref()
            .and_then(|d| d.list_installed(None).ok())
            .map(|models| models.iter().map(|m| m.id.clone()).collect())
            .unwrap_or_default();

        let mut results: Vec<&crate::core::manifest::Manifest> = index.search(&params.q);

        // Filter to user-visible items by default
        if !params.all.unwrap_or(false) {
            results.retain(|m| m.visibility == "user");
        }

        // Filter by type if specified
        if let Some(ref type_filter) = params.type_filter {
            results.retain(|m| m.asset_type.to_string() == *type_filter);
        }

        results
            .into_iter()
            .take(30)
            .map(|m| {
                let size = if !m.variants.is_empty() {
                    m.variants[0].size
                } else {
                    m.file.as_ref().map(|f| f.size).unwrap_or(0)
                };
                SearchResult {
                    id: m.id.clone(),
                    name: m.name.clone(),
                    model_type: m.asset_type.to_string(),
                    author: m.author.clone(),
                    description: m.description.clone(),
                    size_bytes: size,
                    variants: m
                        .variants
                        .iter()
                        .map(|v| SearchVariant {
                            id: v.id.clone(),
                            size_bytes: v.size,
                            precision: v.precision.clone(),
                        })
                        .collect(),
                    installed: installed_ids.contains(&m.id),
                    requires_auth: m.auth.as_ref().is_some_and(|a| a.gated),
                }
            })
            .collect::<Vec<_>>()
    })
    .await
    .unwrap_or_default();

    Json(result)
}

// ---------------------------------------------------------------------------
// Install model
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct InstallRequest {
    id: String,
    variant: Option<String>,
}

pub async fn api_install_model(Json(req): Json<InstallRequest>) -> impl IntoResponse {
    // Database (rusqlite) is not Send, so we run the entire install inside
    // spawn_blocking and block_on the async download from there.
    let handle = tokio::runtime::Handle::current();
    let result = tokio::task::spawn_blocking(move || {
        let index = RegistryIndex::load().map_err(|e| format!("Registry not loaded: {e}"))?;
        let db = Database::open().map_err(|e| format!("DB error: {e}"))?;
        let config = Config::load().map_err(|e| format!("Config error: {e}"))?;
        let auth_store = AuthStore::load().unwrap_or_default();
        let store = Store::new(config.store_root());

        let (plan, vram) = install::resolve_plan(&req.id, req.variant.as_deref(), &index, &db)
            .map_err(|e| format!("Resolve error: {e}"))?;

        let mut installed = Vec::new();
        for item in &plan.items {
            if item.already_installed {
                continue;
            }
            let effective_variant = if item.manifest.id == req.id {
                item.variant_id.as_deref().or(req.variant.as_deref())
            } else {
                item.variant_id.as_deref()
            };
            match handle.block_on(install::install_item(
                &item.manifest,
                effective_variant,
                vram,
                &config,
                &store,
                &auth_store,
                &db,
                false,
            )) {
                Ok(result) => installed.push(result.name),
                Err(e) => {
                    return Err(format!("Failed to install {}: {e}", item.manifest.name));
                }
            }
        }

        Ok::<Vec<String>, String>(installed)
    })
    .await;

    match result {
        Ok(Ok(names)) => (
            StatusCode::OK,
            Json(serde_json::json!({ "installed": names })),
        ),
        Ok(Err(msg)) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": msg })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Internal error: {e}") })),
        ),
    }
}

// ---------------------------------------------------------------------------
// Delete model
// ---------------------------------------------------------------------------

/// Delete an installed model by ID.
pub async fn api_delete_model(Path(id): Path<String>) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
        let db = Database::open().map_err(|e| format!("DB error: {e}"))?;
        let config = Config::load().map_err(|e| format!("Config error: {e}"))?;

        // Check if it's a trained artifact first
        if let Ok(Some(artifact)) = db.find_artifact(&id) {
            let meta: serde_json::Value = artifact
                .metadata
                .as_deref()
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or(serde_json::Value::Null);

            let lora_name = meta
                .get("lora_name")
                .and_then(|v| v.as_str())
                .unwrap_or(&artifact.artifact_id)
                .to_string();

            // Remove symlink from loras dir
            let loras_dir = modl_root().join("loras");
            let symlink_path = loras_dir.join(format!("{lora_name}.safetensors"));
            if symlink_path.is_symlink() {
                std::fs::remove_file(&symlink_path).ok();
            }

            // Remove the store file
            let store_path = std::path::Path::new(&artifact.path);
            if store_path.exists() {
                std::fs::remove_file(store_path).ok();
                if let Some(parent) = store_path.parent() {
                    std::fs::remove_dir(parent).ok();
                }
            }

            // Remove training output
            let training_output_dir = modl_root().join("training_output").join(&lora_name);
            if training_output_dir.is_dir() {
                std::fs::remove_dir_all(&training_output_dir).ok();
            }
            if let Some(parent) = training_output_dir.parent() {
                for suffix in ["-config.yaml", ".log"] {
                    let stale = parent.join(format!("{lora_name}{suffix}"));
                    if stale.exists() {
                        std::fs::remove_file(&stale).ok();
                    }
                }
            }

            db.remove_installed(&artifact.artifact_id)
                .map_err(|e| format!("DB error: {e}"))?;
            db.delete_artifact(&artifact.artifact_id)
                .map_err(|e| format!("DB error: {e}"))?;
            db.delete_jobs_by_lora_name(&lora_name)
                .map_err(|e| format!("DB error: {e}"))?;

            return Ok(lora_name);
        }

        // Regular installed model
        if !db.is_installed(&id).map_err(|e| format!("DB error: {e}"))? {
            return Err(format!("'{id}' is not installed"));
        }

        let models = db
            .list_installed(None)
            .map_err(|e| format!("DB error: {e}"))?;
        if let Some(m) = models.iter().find(|m| m.id == id) {
            // Remove symlinks
            for target in &config.targets {
                if target.symlink {
                    let link_path = crate::compat::symlink_path(
                        &target.path,
                        &target.tool_type,
                        &m.asset_type
                            .parse::<AssetType>()
                            .unwrap_or(AssetType::Checkpoint),
                        &m.file_name,
                    );
                    if link_path.is_symlink() {
                        std::fs::remove_file(&link_path).ok();
                    }
                }
            }
        }

        db.remove_installed(&id)
            .map_err(|e| format!("DB error: {e}"))?;
        Ok(id)
    })
    .await;

    match result {
        Ok(Ok(name)) => (StatusCode::OK, Json(serde_json::json!({ "deleted": name }))),
        Ok(Err(msg)) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": msg })),
        ),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": "Internal error" })),
        ),
    }
}

/// GET /api/model-families — all model families with capabilities, params, and UI metadata.
pub async fn api_model_families() -> impl IntoResponse {
    Json(model_family::families())
}
