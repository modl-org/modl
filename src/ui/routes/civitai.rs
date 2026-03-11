use axum::Json;
use axum::extract::Query;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

use crate::auth::AuthStore;
use crate::core::civitai;
use crate::core::config::Config;
use crate::core::db::{Database, InstalledModelRecord};
use crate::core::download;
use crate::core::store::Store;

// ---------------------------------------------------------------------------
// GET /api/civitai/loras — search CivitAI for LoRAs (proxied to avoid CORS)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct SearchParams {
    #[serde(default)]
    query: String,
    #[serde(default)]
    base_model: Option<String>,
    #[serde(default)]
    sort: Option<String>,
    #[serde(default = "default_page")]
    page: u32,
    #[serde(default = "default_limit")]
    limit: u32,
    #[serde(default)]
    nsfw: Option<bool>,
}

fn default_page() -> u32 {
    1
}
fn default_limit() -> u32 {
    20
}

#[derive(Serialize)]
struct SearchResult {
    items: Vec<LoraItem>,
    total_items: u64,
    total_pages: u64,
    current_page: u64,
}

#[derive(Serialize)]
struct LoraItem {
    /// CivitAI model ID
    id: u64,
    name: String,
    creator: Option<String>,
    /// Best version
    version_id: u64,
    version_name: String,
    base_model: Option<String>,
    /// Normalized base model for modl compatibility
    base_model_hint: Option<String>,
    trigger_words: Vec<String>,
    /// First SFW thumbnail URL
    thumbnail: Option<String>,
    /// File info
    file_name: Option<String>,
    file_size_kb: Option<f64>,
    file_format: Option<String>,
    sha256: Option<String>,
    /// Stats
    download_count: u64,
    rating: Option<f64>,
    rating_count: u64,
    /// Whether this is already installed locally
    installed: bool,
}

pub async fn api_search_loras(Query(params): Query<SearchParams>) -> impl IntoResponse {
    let result = civitai::search_loras(
        &params.query,
        params.base_model.as_deref(),
        params.sort.as_deref(),
        params.page,
        params.limit,
        params.nsfw.unwrap_or(false),
    )
    .await;

    match result {
        Ok(resp) => {
            // Check which are already installed
            let db = Database::open().ok();

            let items: Vec<LoraItem> = resp
                .items
                .into_iter()
                .filter_map(|model| {
                    let version = model.model_versions.first()?;
                    let file = version.files.first();
                    let civitai_id = format!("civitai:{}", version.id);
                    let installed = db
                        .as_ref()
                        .and_then(|d| d.is_installed(&civitai_id).ok())
                        .unwrap_or(false);

                    // Pick first SFW thumbnail (nsfw_level <= 1)
                    let thumbnail = version
                        .images
                        .iter()
                        .find(|img| img.nsfw_level.unwrap_or(0) <= 1)
                        .or_else(|| version.images.first())
                        .and_then(|img| img.url.clone());

                    let base_model_hint = version
                        .base_model
                        .as_deref()
                        .map(civitai::normalize_base_model);

                    Some(LoraItem {
                        id: model.id,
                        name: model.name,
                        creator: model.creator.map(|c| c.username),
                        version_id: version.id,
                        version_name: version.name.clone(),
                        base_model: version.base_model.clone(),
                        base_model_hint,
                        trigger_words: version.trained_words.clone().unwrap_or_default(),
                        thumbnail,
                        file_name: file.map(|f| f.name.clone()),
                        file_size_kb: file.and_then(|f| f.size_kb),
                        file_format: file
                            .and_then(|f| f.metadata.as_ref().and_then(|m| m.format.clone())),
                        sha256: file.and_then(|f| f.hashes.as_ref().and_then(|h| h.sha256.clone())),
                        download_count: model
                            .stats
                            .as_ref()
                            .and_then(|s| s.download_count)
                            .unwrap_or(0),
                        rating: model.stats.as_ref().and_then(|s| s.rating),
                        rating_count: model
                            .stats
                            .as_ref()
                            .and_then(|s| s.rating_count)
                            .unwrap_or(0),
                        installed,
                    })
                })
                .collect();

            (
                StatusCode::OK,
                Json(serde_json::json!(SearchResult {
                    total_items: resp.metadata.total_items.unwrap_or(0),
                    total_pages: resp.metadata.total_pages.unwrap_or(0),
                    current_page: resp.metadata.current_page,
                    items,
                })),
            )
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({ "error": format!("{e}") })),
        ),
    }
}

// ---------------------------------------------------------------------------
// POST /api/civitai/install — download and install a CivitAI LoRA
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct InstallRequest {
    /// CivitAI model version ID
    version_id: u64,
    /// Display name for the LoRA
    name: String,
    /// Trigger words from CivitAI
    #[serde(default)]
    trigger_words: Vec<String>,
    /// Base model hint (e.g. "flux-dev")
    #[serde(default)]
    base_model: Option<String>,
    /// Expected filename
    #[serde(default)]
    file_name: Option<String>,
    /// Expected SHA256 hash
    #[serde(default)]
    sha256: Option<String>,
    /// Expected file size in KB
    #[serde(default)]
    file_size_kb: Option<f64>,
    /// Thumbnail URL from CivitAI
    #[serde(default)]
    thumbnail_url: Option<String>,
}

#[derive(Serialize)]
struct InstallResponse {
    installed: bool,
    id: String,
    name: String,
}

pub async fn api_install_lora(Json(req): Json<InstallRequest>) -> impl IntoResponse {
    let handle = tokio::runtime::Handle::current();

    let result = tokio::task::spawn_blocking(move || {
        let db = Database::open().map_err(|e| format!("DB error: {e}"))?;
        let config = Config::load().map_err(|e| format!("Config error: {e}"))?;
        let auth_store = AuthStore::load().unwrap_or_default();
        let store = Store::new(config.store_root());
        let api_key = auth_store.token_for("civitai");

        let lora_id = format!("civitai:{}", req.version_id);

        // Already installed?
        if db.is_installed(&lora_id).unwrap_or(false) {
            return Ok(InstallResponse {
                installed: true,
                id: lora_id,
                name: req.name,
            });
        }

        // Build download URL
        let url = civitai::download_url(req.version_id, api_key.as_deref());

        // Determine filename
        let file_name = req
            .file_name
            .unwrap_or_else(|| format!("{}.safetensors", req.name.replace(' ', "_")));
        let sha256 = req.sha256.as_deref().unwrap_or("").to_lowercase();
        let expected_size = req.file_size_kb.map(|kb| (kb * 1024.0) as u64);

        // Store path
        let hash_for_path = if sha256.is_empty() {
            // Use version ID as fallback hash prefix
            format!("civitai-{}", req.version_id)
        } else {
            sha256.clone()
        };
        let store_path = store.path_for(
            &crate::core::manifest::AssetType::Lora,
            &hash_for_path,
            &file_name,
        );
        store
            .ensure_dir(&store_path)
            .map_err(|e| format!("Store error: {e}"))?;

        // Download
        handle
            .block_on(download::download_file(
                &url,
                &store_path,
                expected_size,
                api_key.as_deref(),
            ))
            .map_err(|e| format!("Download failed: {e}"))?;

        // Verify hash if available
        if !sha256.is_empty() {
            match Store::verify_hash(&store_path, &sha256) {
                Ok(true) => {}
                Ok(false) => {
                    let _ = std::fs::remove_file(&store_path);
                    return Err("SHA256 hash mismatch after download".to_string());
                }
                Err(e) => {
                    eprintln!("Warning: hash verification failed: {e}");
                }
            }
        }

        let actual_size = std::fs::metadata(&store_path).map(|m| m.len()).unwrap_or(0);

        // Record in DB
        db.insert_installed(&InstalledModelRecord {
            id: &lora_id,
            name: &req.name,
            asset_type: "lora",
            variant: None,
            sha256: &sha256,
            size: actual_size,
            file_name: &file_name,
            store_path: store_path.to_str().unwrap_or(""),
        })
        .map_err(|e| format!("DB error: {e}"))?;

        // Store metadata as artifact (trigger words, base model, thumbnail)
        let metadata = serde_json::json!({
            "trigger_word": req.trigger_words.first().unwrap_or(&String::new()),
            "trigger_words": req.trigger_words,
            "base_model": req.base_model.unwrap_or_default(),
            "source": "civitai",
            "civitai_version_id": req.version_id,
            "thumbnail_url": req.thumbnail_url,
        });

        db.insert_artifact(
            &lora_id,
            None,
            "lora",
            store_path.to_str().unwrap_or(""),
            &sha256,
            actual_size,
            Some(&metadata.to_string()),
        )
        .map_err(|e| format!("DB artifact error: {e}"))?;

        Ok::<InstallResponse, String>(InstallResponse {
            installed: true,
            id: lora_id,
            name: req.name,
        })
    })
    .await;

    match result {
        Ok(Ok(resp)) => (StatusCode::OK, Json(serde_json::json!(resp))),
        Ok(Err(msg)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": msg })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Task failed: {e}") })),
        ),
    }
}
