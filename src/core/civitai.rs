use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const API_BASE: &str = "https://civitai.com/api/v1";

// ---------------------------------------------------------------------------
// API response types (subset of CivitAI's schema)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SearchResponse {
    pub items: Vec<CivitaiModel>,
    pub metadata: PageMetadata,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct PageMetadata {
    #[serde(rename = "totalItems")]
    pub total_items: Option<u64>,
    #[serde(rename = "totalPages")]
    pub total_pages: Option<u64>,
    #[serde(rename = "currentPage")]
    pub current_page: u64,
    #[serde(rename = "pageSize")]
    pub page_size: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CivitaiModel {
    pub id: u64,
    pub name: String,
    #[serde(rename = "type")]
    pub model_type: String,
    pub nsfw: Option<bool>,
    pub tags: Option<Vec<String>>,
    pub creator: Option<Creator>,
    #[serde(rename = "modelVersions")]
    pub model_versions: Vec<ModelVersion>,
    pub stats: Option<ModelStats>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Creator {
    pub username: String,
    pub image: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelVersion {
    pub id: u64,
    pub name: String,
    #[serde(rename = "baseModel")]
    pub base_model: Option<String>,
    #[serde(rename = "trainedWords")]
    pub trained_words: Option<Vec<String>>,
    pub files: Vec<ModelFile>,
    pub images: Vec<ModelImage>,
    pub stats: Option<VersionStats>,
    #[serde(rename = "createdAt")]
    pub created_at: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelFile {
    pub name: String,
    #[serde(rename = "sizeKB")]
    pub size_kb: Option<f64>,
    pub primary: Option<bool>,
    pub metadata: Option<FileMetadata>,
    #[serde(rename = "downloadUrl")]
    pub download_url: Option<String>,
    pub hashes: Option<FileHashes>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FileMetadata {
    pub fp: Option<String>,
    pub size: Option<String>,
    pub format: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FileHashes {
    #[serde(rename = "SHA256")]
    pub sha256: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelImage {
    pub url: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    #[serde(rename = "nsfwLevel")]
    pub nsfw_level: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelStats {
    #[serde(rename = "downloadCount")]
    pub download_count: Option<u64>,
    pub rating: Option<f64>,
    #[serde(rename = "ratingCount")]
    pub rating_count: Option<u64>,
    #[serde(rename = "favoriteCount")]
    pub favorite_count: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VersionStats {
    #[serde(rename = "downloadCount")]
    pub download_count: Option<u64>,
    pub rating: Option<f64>,
    #[serde(rename = "ratingCount")]
    pub rating_count: Option<u64>,
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Search CivitAI for LoRAs. No auth required.
pub async fn search_loras(
    query: &str,
    base_model: Option<&str>,
    sort: Option<&str>,
    page: u32,
    limit: u32,
    nsfw: bool,
) -> Result<SearchResponse> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let mut url =
        format!("{API_BASE}/models?types=LORA&limit={limit}&page={page}&primaryFileOnly=true");

    if !query.is_empty() {
        url.push_str(&format!("&query={}", urlencoding::encode(query)));
    }
    if let Some(bm) = base_model {
        url.push_str(&format!("&baseModels={}", urlencoding::encode(bm)));
    }
    if let Some(s) = sort {
        url.push_str(&format!("&sort={}", urlencoding::encode(s)));
    }
    if !nsfw {
        url.push_str("&nsfw=false");
    }

    let resp = client
        .get(&url)
        .header("User-Agent", "modl/1.0")
        .send()
        .await
        .context("Failed to reach CivitAI API")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("CivitAI API returned {status}: {body}");
    }

    resp.json::<SearchResponse>()
        .await
        .context("Failed to parse CivitAI response")
}

// ---------------------------------------------------------------------------
// Download URL construction
// ---------------------------------------------------------------------------

/// Build a download URL for a CivitAI model version. Appends auth token if provided.
pub fn download_url(model_version_id: u64, api_key: Option<&str>) -> String {
    let base = format!("https://civitai.com/api/download/models/{model_version_id}");
    match api_key {
        Some(key) => format!("{base}?token={key}"),
        None => base,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map CivitAI's base model string to a modl model family hint.
/// E.g. "SDXL 1.0" → "sdxl", "Flux.1 D" → "flux-dev"
///
/// Note: Pony and Illustrious are SDXL fine-tunes (same architecture/tensor shapes),
/// so their LoRAs are architecturally compatible with SDXL models.
pub fn normalize_base_model(civitai_base: &str) -> String {
    let lower = civitai_base.to_lowercase();
    if lower.contains("flux") && lower.contains("dev") {
        "flux-dev".to_string()
    } else if lower.contains("flux") && lower.contains("schnell") {
        "flux-schnell".to_string()
    } else if lower.contains("flux") {
        "flux".to_string()
    } else if lower.contains("pony") || lower.contains("illustrious") || lower.contains("sdxl") {
        "sdxl".to_string()
    } else if lower.contains("sd 1") || lower.contains("sd1") {
        "sd15".to_string()
    } else if lower.contains("hunyuan") {
        "hunyuan".to_string()
    } else {
        lower.replace(' ', "-")
    }
}
