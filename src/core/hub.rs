use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use reqwest::Method;
use serde::{Deserialize, Serialize};

use crate::core::config::Config;

pub const DEFAULT_API_BASE: &str = "https://hub.modl.run";

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct StorageUsage {
    pub used_bytes: u64,
    pub file_count: u64,
    pub limit_bytes: u64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct HubAccount {
    pub account_id: String,
    pub email: String,
    pub username: Option<String>,
    pub plan: String,
    pub created_at: String,
    #[serde(default)]
    pub usage: HashMap<String, u64>,
    pub storage: StorageUsage,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct HubVersion {
    pub id: String,
    pub item_id: String,
    pub version: u32,
    pub r2_key: String,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    pub created_at: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct HubItem {
    pub id: String,
    pub account_id: String,
    pub slug: String,
    #[serde(rename = "type")]
    pub item_type: String,
    pub visibility: String,
    pub description: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub base_model: Option<String>,
    #[serde(default)]
    pub trigger_words: Vec<String>,
    pub downloads: u64,
    pub created_at: String,
    pub updated_at: String,
    pub username: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct HubItemDetail {
    #[serde(flatten)]
    pub item: HubItem,
    #[serde(default)]
    pub versions: Vec<HubVersion>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct HubListResponse {
    pub items: Vec<HubItem>,
    pub count: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct DeviceStartResponse {
    pub device_code: String,
    pub user_code: String,
    pub expires_at: String,
    pub interval: u64,
    #[serde(default)]
    pub verify_url: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct DevicePollResponse {
    pub status: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub account_id: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct PushStartResponse {
    pub upload_url: String,
    pub r2_key: String,
    pub version_id: String,
    pub version: u32,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct PullResponse {
    pub download_url: String,
    pub version: u32,
    pub r2_key: String,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateItemRequest {
    pub slug: String,
    #[serde(rename = "type")]
    pub item_type: String,
    pub visibility: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_model: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub trigger_words: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct HubRef {
    pub username: String,
    pub slug: String,
    pub version: Option<u32>,
}

pub struct HubClient {
    pub api_base: String,
    pub api_key: Option<String>,
    client: reqwest::Client,
}

impl HubClient {
    pub fn from_config(require_auth: bool) -> Result<Self> {
        let config = Config::load().context("Failed to load config")?;
        let api_base = config
            .cloud
            .as_ref()
            .and_then(|c| c.api_base.as_deref())
            .unwrap_or(DEFAULT_API_BASE)
            .trim_end_matches('/')
            .to_string();
        let api_key = config.cloud.and_then(|c| c.api_key);

        if require_auth && api_key.as_deref().is_none_or(|k| k.trim().is_empty()) {
            bail!(
                "No cloud API key configured. Run `modl login` first, or set cloud.api_key in ~/.modl/config.yaml"
            );
        }

        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self {
            api_base,
            api_key,
            client,
        })
    }

    fn auth_header(&self) -> Result<String> {
        let key = self
            .api_key
            .as_deref()
            .filter(|k| !k.trim().is_empty())
            .context("Missing cloud.api_key. Run `modl login` first.")?;
        Ok(format!("Bearer {key}"))
    }

    async fn send_json<B: Serialize + ?Sized, T: for<'de> Deserialize<'de>>(
        &self,
        method: Method,
        path: &str,
        body: Option<&B>,
        auth: bool,
    ) -> Result<T> {
        let url = format!("{}{}", self.api_base, path);
        let mut req = self.client.request(method, &url);

        if auth {
            req = req.header(reqwest::header::AUTHORIZATION, self.auth_header()?);
        }

        if let Some(payload) = body {
            req = req.json(payload);
        }

        let resp = req
            .send()
            .await
            .with_context(|| format!("Request failed: {url}"))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text)
                && let Some(detail) = v.get("detail").and_then(|d| d.as_str())
            {
                bail!("{detail}");
            }
            bail!("HTTP {}: {}", status.as_u16(), text);
        }
        serde_json::from_str(&text).with_context(|| format!("Failed to parse response from {url}"))
    }

    async fn send_empty<B: Serialize + ?Sized>(
        &self,
        method: Method,
        path: &str,
        body: Option<&B>,
        auth: bool,
    ) -> Result<()> {
        let url = format!("{}{}", self.api_base, path);
        let mut req = self.client.request(method, &url);
        if auth {
            req = req.header(reqwest::header::AUTHORIZATION, self.auth_header()?);
        }
        if let Some(payload) = body {
            req = req.json(payload);
        }
        let resp = req
            .send()
            .await
            .with_context(|| format!("Request failed: {url}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text)
                && let Some(detail) = v.get("detail").and_then(|d| d.as_str())
            {
                bail!("{detail}");
            }
            bail!("HTTP {}: {}", status.as_u16(), text);
        }
        Ok(())
    }

    pub async fn device_start(&self) -> Result<DeviceStartResponse> {
        self.send_json::<serde_json::Value, DeviceStartResponse>(
            Method::POST,
            "/auth/device/start",
            None,
            false,
        )
        .await
    }

    pub async fn device_poll(&self, code: &str) -> Result<DevicePollResponse> {
        let path = format!("/auth/device/poll?code={}", urlencoding::encode(code));
        self.send_json::<serde_json::Value, DevicePollResponse>(Method::GET, &path, None, false)
            .await
    }

    pub async fn me(&self) -> Result<HubAccount> {
        self.send_json::<serde_json::Value, HubAccount>(Method::GET, "/auth/me", None, true)
            .await
    }

    #[allow(dead_code)]
    pub async fn list_items(
        &self,
        user: Option<&str>,
        item_type: Option<&str>,
        visibility: Option<&str>,
    ) -> Result<HubListResponse> {
        let mut params: Vec<String> = Vec::new();
        if let Some(v) = user {
            params.push(format!("user={}", urlencoding::encode(v)));
        }
        if let Some(v) = item_type {
            params.push(format!("type={}", urlencoding::encode(v)));
        }
        if let Some(v) = visibility {
            params.push(format!("visibility={}", urlencoding::encode(v)));
        }
        let path = if params.is_empty() {
            "/hub/items".to_string()
        } else {
            format!("/hub/items?{}", params.join("&"))
        };
        self.send_json::<serde_json::Value, HubListResponse>(
            Method::GET,
            &path,
            None,
            self.api_key.is_some(),
        )
        .await
    }

    pub async fn get_item(&self, username: &str, slug: &str) -> Result<HubItemDetail> {
        let path = format!("/hub/{}/{}", username, slug);
        self.send_json::<serde_json::Value, HubItemDetail>(
            Method::GET,
            &path,
            None,
            self.api_key.is_some(),
        )
        .await
    }

    pub async fn create_item(&self, req: &CreateItemRequest) -> Result<HubItem> {
        self.send_json(Method::POST, "/hub/items", Some(req), true)
            .await
    }

    #[allow(dead_code)]
    pub async fn update_item(
        &self,
        username: &str,
        slug: &str,
        fields: &serde_json::Value,
    ) -> Result<()> {
        let path = format!("/hub/{}/{}", username, slug);
        self.send_empty(Method::PATCH, &path, Some(fields), true)
            .await
    }

    #[allow(dead_code)]
    pub async fn delete_item(&self, username: &str, slug: &str) -> Result<()> {
        let path = format!("/hub/{}/{}", username, slug);
        self.send_empty::<serde_json::Value>(Method::DELETE, &path, None, true)
            .await
    }

    pub async fn push_start(&self, username: &str, slug: &str) -> Result<PushStartResponse> {
        let path = format!("/hub/{}/{}/push", username, slug);
        self.send_json::<serde_json::Value, PushStartResponse>(Method::POST, &path, None, true)
            .await
    }

    pub async fn push_complete(
        &self,
        username: &str,
        slug: &str,
        version_id: &str,
        size_bytes: u64,
        sha256: &str,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        let path = format!(
            "/hub/{}/{}/push/complete?version_id={}",
            username,
            slug,
            urlencoding::encode(version_id)
        );
        let body = serde_json::json!({
            "size_bytes": size_bytes,
            "sha256": sha256,
            "metadata": metadata,
        });
        self.send_empty(Method::POST, &path, Some(&body), true)
            .await
    }

    pub async fn pull(
        &self,
        username: &str,
        slug: &str,
        version: Option<u32>,
    ) -> Result<PullResponse> {
        let mut path = format!("/hub/{}/{}/pull", username, slug);
        if let Some(v) = version {
            path.push_str(&format!("?version={v}"));
        }
        self.send_json::<serde_json::Value, PullResponse>(
            Method::GET,
            &path,
            None,
            self.api_key.is_some(),
        )
        .await
    }
}

pub fn parse_hub_ref(input: &str) -> Option<HubRef> {
    if input.contains("://") || input.contains(':') {
        return None;
    }

    let (username, slug_with_ver) = input.split_once('/')?;
    if username.is_empty() || slug_with_ver.is_empty() || slug_with_ver.contains('/') {
        return None;
    }
    if !is_valid_component(username) {
        return None;
    }

    let (slug, version) = if let Some((slug, v)) = slug_with_ver.rsplit_once('@') {
        if slug.is_empty() || v.is_empty() || !v.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        (slug, Some(v.parse::<u32>().ok()?))
    } else {
        (slug_with_ver, None)
    };

    if !is_valid_component(slug) {
        return None;
    }

    Some(HubRef {
        username: username.to_string(),
        slug: slug.to_string(),
        version,
    })
}

fn is_valid_component(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

pub async fn upload_file_presigned(
    upload_url: &str,
    file_path: &Path,
    content_type: &str,
) -> Result<()> {
    let bytes = tokio::fs::read(file_path)
        .await
        .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

    let resp = reqwest::Client::new()
        .put(upload_url)
        .header(reqwest::header::CONTENT_TYPE, content_type)
        .body(bytes)
        .send()
        .await
        .context("Upload request failed")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("Upload failed: HTTP {} {}", status.as_u16(), text);
    }
    Ok(())
}
