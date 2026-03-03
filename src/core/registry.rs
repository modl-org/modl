use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use super::manifest::Manifest;

/// The registry index — compiled from all manifests
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct RegistryIndex {
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub generated_at: Option<String>,
    #[serde(default)]
    pub total_count: Option<u32>,
    #[serde(default)]
    pub type_counts: Option<std::collections::HashMap<String, u32>>,
    #[serde(default)]
    pub cloud_available_count: Option<u32>,
    #[serde(default)]
    pub schema_url: Option<String>,
    pub items: Vec<Manifest>,
}

impl RegistryIndex {
    /// Load the local copy of the registry index
    pub fn load() -> Result<Self> {
        let path = Self::local_path();
        if !path.exists() {
            anyhow::bail!("Registry index not found. Run `modl update` to fetch it.");
        }
        let contents = std::fs::read_to_string(&path).context("Failed to read registry index")?;
        let index: RegistryIndex =
            serde_json::from_str(&contents).context("Failed to parse registry index")?;
        Ok(index)
    }

    /// Save index to local cache
    #[allow(dead_code)]
    pub fn save(&self) -> Result<()> {
        let path = Self::local_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    pub fn local_path() -> PathBuf {
        dirs::home_dir()
            .expect("Could not determine home directory")
            .join(".modl")
            .join("index.json")
    }

    /// Look up a manifest by ID
    pub fn find(&self, id: &str) -> Option<&Manifest> {
        self.items.iter().find(|m| m.id == id)
    }

    /// Search items by query string (matches id, name, tags, description)
    pub fn search(&self, query: &str) -> Vec<&Manifest> {
        let q = query.to_lowercase();
        self.items
            .iter()
            .filter(|m| {
                m.id.to_lowercase().contains(&q)
                    || m.name.to_lowercase().contains(&q)
                    || m.tags.iter().any(|t| t.to_lowercase().contains(&q))
                    || m.description
                        .as_ref()
                        .is_some_and(|d| d.to_lowercase().contains(&q))
            })
            .collect()
    }

    /// Build a lookup map by ID for faster access
    #[allow(dead_code)]
    pub fn as_map(&self) -> HashMap<&str, &Manifest> {
        self.items.iter().map(|m| (m.id.as_str(), m)).collect()
    }

    /// The URL to fetch the latest index from.
    ///
    /// Uses registry.modl.run (Cloudflare-cached) as primary,
    /// with raw GitHub as fallback. Override via MODL_REGISTRY_URL env var.
    pub fn remote_url() -> String {
        std::env::var("MODL_REGISTRY_URL")
            .unwrap_or_else(|_| "https://registry.modl.run/index.json".to_string())
    }

    /// Fallback URL if the primary registry is unreachable
    pub fn fallback_url() -> &'static str {
        "https://raw.githubusercontent.com/modl-org/modl-registry/main/index.json"
    }

    /// Check if the local index is missing or older than `max_age`
    pub fn is_stale(max_age: Duration) -> bool {
        let path = Self::local_path();
        if !path.exists() {
            return true;
        }
        match std::fs::metadata(&path) {
            Ok(meta) => match meta.modified() {
                Ok(modified) => {
                    modified
                        .elapsed()
                        .unwrap_or(max_age + Duration::from_secs(1))
                        > max_age
                }
                Err(_) => true,
            },
            Err(_) => true,
        }
    }

    /// Fetch the index from a URL
    async fn fetch_from(client: &reqwest::Client, url: &str) -> Result<String> {
        let response = client
            .get(url)
            .send()
            .await
            .with_context(|| format!("Failed to connect to {}", url))?;
        if !response.status().is_success() {
            anyhow::bail!("HTTP {} from {}", response.status(), url);
        }
        response
            .text()
            .await
            .with_context(|| format!("Failed to read response from {}", url))
    }

    /// Fetch latest registry from remote and save to local cache
    pub async fn fetch_and_save() -> Result<Self> {
        let primary_url = Self::remote_url();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .unwrap_or_default();

        let body = match Self::fetch_from(&client, &primary_url).await {
            Ok(body) => body,
            Err(e) => {
                let fallback = Self::fallback_url();
                eprintln!(
                    "  ⚠ Primary registry unavailable ({}), trying fallback...",
                    e
                );
                Self::fetch_from(&client, fallback)
                    .await
                    .context("Failed to fetch registry from both primary and fallback URLs")?
            }
        };

        let index: RegistryIndex =
            serde_json::from_str(&body).context("Failed to parse registry index")?;

        let path = Self::local_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &body).context("Failed to write index to disk")?;

        Ok(index)
    }

    /// Load the index from cache if fresh, or auto-fetch if missing/stale (>24h)
    pub async fn load_or_fetch() -> Result<Self> {
        use console::style;

        let max_age = Duration::from_secs(24 * 60 * 60);
        if Self::is_stale(max_age) {
            eprintln!(
                "{} Registry index is {} — updating...",
                style("→").cyan(),
                if Self::local_path().exists() {
                    "stale"
                } else {
                    "missing"
                }
            );
            let index = Self::fetch_and_save().await?;
            eprintln!(
                "{} Registry updated — {} models available",
                style("✓").green(),
                style(index.items.len()).bold()
            );
            Ok(index)
        } else {
            Self::load()
        }
    }
}
