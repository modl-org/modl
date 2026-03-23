use anyhow::{Context, Result};
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
            anyhow::bail!("Registry index not found. Run `modl system update` to fetch it.");
        }
        let contents = std::fs::read_to_string(&path).context("Failed to read registry index")?;
        let index: RegistryIndex =
            serde_json::from_str(&contents).context("Failed to parse registry index")?;
        Ok(index)
    }

    pub fn local_path() -> PathBuf {
        super::paths::modl_root().join("index.json")
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

    /// Suggest similar model IDs for a failed lookup.
    ///
    /// Tries substring matching, prefix matching, and edit distance
    /// to find the closest matches in the registry.
    pub fn suggest(&self, query: &str, max: usize) -> Vec<&Manifest> {
        let q = query.to_lowercase();

        // 1. Substring match on ID (e.g. "sdx" matches "sdxl-base-1.0")
        let mut candidates: Vec<(&Manifest, usize)> = self
            .items
            .iter()
            .filter_map(|m| {
                let id = m.id.to_lowercase();
                let name = m.name.to_lowercase();
                if id.contains(&q) || q.contains(&id) || name.contains(&q) {
                    // Score: shorter edit distance = better
                    Some((m, edit_distance(&q, &id)))
                } else {
                    None
                }
            })
            .collect();

        // 2. If no substring matches, try edit distance on all items
        if candidates.is_empty() {
            candidates = self
                .items
                .iter()
                .map(|m| {
                    let id_dist = edit_distance(&q, &m.id.to_lowercase());
                    let name_dist = edit_distance(&q, &m.name.to_lowercase());
                    (m, id_dist.min(name_dist))
                })
                .filter(|(_, dist)| *dist <= q.len().max(3))
                .collect();
        }

        candidates.sort_by_key(|(_, dist)| *dist);
        candidates.into_iter().take(max).map(|(m, _)| m).collect()
    }

    /// The URL to fetch the latest index from.
    ///
    /// Uses raw GitHub as primary. Override via MODL_REGISTRY_URL env var.
    /// TODO: switch primary to registry.modl.run once the CDN is deployed.
    pub fn remote_url() -> String {
        std::env::var("MODL_REGISTRY_URL").unwrap_or_else(|_| {
            "https://raw.githubusercontent.com/modl-org/modl-registry/main/index.json".to_string()
        })
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

/// Simple Levenshtein edit distance for typo suggestions.
fn edit_distance(a: &str, b: &str) -> usize {
    let a_len = a.len();
    let b_len = b.len();
    let mut matrix = vec![vec![0usize; b_len + 1]; a_len + 1];

    for (i, row) in matrix.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in matrix[0].iter_mut().enumerate() {
        *cell = j;
    }

    for (i, ca) in a.chars().enumerate() {
        for (j, cb) in b.chars().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            matrix[i + 1][j + 1] = (matrix[i][j + 1] + 1)
                .min(matrix[i + 1][j] + 1)
                .min(matrix[i][j] + cost);
        }
    }

    matrix[a_len][b_len]
}
