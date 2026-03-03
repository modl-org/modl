pub mod civitai;
pub mod huggingface;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Auth credentials stored at ~/.modl/auth.yaml
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct AuthStore {
    #[serde(default)]
    pub huggingface: Option<HuggingFaceAuth>,
    #[serde(default)]
    pub civitai: Option<CivitaiAuth>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HuggingFaceAuth {
    pub token: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CivitaiAuth {
    pub api_key: String,
}

impl AuthStore {
    pub fn load() -> Result<Self> {
        let path = Self::path();
        if path.exists() {
            let contents = std::fs::read_to_string(&path).context("Failed to read auth file")?;
            let store: AuthStore =
                serde_yaml::from_str(&contents).context("Failed to parse auth file")?;
            Ok(store)
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let yaml = serde_yaml::to_string(self)?;
        std::fs::write(&path, yaml)?;
        Ok(())
    }

    pub fn path() -> PathBuf {
        dirs::home_dir()
            .expect("Could not determine home directory")
            .join(".modl")
            .join("auth.yaml")
    }

    /// Get the bearer token for a given provider
    pub fn token_for(&self, provider: &str) -> Option<String> {
        match provider {
            "huggingface" => self.huggingface.as_ref().map(|a| a.token.clone()),
            "civitai" => self.civitai.as_ref().map(|a| a.api_key.clone()),
            _ => None,
        }
    }
}
