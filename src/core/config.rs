use anyhow::{Context, Result};
use console::style;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Migrate ~/.mods → ~/.modl if the old directory exists and the new one doesn't.
/// Called automatically on Config::load() so every command benefits.
pub fn migrate_legacy_dir() {
    let Some(home) = dirs::home_dir() else {
        return;
    };
    let old = home.join(".mods");
    let new = home.join(".modl");

    if old.is_dir() && !new.exists() {
        eprintln!(
            "{} Migrating {} → {} …",
            style("↗").cyan(),
            style("~/.mods").dim(),
            style("~/.modl").bold()
        );
        if let Err(e) = std::fs::rename(&old, &new) {
            eprintln!(
                "  {} Could not rename: {}. Copy manually or run:\n    mv ~/.mods ~/.modl",
                style("⚠").yellow(),
                e
            );
        } else {
            eprintln!("  {} Done.", style("✓").green());
        }
    }
}

/// Main configuration for modl, stored at ~/.modl/config.yaml
#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub storage: StorageConfig,
    #[serde(default)]
    pub targets: Vec<TargetConfig>,
    pub gpu: Option<GpuOverride>,
    #[serde(default)]
    pub cloud: Option<CloudConfig>,
}

/// Cloud provider credentials and default settings
#[derive(Debug, Serialize, Deserialize)]
pub struct CloudConfig {
    /// Default provider when --provider is omitted (modal, replicate, runpod)
    #[serde(default)]
    pub default_provider: Option<String>,
    /// Modal token (MODAL_TOKEN_ID)
    #[serde(default)]
    pub modal_token: Option<String>,
    /// Replicate API token
    #[serde(default)]
    pub replicate_token: Option<String>,
    /// RunPod API key
    #[serde(default)]
    pub runpod_key: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StorageConfig {
    pub root: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TargetConfig {
    pub path: PathBuf,
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    #[serde(default = "default_true")]
    pub symlink: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    Comfyui,
    A1111,
    Invokeai,
    Custom,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GpuOverride {
    pub vram_mb: u64,
}

fn default_true() -> bool {
    true
}

impl Config {
    /// Load config from the default path (~/.modl/config.yaml)
    pub fn load() -> Result<Self> {
        migrate_legacy_dir();
        let path = Self::default_path();
        if path.exists() {
            let contents = std::fs::read_to_string(&path).context("Failed to read config file")?;
            let config: Config =
                serde_yaml::from_str(&contents).context("Failed to parse config file")?;
            Ok(config)
        } else {
            Ok(Self::default())
        }
    }

    /// Save config to the default path
    pub fn save(&self) -> Result<()> {
        let path = Self::default_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create config directory")?;
        }
        let yaml = serde_yaml::to_string(self).context("Failed to serialize config")?;
        std::fs::write(&path, yaml).context("Failed to write config file")?;
        Ok(())
    }

    pub fn default_path() -> PathBuf {
        dirs::home_dir()
            .expect("Could not determine home directory")
            .join(".modl")
            .join("config.yaml")
    }

    /// Get the store root, expanding ~ if needed
    pub fn store_root(&self) -> PathBuf {
        expand_tilde(&self.storage.root)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            storage: StorageConfig {
                root: PathBuf::from("~/modl"),
            },
            targets: vec![],
            gpu: None,
            cloud: None,
        }
    }
}

fn expand_tilde(path: &Path) -> PathBuf {
    if let Ok(stripped) = path.strip_prefix("~")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(stripped);
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.storage.root, PathBuf::from("~/modl"));
        assert!(config.targets.is_empty());
        assert!(config.gpu.is_none());
    }

    #[test]
    fn test_config_roundtrip() {
        let config = Config {
            storage: StorageConfig {
                root: PathBuf::from("~/modl"),
            },
            targets: vec![TargetConfig {
                path: PathBuf::from("~/ComfyUI"),
                tool_type: ToolType::Comfyui,
                symlink: true,
            }],
            gpu: None,
            cloud: None,
        };
        let yaml = serde_yaml::to_string(&config).unwrap();
        let parsed: Config = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.storage.root, config.storage.root);
        assert_eq!(parsed.targets.len(), 1);
    }
}
