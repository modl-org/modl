use anyhow::{Context, Result};
use console::style;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Migrate ~/.mods → ~/.modl.
///
/// Handles three scenarios:
///   1. `~/.mods` exists, `~/.modl` does not → atomic rename
///   2. Both exist → merge entries from `~/.mods` into `~/.modl` (existing
///      files in `~/.modl` win), then remove `~/.mods`
///   3. `~/.mods` doesn't exist → nothing to do
///
/// Also cleans up the legacy `mods.db` artifact and rewrites symlinks under
/// `loras/` that still point into `~/.mods/…` → `~/.modl/…`.
///
/// Called automatically on Config::load() so every command benefits.
pub fn migrate_legacy_dir() {
    let Some(home) = dirs::home_dir() else {
        return;
    };
    let old = home.join(".mods");
    let new = home.join(".modl");

    if !old.is_dir() {
        // Nothing to migrate — just fix up any stale symlinks in ~/.modl/loras
        if new.is_dir() {
            rewrite_legacy_symlinks(&new, &old, &new);
        }
        return;
    }

    eprintln!(
        "{} Migrating {} → {} …",
        style("↗").cyan(),
        style("~/.mods").dim(),
        style("~/.modl").bold()
    );

    if !new.exists() {
        // Scenario 1: simple rename
        if let Err(e) = std::fs::rename(&old, &new) {
            eprintln!(
                "  {} Could not rename: {}. Copy manually or run:\n    mv ~/.mods ~/.modl",
                style("⚠").yellow(),
                e
            );
            return;
        }
        eprintln!("  {} Renamed successfully.", style("✓").green());
    } else {
        // Scenario 2: merge — move entries from old into new, skip conflicts
        if let Err(e) = merge_dirs(&old, &new) {
            eprintln!(
                "  {} Merge failed: {}. You can merge manually:\n    rsync -a ~/.mods/ ~/.modl/ && rm -rf ~/.mods",
                style("⚠").yellow(),
                e
            );
            return;
        }
        // Remove the now-empty old dir (best effort)
        let _ = std::fs::remove_dir_all(&old);
        eprintln!("  {} Merged and removed ~/.mods.", style("✓").green());
    }

    // Clean up legacy mods.db (always 0-byte artifact)
    let legacy_db = new.join("mods.db");
    if legacy_db.exists() {
        let _ = std::fs::remove_file(&legacy_db);
    }

    // Migrate config.yaml: rewrite storage.root from ~/mods → ~/modl
    migrate_config_paths(&new);

    // Rename the storage root directory if it still uses the old name.
    // e.g. ~/mods → ~/modl (separate from ~/.mods → ~/.modl config dir)
    migrate_storage_root(&new);

    // Rewrite lora symlinks that still point into ~/.mods/…
    rewrite_legacy_symlinks(&new, &old, &new);

    eprintln!("  {} Migration complete.", style("✓").green());
}

/// Recursively merge `src` into `dst`. Existing files in `dst` are NOT
/// overwritten — only missing entries are moved over.
fn merge_dirs(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let name = entry.file_name();
        let dst_path = dst.join(&name);

        // Skip legacy artifact
        if name == "mods.db" {
            continue;
        }

        if dst_path.exists() {
            if src_path.is_dir() && dst_path.is_dir() {
                // Recurse into sub-directories
                merge_dirs(&src_path, &dst_path)?;
            } else if name == "state.db" {
                // For the database, prefer the larger file (the one with real data)
                let src_size = std::fs::metadata(&src_path).map(|m| m.len()).unwrap_or(0);
                let dst_size = std::fs::metadata(&dst_path).map(|m| m.len()).unwrap_or(0);
                if src_size > dst_size {
                    std::fs::copy(&src_path, &dst_path)?;
                }
            }
            // else: dst already has this file/symlink — keep dst version
        } else {
            // Move the entry (rename is fast on same filesystem)
            if std::fs::rename(&src_path, &dst_path).is_err() {
                // Cross-device: fall back to copy
                if src_path.is_dir() {
                    copy_dir_all(&src_path, &dst_path)?;
                } else {
                    std::fs::copy(&src_path, &dst_path)?;
                }
            }
        }
    }
    Ok(())
}

/// Recursively copy a directory (fallback for cross-device moves).
fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else if src_path.is_symlink() {
            let _target = std::fs::read_link(&src_path)?;
            #[cfg(unix)]
            std::os::unix::fs::symlink(&_target, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

/// Rewrite symlinks in `<base>/loras/` that point into `old_root` so they
/// point into `new_root` instead.
fn rewrite_legacy_symlinks(base: &Path, old_root: &Path, new_root: &Path) {
    let loras_dir = base.join("loras");
    rewrite_symlinks_in_dir(
        &loras_dir,
        &old_root.to_string_lossy(),
        &new_root.to_string_lossy(),
    );
}

/// Rewrite any symlinks in `dir` whose target contains `old_prefix`,
/// replacing that substring with `new_prefix`. Recurses into subdirectories.
fn rewrite_symlinks_in_dir(dir: &Path, old_prefix: &str, new_prefix: &str) {
    if !dir.is_dir() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && !path.is_symlink() {
            rewrite_symlinks_in_dir(&path, old_prefix, new_prefix);
        }
        if !path.is_symlink() {
            continue;
        }
        if let Ok(target) = std::fs::read_link(&path) {
            let target_str = target.to_string_lossy().to_string();
            if target_str.contains(old_prefix) {
                let new_target_str = target_str.replace(old_prefix, new_prefix);
                let _new_target = PathBuf::from(&new_target_str);
                let _ = std::fs::remove_file(&path);
                #[cfg(unix)]
                {
                    let _ = std::os::unix::fs::symlink(&_new_target, &path);
                }
            }
        }
    }
}

/// If config.yaml still references `~/mods` as storage root, update it to `~/modl`.
fn migrate_config_paths(modl_dir: &Path) {
    let config_path = modl_dir.join("config.yaml");
    if !config_path.exists() {
        return;
    }
    if let Ok(contents) = std::fs::read_to_string(&config_path) {
        // Replace legacy storage root references
        let updated = contents
            .replace("~/mods", "~/modl")
            .replace("~/.mods", "~/.modl");
        if updated != contents {
            let _ = std::fs::write(&config_path, &updated);
            eprintln!(
                "  {} Updated config.yaml: storage root → ~/modl",
                style("✓").green()
            );
        }
    }
}

/// Rename the storage root directory from ~/mods → ~/modl (the non-dot dir
/// where model files actually live, separate from the ~/.modl config dir).
fn migrate_storage_root(modl_dir: &Path) {
    let config_path = modl_dir.join("config.yaml");
    if !config_path.exists() {
        return;
    }

    // Parse the config to find the storage root
    let contents = match std::fs::read_to_string(&config_path) {
        Ok(c) => c,
        Err(_) => return,
    };

    // Look for the resolved storage root — after config migration it should
    // already say ~/modl, so we check if the *old* path still exists on disk.
    let Some(home) = dirs::home_dir() else {
        return;
    };
    let old_store = home.join("mods");
    let new_store = home.join("modl");

    if old_store.is_dir() && !new_store.exists() {
        if let Err(e) = std::fs::rename(&old_store, &new_store) {
            eprintln!(
                "  {} Could not rename storage root: {}.\n    Run: mv ~/mods ~/modl",
                style("⚠").yellow(),
                e
            );
        } else {
            eprintln!("  {} Renamed storage ~/mods → ~/modl", style("✓").green());
        }
    } else if old_store.is_dir() && new_store.is_dir() {
        // Both exist — merge old into new
        if let Err(e) = merge_dirs(&old_store, &new_store) {
            eprintln!(
                "  {} Could not merge storage dirs: {}.\n    Run: rsync -a ~/mods/ ~/modl/ && rm -rf ~/mods",
                style("⚠").yellow(),
                e
            );
        } else {
            let _ = std::fs::remove_dir_all(&old_store);
            eprintln!("  {} Merged storage ~/mods → ~/modl", style("✓").green());
        }
    }

    // Rewrite any symlinks in target dirs that point into ~/mods/store/…
    if let Ok(parsed) = serde_yaml::from_str::<serde_yaml::Value>(&contents)
        && let Some(targets) = parsed.get("targets").and_then(|t| t.as_sequence())
    {
        for target in targets {
            if let Some(path_str) = target.get("path").and_then(|p| p.as_str()) {
                let target_dir = if path_str.starts_with('~') {
                    home.join(path_str.trim_start_matches("~/"))
                } else {
                    PathBuf::from(path_str)
                };
                rewrite_symlinks_in_dir(
                    &target_dir,
                    &old_store.to_string_lossy(),
                    &new_store.to_string_lossy(),
                );
            }
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
    /// Hub/cloud API base URL (default: https://hub.modl.run)
    #[serde(default)]
    pub api_base: Option<String>,
    /// Hub/cloud API key (Bearer modl_...)
    #[serde(default)]
    pub api_key: Option<String>,
    /// Last known username for hub operations
    #[serde(default)]
    pub username: Option<String>,
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
        super::paths::modl_root().join("config.yaml")
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
