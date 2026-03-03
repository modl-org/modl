use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

use super::manifest::AssetType;

/// Content-addressed store at ~/modl/store/
pub struct Store {
    root: PathBuf,
}

impl Store {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root: root.join("store"),
        }
    }

    /// Get the storage path for a file: store/<type>/<sha256_prefix>/<filename>
    pub fn path_for(&self, asset_type: &AssetType, sha256: &str, file_name: &str) -> PathBuf {
        let prefix = if sha256.len() >= 16 {
            &sha256[..16]
        } else {
            sha256
        };
        self.root
            .join(asset_type.to_string())
            .join(prefix)
            .join(file_name)
    }

    /// Check if a file exists in the store with matching hash
    #[allow(dead_code)]
    pub fn has(&self, asset_type: &AssetType, sha256: &str, file_name: &str) -> bool {
        self.path_for(asset_type, sha256, file_name).exists()
    }

    /// Ensure the directory for a store path exists
    pub fn ensure_dir(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create store directory")?;
        }
        Ok(())
    }

    /// Verify SHA256 hash of a file
    pub fn verify_hash(path: &Path, expected: &str) -> Result<bool> {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open file for verification: {}", path.display()))?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        let hash = format!("{:x}", hasher.finalize());
        Ok(hash == expected)
    }

    /// Compute SHA256 hash of a file
    pub fn hash_file(path: &Path) -> Result<String> {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open file for hashing: {}", path.display()))?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        Ok(format!("{:x}", hasher.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_path() {
        let store = Store::new(PathBuf::from("/tmp/modl"));
        let path = store.path_for(
            &AssetType::Checkpoint,
            "abcdef1234567890abcdef",
            "model.safetensors",
        );
        assert_eq!(
            path,
            PathBuf::from("/tmp/modl/store/checkpoint/abcdef1234567890/model.safetensors")
        );
    }

    #[test]
    fn test_hash_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"test content").unwrap();
        let hash = Store::hash_file(tmp.path()).unwrap();
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64); // SHA256 = 64 hex chars
    }
}
