use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use crate::core::db::Database;
use crate::core::manifest::AssetType;
use crate::core::store::Store;
use crate::core::symlink;

/// Collected LoRA artifact info
#[derive(Debug)]
#[allow(dead_code)]
pub struct CollectedLora {
    pub artifact_id: String,
    pub store_path: PathBuf,
    pub sha256: String,
    pub size_bytes: u64,
    pub symlinks: Vec<PathBuf>,
}

/// Collect a training output LoRA: hash, store, register, symlink.
///
/// 1. Hash the .safetensors file
/// 2. Move to content-addressed store
/// 3. Register in `installed` table as AssetType::Lora
/// 4. Register in `artifacts` table (link to job)
/// 5. Create symlinks to configured targets
pub fn collect_lora(
    output_path: &Path,
    lora_name: &str,
    base_model: &str,
    trigger_word: &str,
    job_id: &str,
    db: &Database,
    store_root: &Path,
) -> Result<CollectedLora> {
    if !output_path.exists() {
        anyhow::bail!("Output artifact not found: {}", output_path.display());
    }

    let file_name = output_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("output.safetensors");

    // 1. Hash the file
    let sha256 = Store::hash_file(output_path)
        .with_context(|| format!("Failed to hash {}", output_path.display()))?;

    let size_bytes = std::fs::metadata(output_path)
        .with_context(|| format!("Failed to stat {}", output_path.display()))?
        .len();

    // 2. Move to content-addressed store
    let store = Store::new(store_root.to_path_buf());
    let store_path = store.path_for(&AssetType::Lora, &sha256, file_name);
    store.ensure_dir(&store_path)?;

    if !store_path.exists() {
        std::fs::copy(output_path, &store_path).with_context(|| {
            format!(
                "Failed to copy {} → {}",
                output_path.display(),
                store_path.display()
            )
        })?;
    }

    // 3. Register in installed table
    let artifact_id = format!("train:{lora_name}:{}", &sha256[..16]);
    db.insert_installed(
        &artifact_id,
        lora_name,
        "lora",
        None,
        &sha256,
        size_bytes,
        file_name,
        &store_path.to_string_lossy(),
    )
    .context("Failed to register LoRA in installed table")?;

    // 4. Register in artifacts table
    let metadata = serde_json::json!({
        "base_model": base_model,
        "trigger_word": trigger_word,
        "lora_name": lora_name,
    });

    db.insert_artifact(
        &artifact_id,
        Some(job_id),
        "lora",
        &store_path.to_string_lossy(),
        &sha256,
        size_bytes,
        Some(&metadata.to_string()),
    )
    .context("Failed to register artifact")?;

    // 5. Create symlinks to configured lora directories
    let symlinks = create_lora_symlinks(&store_path, lora_name, file_name)?;

    Ok(CollectedLora {
        artifact_id,
        store_path,
        sha256,
        size_bytes,
        symlinks,
    })
}

/// Create symlinks for a LoRA file in common tool directories.
/// Currently symlinks into any configured ComfyUI lora dirs.
fn create_lora_symlinks(
    store_path: &Path,
    lora_name: &str,
    file_name: &str,
) -> Result<Vec<PathBuf>> {
    let mut links = Vec::new();

    // Symlink with friendly name in the modl loras dir
    let modl_lora_dir = dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
        .join("loras");

    if let Err(e) = std::fs::create_dir_all(&modl_lora_dir) {
        eprintln!(
            "Warning: could not create loras dir {}: {e}",
            modl_lora_dir.display()
        );
        return Ok(links);
    }

    // Use lora_name as the symlink name, preserving the file extension
    let ext = Path::new(file_name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("safetensors");
    let link_name = format!("{lora_name}.{ext}");
    let link_path = modl_lora_dir.join(&link_name);

    match symlink::create(&link_path, store_path) {
        Ok(()) => links.push(link_path),
        Err(e) => {
            eprintln!(
                "Warning: could not create symlink {}: {e}",
                link_path.display()
            );
        }
    }

    Ok(links)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_lora_nonexistent_path() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let db = Database::open_at(tmp.path()).unwrap();
        let store_root = tempfile::TempDir::new().unwrap();

        let result = collect_lora(
            Path::new("/nonexistent/output.safetensors"),
            "test-lora",
            "flux-schnell",
            "OHWX",
            "job-123",
            &db,
            store_root.path(),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_collect_lora_success() {
        // Create a fake safetensors file
        let output_dir = tempfile::TempDir::new().unwrap();
        let output_file = output_dir.path().join("lora_output.safetensors");
        std::fs::write(&output_file, b"fake safetensors content for testing").unwrap();

        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db = Database::open_at(db_file.path()).unwrap();
        let store_root = tempfile::TempDir::new().unwrap();

        // Insert a job record so the FK on artifacts is satisfied
        db.insert_job("job-test-123", "train", "running", "{}", "local", None)
            .unwrap();

        let result = collect_lora(
            &output_file,
            "my-lora",
            "flux-schnell",
            "OHWX",
            "job-test-123",
            &db,
            store_root.path(),
        )
        .unwrap();

        assert!(!result.sha256.is_empty());
        assert!(result.store_path.exists());
        assert!(result.size_bytes > 0);
        assert!(result.artifact_id.starts_with("train:my-lora:"));

        // Verify DB record
        assert!(db.is_installed(&result.artifact_id).unwrap());
    }
}
