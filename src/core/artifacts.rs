use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use crate::core::db::Database;
use crate::core::manifest::AssetType;
use crate::core::store::Store;
use crate::core::symlink;

/// Collected LoRA artifact info
#[derive(Debug)]
pub struct CollectedLora {
    #[allow(dead_code)]
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
    let store_path_str = store_path.to_string_lossy();
    db.insert_installed(&crate::core::db::InstalledModelRecord {
        id: &artifact_id,
        name: lora_name,
        asset_type: "lora",
        variant: None,
        sha256: &sha256,
        size: size_bytes,
        file_name,
        store_path: &store_path_str,
    })
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
    let modl_lora_dir = super::paths::modl_root().join("loras");

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
// Promote LoRA to library
// ---------------------------------------------------------------------------

use super::training::parse_step_from_filename;

/// Parameters for promoting a training output LoRA to the library.
pub struct PromoteLoraParams {
    pub name: String,
    pub trigger_word: Option<String>,
    pub base_model: Option<String>,
    pub lora_path: String,
    pub thumbnail: Option<String>,
    pub step: Option<u64>,
    pub training_run: Option<String>,
    pub config_json: Option<String>,
    pub tags: Option<String>,
}

/// Promote a trained LoRA to the library: validate, register in DB, copy to
/// content-addressed store, and register as installed artifact.
///
/// Returns the library LoRA ID on success.
pub fn promote_lora(params: PromoteLoraParams) -> Result<String> {
    let modl_root = super::paths::modl_root();
    let lora_path = std::path::Path::new(&params.lora_path);

    // Validate that lora_path is under the modl training_output directory
    let allowed_dir = modl_root.join("training_output");
    let canonical_lora = lora_path
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("Invalid lora_path: {e}"))?;
    let canonical_allowed = allowed_dir.canonicalize().unwrap_or(allowed_dir.clone());
    if !canonical_lora.starts_with(&canonical_allowed) {
        anyhow::bail!("lora_path must be under the training output directory");
    }

    let size_bytes = lora_path.metadata().map(|m| m.len()).unwrap_or(0);

    // Auto-resolve thumbnail from training samples when not provided
    let thumbnail = params.thumbnail.or_else(|| {
        let run_name = params.training_run.as_deref()?;
        let samples_dir = modl_root
            .join("training_output")
            .join(run_name)
            .join(run_name)
            .join("samples");
        if !samples_dir.exists() {
            return None;
        }
        let mut best: Option<(u64, String)> = None;
        if let Ok(entries) = std::fs::read_dir(&samples_dir) {
            for entry in entries.flatten() {
                let fname = entry.file_name().to_string_lossy().to_string();
                if let Some(step) = parse_step_from_filename(&fname)
                    && best.as_ref().is_none_or(|(s, _)| step > *s)
                {
                    let rel = format!("training_output/{run_name}/{run_name}/samples/{fname}");
                    best = Some((step, rel));
                }
            }
        }
        best.map(|(_, path)| path)
    });

    let id = format!(
        "lib:{}:{}",
        slug(&params.name),
        &uuid::Uuid::new_v4().to_string()[..8]
    );

    // Fetch job spec for reproducibility metadata
    let job_spec = params.training_run.as_deref().and_then(|run_name| {
        let db = Database::open().ok()?;
        let jobs = db.find_jobs_by_lora_name(run_name).ok()?;
        let spec: serde_json::Value = serde_json::from_str(&jobs.first()?.spec_json).ok()?;
        Some(spec)
    });

    // Merge ai-toolkit config + job spec into one JSON blob
    let merged_config = {
        let mut obj = serde_json::Map::new();
        if let Some(ref cfg) = params.config_json
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(cfg)
        {
            obj.insert("toolkit_config".into(), v);
        }
        if let Some(ref spec) = job_spec {
            obj.insert("job_spec".into(), spec.clone());
        }
        if obj.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(obj).to_string())
        }
    };

    let auto_tag = job_spec.as_ref().and_then(|spec| {
        spec.pointer("/params/lora_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    });

    let record = crate::core::db::LibraryLoraRecord {
        id: id.clone(),
        name: params.name.clone(),
        trigger_word: params.trigger_word,
        base_model: params.base_model.clone(),
        lora_path: params.lora_path.clone(),
        thumbnail,
        step: params.step,
        training_run: params.training_run.clone(),
        config_json: merged_config,
        tags: params.tags.or(auto_tag),
        notes: None,
        size_bytes,
        created_at: String::new(),
    };

    let db = Database::open()?;
    db.insert_library_lora(&record)?;

    // Copy to content-addressed store so the LoRA survives training run deletion
    let install_id = record.id.clone();
    let already_installed = db.is_installed(&install_id).unwrap_or(true);
    if !already_installed {
        let file_name = lora_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("checkpoint.safetensors");

        let sha256 = Store::hash_file(lora_path).unwrap_or_else(|_| "unknown".to_string());

        let store = Store::new(modl_root.join("store"));
        let store_path = store.path_for(&AssetType::Lora, &sha256, file_name);
        store.ensure_dir(&store_path)?;
        if !store_path.exists() {
            std::fs::copy(lora_path, &store_path)?;
        }
        let store_path_str = store_path.to_string_lossy();

        db.update_library_lora_path(&record.id, &store_path_str)?;

        db.insert_installed(&crate::core::db::InstalledModelRecord {
            id: &install_id,
            name: &record.name,
            asset_type: "lora",
            variant: None,
            sha256: &sha256,
            size: size_bytes,
            file_name,
            store_path: &store_path_str,
        })?;

        let meta = serde_json::json!({
            "base_model": record.base_model,
            "trigger_word": record.trigger_word,
            "lora_name": record.name,
        });
        db.insert_artifact(
            &install_id,
            None,
            "lora",
            &store_path_str,
            &sha256,
            size_bytes,
            Some(&meta.to_string()),
        )?;
    }

    Ok(id)
}

/// Slugify a name for use in IDs.
fn slug(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
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
        db.insert_job(
            "job-test-123",
            "train",
            "running",
            "{}",
            "local",
            None,
            None,
        )
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
