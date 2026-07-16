use anyhow::{Context, Result};

use std::path::PathBuf;

use super::config::Config;
use super::db::{Database, InstalledModelRecord};
use super::download;
use super::gpu;
use super::huggingface;
use super::manifest::{AssetType, Manifest, Variant};
use super::registry::RegistryIndex;
use super::resolver::{self, InstallPlan};
use super::store::Store;
use super::symlink;
use crate::auth::AuthStore;
use crate::compat;

/// Information about a resolved variant for display/download purposes.
pub struct ResolvedFileInfo {
    pub file_name: String,
    pub size: u64,
    pub variant_label: Option<String>,
    pub url: String,
    pub sha256: String,
}

/// Result of installing a single item.
pub struct InstallItemResult {
    pub name: String,
    pub adopted: bool,
}

/// Result of an HF direct pull.
#[allow(dead_code)]
pub struct HfPullResult {
    pub display_name: String,
    pub asset_type: AssetType,
    pub size: u64,
    pub already_installed: bool,
}

/// Select the best variant from a manifest based on user request or VRAM.
pub fn select_variant<'a>(
    manifest: &'a Manifest,
    requested: Option<&str>,
    vram: Option<u64>,
) -> Option<&'a Variant> {
    if manifest.variants.is_empty() {
        return None;
    }

    if let Some(req) = requested {
        return manifest.variants.iter().find(|v| v.id == req);
    }

    if let Some(vram_mb) = vram {
        // On MPS (Apple Silicon), exclude fp8 variants — float8 dtype requires CUDA.
        // GGUF is fine: weights are dequantized to float16/bfloat16 at load time.
        let variant_info: Vec<(String, u64)> = manifest
            .variants
            .iter()
            .filter(|v| gpu::variant_compatible_with_device(&v.id, v.precision.as_deref()))
            .map(|v| (v.id.clone(), v.vram_required.unwrap_or(0)))
            .collect();
        if let Some(selected_id) = gpu::select_variant(vram_mb, &variant_info) {
            return manifest.variants.iter().find(|v| v.id == selected_id);
        }
    }

    manifest.variants.first()
}

/// Get file info (name, size, variant label, url, sha256) for a manifest.
pub fn resolve_file_info(
    manifest: &Manifest,
    requested_variant: Option<&str>,
    vram: Option<u64>,
) -> ResolvedFileInfo {
    if let Some(variant) = select_variant(manifest, requested_variant, vram) {
        ResolvedFileInfo {
            file_name: variant.file.clone(),
            size: variant.size,
            variant_label: Some(variant.id.clone()),
            url: variant.url.clone(),
            sha256: variant.sha256.clone(),
        }
    } else if let Some(ref file) = manifest.file {
        ResolvedFileInfo {
            file_name: manifest.id.clone() + "." + file.format.as_deref().unwrap_or("safetensors"),
            size: file.size,
            variant_label: None,
            url: file.url.clone(),
            sha256: file.sha256.clone(),
        }
    } else {
        ResolvedFileInfo {
            file_name: format!("{}.safetensors", manifest.id),
            size: 0,
            variant_label: None,
            url: String::new(),
            sha256: String::new(),
        }
    }
}

/// Resolve the full install plan for a registry model.
pub fn resolve_plan(
    id: &str,
    variant: Option<&str>,
    index: &RegistryIndex,
    db: &Database,
) -> Result<(InstallPlan, Option<u64>)> {
    let config = Config::load()?;
    let installed_list = db.list_installed(None)?;
    let installed_map: std::collections::HashMap<String, Option<String>> = installed_list
        .iter()
        .map(|m| (m.id.clone(), m.variant.clone()))
        .collect();

    let plan = resolver::resolve(id, variant, index, &installed_map)?;

    let vram = config
        .gpu
        .as_ref()
        .map(|g| g.vram_mb)
        .or_else(|| gpu::detect().map(|g| g.vram_mb));

    Ok((plan, vram))
}

/// Download and install a single manifest item into the store.
///
/// This handles: auth token resolution, existing file adoption from targets,
/// download, SHA256 verification, symlink creation, and DB registration.
#[allow(clippy::too_many_arguments)]
pub async fn install_item(
    manifest: &Manifest,
    effective_variant: Option<&str>,
    vram: Option<u64>,
    config: &Config,
    store: &Store,
    auth_store: &AuthStore,
    db: &Database,
    force: bool,
) -> Result<InstallItemResult> {
    let info = resolve_file_info(manifest, effective_variant, vram);

    let store_path = store.path_for(&manifest.asset_type, &info.sha256, &info.file_name);
    store
        .ensure_dir(&store_path)
        .context("Failed to create store directory")?;

    // Resolve auth token
    let auth_token = manifest
        .auth
        .as_ref()
        .and_then(|a| auth_store.token_for(&a.provider))
        .or_else(|| {
            if info.url.contains("huggingface.co") {
                auth_store.token_for("huggingface")
            } else {
                None
            }
        });

    // Check gated auth requirement
    if manifest.auth.as_ref().is_some_and(|a| a.gated) && auth_token.is_none() {
        let provider = &manifest.auth.as_ref().unwrap().provider;
        anyhow::bail!(
            "{} requires authentication with {}. Run `modl auth {}` first.",
            manifest.name,
            provider,
            provider
        );
    }

    let mut adopted = false;

    if !store_path.exists() || force {
        // Check if file exists at any target path (adopt instead of download)
        if !force {
            for target in &config.targets {
                if !target.symlink {
                    continue;
                }
                let target_path = compat::symlink_path(
                    &target.path,
                    &target.tool_type,
                    &manifest.asset_type,
                    &info.file_name,
                );
                if target_path.exists()
                    && !target_path.is_symlink()
                    && Store::verify_hash(&target_path, &info.sha256)?
                {
                    std::fs::rename(&target_path, &store_path).with_context(|| {
                        format!("Failed to move {} to store", target_path.display())
                    })?;
                    symlink::create(&target_path, &store_path)?;
                    adopted = true;
                    break;
                }
            }
        }

        if !adopted {
            download::download_file(
                &info.url,
                &store_path,
                Some(info.size),
                auth_token.as_deref(),
            )
            .await
            .with_context(|| format!("Failed to download {}", manifest.name))?;

            // Verify hash — warn on mismatch but don't block install.
            // Registry hashes can go stale when upstream files are re-published.
            if !info.sha256.is_empty() {
                match Store::verify_hash(&store_path, &info.sha256) {
                    Ok(true) => {}
                    Ok(false) => {
                        let actual = Store::hash_file(&store_path).unwrap_or_default();
                        eprintln!(
                            "  {} SHA256 mismatch for {}. File kept — upstream may have changed.",
                            console::style("⚠").yellow(),
                            manifest.name,
                        );
                        eprintln!("    expected: {}\n    got:      {}", info.sha256, actual,);
                    }
                    Err(e) => {
                        eprintln!(
                            "  {} Could not verify hash for {}: {}",
                            console::style("⚠").yellow(),
                            manifest.name,
                            e,
                        );
                    }
                }
            }
        }
    }

    // Create symlinks to all configured targets
    for target in &config.targets {
        if target.symlink {
            let link_path = compat::symlink_path(
                &target.path,
                &target.tool_type,
                &manifest.asset_type,
                &info.file_name,
            );
            symlink::create(&link_path, &store_path).ok();
        }
    }

    // Record in database
    let actual_size = std::fs::metadata(&store_path)
        .map(|m| m.len())
        .unwrap_or(info.size);
    db.insert_installed(&InstalledModelRecord {
        id: &manifest.id,
        name: &manifest.name,
        asset_type: &manifest.asset_type.to_string(),
        variant: info.variant_label.as_deref(),
        sha256: &info.sha256,
        size: actual_size,
        file_name: &info.file_name,
        store_path: &store_path.to_string_lossy(),
    })?;

    Ok(InstallItemResult {
        name: manifest.name.clone(),
        adopted,
    })
}

/// Pull a model directly from HuggingFace (hf:owner/repo).
pub async fn hf_pull(
    repo_id: &str,
    variant: Option<&str>,
    force: bool,
) -> Result<(HfPullResult, PathBuf)> {
    let config = Config::load()?;
    let db = Database::open()?;
    let auth_store = AuthStore::load().unwrap_or_default();
    let hf_token = auth_store.token_for("huggingface");

    let model = huggingface::get_model(repo_id, hf_token.as_deref()).await?;
    let resolved = huggingface::resolve_download(repo_id, variant, hf_token.as_deref()).await?;

    let asset_type_str = huggingface::guess_asset_type(&model, &resolved.filename);
    let asset_type: AssetType = asset_type_str.parse().unwrap_or(AssetType::Checkpoint);

    let display_name = repo_id
        .split('/')
        .next_back()
        .unwrap_or(repo_id)
        .to_string();

    let local_id = format!("hf:{}", repo_id);

    // Check if already installed
    let installed_list = db.list_installed(None)?;
    if !force && installed_list.iter().any(|m| m.id == local_id) {
        return Ok((
            HfPullResult {
                display_name,
                asset_type,
                size: resolved.size,
                already_installed: true,
            },
            PathBuf::new(),
        ));
    }

    let store = Store::new(config.store_root());

    // Use a temp prefix derived from repo ID
    let temp_prefix = format!("{:0>16x}", {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        repo_id.hash(&mut hasher);
        hasher.finish()
    });

    let store_path = store.path_for(&asset_type, &temp_prefix, &resolved.filename);
    store.ensure_dir(&store_path)?;

    if !store_path.exists() || force {
        download::download_file(
            &resolved.url,
            &store_path,
            if resolved.size > 0 {
                Some(resolved.size)
            } else {
                None
            },
            hf_token.as_deref(),
        )
        .await
        .with_context(|| format!("Failed to download {}", display_name))?;
    }

    // Compute real SHA256 and move to content-addressed path
    let sha256 =
        Store::hash_file(&store_path).context("Failed to compute SHA256 of downloaded file")?;

    let real_prefix = &sha256[..16];
    let final_path = if temp_prefix != real_prefix {
        let real_path = store.path_for(&asset_type, &sha256, &resolved.filename);
        if real_path != store_path {
            store.ensure_dir(&real_path)?;
            if real_path.exists() {
                std::fs::remove_file(&store_path).ok();
            } else {
                std::fs::rename(&store_path, &real_path)
                    .or_else(|_| {
                        std::fs::copy(&store_path, &real_path)?;
                        std::fs::remove_file(&store_path)?;
                        Ok::<(), std::io::Error>(())
                    })
                    .context("Failed to move file to content-addressed path")?;
            }
            if let Some(parent) = store_path.parent() {
                std::fs::remove_dir(parent).ok();
            }
        }
        store.path_for(&asset_type, &sha256, &resolved.filename)
    } else {
        store_path
    };

    let actual_size = std::fs::metadata(&final_path)
        .map(|m| m.len())
        .unwrap_or(resolved.size);

    // Create symlinks
    for target in &config.targets {
        if target.symlink {
            let link_path = compat::symlink_path(
                &target.path,
                &target.tool_type,
                &asset_type,
                &resolved.filename,
            );
            symlink::create(&link_path, &final_path).ok();
        }
    }

    // Record in database
    db.insert_installed(&InstalledModelRecord {
        id: &local_id,
        name: &display_name,
        asset_type: &asset_type.to_string(),
        variant: None,
        sha256: &sha256,
        size: actual_size,
        file_name: &resolved.filename,
        store_path: &final_path.to_string_lossy(),
    })?;

    Ok((
        HfPullResult {
            display_name,
            asset_type,
            size: actual_size,
            already_installed: false,
        },
        final_path,
    ))
}

/// Result of `register_file`.
#[derive(Debug)]
pub struct RegisterFileResult {
    pub id: String,
    pub sha256: String,
    pub store_path: PathBuf,
    /// True when the id was already registered with this content — the call
    /// was a no-op.
    pub already_registered: bool,
}

/// Register a local model file into the content-addressed store: hash,
/// verify, copy into `store/<type>/<sha16>/<file>` (no-op when the file
/// already lives there), record in SQLite and the shared store index.
///
/// This is how a file that arrived outside `modl pull` becomes referenceable
/// by name (`--lora <name>`, workflow `lora:`). It is also the pod-side half
/// of LoRA push: the file is rsynced straight into the store path, then this
/// registers it. Idempotent — re-registering the same content is a no-op.
pub fn register_file(
    path: &std::path::Path,
    name: &str,
    id: Option<&str>,
    asset_type: &AssetType,
    expected_sha256: Option<&str>,
    db: &Database,
    store_root: &std::path::Path,
) -> Result<RegisterFileResult> {
    if !path.is_file() {
        anyhow::bail!("File not found: {}", path.display());
    }
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .context("File has no valid file name")?
        .to_string();

    let sha256 =
        Store::hash_file(path).with_context(|| format!("Failed to hash {}", path.display()))?;
    if let Some(expected) = expected_sha256
        && !sha256.eq_ignore_ascii_case(expected)
    {
        anyhow::bail!(
            "SHA256 mismatch for {}: expected {expected}, got {sha256} — transfer corrupted?",
            path.display()
        );
    }

    let store = Store::new(store_root.to_path_buf());
    let store_path = store.path_for(asset_type, &sha256, &file_name);
    // Copy into the store unless the source already IS the store path
    // (LoRA push rsyncs directly there before registering).
    let already_in_place = path
        .canonicalize()
        .ok()
        .is_some_and(|p| p == store_path || store_path.exists());
    if !already_in_place {
        store.ensure_dir(&store_path)?;
        std::fs::copy(path, &store_path).with_context(|| {
            format!(
                "Failed to copy {} → {}",
                path.display(),
                store_path.display()
            )
        })?;
    }

    let id = id
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("local/{asset_type}/{name}"));
    let already_registered = db
        .list_installed(None)?
        .iter()
        .any(|m| m.id == id && m.sha256 == sha256);
    if !already_registered {
        let size = std::fs::metadata(&store_path).map(|m| m.len()).unwrap_or(0);
        db.insert_installed(&InstalledModelRecord {
            id: &id,
            name,
            asset_type: &asset_type.to_string(),
            variant: None,
            sha256: &sha256,
            size,
            file_name: &file_name,
            store_path: &store_path.to_string_lossy(),
        })?;
    }

    if matches!(asset_type, AssetType::Lora) {
        let _ = super::artifacts::create_lora_symlinks(&store_path, name, &file_name);
    }

    Ok(RegisterFileResult {
        id,
        sha256,
        store_path,
        already_registered,
    })
}

#[cfg(test)]
mod register_tests {
    use super::*;
    use std::path::Path;

    fn setup() -> (tempfile::TempDir, Database, std::path::PathBuf) {
        let dir = tempfile::TempDir::new().unwrap();
        let db = Database::open_at(&dir.path().join("db.sqlite")).unwrap();
        let file = dir.path().join("alice.safetensors");
        std::fs::write(&file, b"fake lora bytes").unwrap();
        (dir, db, file)
    }

    #[test]
    fn registers_copies_and_is_idempotent() {
        let (dir, db, file) = setup();

        let r =
            register_file(&file, "alice", None, &AssetType::Vae, None, &db, dir.path()).unwrap();
        assert!(!r.already_registered);
        assert_eq!(r.id, "local/vae/alice");
        assert!(r.store_path.exists());
        assert!(r.store_path.starts_with(dir.path()));
        let installed = db.list_installed(Some("vae")).unwrap();
        assert_eq!(installed.len(), 1);
        assert_eq!(installed[0].name, "alice");
        assert_eq!(installed[0].sha256, r.sha256);

        // Second call: same content, same id — no-op.
        let r2 =
            register_file(&file, "alice", None, &AssetType::Vae, None, &db, dir.path()).unwrap();
        assert!(r2.already_registered);
        assert_eq!(db.list_installed(Some("vae")).unwrap().len(), 1);
    }

    #[test]
    fn explicit_id_and_sha_verification() {
        let (dir, db, file) = setup();

        let err = register_file(
            &file,
            "alice",
            Some("train:alice:1234"),
            &AssetType::Vae,
            Some(&"0".repeat(64)),
            &db,
            dir.path(),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("SHA256 mismatch"), "{err}");
        assert!(db.list_installed(None).unwrap().is_empty());

        let sha = Store::hash_file(&file).unwrap();
        let r = register_file(
            &file,
            "alice",
            Some("train:alice:1234"),
            &AssetType::Vae,
            Some(&sha),
            &db,
            dir.path(),
        )
        .unwrap();
        assert_eq!(r.id, "train:alice:1234");
    }

    #[test]
    fn missing_file_errors() {
        let (dir, db, _) = setup();
        assert!(
            register_file(
                Path::new("/nonexistent/x.safetensors"),
                "x",
                None,
                &AssetType::Vae,
                None,
                &db,
                dir.path(),
            )
            .is_err()
        );
    }
}
