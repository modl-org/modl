use anyhow::{Context, Result};
use std::collections::HashSet;
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
        // On MPS (Apple Silicon), exclude fp8 and GGUF variants — they require
        // CUDA-specific kernels (float8 dtype, ggml quantization).
        let is_mps = gpu::detect()
            .map(|g| g.device == gpu::DeviceType::Mps)
            .unwrap_or(false);

        let variant_info: Vec<(String, u64)> = manifest
            .variants
            .iter()
            .filter(|v| {
                if is_mps {
                    let id = v.id.to_lowercase();
                    !id.contains("fp8") && !id.contains("gguf")
                } else {
                    true
                }
            })
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
    let installed_ids: HashSet<String> = installed_list.iter().map(|m| m.id.clone()).collect();

    let plan = resolver::resolve(id, variant, index, &installed_ids)?;

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
