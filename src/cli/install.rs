use anyhow::{Context, Result};
use console::style;
use indicatif::HumanBytes;
use std::collections::HashSet;

use crate::auth::AuthStore;
use crate::compat;
use crate::core::config::Config;
use crate::core::db::Database;
use crate::core::download;
use crate::core::gpu;
use crate::core::manifest::Manifest;
use crate::core::registry::RegistryIndex;
use crate::core::resolver;
use crate::core::store::Store;
use crate::core::symlink;

pub async fn run(id: &str, variant: Option<&str>, dry_run: bool, force: bool) -> Result<()> {
    let config = Config::load()?;
    let index = RegistryIndex::load()?;
    let db = Database::open()?;

    // Get set of installed model IDs
    let installed_list = db.list_installed(None)?;
    let installed_ids: HashSet<String> = installed_list.iter().map(|m| m.id.clone()).collect();

    // Resolve dependency tree
    let plan = resolver::resolve(id, variant, &index, &installed_ids)?;

    // Auto-select variants based on GPU if not specified
    let vram = config
        .gpu
        .as_ref()
        .map(|g| g.vram_mb)
        .or_else(|| gpu::detect().map(|g| g.vram_mb));

    // Display the plan
    println!(
        "{} Install plan for {}:",
        style("→").cyan(),
        style(id).bold()
    );
    println!();

    let mut total_download: u64 = 0;
    for item in &plan.items {
        let (_file_name, size, variant_label) = get_file_info(&item.manifest, variant, vram);
        let status = if item.already_installed {
            style("installed").green().to_string()
        } else {
            total_download += size;
            style(HumanBytes(size).to_string()).yellow().to_string()
        };
        println!(
            "  {} {} {} {}",
            if item.already_installed {
                style("✓").green()
            } else {
                style("↓").cyan()
            },
            style(&item.manifest.name).bold(),
            style(format!("({})", item.manifest.asset_type)).dim(),
            if let Some(ref v) = variant_label {
                format!("[{}] {}", v, status)
            } else {
                status
            }
        );
    }

    println!();
    if total_download > 0 {
        println!(
            "  Total download: {}",
            style(HumanBytes(total_download)).bold()
        );
    }

    if dry_run {
        println!();
        println!("{}", style("Dry run — nothing downloaded.").dim());
        return Ok(());
    }

    let items_to_install: Vec<_> = plan.items.iter().filter(|i| !i.already_installed).collect();

    if items_to_install.is_empty() {
        println!("{} Everything is already installed.", style("✓").green());
        return Ok(());
    }

    println!();

    // Load auth tokens
    let auth_store = AuthStore::load().unwrap_or_default();
    let store = Store::new(config.store_root());

    // Download each item
    for item in &items_to_install {
        let (file_name, size, selected_variant) = get_file_info(&item.manifest, variant, vram);
        let url = get_download_url(&item.manifest, variant, vram);
        let sha256 = get_sha256(&item.manifest, variant, vram);

        let store_path = store.path_for(&item.manifest.asset_type, &sha256, &file_name);
        store
            .ensure_dir(&store_path)
            .context("Failed to create store directory")?;

        // Get auth token if needed
        let auth_token = item
            .manifest
            .auth
            .as_ref()
            .and_then(|a| auth_store.token_for(&a.provider));

        if item.manifest.auth.as_ref().is_some_and(|a| a.gated) && auth_token.is_none() {
            let provider = &item.manifest.auth.as_ref().unwrap().provider;
            println!(
                "  {} {} requires authentication with {}",
                style("!").yellow(),
                item.manifest.name,
                provider
            );
            println!(
                "    Run: {}",
                style(format!("mods auth {}", provider)).cyan()
            );
            if let Some(ref terms) = item.manifest.auth.as_ref().unwrap().terms_url {
                println!("    Accept terms at: {}", style(terms).underlined());
            }
            anyhow::bail!(
                "Authentication required. Run `mods auth {}` first.",
                provider
            );
        }

        // Download
        if !store_path.exists() || force {
            // Before downloading, check if the file already exists at any target path
            let mut adopted = false;
            if !force {
                for target in &config.targets {
                    if !target.symlink {
                        continue;
                    }
                    let target_path = compat::symlink_path(
                        &target.path,
                        &target.tool_type,
                        &item.manifest.asset_type,
                        &file_name,
                    );
                    if target_path.exists() && !target_path.is_symlink() {
                        // Real file exists at target — check its hash
                        print!(
                            "  {} Found existing {} — verifying... ",
                            style("?").yellow(),
                            target_path.display()
                        );
                        if Store::verify_hash(&target_path, &sha256)? {
                            println!("{}", style("match!").green());
                            // Move file to store, create symlink
                            std::fs::rename(&target_path, &store_path)
                                .with_context(|| format!(
                                    "Failed to move {} to store",
                                    target_path.display()
                                ))?;
                            symlink::create(&target_path, &store_path)?;
                            println!(
                                "  {} Adopted → {}",
                                style("→").dim(),
                                store_path.display()
                            );
                            adopted = true;
                            break;
                        } else {
                            println!("{}", style("hash mismatch, downloading fresh").yellow());
                        }
                    }
                }
            }

            if !adopted {
                download::download_file(&url, &store_path, Some(size), auth_token.as_deref())
                    .await
                    .with_context(|| format!("Failed to download {}", item.manifest.name))?;

                // Verify hash
                if !Store::verify_hash(&store_path, &sha256)? {
                    std::fs::remove_file(&store_path).ok();
                    anyhow::bail!(
                        "SHA256 mismatch for {}. File deleted. Try again.",
                        item.manifest.name
                    );
                }
            }
        } else {
            println!(
                "  {} {} already in store, skipping download",
                style("✓").green(),
                style(&file_name).dim()
            );
        }

        // Create symlinks to all configured targets
        for target in &config.targets {
            if target.symlink {
                let link_path = compat::symlink_path(
                    &target.path,
                    &target.tool_type,
                    &item.manifest.asset_type,
                    &file_name,
                );
                match symlink::create(&link_path, &store_path) {
                    Ok(()) => {
                        println!("  {} Linked → {}", style("→").dim(), link_path.display());
                    }
                    Err(e) => {
                        eprintln!("  {} Symlink failed: {}", style("!").yellow(), e);
                    }
                }
            }
        }

        // Record in database
        db.insert_installed(
            &item.manifest.id,
            &item.manifest.name,
            &item.manifest.asset_type.to_string(),
            selected_variant.as_deref(),
            &sha256,
            size,
            &file_name,
            &store_path.to_string_lossy(),
        )?;

        println!(
            "  {} {}",
            style("✓").green(),
            style(&item.manifest.name).bold()
        );
    }

    println!();
    println!(
        "{} Installed {} successfully.",
        style("✓").green().bold(),
        style(id).bold()
    );

    Ok(())
}

/// Get file name, size, and variant label for a manifest
fn get_file_info(
    manifest: &Manifest,
    requested_variant: Option<&str>,
    vram: Option<u64>,
) -> (String, u64, Option<String>) {
    if let Some(variant) = select_variant(manifest, requested_variant, vram) {
        (variant.file.clone(), variant.size, Some(variant.id.clone()))
    } else if let Some(ref file) = manifest.file {
        (
            manifest.id.clone() + "." + file.format.as_deref().unwrap_or("safetensors"),
            file.size,
            None,
        )
    } else {
        (format!("{}.safetensors", manifest.id), 0, None)
    }
}

fn get_download_url(
    manifest: &Manifest,
    requested_variant: Option<&str>,
    vram: Option<u64>,
) -> String {
    if let Some(variant) = select_variant(manifest, requested_variant, vram) {
        variant.url.clone()
    } else if let Some(ref file) = manifest.file {
        file.url.clone()
    } else {
        String::new()
    }
}

fn get_sha256(manifest: &Manifest, requested_variant: Option<&str>, vram: Option<u64>) -> String {
    if let Some(variant) = select_variant(manifest, requested_variant, vram) {
        variant.sha256.clone()
    } else if let Some(ref file) = manifest.file {
        file.sha256.clone()
    } else {
        String::new()
    }
}

fn select_variant<'a>(
    manifest: &'a Manifest,
    requested: Option<&str>,
    vram: Option<u64>,
) -> Option<&'a crate::core::manifest::Variant> {
    if manifest.variants.is_empty() {
        return None;
    }

    // If user requested a specific variant
    if let Some(req) = requested {
        return manifest.variants.iter().find(|v| v.id == req);
    }

    // Auto-select based on VRAM
    if let Some(vram_mb) = vram {
        let variant_info: Vec<(String, u64)> = manifest
            .variants
            .iter()
            .map(|v| (v.id.clone(), v.vram_required.unwrap_or(0)))
            .collect();
        if let Some(selected_id) = gpu::select_variant(vram_mb, &variant_info) {
            return manifest.variants.iter().find(|v| v.id == selected_id);
        }
    }

    // Fallback: pick the first (usually smallest or most common)
    manifest.variants.first()
}
