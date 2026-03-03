use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

use crate::core::config::Config;
use crate::core::db::Database;
use crate::core::symlink;

pub async fn run(key: Option<&str>, value: Option<&str>) -> Result<()> {
    match (key, value) {
        // modl config — show current config
        (None, _) => show_config(),

        // modl config set <key> <value>
        (Some(key), Some(value)) => set_config(key, value),

        // modl config <key> — show single value
        (Some(key), None) => show_key(key),
    }
}

fn show_config() -> Result<()> {
    let config = Config::load()?;

    println!("{}", style("storage.root").cyan());
    println!("  {}", config.store_root().display());

    if !config.targets.is_empty() {
        println!();
        println!("{}", style("targets").cyan());
        for t in &config.targets {
            let tool = match t.tool_type {
                crate::core::config::ToolType::Comfyui => "comfyui",
                crate::core::config::ToolType::A1111 => "a1111",
                crate::core::config::ToolType::Invokeai => "invokeai",
                crate::core::config::ToolType::Custom => "custom",
            };
            println!("  {} ({})", t.path.display(), tool);
        }
    }

    if let Some(ref gpu) = config.gpu {
        println!();
        println!("{}", style("gpu.vram_mb").cyan());
        println!("  {}", gpu.vram_mb);
    }

    Ok(())
}

fn show_key(key: &str) -> Result<()> {
    let config = Config::load()?;
    match key {
        "storage.root" => println!("{}", config.store_root().display()),
        "gpu.vram_mb" => match config.gpu {
            Some(ref gpu) => println!("{}", gpu.vram_mb),
            None => println!("(auto-detected)"),
        },
        _ => anyhow::bail!(
            "Unknown config key: {}. Available: storage.root, gpu.vram_mb",
            key
        ),
    }
    Ok(())
}

fn set_config(key: &str, value: &str) -> Result<()> {
    let mut config = Config::load()?;

    match key {
        "storage.root" => {
            let new_root = PathBuf::from(shellexpand::tilde(value).to_string());
            let old_root = config.store_root();

            if new_root == old_root {
                println!("Storage root is already set to {}", new_root.display());
                return Ok(());
            }

            let old_store = old_root.join("store");
            let has_existing_files = old_store.exists()
                && std::fs::read_dir(&old_store)
                    .map(|mut d| d.next().is_some())
                    .unwrap_or(false);

            if has_existing_files {
                migrate_store(&old_root, &new_root, &config)?;
            }

            // Update config
            config.storage.root = PathBuf::from(value);
            config.save()?;

            println!(
                "{} Storage root set to {}",
                style("✓").green().bold(),
                new_root.display()
            );
        }
        "gpu.vram_mb" => {
            let vram: u64 = value
                .parse()
                .context("gpu.vram_mb must be a number (in MB)")?;
            config.gpu = Some(crate::core::config::GpuOverride { vram_mb: vram });
            config.save()?;
            println!(
                "{} GPU VRAM override set to {} MB",
                style("✓").green().bold(),
                vram
            );
        }
        _ => anyhow::bail!(
            "Unknown config key: {}. Available: storage.root, gpu.vram_mb",
            key
        ),
    }

    Ok(())
}

fn migrate_store(
    old_root: &std::path::Path,
    new_root: &std::path::Path,
    config: &Config,
) -> Result<()> {
    let old_store = old_root.join("store");
    let new_store = new_root.join("store");

    println!(
        "{} Migrating store from {} to {}...",
        style("→").cyan(),
        old_store.display(),
        new_store.display()
    );

    // Create new store directory
    std::fs::create_dir_all(&new_store)
        .with_context(|| format!("Failed to create directory: {}", new_store.display()))?;

    // Move the store contents
    // Use rename if same filesystem, otherwise we'd need a recursive copy
    match std::fs::rename(&old_store, &new_store) {
        Ok(()) => {}
        Err(e) => {
            // Cross-device move — need to do recursive copy + delete
            if e.raw_os_error() == Some(18) {
                // EXDEV
                println!(
                    "  {} Cross-device move — copying files (this may take a while)...",
                    style("i").dim()
                );
                copy_dir_recursive(&old_store, &new_store)?;
                std::fs::remove_dir_all(&old_store)
                    .context("Failed to remove old store after copy")?;
            } else {
                return Err(e).context("Failed to move store directory");
            }
        }
    }

    println!("  {} Store files moved", style("✓").green());

    // Update symlinks in all targets to point to new store location
    let db = Database::open()?;
    let models = db.list_installed(None)?;
    let mut updated_links = 0;

    for m in &models {
        let old_path = &m.store_path;
        if let Some(new_path) = old_path
            .strip_prefix(old_store.to_string_lossy().as_ref())
            .or_else(|| old_path.strip_prefix(old_root.to_string_lossy().as_ref()))
        {
            let new_store_path = format!("{}{}", new_root.join("store").display(), new_path);

            // Update all symlinks for this model across targets
            for target in &config.targets {
                if target.symlink {
                    let link_path = crate::compat::symlink_path(
                        &target.path,
                        &target.tool_type,
                        &m.asset_type
                            .parse::<crate::core::manifest::AssetType>()
                            .unwrap_or(crate::core::manifest::AssetType::Checkpoint),
                        &m.file_name,
                    );
                    if link_path.is_symlink() {
                        std::fs::remove_file(&link_path).ok();
                        let new_target = std::path::PathBuf::from(&new_store_path);
                        symlink::create(&link_path, &new_target)?;
                        updated_links += 1;
                    }
                }
            }

            // Update DB
            db.update_store_path(&m.id, &new_store_path)?;
        }
    }

    if updated_links > 0 {
        println!(
            "  {} Updated {} symlinks",
            style("✓").green(),
            updated_links
        );
    }

    // Clean up old root if empty
    if old_root.exists() {
        let is_empty = std::fs::read_dir(old_root)
            .map(|mut d| d.next().is_none())
            .unwrap_or(false);
        if is_empty {
            std::fs::remove_dir(old_root).ok();
        }
    }

    Ok(())
}

fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src).context("Failed to read source directory")? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path).with_context(|| {
                format!(
                    "Failed to copy {} to {}",
                    src_path.display(),
                    dst_path.display()
                )
            })?;
        }
    }
    Ok(())
}
