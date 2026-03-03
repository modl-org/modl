use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};

use crate::core::config::{Config, TargetConfig, ToolType};
use crate::core::db::Database;
use crate::core::manifest::Manifest;
use crate::core::registry::RegistryIndex;
use crate::core::store::Store;
use crate::core::symlink;

pub async fn run(comfyui: Option<&str>, a1111: Option<&str>) -> Result<()> {
    if comfyui.is_none() && a1111.is_none() {
        anyhow::bail!(
            "Specify a tool to link:\n\n  \
             modl link --comfyui ~/ComfyUI\n  \
             modl link --a1111 ~/stable-diffusion-webui\n\n\
             This scans the tool's model folders and adopts recognized files into the modl store."
        );
    }

    let mut config = Config::load()?;
    let db = Database::open()?;

    // Try to load registry for matching, but don't require it
    let index = RegistryIndex::load().ok();

    if let Some(path) = comfyui {
        link_tool(path, ToolType::Comfyui, &mut config, &db, index.as_ref()).await?;
    }
    if let Some(path) = a1111 {
        link_tool(path, ToolType::A1111, &mut config, &db, index.as_ref()).await?;
    }

    config.save()?;
    println!();
    println!(
        "{} Config updated. Future installs will symlink to these targets.",
        style("✓").green().bold()
    );

    Ok(())
}

/// Info about a matched file: which manifest and variant it belongs to
struct MatchedFile {
    manifest: Manifest,
    variant_id: Option<String>,
    hash: String,
    file_path: PathBuf,
    size: u64,
}

async fn link_tool(
    path_str: &str,
    tool_type: ToolType,
    config: &mut Config,
    db: &Database,
    index: Option<&RegistryIndex>,
) -> Result<()> {
    let path = PathBuf::from(shellexpand::tilde(path_str).to_string());

    if !path.exists() {
        anyhow::bail!("Path does not exist: {}", path.display());
    }

    let tool_name = match tool_type {
        ToolType::Comfyui => "ComfyUI",
        ToolType::A1111 => "A1111",
        ToolType::Invokeai => "InvokeAI",
        ToolType::Custom => "Custom",
    };

    println!(
        "{} Scanning {} at {}...",
        style("→").cyan(),
        tool_name,
        path.display()
    );

    // Find model files (skip files that are already symlinks — already managed)
    let models_dir = path.join("models");
    if !models_dir.exists() {
        println!(
            "  {} No 'models' directory found at {}",
            style("!").yellow(),
            path.display()
        );
    }

    let files = find_model_files(&models_dir)?;
    println!("  Found {} model files", files.len());

    let store = Store::new(config.store_root());
    let mut matched_files: Vec<MatchedFile> = Vec::new();

    if !files.is_empty() {
        // Build a hash → (manifest, variant_id) lookup and a set of known file sizes
        let mut known_sizes: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut hash_map: std::collections::HashMap<String, (&Manifest, Option<&str>)> =
            std::collections::HashMap::new();

        if let Some(idx) = index {
            for m in &idx.items {
                for v in &m.variants {
                    hash_map.insert(v.sha256.clone(), (m, Some(v.id.as_str())));
                    known_sizes.insert(v.size);
                }
                if let Some(ref f) = m.file {
                    hash_map.insert(f.sha256.clone(), (m, None));
                    known_sizes.insert(f.size);
                }
            }
        }

        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Hashing [{bar:30}] {pos}/{len}")
                .unwrap(),
        );

        let mut skipped = 0u64;
        for file in &files {
            pb.inc(1);

            // Size pre-filter: skip hashing files whose size doesn't match any known variant
            if let Ok(meta) = std::fs::metadata(file)
                && !known_sizes.is_empty()
                && !known_sizes.contains(&meta.len())
            {
                skipped += 1;
                continue;
            }

            if let Ok(hash) = Store::hash_file(file)
                && let Some((manifest, variant_id)) = hash_map.get(&hash)
            {
                let size = std::fs::metadata(file).map(|m| m.len()).unwrap_or(0);
                matched_files.push(MatchedFile {
                    manifest: (*manifest).clone(),
                    variant_id: variant_id.map(String::from),
                    hash,
                    file_path: file.clone(),
                    size,
                });
            }
        }

        pb.finish_and_clear();
        if skipped > 0 {
            println!(
                "  {} Skipped {} files (size mismatch)",
                style("i").dim(),
                skipped
            );
        }
        println!(
            "  {} Matched {} files to registry entries",
            style("✓").green(),
            matched_files.len()
        );
    }

    // Migrate matched files: move to store, replace with symlink
    if !matched_files.is_empty() {
        println!();
        println!("{} Adopting matched files into store...", style("→").cyan());

        for mf in &matched_files {
            let file_name = mf
                .file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            let store_path = store.path_for(&mf.manifest.asset_type, &mf.hash, file_name);

            if store_path.exists() {
                // Already in store (e.g., from a previous link or install).
                // Just replace the original with a symlink if it's not one already.
                if !mf.file_path.is_symlink() {
                    std::fs::remove_file(&mf.file_path)?;
                    symlink::create(&mf.file_path, &store_path)?;
                    println!(
                        "  {} {} (already in store, linked)",
                        style("→").dim(),
                        mf.manifest.name,
                    );
                } else {
                    println!(
                        "  {} {} (already managed)",
                        style("i").dim(),
                        mf.manifest.name,
                    );
                }
            } else {
                // Move file to store, replace with symlink
                store.ensure_dir(&store_path)?;
                std::fs::rename(&mf.file_path, &store_path).with_context(|| {
                    format!(
                        "Failed to move {} to store at {}",
                        mf.file_path.display(),
                        store_path.display()
                    )
                })?;
                symlink::create(&mf.file_path, &store_path)?;
                println!("  {} {} → store", style("✓").green(), mf.manifest.name,);
            }

            // Record in DB
            db.insert_installed(
                &mf.manifest.id,
                &mf.manifest.name,
                &mf.manifest.asset_type.to_string(),
                mf.variant_id.as_deref(),
                &mf.hash,
                mf.size,
                file_name,
                &store_path.to_string_lossy(),
            )?;
        }
    }

    // Add to config targets if not already present
    let already_targeted = config.targets.iter().any(|t| t.path == path);

    if !already_targeted {
        config.targets.push(TargetConfig {
            path,
            tool_type,
            symlink: true,
        });
        println!("  {} Added as symlink target", style("✓").green());
    } else {
        println!("  {} Already configured as target", style("i").dim());
    }

    Ok(())
}

fn find_model_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if !dir.exists() {
        return Ok(files);
    }
    find_model_files_recursive(dir, &mut files)?;
    Ok(files)
}

fn find_model_files_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir).context("Failed to read directory")? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && !path.is_symlink() {
            find_model_files_recursive(&path, files)?;
        } else if path.is_symlink() {
            // Skip files already managed by modl
            continue;
        } else if let Some("safetensors" | "ckpt" | "pt" | "pth" | "bin" | "gguf") =
            path.extension().and_then(|e| e.to_str())
        {
            files.push(path);
        }
    }
    Ok(())
}
