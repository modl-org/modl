use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};

use crate::core::config::{Config, TargetConfig, ToolType};
use crate::core::db::Database;
use crate::core::registry::RegistryIndex;
use crate::core::store::Store;

pub async fn run(comfyui: Option<&str>, a1111: Option<&str>) -> Result<()> {
    if comfyui.is_none() && a1111.is_none() {
        anyhow::bail!("Specify at least one: --comfyui <path> or --a1111 <path>");
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

    // Find model files
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

    if !files.is_empty() {
        // Build a hash → manifest lookup and a set of known file sizes for pre-filtering
        let mut known_sizes: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let hash_map: std::collections::HashMap<String, &crate::core::manifest::Manifest> =
            if let Some(idx) = index {
                let mut map = std::collections::HashMap::new();
                for m in &idx.items {
                    for v in &m.variants {
                        map.insert(v.sha256.clone(), m);
                        known_sizes.insert(v.size);
                    }
                    if let Some(ref f) = m.file {
                        map.insert(f.sha256.clone(), m);
                        known_sizes.insert(f.size);
                    }
                }
                map
            } else {
                std::collections::HashMap::new()
            };

        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Hashing [{bar:30}] {pos}/{len}")
                .unwrap(),
        );

        let mut matched = 0;
        let mut skipped = 0u64;
        for file in &files {
            pb.inc(1);

            // Size pre-filter: skip hashing files whose size doesn't match any known variant
            if let Ok(meta) = std::fs::metadata(file) {
                if !known_sizes.is_empty() && !known_sizes.contains(&meta.len()) {
                    skipped += 1;
                    continue;
                }
            }

            if let Ok(hash) = Store::hash_file(file)
                && let Some(manifest) = hash_map.get(&hash)
            {
                let size = std::fs::metadata(file).map(|m| m.len()).unwrap_or(0);
                let file_name = file
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");

                // Register in DB
                db.insert_installed(
                    &manifest.id,
                    &manifest.name,
                    &manifest.asset_type.to_string(),
                    None,
                    &hash,
                    size,
                    file_name,
                    &file.to_string_lossy(),
                )?;
                matched += 1;
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
            matched
        );
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
        } else if let Some("safetensors" | "ckpt" | "pt" | "pth" | "bin" | "gguf") =
            path.extension().and_then(|e| e.to_str())
        {
            files.push(path);
        }
    }
    Ok(())
}
