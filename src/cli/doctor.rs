use anyhow::Result;
use console::style;
use std::collections::{HashMap, HashSet};

use crate::core::config::Config;
use crate::core::db::Database;
use crate::core::registry::RegistryIndex;
use crate::core::store::Store;
use crate::core::symlink;

pub async fn run(verify_hashes: bool, repair: bool) -> Result<()> {
    println!(
        "{}",
        style("modl doctor — running diagnostics").bold().cyan()
    );
    println!();

    let config = Config::load()?;
    let db = Database::open()?;
    let models = db.list_installed(None)?;
    let mut issues = 0;

    // 1. Check for broken symlinks in target directories
    println!("{} Checking symlinks...", style("→").cyan());
    for target in &config.targets {
        let broken = symlink::find_broken(&target.path)?;
        if broken.is_empty() {
            println!(
                "  {} {} — all symlinks valid",
                style("✓").green(),
                target.path.display()
            );
        } else {
            for b in &broken {
                println!("  {} Broken symlink: {}", style("✗").red(), b.display());
                issues += 1;
            }
        }
    }

    // 2. Verify store files exist with correct size
    println!();
    println!("{} Checking store files...", style("→").cyan());
    let mut store_ok = true;
    let tracked_paths: HashSet<String> = models.iter().map(|m| m.store_path.clone()).collect();

    for m in &models {
        let path = std::path::Path::new(&m.store_path);
        if !path.exists() {
            println!(
                "  {} Missing store file for '{}': {}",
                style("✗").red(),
                m.name,
                m.store_path
            );
            issues += 1;
            store_ok = false;
        } else if let Ok(meta) = std::fs::metadata(path)
            && meta.len() != m.size
        {
            println!(
                "  {} Size mismatch for '{}' — expected {} bytes, got {}",
                style("✗").red(),
                m.name,
                m.size,
                meta.len()
            );
            println!(
                "    Fix: {}",
                style(format!("modl uninstall {} && modl install {}", m.id, m.id)).cyan()
            );
            issues += 1;
            store_ok = false;
        }
    }
    if store_ok && !models.is_empty() {
        println!(
            "  {} All {} tracked store files present and sizes match",
            style("✓").green(),
            models.len()
        );
    } else if models.is_empty() {
        println!("  {} No models tracked in database", style("i").dim());
    }

    // 3. Check for orphaned store files (on disk but not in DB)
    println!();
    println!("{} Checking for orphaned store files...", style("→").cyan());
    let mut orphans = Vec::new();

    // Scan the main storage root (e.g. ~/modl/store/)
    let store_dir = config.store_root().join("store");
    scan_orphans(&store_dir, &tracked_paths, &mut orphans);

    // Also scan the config dir store (e.g. ~/.modl/store/) — training outputs
    // and locally-trained LoRAs may live here
    let config_dir_store = Config::default_path()
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("store");
    if config_dir_store != store_dir {
        scan_orphans(&config_dir_store, &tracked_paths, &mut orphans);
    }

    if orphans.is_empty() {
        println!("  {} No orphaned files", style("✓").green());
    } else {
        let total_size: u64 = orphans.iter().map(|(_, _, _, _, s)| s).sum();
        println!(
            "  {} Found {} orphaned file{} ({}) in store not tracked by database:",
            style("!").yellow(),
            orphans.len(),
            if orphans.len() == 1 { "" } else { "s" },
            format_size(total_size)
        );
        for (path, asset_type, _hash, name, size) in &orphans {
            println!(
                "    {} [{}] {} ({})",
                style("·").dim(),
                asset_type,
                name,
                format_size(*size)
            );
            let _ = path; // used below in repair
        }

        if !repair {
            println!();
            println!(
                "  {} Run {} to re-register these files in the database",
                style("i").dim(),
                style("modl doctor --repair").cyan()
            );
        }
        issues += orphans.len();
    }

    // 4. Repair: re-populate DB from orphan files
    if repair && !orphans.is_empty() {
        println!();
        println!(
            "{} Repairing — registering orphaned files...",
            style("→").cyan()
        );
        let mut repaired = 0;

        // Build a hash-prefix → (id, name, variant) lookup from the registry
        let registry_lookup = build_registry_lookup();

        for (store_path, asset_type, hash_prefix, file_name, size) in &orphans {
            // Use the directory name as the SHA256 prefix — the store is
            // content-addressed so the dir name IS the hash. Full verification
            // can be done later with --verify-hashes. This avoids hashing
            // multi-GB files during repair.
            let sha256 = hash_prefix.clone();

            // Try to match against the registry for proper ID/name
            let (id, display_name, variant) = if let Some((reg_id, reg_name, reg_variant)) =
                registry_lookup.get(hash_prefix.as_str())
            {
                (reg_id.clone(), reg_name.clone(), reg_variant.clone())
            } else {
                // Fallback: derive from filename
                let stem = file_name
                    .strip_suffix(".safetensors")
                    .or_else(|| file_name.strip_suffix(".ckpt"))
                    .or_else(|| file_name.strip_suffix(".bin"))
                    .or_else(|| file_name.strip_suffix(".pt"))
                    .unwrap_or(file_name);
                let id = format!("local/{}/{}", asset_type, stem);
                let display_name = stem.to_string();
                (id, display_name, None)
            };

            if let Err(e) = db.insert_installed(&crate::core::db::InstalledModelRecord {
                id: &id,
                name: &display_name,
                asset_type,
                variant: variant.as_deref(),
                sha256: &sha256,
                size: *size,
                file_name,
                store_path,
            }) {
                println!(
                    "  {} Failed to register {}: {}",
                    style("✗").red(),
                    file_name,
                    e
                );
            } else {
                println!(
                    "  {} Registered [{}] {} ({})",
                    style("✓").green(),
                    asset_type,
                    display_name,
                    format_size(*size)
                );
                repaired += 1;
            }
        }

        println!();
        if repaired > 0 {
            println!(
                "  {} Recovered {} model{}. Run {} to verify.",
                style("✓").green(),
                repaired,
                if repaired == 1 { "" } else { "s" },
                style("modl model list").cyan()
            );
        }
    }

    // 5. Verify hashes (opt-in, slow for large files)
    if verify_hashes {
        println!();
        println!("{} Verifying file hashes...", style("→").cyan());
        let mut hash_issues = 0;
        for m in &models {
            let path = std::path::Path::new(&m.store_path);
            if path.exists() {
                match Store::verify_hash(path, &m.sha256) {
                    Ok(true) => {}
                    Ok(false) => {
                        println!(
                            "  {} Hash mismatch for '{}' — file may be corrupted",
                            style("✗").red(),
                            m.name
                        );
                        println!(
                            "    Fix: {}",
                            style(format!("modl uninstall {} && modl install {}", m.id, m.id))
                                .cyan()
                        );
                        hash_issues += 1;
                    }
                    Err(e) => {
                        println!(
                            "  {} Could not verify '{}': {}",
                            style("!").yellow(),
                            m.name,
                            e
                        );
                    }
                }
            }
        }
        issues += hash_issues;
        if hash_issues == 0 {
            println!("  {} All hashes verified", style("✓").green());
        }
    } else {
        println!();
        println!(
            "  {} Hash verification skipped (use {} for full check)",
            style("i").dim(),
            style("--verify-hashes").cyan()
        );
    }

    // Summary
    println!();
    if issues == 0 {
        println!(
            "{} No issues found. Everything looks good!",
            style("✓").green().bold()
        );
    } else if repair {
        println!("{} Repair complete.", style("✓").green().bold());
    } else {
        println!(
            "{} Found {} issue{}.",
            style("!").yellow().bold(),
            issues,
            if issues == 1 { "" } else { "s" }
        );
    }

    Ok(())
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Walk `store_dir/<asset_type>/<hash_prefix>/<filename>` and collect any files
/// not present in `tracked_paths`.
fn scan_orphans(
    store_dir: &std::path::Path,
    tracked_paths: &HashSet<String>,
    orphans: &mut Vec<(String, String, String, String, u64)>,
) {
    if !store_dir.is_dir() {
        return;
    }
    let Ok(type_dirs) = std::fs::read_dir(store_dir) else {
        return;
    };
    for type_entry in type_dirs.flatten() {
        if !type_entry.path().is_dir() {
            continue;
        }
        let Ok(hash_dirs) = std::fs::read_dir(type_entry.path()) else {
            continue;
        };
        for hash_entry in hash_dirs.flatten() {
            if !hash_entry.path().is_dir() {
                continue;
            }
            let Ok(files) = std::fs::read_dir(hash_entry.path()) else {
                continue;
            };
            for file_entry in files.flatten() {
                let file_path = file_entry.path();
                if file_path.is_file() {
                    let path_str = file_path.to_string_lossy().to_string();
                    if !tracked_paths.contains(&path_str) {
                        let size = std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);
                        let asset_type = type_entry.file_name().to_string_lossy().to_string();
                        let hash_prefix = hash_entry.file_name().to_string_lossy().to_string();
                        let file_name = file_entry.file_name().to_string_lossy().to_string();
                        orphans.push((path_str, asset_type, hash_prefix, file_name, size));
                    }
                }
            }
        }
    }
}

/// Build a HashMap from SHA256 prefix (16 chars) → (id, name, variant_label)
/// by scanning the local registry index. Returns empty map if index is unavailable.
fn build_registry_lookup() -> HashMap<String, (String, String, Option<String>)> {
    let mut map = HashMap::new();
    let Ok(index) = RegistryIndex::load() else {
        return map;
    };
    for manifest in &index.items {
        // Single-file models (LoRAs, VAEs, etc.)
        if let Some(ref file) = manifest.file
            && file.sha256.len() >= 16
        {
            map.insert(
                file.sha256[..16].to_string(),
                (manifest.id.clone(), manifest.name.clone(), None),
            );
        }
        // Multi-variant models (checkpoints, text encoders)
        for variant in &manifest.variants {
            if variant.sha256.len() >= 16 {
                map.insert(
                    variant.sha256[..16].to_string(),
                    (
                        manifest.id.clone(),
                        manifest.name.clone(),
                        Some(variant.id.clone()),
                    ),
                );
            }
        }
    }
    map
}
