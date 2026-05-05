use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Fast reconciliation from the shared store index.yaml instead of a directory scan.
///
/// For new users with a symlinked store, this populates the local SQLite from
/// the YAML that install operations maintain. O(index entries), not O(files).
/// Returns the number of newly-registered entries.
pub fn reconcile_from_index(db: &Database) -> Result<usize> {
    let entries = crate::core::store_index::load();
    if entries.is_empty() {
        return Ok(0);
    }
    let tracked: HashSet<String> = db
        .list_installed(None)?
        .into_iter()
        .map(|m| m.store_path)
        .collect();

    let mut registered = 0;
    for entry in &entries {
        if !tracked.contains(&entry.store_path)
            && db
                .insert_installed(&crate::core::db::InstalledModelRecord {
                    id: &entry.id,
                    name: &entry.name,
                    asset_type: &entry.asset_type,
                    variant: entry.variant.as_deref(),
                    sha256: &entry.sha256,
                    size: entry.size,
                    file_name: &entry.file_name,
                    store_path: &entry.store_path,
                })
                .is_ok()
        {
            registered += 1;
        }
    }
    Ok(registered)
}

use crate::core::config::Config;
use crate::core::db::Database;
use crate::core::registry::RegistryIndex;

/// Scan store directories and register any files not already tracked in the DB.
///
/// Uses `canonicalize()` on each file path so symlinked stores (e.g. a new OS
/// user whose `~/modl/store` points to another user's store) produce canonical
/// paths that survive across users.
///
/// Returns the number of newly-registered models.
pub fn reconcile_store(db: &Database) -> Result<usize> {
    let config = Config::load()?;

    let tracked: HashSet<String> = db
        .list_installed(None)?
        .into_iter()
        .map(|m| m.store_path)
        .collect();

    let mut candidates: Vec<(String, String, String, String, u64)> = Vec::new();

    let store_dir = config.store_root().join("store");
    scan_dir(&store_dir, &tracked, &mut candidates);

    // Also cover the config-dir store (~/.modl/store) for locally-trained LoRAs.
    let config_dir_store = Config::default_path()
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("store");
    if config_dir_store != store_dir {
        scan_dir(&config_dir_store, &tracked, &mut candidates);
    }

    if candidates.is_empty() {
        return Ok(0);
    }

    let registry_lookup = build_registry_lookup();
    let mut registered = 0;

    for (store_path, asset_type, hash_prefix, file_name, size) in &candidates {
        let (id, name, variant) = registry_lookup
            .get(hash_prefix.as_str())
            .cloned()
            .unwrap_or_else(|| {
                let stem = file_name
                    .strip_suffix(".safetensors")
                    .or_else(|| file_name.strip_suffix(".ckpt"))
                    .or_else(|| file_name.strip_suffix(".bin"))
                    .or_else(|| file_name.strip_suffix(".pt"))
                    .unwrap_or(file_name);
                (format!("local/{asset_type}/{stem}"), stem.to_string(), None)
            });

        if db
            .insert_installed(&crate::core::db::InstalledModelRecord {
                id: &id,
                name: &name,
                asset_type,
                variant: variant.as_deref(),
                sha256: hash_prefix,
                size: *size,
                file_name,
                store_path,
            })
            .is_ok()
        {
            registered += 1;
        }
    }

    Ok(registered)
}

fn scan_dir(
    store_dir: &std::path::Path,
    tracked: &HashSet<String>,
    out: &mut Vec<(String, String, String, String, u64)>,
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
                if !file_path.is_file() {
                    continue;
                }
                // Use the canonical path so symlinked stores produce stable DB entries.
                let canonical =
                    std::fs::canonicalize(&file_path).unwrap_or_else(|_| file_path.clone());
                let path_str = canonical.to_string_lossy().to_string();
                if tracked.contains(&path_str) {
                    continue;
                }
                let size = std::fs::metadata(&canonical).map(|m| m.len()).unwrap_or(0);
                let asset_type = type_entry.file_name().to_string_lossy().to_string();
                let hash_prefix = hash_entry.file_name().to_string_lossy().to_string();
                let file_name = file_entry.file_name().to_string_lossy().to_string();
                out.push((path_str, asset_type, hash_prefix, file_name, size));
            }
        }
    }
}

fn build_registry_lookup() -> HashMap<String, (String, String, Option<String>)> {
    let mut map = HashMap::new();
    let Ok(index) = RegistryIndex::load() else {
        return map;
    };
    for manifest in &index.items {
        if let Some(ref file) = manifest.file
            && file.sha256.len() >= 16
        {
            map.insert(
                file.sha256[..16].to_string(),
                (manifest.id.clone(), manifest.name.clone(), None),
            );
        }
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
