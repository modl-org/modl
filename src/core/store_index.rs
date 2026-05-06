use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::core::db::InstalledModelRecord;

/// Shared store index — a YAML manifest at `<store_root>/store/index.yaml`.
///
/// Written by every install/uninstall operation so the index is always current.
/// On `modl ls`, a new user whose `~/modl/store` is a symlink to a shared store
/// reads this file and populates their local SQLite in milliseconds instead of
/// walking the entire directory tree.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IndexEntry {
    pub id: String,
    pub name: String,
    pub asset_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    pub sha256: String,
    pub size: u64,
    pub file_name: String,
    pub store_path: String,
}

fn index_path() -> PathBuf {
    // Prefer config-declared store root; fall back to ~/.modl/store.
    crate::core::config::Config::load()
        .ok()
        .map(|c| c.store_root().join("store").join("index.yaml"))
        .unwrap_or_else(|| {
            crate::core::paths::modl_root()
                .join("store")
                .join("index.yaml")
        })
}

pub fn load() -> Vec<IndexEntry> {
    let path = index_path();
    if !path.exists() {
        return vec![];
    }
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_yaml::from_str(&s).ok())
        .unwrap_or_default()
}

fn save(entries: &[IndexEntry]) {
    let path = index_path();
    let Some(parent) = path.parent() else { return };
    if let Err(e) = std::fs::create_dir_all(parent) {
        eprintln!(
            "warning: failed to create store index directory {}: {e}",
            parent.display()
        );
        return;
    }
    let Ok(yaml) = serde_yaml::to_string(entries) else {
        return;
    };
    // Write to a sibling tempfile then rename atomically so concurrent writers
    // on different OS users never produce a partial or interleaved index.
    if let Ok(mut tmp) = tempfile::Builder::new()
        .prefix(".index-")
        .suffix(".yaml.tmp")
        .tempfile_in(parent)
        && std::io::Write::write_all(&mut tmp, yaml.as_bytes()).is_ok()
        && let Err(e) = tmp.persist(&path)
    {
        eprintln!(
            "warning: failed to write store index {}: {e}",
            path.display()
        );
    }
}

/// Add or update an entry (keyed by `id`). Errors are intentionally swallowed
/// — the index is a best-effort cache; SQLite is the source of truth.
pub fn upsert(record: &InstalledModelRecord) {
    let mut entries = load();
    entries.retain(|e| e.id != record.id);
    entries.push(IndexEntry {
        id: record.id.to_string(),
        name: record.name.to_string(),
        asset_type: record.asset_type.to_string(),
        variant: record.variant.map(|v| v.to_string()),
        sha256: record.sha256.to_string(),
        size: record.size,
        file_name: record.file_name.to_string(),
        store_path: record.store_path.to_string(),
    });
    entries.sort_by(|a, b| a.id.cmp(&b.id));
    save(&entries);
}

/// Remove the entry for `id`. No-op if not found.
pub fn remove_by_id(id: &str) {
    let mut entries = load();
    let before = entries.len();
    entries.retain(|e| e.id != id);
    if entries.len() != before {
        save(&entries);
    }
}
