//! Export/import service for modl data backup and restore.
//!
//! Creates `.tar.zst` archives containing the SQLite database, trained LoRAs,
//! generation outputs, and auth tokens. Import merges data into the current
//! installation using INSERT OR IGNORE for DB records.

use std::collections::HashSet;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::db::Database;
use super::paths;

// ---------------------------------------------------------------------------
// Options & results
// ---------------------------------------------------------------------------

pub struct ExportOptions {
    pub output_path: PathBuf,
    pub include_outputs: bool,
    pub since: Option<String>,
}

pub struct ImportOptions {
    pub archive_path: PathBuf,
    pub dry_run: bool,
    pub overwrite: bool,
}

pub struct ExportResult {
    #[allow(dead_code)]
    pub path: PathBuf,
    pub manifest: ArchiveManifest,
}

#[derive(Default)]
pub struct ImportResult {
    pub db_merged: bool,
    pub outputs_restored: usize,
    pub outputs_skipped: usize,
    pub loras_restored: usize,
    pub loras_skipped: usize,
    pub auth_restored: bool,
}

// ---------------------------------------------------------------------------
// Archive manifest
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub struct ArchiveManifest {
    pub version: u32,
    pub created_at: String,
    pub modl_version: String,
    pub contents: ArchiveContents,
    pub stats: ArchiveStats,
}

#[derive(Serialize, Deserialize)]
pub struct ArchiveContents {
    pub db: bool,
    pub outputs: bool,
    pub trained_loras: usize,
    pub auth: bool,
}

#[derive(Serialize, Deserialize)]
pub struct ArchiveStats {
    pub total_size: u64,
    pub output_count: usize,
    pub job_count: usize,
    pub lora_count: usize,
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

pub fn export(opts: &ExportOptions, progress: &dyn ExportProgress) -> Result<ExportResult> {
    let root = paths::modl_root();
    let db_path = root.join("state.db");
    let auth_path = root.join("auth.yaml");
    let outputs_dir = root.join("outputs");
    let store_lora_dir = root.join("store").join("lora");

    if !db_path.exists() {
        bail!(
            "No modl database found at {}. Nothing to export.",
            db_path.display()
        );
    }

    // Collect trained LoRA files
    let trained_lora_files = collect_trained_loras(&db_path, &store_lora_dir)?;

    // Collect output files
    let output_files = if opts.include_outputs {
        collect_outputs(&outputs_dir, opts.since.as_deref())?
    } else {
        Vec::new()
    };

    // Count jobs for stats
    let db = Database::open()?;
    let job_count = db.count_jobs()?;
    drop(db);

    let has_auth = auth_path.exists();

    // Build manifest
    let manifest = ArchiveManifest {
        version: 1,
        created_at: chrono::Utc::now().to_rfc3339(),
        modl_version: env!("CARGO_PKG_VERSION").to_string(),
        contents: ArchiveContents {
            db: true,
            outputs: opts.include_outputs,
            trained_loras: trained_lora_files.len(),
            auth: has_auth,
        },
        stats: ArchiveStats {
            total_size: 0, // filled after archiving
            output_count: output_files.len(),
            job_count,
            lora_count: trained_lora_files.len(),
        },
    };

    // Calculate total items for progress
    let total_items = 1 // manifest
        + 1 // db
        + if has_auth { 1 } else { 0 }
        + trained_lora_files.len()
        + output_files.len();
    progress.set_total(total_items);

    // Create tar.zst archive
    let file = std::fs::File::create(&opts.output_path)
        .with_context(|| format!("Failed to create {}", opts.output_path.display()))?;
    let zst = zstd::Encoder::new(file, 3)?.auto_finish();
    let mut tar = tar::Builder::new(zst);

    // 1. manifest.json
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    append_bytes(&mut tar, "manifest.json", manifest_json.as_bytes())?;
    progress.tick("manifest.json");

    // 2. state.db — copy to temp first (avoid locking issues)
    let tmp_dir = tempfile::tempdir()?;
    let tmp_db = tmp_dir.path().join("state.db");
    std::fs::copy(&db_path, &tmp_db).context("Failed to copy state.db")?;
    append_file(&mut tar, "state.db", &tmp_db)?;
    progress.tick("state.db");

    // 3. auth.yaml
    if has_auth {
        append_file(&mut tar, "auth.yaml", &auth_path)?;
        progress.tick("auth.yaml");
    }

    // 4. Trained LoRAs
    for (archive_path, disk_path) in &trained_lora_files {
        append_file(&mut tar, archive_path, disk_path)?;
        progress.tick(archive_path);
    }

    // 5. Outputs
    for (archive_path, disk_path) in &output_files {
        append_file(&mut tar, archive_path, disk_path)?;
        progress.tick(archive_path);
    }

    tar.finish()?;
    drop(tar);

    // Update total_size in the returned manifest
    let final_size = std::fs::metadata(&opts.output_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let mut final_manifest = manifest;
    final_manifest.stats.total_size = final_size;

    Ok(ExportResult {
        path: opts.output_path.clone(),
        manifest: final_manifest,
    })
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

pub fn import(opts: &ImportOptions, progress: &dyn ImportProgress) -> Result<ImportResult> {
    let root = paths::modl_root();
    let archive_path = &opts.archive_path;

    if !archive_path.exists() {
        bail!("Archive not found: {}", archive_path.display());
    }

    let file = std::fs::File::open(archive_path)
        .with_context(|| format!("Failed to open {}", archive_path.display()))?;
    let zst = zstd::Decoder::new(file)?;
    let mut tar = tar::Archive::new(zst);

    // First pass: read manifest and collect entries
    let tmp_dir = tempfile::tempdir()?;
    let mut manifest: Option<ArchiveManifest> = None;
    let mut has_db = false;
    let mut has_auth = false;
    let mut lora_entries: Vec<PathBuf> = Vec::new();
    let mut output_entries: Vec<PathBuf> = Vec::new();

    // Extract everything to temp dir
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_path_buf();
        let path_str = path.to_string_lossy().to_string();

        if path_str == "manifest.json" {
            let mut buf = String::new();
            entry.read_to_string(&mut buf)?;
            manifest = Some(serde_json::from_str(&buf).context("Invalid manifest.json")?);
        } else {
            // Extract to temp dir
            let dest = tmp_dir.path().join(&path);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut out = std::fs::File::create(&dest)?;
            std::io::copy(&mut entry, &mut out)?;

            if path_str == "state.db" {
                has_db = true;
            } else if path_str == "auth.yaml" {
                has_auth = true;
            } else if path_str.starts_with("loras/") {
                lora_entries.push(path);
            } else if path_str.starts_with("outputs/") {
                output_entries.push(path);
            }
        }
    }

    let manifest = manifest.context("Archive missing manifest.json — not a valid modl backup")?;
    if manifest.version != 1 {
        bail!(
            "Unsupported archive version {}. This modl only supports version 1.",
            manifest.version
        );
    }

    progress.summary(
        &manifest,
        has_db,
        has_auth,
        lora_entries.len(),
        output_entries.len(),
    );

    if opts.dry_run {
        return Ok(ImportResult::default());
    }

    let mut result = ImportResult::default();

    // 1. Merge DB
    if has_db {
        let src_db_path = tmp_dir.path().join("state.db");
        merge_database(&src_db_path, &root)?;
        result.db_merged = true;
        progress.tick("database merged");
    }

    // 2. Restore auth.yaml
    if has_auth {
        let src_auth = tmp_dir.path().join("auth.yaml");
        let dest_auth = root.join("auth.yaml");
        if !dest_auth.exists() || opts.overwrite {
            std::fs::copy(&src_auth, &dest_auth)?;
            result.auth_restored = true;
            progress.tick("auth.yaml restored");
        } else {
            progress.tick("auth.yaml skipped (exists)");
        }
    }

    // 3. Restore LoRAs
    for lora_path in &lora_entries {
        let src = tmp_dir.path().join(lora_path);
        // loras/<hash_prefix>/<filename> → store/lora/<hash_prefix>/<filename>
        let relative = lora_path.strip_prefix("loras/").unwrap_or(lora_path);
        let dest = root.join("store").join("lora").join(relative);
        if !dest.exists() || opts.overwrite {
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(&src, &dest)?;
            result.loras_restored += 1;
        } else {
            result.loras_skipped += 1;
        }
    }
    if !lora_entries.is_empty() {
        progress.tick(&format!(
            "{} LoRAs restored, {} skipped",
            result.loras_restored, result.loras_skipped
        ));
    }

    // 4. Restore outputs
    for output_path in &output_entries {
        let src = tmp_dir.path().join(output_path);
        let dest = root.join(output_path);
        if !dest.exists() || opts.overwrite {
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(&src, &dest)?;
            result.outputs_restored += 1;
        } else {
            result.outputs_skipped += 1;
        }
    }
    if !output_entries.is_empty() {
        progress.tick(&format!(
            "{} outputs restored, {} skipped",
            result.outputs_restored, result.outputs_skipped
        ));
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Progress traits (CLI provides the implementation)
// ---------------------------------------------------------------------------

pub trait ExportProgress {
    fn set_total(&self, total: usize);
    fn tick(&self, item: &str);
}

pub trait ImportProgress {
    fn summary(
        &self,
        manifest: &ArchiveManifest,
        has_db: bool,
        has_auth: bool,
        lora_count: usize,
        output_count: usize,
    );
    fn tick(&self, msg: &str);
}

// ---------------------------------------------------------------------------
// DB merge via ATTACH
// ---------------------------------------------------------------------------

fn merge_database(src_path: &Path, modl_root: &Path) -> Result<()> {
    let dest_path = modl_root.join("state.db");

    // Ensure destination DB exists with schema
    let _db = Database::open()?;
    drop(_db);

    let conn = rusqlite::Connection::open(&dest_path).context("Failed to open target database")?;

    conn.execute(
        "ATTACH DATABASE ?1 AS src",
        rusqlite::params![src_path.to_string_lossy().as_ref()],
    )
    .context("Failed to attach source database")?;

    // Merge user data tables (skip installed, symlinks, dependencies — rebuilt by modl pull)
    let merge_tables = [
        "jobs",
        "job_events",
        "artifacts",
        "favorites",
        "lora_library",
        "training_queue",
        "studio_sessions",
        "session_events",
        "session_images",
    ];

    for table in &merge_tables {
        // Get column names from the source table
        let cols: String = {
            let mut stmt = conn
                .prepare(&format!("PRAGMA src.table_info({})", table))
                .with_context(|| format!("Failed to read schema for {}", table))?;
            let col_names: Vec<String> = stmt
                .query_map([], |row| row.get::<_, String>(1))?
                .filter_map(|r| r.ok())
                .collect();
            if col_names.is_empty() {
                continue; // Table doesn't exist in source
            }
            col_names.join(", ")
        };

        conn.execute_batch(&format!(
            "INSERT OR IGNORE INTO main.{table} ({cols}) SELECT {cols} FROM src.{table};"
        ))
        .with_context(|| format!("Failed to merge table {}", table))?;
    }

    // Rewrite artifact paths: replace source modl_root with current modl_root
    // Detect old root from artifact paths in the source DB
    let old_root: Option<String> = conn
        .query_row(
            "SELECT path FROM src.artifacts WHERE path LIKE '%/.modl/%' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .ok();

    if let Some(old_path) = old_root {
        let current_root = modl_root.to_string_lossy().to_string();
        // Extract old root: everything up to and including ".modl"
        if let Some(idx) = old_path.find("/.modl/") {
            let old_root_prefix = &old_path[..idx + 6]; // includes "/.modl"
            if old_root_prefix != current_root {
                conn.execute(
                    "UPDATE main.artifacts SET path = REPLACE(path, ?1, ?2) WHERE path LIKE ?3",
                    rusqlite::params![
                        old_root_prefix,
                        &current_root,
                        format!("{}%", old_root_prefix),
                    ],
                )?;
                // Also rewrite lora_library paths
                conn.execute(
                    "UPDATE main.lora_library SET lora_path = REPLACE(lora_path, ?1, ?2) WHERE lora_path LIKE ?3",
                    rusqlite::params![
                        old_root_prefix,
                        &current_root,
                        format!("{}%", old_root_prefix),
                    ],
                )?;
                // Rewrite favorites paths
                // favorites uses path as PRIMARY KEY, so we need to handle carefully
                conn.execute(
                    "UPDATE main.favorites SET path = REPLACE(path, ?1, ?2) WHERE path LIKE ?3",
                    rusqlite::params![
                        old_root_prefix,
                        &current_root,
                        format!("{}%", old_root_prefix),
                    ],
                )?;
            }
        }
    }

    conn.execute_batch("DETACH DATABASE src;")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collect trained LoRA files (those NOT from the registry).
/// Returns (archive_path, disk_path) pairs.
fn collect_trained_loras(db_path: &Path, store_lora_dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    if !store_lora_dir.exists() {
        return Ok(Vec::new());
    }

    // Get IDs of registry-installed LoRAs from the DB
    let conn = rusqlite::Connection::open(db_path)?;
    let mut stmt = conn.prepare("SELECT sha256 FROM installed WHERE asset_type = 'lora'")?;
    let installed_hashes: HashSet<String> = stmt
        .query_map([], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    let mut results = Vec::new();

    // Walk store/lora/<hash_prefix>/
    for entry in std::fs::read_dir(store_lora_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let hash_prefix = entry.file_name().to_string_lossy().to_string();

        // Check if any installed LoRA has this hash prefix
        let is_installed = installed_hashes.iter().any(|h| h.starts_with(&hash_prefix));
        if is_installed {
            continue; // Skip registry LoRAs — they're re-pullable
        }

        // Include all files in this directory
        for file_entry in std::fs::read_dir(entry.path())? {
            let file_entry = file_entry?;
            if file_entry.file_type()?.is_file() {
                let filename = file_entry.file_name().to_string_lossy().to_string();
                let archive_path = format!("loras/{}/{}", hash_prefix, filename);
                results.push((archive_path, file_entry.path()));
            }
        }
    }

    Ok(results)
}

/// Collect output files, optionally filtered by date.
/// Returns (archive_path, disk_path) pairs.
fn collect_outputs(outputs_dir: &Path, since: Option<&str>) -> Result<Vec<(String, PathBuf)>> {
    if !outputs_dir.exists() {
        return Ok(Vec::new());
    }

    let mut results = Vec::new();

    // outputs/<date>/<file>
    for date_entry in std::fs::read_dir(outputs_dir)? {
        let date_entry = date_entry?;
        if !date_entry.file_type()?.is_dir() {
            continue;
        }
        let date_name = date_entry.file_name().to_string_lossy().to_string();

        // Filter by --since (compare as strings — both YYYY-MM-DD or YYYYMMDD sort lexically)
        if let Some(since_date) = since {
            let normalized = since_date.replace('-', "");
            let date_cmp = date_name.replace('-', "");
            if date_cmp < normalized {
                continue;
            }
        }

        for file_entry in std::fs::read_dir(date_entry.path())? {
            let file_entry = file_entry?;
            if file_entry.file_type()?.is_file() {
                let filename = file_entry.file_name().to_string_lossy().to_string();
                let archive_path = format!("outputs/{}/{}", date_name, filename);
                results.push((archive_path, file_entry.path()));
            }
        }
    }

    Ok(results)
}

/// Append raw bytes as a file entry in the tar archive.
fn append_bytes<W: Write>(tar: &mut tar::Builder<W>, path: &str, data: &[u8]) -> Result<()> {
    let mut header = tar::Header::new_gnu();
    header.set_size(data.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    tar.append_data(&mut header, path, data)
        .with_context(|| format!("Failed to add {} to archive", path))?;
    Ok(())
}

/// Append a file from disk to the tar archive.
fn append_file<W: Write>(
    tar: &mut tar::Builder<W>,
    archive_path: &str,
    disk_path: &Path,
) -> Result<()> {
    let mut file = std::fs::File::open(disk_path)
        .with_context(|| format!("Failed to open {}", disk_path.display()))?;
    let meta = file.metadata()?;
    let mut header = tar::Header::new_gnu();
    header.set_size(meta.len());
    header.set_mode(0o644);
    header.set_cksum();
    tar.append_data(&mut header, archive_path, &mut file)
        .with_context(|| format!("Failed to add {} to archive", archive_path))?;
    Ok(())
}
