//! Service layer for output management.
//!
//! All output operations (list, delete, favorite) go through this module.
//! Both the CLI and the web UI server call these functions — neither should
//! talk to the database or filesystem directly for output operations.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::db::{ArtifactRecord, Database};
use super::paths;

// ---------------------------------------------------------------------------
// Sidecar YAML metadata
// ---------------------------------------------------------------------------

/// Metadata written as a YAML sidecar file alongside generated/edited images.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarMetadata {
    pub prompt: String,
    pub base_model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub steps: u32,
    pub guidance: f32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub size: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_strength: Option<f32>,
    pub created_at: String,
    pub source: String,
}

/// Write a YAML sidecar file next to an image.
///
/// The sidecar is named `<stem>.yaml` where `<stem>` is the image filename
/// without its extension. Failures are logged as warnings but never propagated
/// — sidecar writing must not break generation/editing.
pub fn write_sidecar_yaml(image_path: &str, metadata: &SidecarMetadata) {
    let path = Path::new(image_path);
    let sidecar_path = path.with_extension("yaml");
    match serde_yaml::to_string(metadata) {
        Ok(yaml) => {
            if let Err(e) = std::fs::write(&sidecar_path, &yaml) {
                eprintln!(
                    "Warning: failed to write sidecar YAML {}: {}",
                    sidecar_path.display(),
                    e
                );
            }
        }
        Err(e) => {
            eprintln!("Warning: failed to serialize sidecar YAML: {}", e);
        }
    }
}

/// Read the YAML sidecar next to an image, if present and parseable.
pub fn read_sidecar(image_path: &Path) -> Option<SidecarMetadata> {
    let yaml = std::fs::read_to_string(image_path.with_extension("yaml")).ok()?;
    serde_yaml::from_str(&yaml).ok()
}

/// Convert a sidecar into the artifact-metadata JSON shape that
/// `parse_output_meta` (and therefore `modl outputs` + the UI) reads.
fn sidecar_to_artifact_meta(s: &SidecarMetadata) -> String {
    let (width, height) = s
        .size
        .split_once('x')
        .map(|(w, h)| (w.parse::<u32>().ok(), h.parse::<u32>().ok()))
        .unwrap_or((None, None));
    serde_json::json!({
        "generated_with": s.source,
        "prompt": s.prompt,
        "base_model_id": s.base_model,
        "lora_name": s.lora,
        "lora_strength": s.lora_strength,
        "seed": s.seed,
        "steps": s.steps,
        "guidance": s.guidance,
        "width": width,
        "height": height,
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// Run export + import — files and their sidecars are the sync payload;
// SQLite is a per-machine rebuildable cache (same doctrine as store/index.yaml).
// ---------------------------------------------------------------------------

/// One artifact to ship when exporting a run: the image plus its sidecar
/// YAML when one exists on disk.
pub struct RunExportFile {
    pub image: PathBuf,
    pub sidecar: Option<PathBuf>,
}

pub enum RunExportFiles {
    /// No jobs recorded under this run id.
    NoSuchRun,
    /// Run exists; may be empty if no artifacts landed yet.
    Files(Vec<RunExportFile>),
}

/// Collect a run's exportable files: every image artifact on disk, paired
/// with its sidecar. Shared by `modl outputs export` and the ZIP endpoint
/// so both always carry the metadata needed to reconcile on another machine.
pub fn collect_run_export_files(db: &Database, run_id: &str) -> Result<RunExportFiles> {
    let jobs = db.list_jobs_by_run_id(run_id)?;
    if jobs.is_empty() {
        return Ok(RunExportFiles::NoSuchRun);
    }
    let mut files = Vec::new();
    for job in &jobs {
        for a in db.list_artifacts(Some(&job.job_id))? {
            let image = PathBuf::from(&a.path);
            if !image.is_file() {
                continue;
            }
            let sidecar = image.with_extension("yaml");
            files.push(RunExportFile {
                sidecar: sidecar.is_file().then_some(sidecar),
                image,
            });
        }
    }
    Ok(RunExportFiles::Files(files))
}

/// Register any image under `~/.modl/outputs` that has a sidecar YAML but no
/// artifact row. This is how outputs synced in from another machine (pod
/// pulls, manual copies) show up in `modl outputs` and the UI with full
/// metadata. Idempotent: the artifact id is derived from the image content,
/// and paths that already have any artifact row are skipped.
pub fn reconcile_outputs(db: &Database) -> Result<usize> {
    reconcile_outputs_in(db, &paths::modl_root().join("outputs"))
}

fn reconcile_outputs_in(db: &Database, outputs_root: &Path) -> Result<usize> {
    let known: std::collections::HashSet<String> = db
        .list_artifacts(None)?
        .into_iter()
        .map(|a| a.path)
        .collect();

    let mut registered = 0usize;
    let Ok(dates) = std::fs::read_dir(outputs_root) else {
        return Ok(0);
    };
    for date_entry in dates.filter_map(|e| e.ok()) {
        let date_path = date_entry.path();
        if !date_path.is_dir() {
            continue;
        }
        let Ok(files) = std::fs::read_dir(&date_path) else {
            continue;
        };
        for entry in files.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !is_image_file(&path) {
                continue;
            }
            let abs = path.to_string_lossy().to_string();
            if known.contains(&abs) {
                continue;
            }
            let Some(sidecar) = read_sidecar(&path) else {
                continue;
            };
            let Ok(sha256) = super::store::Store::hash_file(&path) else {
                continue;
            };
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            db.insert_artifact(
                &format!("import:{}", &sha256[..16]),
                None,
                "image",
                &abs,
                &sha256,
                size,
                Some(&sidecar_to_artifact_meta(&sidecar)),
            )?;
            registered += 1;
        }
    }
    Ok(registered)
}

/// Import a directory of exported run files (images + sidecars) into the
/// outputs tree and register them. Each image lands under
/// `outputs/<date>/` where the date comes from its sidecar's `created_at`
/// (falling back to today), so synced outputs sort naturally alongside
/// local ones. Idempotent: an already-imported image (same content at the
/// destination) is skipped. Returns the destination paths of all images.
pub fn import_run_dir(src: &Path, db: &Database) -> Result<Vec<PathBuf>> {
    import_run_dir_into(src, db, &paths::modl_root().join("outputs"))
}

fn import_run_dir_into(src: &Path, db: &Database, outputs_root: &Path) -> Result<Vec<PathBuf>> {
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    let mut imported = Vec::new();

    let entries =
        std::fs::read_dir(src).with_context(|| format!("Failed to read {}", src.display()))?;
    let mut images: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| is_image_file(p))
        .collect();
    images.sort();

    for image in images {
        let sidecar = read_sidecar(&image);
        // `created_at` is RFC 3339; its first 10 chars are the date.
        let date = sidecar
            .as_ref()
            .map(|s| s.created_at.chars().take(10).collect::<String>())
            .filter(|d| d.len() == 10 && d.chars().filter(|c| *c == '-').count() == 2)
            .unwrap_or_else(|| today.clone());
        let dest_dir = outputs_root.join(&date);
        std::fs::create_dir_all(&dest_dir)
            .with_context(|| format!("Failed to create {}", dest_dir.display()))?;

        let file_name = image
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "image.png".to_string());
        let (dest, already_present) = unique_dest(&dest_dir.join(&file_name), &image)?;
        if !already_present {
            std::fs::copy(&image, &dest)
                .with_context(|| format!("Failed to copy {}", image.display()))?;
            let src_sidecar = image.with_extension("yaml");
            if src_sidecar.is_file() {
                std::fs::copy(&src_sidecar, dest.with_extension("yaml"))
                    .with_context(|| format!("Failed to copy sidecar {}", src_sidecar.display()))?;
            }
        }
        imported.push(dest);
    }

    reconcile_outputs_in(db, outputs_root)?;
    Ok(imported)
}

fn is_image_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "webp"))
}

/// Pick a destination for `source` under its preferred name: the original
/// name when free, `<stem>-2.<ext>`, `<stem>-3.<ext>`, … when taken by a
/// different file. Returns `(path, true)` when some candidate already holds
/// identical content (import is a no-op for it).
fn unique_dest(preferred: &Path, source: &Path) -> Result<(PathBuf, bool)> {
    use super::store::Store;
    if !preferred.exists() {
        return Ok((preferred.to_path_buf(), false));
    }
    let src_sha = Store::hash_file(source)?;
    if Store::hash_file(preferred)? == src_sha {
        return Ok((preferred.to_path_buf(), true));
    }
    let stem = preferred
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "image".to_string());
    let ext = preferred
        .extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_else(|| "png".to_string());
    let dir = preferred.parent().unwrap_or(Path::new("."));
    for n in 2..1000 {
        let candidate = dir.join(format!("{stem}-{n}.{ext}"));
        if !candidate.exists() {
            return Ok((candidate, false));
        }
        if Store::hash_file(&candidate)? == src_sha {
            return Ok((candidate, true));
        }
    }
    bail!(
        "Could not find a free destination name for {}",
        preferred.display()
    );
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize)]
pub struct GeneratedOutput {
    pub date: String,
    pub images: Vec<GeneratedImage>,
}

#[derive(Clone, Serialize)]
pub struct GeneratedImage {
    /// Relative path usable as /files/<path>
    pub path: String,
    /// Filename without directory
    pub filename: String,
    /// mtime as unix timestamp (seconds)
    pub modified: u64,
    /// Artifact ID in DB, if tracked
    pub artifact_id: Option<String>,
    /// Job ID that produced the image, if tracked
    pub job_id: Option<String>,
    /// Prompt used to generate the image, if available
    pub prompt: Option<String>,
    /// Base model ID used for generation, if available
    pub base_model_id: Option<String>,
    /// LoRA name used, if any
    pub lora_name: Option<String>,
    /// LoRA strength used, if any
    pub lora_strength: Option<f64>,
    /// Per-image seed, if available
    pub seed: Option<u64>,
    /// Inference steps, if available
    pub steps: Option<u32>,
    /// Guidance scale, if available
    pub guidance: Option<f64>,
    /// Output width, if available
    pub width: Option<u32>,
    /// Output height, if available
    pub height: Option<u32>,
    /// Stored artifact size, if tracked
    pub size_bytes: Option<u64>,
    /// Marker embedded by generator
    pub generated_with: Option<String>,
    /// Whether the user has starred this image
    pub favorited: bool,
}

pub struct DeleteOutputResult {
    pub deleted_file: bool,
    pub deleted_records: usize,
}

pub struct BatchDeleteResult {
    pub deleted_files: usize,
    pub deleted_records: usize,
    pub errors: Vec<String>,
}

pub struct ToggleFavoriteResult {
    pub favorited: bool,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[derive(Default)]
struct OutputMetaSummary {
    prompt: Option<String>,
    base_model_id: Option<String>,
    lora_name: Option<String>,
    lora_strength: Option<f64>,
    seed: Option<u64>,
    steps: Option<u32>,
    guidance: Option<f64>,
    width: Option<u32>,
    height: Option<u32>,
    generated_with: Option<String>,
}

fn parse_output_meta(metadata: Option<&str>) -> OutputMetaSummary {
    let Some(raw) = metadata else {
        return OutputMetaSummary::default();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) else {
        return OutputMetaSummary::default();
    };

    OutputMetaSummary {
        prompt: v
            .get("prompt")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
        base_model_id: v
            .get("base_model_id")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
        lora_name: v
            .get("lora_name")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
        lora_strength: v.get("lora_strength").and_then(|x| x.as_f64()),
        seed: v.get("seed").and_then(|x| x.as_u64()),
        steps: v.get("steps").and_then(|x| x.as_u64()).map(|n| n as u32),
        guidance: v.get("guidance").and_then(|x| x.as_f64()),
        width: v.get("width").and_then(|x| x.as_u64()).map(|n| n as u32),
        height: v.get("height").and_then(|x| x.as_u64()).map(|n| n as u32),
        generated_with: v
            .get("generated_with")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
    }
}

fn parse_generate_job_spec_meta(spec_json: &str) -> Option<OutputMetaSummary> {
    let spec: serde_json::Value = serde_json::from_str(spec_json).ok()?;
    Some(OutputMetaSummary {
        prompt: spec
            .get("prompt")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
        base_model_id: spec
            .pointer("/model/base_model_id")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
        lora_name: spec
            .pointer("/lora/name")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
        lora_strength: spec.pointer("/lora/weight").and_then(|x| x.as_f64()),
        seed: spec.pointer("/params/seed").and_then(|x| x.as_u64()),
        steps: spec
            .pointer("/params/steps")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
        guidance: spec.pointer("/params/guidance").and_then(|x| x.as_f64()),
        width: spec
            .pointer("/params/width")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
        height: spec
            .pointer("/params/height")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
        generated_with: Some("modl.run".to_string()),
    })
}

struct OutputArtifactInfo {
    artifact_id: String,
    job_id: Option<String>,
    size_bytes: u64,
    metadata: Option<String>,
}

fn load_output_artifact_index() -> HashMap<String, OutputArtifactInfo> {
    let mut by_path: HashMap<String, OutputArtifactInfo> = HashMap::new();
    let Ok(db) = Database::open() else {
        return by_path;
    };
    let Ok(artifacts) = db.list_artifacts(None) else {
        return by_path;
    };

    for artifact in artifacts {
        if artifact.kind != "image" || artifact.path.is_empty() {
            continue;
        }
        if by_path.contains_key(&artifact.path) {
            continue;
        }

        // Fallback for older rows that didn't store per-image metadata.
        let metadata =
            if artifact.metadata.is_none() || artifact.metadata.as_deref() == Some("null") {
                if let Some(job_id) = &artifact.job_id {
                    if let Ok(Some(job)) = db.get_job(job_id) {
                        parse_generate_job_spec_meta(&job.spec_json).map(|m| {
                            serde_json::json!({
                                "generated_with": m.generated_with,
                                "prompt": m.prompt,
                                "base_model_id": m.base_model_id,
                                "lora_name": m.lora_name,
                                "lora_strength": m.lora_strength,
                                "seed": m.seed,
                                "steps": m.steps,
                                "guidance": m.guidance,
                                "width": m.width,
                                "height": m.height,
                            })
                            .to_string()
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                artifact.metadata.clone()
            };

        by_path.insert(
            artifact.path.clone(),
            OutputArtifactInfo {
                artifact_id: artifact.artifact_id,
                job_id: artifact.job_id,
                size_bytes: artifact.size_bytes,
                metadata,
            },
        );
    }

    by_path
}

/// Remove cached thumbnails for a source image (all widths).
fn cleanup_thumbs(source: &Path) {
    let cache_dir = paths::modl_root().join("cache").join("thumbs");
    if !cache_dir.exists() {
        return;
    }
    // Thumbnails are cached as <hash[..16]>.webp where hash = sha256("path:width")
    // We check all known thumb widths used by the UI, and clean up both old
    // .jpg and current .webp cached thumbnails.
    let widths = [200u32, 320, 480];
    for w in widths {
        let hash_input = format!("{}:{}", source.to_string_lossy(), w);
        let hash = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(hash_input.as_bytes());
            format!("{:x}", h.finalize())
        };
        let prefix = &hash[..16];
        let _ = std::fs::remove_file(cache_dir.join(format!("{prefix}.webp")));
        let _ = std::fs::remove_file(cache_dir.join(format!("{prefix}.jpg")));
    }
}

fn is_within_outputs_root(path: &Path, outputs_root: &Path) -> bool {
    if path.exists() {
        let Ok(path_canon) = path.canonicalize() else {
            return false;
        };
        let Ok(root_canon) = outputs_root.canonicalize() else {
            return false;
        };
        path_canon.starts_with(root_canon)
    } else {
        path.starts_with(outputs_root)
    }
}

// ---------------------------------------------------------------------------
// Public API — all output operations go through these functions
// ---------------------------------------------------------------------------

/// Scan ~/.modl/outputs/ for generated images, grouped by date.
pub fn list_outputs() -> Vec<GeneratedOutput> {
    let outputs_root = paths::modl_root().join("outputs");
    let mut result: Vec<GeneratedOutput> = Vec::new();
    let artifacts_by_path = load_output_artifact_index();
    let favorites = Database::open()
        .ok()
        .and_then(|db| db.get_favorite_paths().ok())
        .unwrap_or_default();

    let Ok(dates) = std::fs::read_dir(&outputs_root) else {
        return result;
    };

    let mut date_entries: Vec<_> = dates.filter_map(|e| e.ok()).collect();
    date_entries.sort_by_key(|e| std::cmp::Reverse(e.file_name()));

    for date_entry in date_entries {
        let date_path = date_entry.path();
        if !date_path.is_dir() {
            continue;
        }
        let date_str = date_entry.file_name().to_string_lossy().to_string();

        let Ok(files) = std::fs::read_dir(&date_path) else {
            continue;
        };

        let mut images: Vec<GeneratedImage> = files
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                name.ends_with(".png") || name.ends_with(".jpg") || name.ends_with(".webp")
            })
            .map(|e| {
                let filename = e.file_name().to_string_lossy().to_string();
                let rel = format!("outputs/{}/{}", date_str, filename);
                let abs = date_path.join(&filename).to_string_lossy().to_string();
                let modified = e
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let artifact = artifacts_by_path.get(&abs);
                let meta = parse_output_meta(artifact.and_then(|a| a.metadata.as_deref()));
                GeneratedImage {
                    path: rel.clone(),
                    filename,
                    modified,
                    artifact_id: artifact.map(|a| a.artifact_id.clone()),
                    job_id: artifact.and_then(|a| a.job_id.clone()),
                    prompt: meta.prompt,
                    base_model_id: meta.base_model_id,
                    lora_name: meta.lora_name,
                    lora_strength: meta.lora_strength,
                    seed: meta.seed,
                    steps: meta.steps,
                    guidance: meta.guidance,
                    width: meta.width,
                    height: meta.height,
                    size_bytes: artifact.and_then(|a| (a.size_bytes > 0).then_some(a.size_bytes)),
                    generated_with: meta.generated_with,
                    favorited: favorites.contains(&rel),
                }
            })
            .collect();

        images.sort_by_key(|i| std::cmp::Reverse(i.modified));

        if !images.is_empty() {
            result.push(GeneratedOutput {
                date: date_str,
                images,
            });
        }
    }

    result
}

/// Delete an output by artifact_id and/or relative path.
///
/// At least one of `artifact_id` or `rel_path` must be provided.
/// Returns details about what was deleted, or an error.
pub fn delete_output(
    artifact_id: Option<&str>,
    rel_path: Option<&str>,
) -> Result<DeleteOutputResult> {
    let db = Database::open().context("Failed to open database")?;

    let mut target_file: Option<PathBuf> = None;
    let mut deleted_records = 0usize;

    // If we have an artifact_id, look it up and use its path
    if let Some(aid) = artifact_id
        && !aid.trim().is_empty()
    {
        match db.get_artifact_exact(aid) {
            Ok(Some(artifact)) => {
                target_file = Some(PathBuf::from(&artifact.path));
                db.delete_artifact(aid)
                    .context("Failed to delete artifact record")?;
                deleted_records += 1;
            }
            Ok(None) => {}
            Err(e) => bail!("Failed to query artifact: {e}"),
        }
    }

    // Fall back to relative path
    if target_file.is_none()
        && let Some(rp) = rel_path
    {
        if !rp.starts_with("outputs/") {
            bail!("Path must be under outputs/");
        }
        target_file = Some(paths::modl_root().join(rp));
    }

    let Some(target_file) = target_file else {
        bail!("Missing artifact_id or path");
    };

    // Safety check: path must be within outputs/
    let outputs_root = paths::modl_root().join("outputs");
    if !is_within_outputs_root(&target_file, &outputs_root) {
        bail!("Path must be within the outputs directory");
    }

    // Delete the file from disk
    let deleted_file = if target_file.exists() {
        std::fs::remove_file(&target_file).context("Failed to delete output file")?;
        true
    } else {
        false
    };

    // Clean up sidecar YAML if present
    let sidecar_path = target_file.with_extension("yaml");
    let _ = std::fs::remove_file(sidecar_path);

    // Clean up cached thumbnails
    cleanup_thumbs(&target_file);

    // Delete any artifact records that reference this path
    let target_str = target_file.to_string_lossy().to_string();
    match db.delete_artifacts_by_path(&target_str) {
        Ok(n) => deleted_records += n,
        Err(e) => bail!("Failed to delete artifact records by path: {e}"),
    }

    // Clean up favorite entry
    let _ = db.set_favorite(&target_str, false);

    Ok(DeleteOutputResult {
        deleted_file,
        deleted_records,
    })
}

/// Delete multiple outputs at once.
pub fn batch_delete_outputs(items: Vec<(Option<String>, Option<String>)>) -> BatchDeleteResult {
    let mut deleted_files = 0usize;
    let mut deleted_records = 0usize;
    let mut errors = Vec::new();

    for (artifact_id, path) in items {
        match delete_output(artifact_id.as_deref(), path.as_deref()) {
            Ok(result) => {
                if result.deleted_file {
                    deleted_files += 1;
                }
                deleted_records += result.deleted_records;
            }
            Err(e) => {
                let label = artifact_id
                    .as_deref()
                    .or(path.as_deref())
                    .unwrap_or("unknown");
                errors.push(format!("{label}: {e}"));
            }
        }
    }

    BatchDeleteResult {
        deleted_files,
        deleted_records,
        errors,
    }
}

/// Delete an output found by artifact ID prefix match.
///
/// Finds a unique artifact whose ID starts with `prefix`, then deletes it.
#[allow(dead_code)]
pub fn delete_output_by_prefix(prefix: &str) -> Result<(ArtifactRecord, DeleteOutputResult)> {
    let db = Database::open().context("Failed to open database")?;
    let artifact = find_artifact_by_prefix(prefix, &db)?;
    let artifact_id = artifact.artifact_id.clone();

    // Determine the relative path for favorites cleanup
    let rel_path = artifact
        .path
        .strip_prefix(&paths::modl_root().to_string_lossy().to_string())
        .map(|s| s.trim_start_matches('/').to_string());

    let result = delete_output(Some(&artifact_id), rel_path.as_deref())?;
    Ok((artifact, result))
}

/// Toggle the favorite state for an output path.
///
/// `rel_path` must be a path starting with `outputs/`.
/// Returns the new favorite state.
pub fn toggle_favorite(rel_path: &str) -> Result<ToggleFavoriteResult> {
    if !rel_path.starts_with("outputs/") {
        bail!("Path must be under outputs/");
    }
    let db = Database::open().context("Failed to open database")?;
    let favorited = db
        .toggle_favorite(rel_path)
        .context("Failed to toggle favorite")?;
    Ok(ToggleFavoriteResult { favorited })
}

/// Set the favorite state for an output path (idempotent, non-toggle).
///
/// Returns `true` if the state changed.
pub fn set_favorite(path: &str, favorited: bool) -> Result<bool> {
    let db = Database::open().context("Failed to open database")?;
    db.set_favorite(path, favorited)
}

/// Check whether a path is favorited.
pub fn is_favorite(path: &str) -> Result<bool> {
    let db = Database::open().context("Failed to open database")?;
    db.is_favorite(path)
}

/// Find an artifact by prefix match on artifact_id.
pub fn find_artifact_by_prefix(prefix: &str, db: &Database) -> Result<ArtifactRecord> {
    let artifacts = db.list_artifacts(None)?;
    let matches: Vec<_> = artifacts
        .into_iter()
        .filter(|a| a.artifact_id.starts_with(prefix))
        .collect();

    match matches.len() {
        0 => bail!("No output found matching '{prefix}'."),
        1 => Ok(matches.into_iter().next().unwrap()),
        n => {
            let ids: Vec<_> = matches
                .iter()
                .map(|a| {
                    if a.artifact_id.len() > 12 {
                        a.artifact_id[..12].to_string()
                    } else {
                        a.artifact_id.clone()
                    }
                })
                .collect();
            bail!(
                "Ambiguous ID '{prefix}' matches {n} outputs: {}. Be more specific.",
                ids.join(", ")
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sidecar(prompt: &str, date: &str) -> SidecarMetadata {
        SidecarMetadata {
            prompt: prompt.to_string(),
            base_model: "z-image-turbo".to_string(),
            seed: Some(7),
            steps: 8,
            guidance: 1.0,
            size: "1024x768".to_string(),
            lora: Some("alice".to_string()),
            lora_strength: Some(1.0),
            created_at: format!("{date}T10:00:00.000000000+00:00"),
            source: "workflow".to_string(),
        }
    }

    fn write_image_with_sidecar(dir: &Path, name: &str, content: &[u8], meta: &SidecarMetadata) {
        let img = dir.join(name);
        std::fs::write(&img, content).unwrap();
        write_sidecar_yaml(&img.to_string_lossy(), meta);
    }

    #[test]
    fn sidecar_roundtrip_and_meta_json() {
        let dir = tempfile::TempDir::new().unwrap();
        write_image_with_sidecar(dir.path(), "a.png", b"img", &sidecar("hello", "2026-07-16"));

        let read = read_sidecar(&dir.path().join("a.png")).unwrap();
        assert_eq!(read.prompt, "hello");
        assert_eq!(read.size, "1024x768");

        let meta: serde_json::Value =
            serde_json::from_str(&sidecar_to_artifact_meta(&read)).unwrap();
        assert_eq!(meta["base_model_id"], "z-image-turbo");
        assert_eq!(meta["width"], 1024);
        assert_eq!(meta["height"], 768);
        assert_eq!(meta["lora_name"], "alice");
        assert_eq!(meta["seed"], 7);
    }

    #[test]
    fn reconcile_registers_sidecar_images_once() {
        let dir = tempfile::TempDir::new().unwrap();
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db = Database::open_at(db_file.path()).unwrap();

        let date_dir = dir.path().join("2026-07-16");
        std::fs::create_dir_all(&date_dir).unwrap();
        write_image_with_sidecar(&date_dir, "a.png", b"imgA", &sidecar("pa", "2026-07-16"));
        write_image_with_sidecar(&date_dir, "b.png", b"imgB", &sidecar("pb", "2026-07-16"));
        // No sidecar → not registered.
        std::fs::write(date_dir.join("c.png"), b"imgC").unwrap();

        assert_eq!(reconcile_outputs_in(&db, dir.path()).unwrap(), 2);
        // Idempotent.
        assert_eq!(reconcile_outputs_in(&db, dir.path()).unwrap(), 0);

        let arts = db.list_artifacts(None).unwrap();
        assert_eq!(arts.len(), 2);
        assert!(arts.iter().all(|a| a.artifact_id.starts_with("import:")));
        assert!(arts.iter().all(|a| a.kind == "image"));
        let meta: serde_json::Value =
            serde_json::from_str(arts[0].metadata.as_deref().unwrap()).unwrap();
        assert_eq!(meta["base_model_id"], "z-image-turbo");
    }

    #[test]
    fn import_lands_by_sidecar_date_and_is_idempotent() {
        let staging = tempfile::TempDir::new().unwrap();
        let outputs = tempfile::TempDir::new().unwrap();
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db = Database::open_at(db_file.path()).unwrap();

        write_image_with_sidecar(
            staging.path(),
            "x.png",
            b"imgX",
            &sidecar("px", "2026-07-15"),
        );

        let imported = import_run_dir_into(staging.path(), &db, outputs.path()).unwrap();
        assert_eq!(imported.len(), 1);
        let dest = outputs.path().join("2026-07-15").join("x.png");
        assert_eq!(imported[0], dest);
        assert!(dest.exists());
        assert!(dest.with_extension("yaml").exists());
        assert_eq!(db.list_artifacts(None).unwrap().len(), 1);

        // Re-import: same content → no duplicate file, no duplicate row.
        let again = import_run_dir_into(staging.path(), &db, outputs.path()).unwrap();
        assert_eq!(again, vec![dest.clone()]);
        assert_eq!(db.list_artifacts(None).unwrap().len(), 1);

        // Different content under the same name → suffixed, both kept.
        std::fs::write(staging.path().join("x.png"), b"imgX-v2").unwrap();
        let third = import_run_dir_into(staging.path(), &db, outputs.path()).unwrap();
        assert_eq!(third[0], outputs.path().join("2026-07-15").join("x-2.png"));
        assert!(third[0].with_extension("yaml").exists());
        assert_eq!(db.list_artifacts(None).unwrap().len(), 2);
    }

    #[test]
    fn import_falls_back_to_today_without_sidecar() {
        let staging = tempfile::TempDir::new().unwrap();
        let outputs = tempfile::TempDir::new().unwrap();
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db = Database::open_at(db_file.path()).unwrap();

        std::fs::write(staging.path().join("bare.png"), b"img").unwrap();
        let imported = import_run_dir_into(staging.path(), &db, outputs.path()).unwrap();
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        assert_eq!(imported[0], outputs.path().join(today).join("bare.png"));
        // No sidecar → file lands but nothing to register.
        assert_eq!(db.list_artifacts(None).unwrap().len(), 0);
    }
}
