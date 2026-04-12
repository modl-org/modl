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
    #[serde(skip_serializing_if = "String::is_empty")]
    pub size: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_strength: Option<f32>,
    pub created_at: String,
    pub source: String,
    /// Number of video frames (video outputs only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_frames: Option<u32>,
    /// Video frame rate (video outputs only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
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
    /// Artifact kind: "image" or "video"
    pub kind: String,
    /// Number of video frames (video only)
    pub num_frames: Option<u32>,
    /// Video frame rate (video only)
    pub fps: Option<u32>,
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
    num_frames: Option<u32>,
    fps: Option<u32>,
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
        num_frames: v
            .get("num_frames")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
        fps: v.get("fps").and_then(|x| x.as_u64()).map(|n| n as u32),
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
        num_frames: spec
            .pointer("/params/num_frames")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
        fps: spec
            .pointer("/params/fps")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
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
        if (artifact.kind != "image" && artifact.kind != "video") || artifact.path.is_empty() {
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
                                "num_frames": m.num_frames,
                                "fps": m.fps,
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
    // Thumbnails are cached as <hash[..16]>.jpg where hash = sha256("path:width")
    // We check all known thumb widths used by the UI.
    let widths = [200u32, 320, 480];
    for w in widths {
        let hash_input = format!("{}:{}", source.to_string_lossy(), w);
        let hash = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(hash_input.as_bytes());
            format!("{:x}", h.finalize())
        };
        let thumb_path = cache_dir.join(format!("{}.jpg", &hash[..16]));
        let _ = std::fs::remove_file(thumb_path);
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
                name.ends_with(".png")
                    || name.ends_with(".jpg")
                    || name.ends_with(".webp")
                    || name.ends_with(".mp4")
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
                let kind = if filename.ends_with(".mp4") {
                    "video".to_string()
                } else {
                    "image".to_string()
                };
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
                    kind,
                    num_frames: meta.num_frames,
                    fps: meta.fps,
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
