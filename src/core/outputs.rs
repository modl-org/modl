//! Service layer for output management.
//!
//! All output operations (list, delete, favorite) go through this module.
//! Both the CLI and the web UI server call these functions — neither should
//! talk to the database or filesystem directly for output operations.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use super::db::{ArtifactRecord, Database};

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

pub struct ToggleFavoriteResult {
    pub favorited: bool,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn modl_root() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
}

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
    let outputs_root = modl_root().join("outputs");
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
        target_file = Some(modl_root().join(rp));
    }

    let Some(target_file) = target_file else {
        bail!("Missing artifact_id or path");
    };

    // Safety check: path must be within outputs/
    let outputs_root = modl_root().join("outputs");
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
        .strip_prefix(&modl_root().to_string_lossy().to_string())
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
