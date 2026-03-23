use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

use crate::core::cloud::{CloudExecutor, CloudProvider};
use crate::core::db::Database;
use crate::core::executor::{Executor, LocalExecutor};
use crate::core::job::*;
use crate::core::model_family;
use crate::core::outputs::{SidecarMetadata, write_sidecar_yaml};
use crate::core::preflight;
use crate::core::runtime;

/// All arguments for `modl edit`, used by both CLI and web UI.
pub struct EditArgs<'a> {
    pub prompt: &'a str,
    pub images: &'a [String],
    pub base: Option<&'a str>,
    pub seed: Option<u64>,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub count: u32,
    pub size: Option<&'a str>,
    pub fast: bool,
    pub cloud: bool,
    pub provider: Option<CloudProvider>,
    pub no_worker: bool,
    pub json: bool,
}

/// Resolve an image input: local path or URL → local path.
/// URLs are downloaded to ~/.modl/tmp/.
async fn resolve_image_input(input: &str) -> Result<String> {
    if input.starts_with("http://") || input.starts_with("https://") {
        let tmp_dir = crate::core::paths::modl_root().join("tmp");
        std::fs::create_dir_all(&tmp_dir)?;

        // Use a hash-based filename to avoid collisions
        let hash = &format!("{:x}", md5_hash(input))[..12];
        let ext = url_extension(input).unwrap_or("png");
        let dest = tmp_dir.join(format!("{}.{}", hash, ext));

        if !dest.exists() {
            eprintln!("  {} Downloading image from URL...", style("↓").cyan());
            crate::core::download::download_file(input, &dest, None, None)
                .await
                .with_context(|| format!("Failed to download image: {input}"))?;
        }
        Ok(dest.to_string_lossy().to_string())
    } else {
        let path = PathBuf::from(input);
        if !path.exists() {
            anyhow::bail!("Image not found: {input}");
        }
        Ok(input.to_string())
    }
}

/// Simple hash for URL → filename mapping.
fn md5_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Extract file extension from a URL (before query params).
fn url_extension(url: &str) -> Option<&str> {
    let path = url.split('?').next()?;
    let filename = path.rsplit('/').next()?;
    let ext = filename.rsplit('.').next()?;
    if ext.len() <= 4 && ext.chars().all(|c| c.is_alphanumeric()) {
        Some(ext)
    } else {
        None
    }
}

/// Resolve base model path from installed models.
fn resolve_base_model_path(base_model: &str, db: &Database) -> Option<String> {
    let installed = db.list_installed(None).ok()?;
    for model in &installed {
        if (model.name == base_model || model.id == base_model)
            && (model.asset_type == "checkpoint" || model.asset_type == "diffusion_model")
        {
            return Some(model.store_path.clone());
        }
    }
    None
}

/// Resolve a LoRA name to its store path by looking in the DB.
fn resolve_lora(name: &str, weight: f32, db: &Database) -> Result<Option<LoraRef>> {
    let path = PathBuf::from(name);
    if path.exists() && path.extension().is_some_and(|e| e == "safetensors") {
        return Ok(Some(LoraRef {
            name: path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            path: path.to_string_lossy().to_string(),
            weight,
        }));
    }

    let installed = db.list_installed(None)?;
    for model in &installed {
        if (model.name == name || model.id == name) && model.asset_type == "lora" {
            return Ok(Some(LoraRef {
                name: model.name.clone(),
                path: model.store_path.clone(),
                weight,
            }));
        }
    }
    Ok(None)
}

const SIZE_PRESETS: &[(&str, u32, u32)] = &[
    ("1:1", 1024, 1024),
    ("16:9", 1344, 768),
    ("9:16", 768, 1344),
    ("4:3", 1152, 896),
    ("3:4", 896, 1152),
];

fn resolve_edit_size(size: &str) -> Result<(u32, u32)> {
    for &(name, w, h) in SIZE_PRESETS {
        if size == name {
            return Ok((w, h));
        }
    }
    if let Some((w, h)) = size.split_once('x') {
        let w: u32 = w.parse().context("Invalid width in size")?;
        let h: u32 = h.parse().context("Invalid height in size")?;
        return Ok((w, h));
    }
    anyhow::bail!(
        "Unknown size: {size}. Use a preset (1:1, 16:9, 9:16, 4:3, 3:4) or WxH (e.g. 1820x1024)"
    );
}

pub async fn run(args: EditArgs<'_>) -> Result<()> {
    let db = Database::open()?;

    let EditArgs {
        prompt,
        images,
        base,
        seed,
        steps,
        guidance,
        count,
        size,
        fast,
        cloud,
        provider,
        no_worker,
        json,
    } = args;

    // -------------------------------------------------------------------
    // Validate inputs
    // -------------------------------------------------------------------
    if images.is_empty() {
        anyhow::bail!("At least one --image is required for editing");
    }

    // -------------------------------------------------------------------
    // Resolve base model
    // -------------------------------------------------------------------
    let base_model = base.unwrap_or("qwen-image-edit").to_string();

    // -------------------------------------------------------------------
    // Pre-flight checks
    // -------------------------------------------------------------------
    if !cloud {
        preflight::for_generation(&base_model)?;
    }

    // Check model supports edit mode
    if let Err(msg) = model_family::validate_mode(&base_model, "edit") {
        anyhow::bail!(msg);
    }

    let base_model_path = resolve_base_model_path(&base_model, &db);

    // -------------------------------------------------------------------
    // Resolve --fast (Lightning LoRA)
    // -------------------------------------------------------------------
    let (fast_lora, fast_steps, fast_guidance) = if fast {
        let lightning = model_family::lightning_config(&base_model).with_context(|| {
            let supported: Vec<&str> = model_family::LIGHTNING_CONFIGS
                .iter()
                .map(|c| c.base_model_id)
                .collect();
            format!(
                "--fast is not yet supported for '{}'. Supported: {}",
                base_model,
                supported.join(", ")
            )
        })?;

        let lora_ref = resolve_lora(lightning.lora_registry_id, 1.0, &db).with_context(|| {
            format!(
                "Lightning LoRA '{}' is not installed.\n\n  \
                 Install it:\n\n    modl pull {} --variant {}\n",
                lightning.lora_registry_id, lightning.lora_registry_id, lightning.lora_variant,
            )
        })?;

        (lora_ref, Some(lightning.steps), Some(lightning.guidance))
    } else {
        (None, None, None)
    };

    // -------------------------------------------------------------------
    // Resolve image inputs (download URLs if needed)
    // -------------------------------------------------------------------
    let mut resolved_paths = Vec::new();
    for img in images {
        let path = resolve_image_input(img).await?;
        resolved_paths.push(path);
    }

    // -------------------------------------------------------------------
    // Build output directory
    // -------------------------------------------------------------------
    let date = chrono::Local::now().format("%Y-%m-%d");
    let output_dir = crate::core::paths::modl_root()
        .join("outputs")
        .join(date.to_string());
    std::fs::create_dir_all(&output_dir)?;

    // -------------------------------------------------------------------
    // Resolve defaults (--fast overrides, then explicit --steps/--guidance)
    // -------------------------------------------------------------------
    let (default_steps, default_guidance) = model_family::model_defaults(&base_model);
    let steps = steps.or(fast_steps).unwrap_or(default_steps);
    let guidance = guidance.or(fast_guidance).unwrap_or(default_guidance);

    // -------------------------------------------------------------------
    // Resolve output size (for outpainting)
    // -------------------------------------------------------------------
    let output_size: Option<(u32, u32)> = if let Some(s) = size {
        Some(resolve_edit_size(s)?)
    } else {
        None
    };

    // -------------------------------------------------------------------
    // Build spec
    // -------------------------------------------------------------------
    let spec = EditJobSpec {
        prompt: prompt.to_string(),
        model: ModelRef {
            base_model_id: base_model.clone(),
            base_model_path,
        },
        lora: fast_lora,
        output: GenerateOutputRef {
            output_dir: output_dir.to_string_lossy().to_string(),
        },
        params: EditParams {
            image_paths: resolved_paths.clone(),
            steps,
            guidance,
            seed,
            count,
            width: output_size.map(|(w, _)| w),
            height: output_size.map(|(_, h)| h),
        },
        runtime: RuntimeRef {
            profile: runtime::resolved_generation_profile().to_string(),
            python_version: Some("3.11.12".to_string()),
        },
        target: if cloud {
            ExecutionTarget::Cloud
        } else {
            ExecutionTarget::Local
        },
        labels: std::collections::HashMap::new(),
    };

    // -------------------------------------------------------------------
    // Print summary
    // -------------------------------------------------------------------
    if !json {
        println!("{} Editing image(s)...", style("→").cyan());
        println!("  Prompt: {}", style(prompt).italic());
        println!("  Model:  {}", base_model);
        if fast {
            println!(
                "  Mode:   {}",
                style("fast (Lightning LoRA)").green().bold()
            );
        }
        for (i, path) in resolved_paths.iter().enumerate() {
            println!("  Image {}: {}", i + 1, path);
        }
        println!("  Steps:  {}", steps);
        if let Some(s) = seed {
            println!("  Seed:   {}", s);
        }
        if count > 1 {
            println!("  Count:  {}", count);
        }
    }

    // -------------------------------------------------------------------
    // Execute
    // -------------------------------------------------------------------
    execute_edit(spec, cloud, provider, no_worker, json).await
}

async fn execute_edit(
    spec: EditJobSpec,
    cloud: bool,
    provider: Option<CloudProvider>,
    no_worker: bool,
    json: bool,
) -> Result<()> {
    let db = Database::open()?;
    let spec_json = serde_json::to_string(&spec)?;
    let target_str = serde_json::to_string(&spec.target)?;

    // Bootstrap executor
    let mut executor: Box<dyn Executor> = if cloud {
        let cloud_provider = resolve_cloud_provider(provider);
        Box::new(CloudExecutor::new(cloud_provider)?)
    } else {
        if !json {
            println!("{} Preparing runtime...", style("→").cyan());
        }
        let mut executor = LocalExecutor::for_generation().await?;
        if no_worker {
            executor.use_worker = false;
        }
        Box::new(executor)
    };

    // Submit job
    let handle = executor.submit_edit(&spec)?;
    let job_id = &handle.job_id;

    db.insert_job(
        job_id,
        "edit",
        "queued",
        &spec_json,
        target_str.trim_matches('"'),
        None,
    )?;

    // Event loop
    let rx = executor.events(job_id)?;
    db.update_job_status(job_id, "running")?;

    let pb = if json {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(spec.params.count as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} images {msg}",
            )?
            .progress_chars("█▓░"),
        );
        pb
    };

    struct EditedArtifact {
        path: String,
        sha256: Option<String>,
        size_bytes: Option<u64>,
    }

    let mut artifacts: Vec<EditedArtifact> = Vec::new();
    let mut final_status = "completed";

    for event in rx {
        match &event.event {
            EventPayload::Progress {
                step, total_steps, ..
            } => {
                pb.set_length(*total_steps as u64);
                pb.set_position(*step as u64);
            }
            EventPayload::Artifact {
                path,
                sha256,
                size_bytes,
            } => {
                artifacts.push(EditedArtifact {
                    path: path.clone(),
                    sha256: sha256.clone(),
                    size_bytes: *size_bytes,
                });
            }
            EventPayload::Completed { message } => {
                pb.finish_with_message(message.as_deref().unwrap_or("done").to_string());
                break;
            }
            EventPayload::Error { code, message, .. } => {
                pb.abandon_with_message(format!("error: {code}"));
                println!("{} Edit failed: {message}", style("✗").red().bold());
                final_status = "error";
                break;
            }
            EventPayload::Log { message, level } => {
                if level == "info" {
                    pb.println(format!("  {} {}", style("[log]").dim(), message));
                }
            }
            EventPayload::Warning { message, .. } => {
                pb.println(format!("  {} {}", style("[warn]").yellow(), message));
            }
            EventPayload::JobAccepted { .. } | EventPayload::JobStarted { .. } => {}
            EventPayload::Cancelled => {
                pb.abandon_with_message("cancelled".to_string());
                final_status = "cancelled";
                break;
            }
            EventPayload::Heartbeat | EventPayload::Result { .. } => {}
        }

        let event_json = serde_json::to_string(&event).unwrap_or_default();
        let _ = db.insert_job_event(job_id, event.sequence, &event_json);
    }

    db.update_job_status(job_id, final_status)?;

    // Print results
    if final_status == "completed" && !artifacts.is_empty() {
        for (i, artifact) in artifacts.iter().enumerate() {
            let artifact_id = format!("{}-img-{}", job_id, i);
            let image_seed = spec.params.seed.map(|s| s + i as u64);
            let metadata = serde_json::json!({
                "generated_with": "modl.run",
                "mode": "edit",
                "prompt": spec.prompt,
                "base_model_id": spec.model.base_model_id,
                "input_images": spec.params.image_paths,
                "steps": spec.params.steps,
                "guidance": spec.params.guidance,
                "seed": image_seed,
            });
            let metadata_str = metadata.to_string();
            let _ = db.insert_artifact(
                &artifact_id,
                Some(job_id),
                "image",
                &artifact.path,
                artifact.sha256.as_deref().unwrap_or(""),
                artifact.size_bytes.unwrap_or(0),
                Some(&metadata_str),
            );

            // Write YAML sidecar file next to the image
            let sidecar = SidecarMetadata {
                prompt: spec.prompt.clone(),
                base_model: spec.model.base_model_id.clone(),
                seed: image_seed,
                steps: spec.params.steps,
                guidance: spec.params.guidance,
                size: String::new(), // edit mode does not specify output size
                lora: spec.lora.as_ref().map(|l| l.name.clone()),
                lora_strength: spec.lora.as_ref().map(|l| l.weight),
                created_at: chrono::Utc::now().to_rfc3339(),
                source: "edit".to_string(),
            };
            write_sidecar_yaml(&artifact.path, &sidecar);
        }

        let artifact_paths: Vec<String> = artifacts.iter().map(|a| a.path.clone()).collect();
        if json {
            let output = serde_json::json!({
                "status": "completed",
                "job_id": job_id,
                "images": artifact_paths,
            });
            println!("{}", serde_json::to_string(&output)?);
        } else {
            println!();
            println!(
                "{} Edited {} image(s):",
                style("✓").green().bold(),
                artifact_paths.len()
            );
            for path in &artifact_paths {
                println!("  {}", path);
            }
        }
    } else if json {
        let artifact_paths: Vec<String> = artifacts.iter().map(|a| a.path.clone()).collect();
        println!(
            "{}",
            serde_json::json!({"status": final_status, "images": artifact_paths})
        );
    }

    if final_status == "error" {
        anyhow::bail!("Edit failed");
    }

    Ok(())
}

fn resolve_cloud_provider(provider: Option<CloudProvider>) -> CloudProvider {
    if let Some(p) = provider {
        return p;
    }
    if let Ok(config) = crate::core::config::Config::load()
        && let Some(ref cloud) = config.cloud
        && let Some(ref default) = cloud.default_provider
        && let Ok(p) = default.parse()
    {
        return p;
    }
    CloudProvider::Modal
}
