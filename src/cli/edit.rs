use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

use crate::core::cloud::{CloudExecutor, CloudProvider};
use crate::core::db::Database;
use crate::core::executor::{Executor, LocalExecutor};
use crate::core::gpu_session;
use crate::core::job::*;
use crate::core::model_family;
use crate::core::model_resolve;
use crate::core::outputs::{SidecarMetadata, write_sidecar_yaml};
use crate::core::preflight;
use crate::core::remote_executor::RemoteExecutor;
use crate::core::runtime;

/// All arguments for `modl edit`, used by both CLI and web UI.
pub struct EditArgs<'a> {
    pub prompt: &'a str,
    pub images: &'a [String],
    pub mask: Option<&'a str>,
    pub blend: &'a str,
    pub lora: Option<&'a str>,
    pub lora_strength: f32,
    pub base: Option<&'a str>,
    pub seed: Option<u64>,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub count: u32,
    pub size: Option<&'a str>,
    pub fast: Option<u32>,
    pub cloud: bool,
    pub provider: Option<CloudProvider>,
    pub no_worker: bool,
    pub attach_gpu: bool,
    pub gpu_type: &'a str,
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

fn resolve_base_model_path(base_model: &str, db: &Database) -> Option<String> {
    model_resolve::resolve_base_model_path(base_model, db)
}

fn resolve_lora(name: &str, weight: f32, db: &Database) -> Result<Option<LoraRef>> {
    model_resolve::resolve_lora(name, weight, db)
}

fn resolve_edit_size(size: &str) -> Result<(u32, u32)> {
    model_resolve::resolve_size(size)
}

pub async fn run(args: EditArgs<'_>) -> Result<()> {
    let db = Database::open()?;

    let EditArgs {
        prompt,
        images,
        mask,
        blend,
        lora,
        lora_strength,
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
        attach_gpu,
        gpu_type,
        json,
    } = args;

    // -------------------------------------------------------------------
    // Validate inputs
    // -------------------------------------------------------------------
    if images.is_empty() {
        anyhow::bail!("At least one --image is required for editing");
    }

    // Validate --blend
    if blend != "pixel" && blend != "latent" {
        anyhow::bail!("--blend must be 'pixel' or 'latent', got '{}'", blend);
    }

    // -------------------------------------------------------------------
    // Resolve base model
    // -------------------------------------------------------------------
    let base_model = base.unwrap_or("qwen-image-edit-2511").to_string();

    // -------------------------------------------------------------------
    // Pre-flight checks
    // -------------------------------------------------------------------
    if !cloud && !attach_gpu {
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
    let (fast_lora, fast_steps, fast_guidance, scheduler_overrides) = if let Some(fast_steps) = fast
    {
        let lightning = model_family::lightning_config(&base_model).with_context(|| {
            let supported: Vec<&str> = model_family::lightning_configs()
                .iter()
                .map(|c| c.base_model_id.as_str())
                .collect();
            format!(
                "--fast is not yet supported for '{}'. Supported: {}",
                base_model,
                supported.join(", ")
            )
        })?;

        let (variant, resolved_steps) = lightning.resolve(fast_steps);
        let lora_ref = resolve_lora(&lightning.lora_registry_id, 1.0, &db).with_context(|| {
            format!(
                "Lightning LoRA '{}' is not installed.\n\n  \
                 Install it:\n\n    modl pull {} --variant {}\n",
                lightning.lora_registry_id, lightning.lora_registry_id, variant,
            )
        })?;

        let sched_overrides: std::collections::HashMap<String, serde_json::Value> = lightning
            .scheduler_overrides
            .iter()
            .map(|(k, v)| {
                let val = if *v == "null" {
                    serde_json::Value::Null
                } else if let Ok(f) = v.parse::<f64>() {
                    serde_json::json!(f)
                } else {
                    serde_json::Value::String(v.to_string())
                };
                (k.to_string(), val)
            })
            .collect();

        (
            lora_ref,
            Some(resolved_steps),
            Some(lightning.guidance),
            sched_overrides,
        )
    } else {
        (None, None, None, std::collections::HashMap::new())
    };

    // -------------------------------------------------------------------
    // Resolve user --lora (takes priority over Lightning LoRA)
    // -------------------------------------------------------------------
    let user_lora = if let Some(lora_name) = lora {
        if fast.is_some() {
            anyhow::bail!(
                "Cannot use --lora and --fast together. --fast applies a Lightning LoRA which conflicts with a user LoRA."
            );
        }
        resolve_lora(lora_name, lora_strength, &db)?
    } else {
        None
    };

    // User LoRA takes priority; fall back to Lightning LoRA
    let resolved_lora = user_lora.or(fast_lora);

    // -------------------------------------------------------------------
    // Resolve image inputs (download URLs if needed)
    // -------------------------------------------------------------------
    let mut resolved_paths = Vec::new();
    for img in images {
        let path = resolve_image_input(img).await?;
        resolved_paths.push(path);
    }

    // -------------------------------------------------------------------
    // Resolve mask input
    // -------------------------------------------------------------------
    let resolved_mask: Option<String> = if let Some(m) = mask {
        match m {
            // Special values passed through — Python side handles derivation
            "auto" | "from-alpha" => Some(m.to_string()),
            // File path or URL
            _ => Some(resolve_image_input(m).await?),
        }
    } else {
        None
    };

    // Validate --blend latent requires a mask
    if blend == "latent" && resolved_mask.is_none() {
        anyhow::bail!("--blend latent requires --mask");
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
            arch_key: model_family::find_model(&base_model).map(|m| m.arch_key.to_string()),
        },
        lora: resolved_lora,
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
            mask_path: resolved_mask.clone(),
            blend_mode: blend.to_string(),
            scheduler_overrides,
        },
        runtime: RuntimeRef {
            profile: runtime::resolved_generation_profile().to_string(),
            python_version: Some("3.11.12".to_string()),
        },
        target: if attach_gpu {
            ExecutionTarget::Remote
        } else if cloud {
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
        if fast.is_some() {
            println!(
                "  Mode:   {}",
                style("fast (Lightning LoRA)").green().bold()
            );
        }
        if let Some(lora_name) = lora {
            println!("  LoRA:   {} (strength: {})", lora_name, lora_strength);
        }
        for (i, path) in resolved_paths.iter().enumerate() {
            println!("  Image {}: {}", i + 1, path);
        }
        if let Some(ref m) = resolved_mask {
            println!("  Mask:   {}", m);
            if blend == "latent" {
                println!("  Blend:  {}", style("latent").yellow());
            }
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
    execute_edit(spec, cloud, provider, no_worker, attach_gpu, gpu_type, json).await
}

async fn execute_edit(
    spec: EditJobSpec,
    cloud: bool,
    provider: Option<CloudProvider>,
    no_worker: bool,
    attach_gpu: bool,
    gpu_type: &str,
    json: bool,
) -> Result<()> {
    let db = Database::open()?;
    let spec_json = serde_json::to_string(&spec)?;
    let target_str = serde_json::to_string(&spec.target)?;

    // Bootstrap executor
    let mut executor: Box<dyn Executor> = if attach_gpu {
        if !json {
            println!(
                "{} Connecting to remote GPU ({})...",
                style("→").cyan(),
                style(gpu_type).bold()
            );
        }
        let session = gpu_session::ensure_session(
            gpu_type,
            "30m",
            std::slice::from_ref(&spec.model.base_model_id),
        )
        .await?;
        if !json {
            println!(
                "  {} Session {} ({})",
                style("✓").green(),
                style(&session.session_id).bold(),
                session.state,
            );
        }
        Box::new(RemoteExecutor::new(session))
    } else if cloud {
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
