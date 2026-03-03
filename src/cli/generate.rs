use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

use crate::core::cloud::{CloudExecutor, CloudProvider};
use crate::core::db::Database;
use crate::core::executor::{Executor, LocalExecutor};
use crate::core::job::*;
use crate::core::preflight;

/// Size presets: aspect ratio → (width, height)
const SIZE_PRESETS: &[(&str, u32, u32)] = &[
    ("1:1", 1024, 1024),
    ("16:9", 1344, 768),
    ("9:16", 768, 1344),
    ("4:3", 1152, 896),
    ("3:4", 896, 1152),
];

/// Resolve a size preset string to (width, height).
fn resolve_size(size: &str) -> Result<(u32, u32)> {
    // Check presets
    for &(name, w, h) in SIZE_PRESETS {
        if size == name {
            return Ok((w, h));
        }
    }

    // Try WxH format
    if let Some((w, h)) = size.split_once('x') {
        let w: u32 = w.parse().context("Invalid width in size")?;
        let h: u32 = h.parse().context("Invalid height in size")?;
        return Ok((w, h));
    }

    anyhow::bail!(
        "Unknown size: {size}. Use a preset (1:1, 16:9, 9:16, 4:3, 3:4) or WxH (e.g. 1024x1024)"
    );
}

/// Resolve a LoRA name to its store path by looking in the DB.
fn resolve_lora(name: &str, db: &Database) -> Result<Option<LoraRef>> {
    // Check if the name is a direct path to a .safetensors file
    let path = PathBuf::from(name);
    if path.exists() && path.extension().is_some_and(|e| e == "safetensors") {
        return Ok(Some(LoraRef {
            name: path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            path: path.to_string_lossy().to_string(),
            weight: 1.0,
        }));
    }

    // Look up in installed models
    let installed = db.list_installed(None)?;
    for model in &installed {
        if model.name == name && model.asset_type == "lora" {
            return Ok(Some(LoraRef {
                name: model.name.clone(),
                path: model.store_path.clone(),
                weight: 1.0,
            }));
        }
    }

    anyhow::bail!(
        "LoRA not found: {name}. Use `modl model ls --type lora` to see installed LoRAs, or provide a file path."
    );
}

/// Default inference steps based on model type.
fn default_steps(base_model: &str) -> u32 {
    let lower = base_model.to_lowercase();
    if lower.contains("schnell") || lower.contains("turbo") || lower.contains("lightning") {
        4
    } else {
        28
    }
}

/// Default guidance scale based on model type.
fn default_guidance(base_model: &str) -> f32 {
    let lower = base_model.to_lowercase();
    if lower.contains("schnell") || lower.contains("turbo") || lower.contains("lightning") {
        0.0
    } else {
        3.5
    }
}

/// Resolve base model path from installed models.
fn resolve_base_model_path(base_model: &str, db: &Database) -> Option<String> {
    let installed = db.list_installed(None).ok()?;
    for model in &installed {
        if model.name == base_model
            && (model.asset_type == "checkpoint" || model.asset_type == "diffusion_model")
        {
            return Some(model.store_path.clone());
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub async fn run(
    prompt: &str,
    base: Option<&str>,
    lora: Option<&str>,
    seed: Option<u64>,
    size: &str,
    steps: Option<u32>,
    guidance: Option<f32>,
    count: u32,
    cloud: bool,
    provider: Option<CloudProvider>,
) -> Result<()> {
    let db = Database::open()?;

    // -------------------------------------------------------------------
    // Resolve base model
    // -------------------------------------------------------------------
    let base_model = base.unwrap_or("flux-schnell").to_string();

    // -------------------------------------------------------------------
    // Pre-flight checks (fail fast with actionable hints)
    // -------------------------------------------------------------------
    if !cloud {
        preflight::for_generation(&base_model)?;
    }

    let base_model_path = resolve_base_model_path(&base_model, &db);

    // -------------------------------------------------------------------
    // Resolve size
    // -------------------------------------------------------------------
    let (width, height) = resolve_size(size)?;

    // -------------------------------------------------------------------
    // Resolve LoRA
    // -------------------------------------------------------------------
    let lora_ref = match lora {
        Some(name) => Some(resolve_lora(name, &db)?.context("LoRA resolution returned None")?),
        None => None,
    };

    // -------------------------------------------------------------------
    // Build output directory: ~/.modl/outputs/<date>/
    // -------------------------------------------------------------------
    let date = chrono::Local::now().format("%Y-%m-%d");
    let output_dir = dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
        .join("outputs")
        .join(date.to_string());
    std::fs::create_dir_all(&output_dir)?;

    // -------------------------------------------------------------------
    // Build spec
    // -------------------------------------------------------------------
    let steps = steps.unwrap_or_else(|| default_steps(&base_model));
    let guidance = guidance.unwrap_or_else(|| default_guidance(&base_model));

    let spec = GenerateJobSpec {
        prompt: prompt.to_string(),
        model: ModelRef {
            base_model_id: base_model.clone(),
            base_model_path,
        },
        lora: lora_ref.clone(),
        output: GenerateOutputRef {
            output_dir: output_dir.to_string_lossy().to_string(),
        },
        params: GenerateParams {
            width,
            height,
            steps,
            guidance,
            seed,
            count,
        },
        runtime: RuntimeRef {
            profile: "trainer-cu124".to_string(),
            python_version: Some("3.11.11".to_string()),
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
    println!("{} Generating image(s)...", style("→").cyan());
    println!("  Prompt: {}", style(prompt).italic());
    println!("  Model:  {}", base_model);
    if let Some(ref lr) = lora_ref {
        println!("  LoRA:   {}", lr.name);
    }
    println!("  Size:   {}×{}", width, height);
    println!("  Steps:  {}", steps);
    if let Some(s) = seed {
        println!("  Seed:   {}", s);
    }
    if count > 1 {
        println!("  Count:  {}", count);
    }

    // -------------------------------------------------------------------
    // Execute
    // -------------------------------------------------------------------
    execute_generate(spec, cloud, provider).await
}

async fn execute_generate(
    spec: GenerateJobSpec,
    cloud: bool,
    provider: Option<CloudProvider>,
) -> Result<()> {
    let db = Database::open()?;
    let spec_json = serde_json::to_string(&spec)?;
    let target_str = serde_json::to_string(&spec.target)?;

    // -------------------------------------------------------------------
    // 1. Bootstrap executor
    // -------------------------------------------------------------------
    let mut executor: Box<dyn Executor> = if cloud {
        let cloud_provider = resolve_cloud_provider(provider);
        println!(
            "{} Preparing cloud generation via {}...",
            style("→").cyan(),
            style(cloud_provider.to_string()).bold()
        );
        Box::new(CloudExecutor::new(cloud_provider)?)
    } else {
        println!("{} Preparing runtime...", style("→").cyan());
        Box::new(LocalExecutor::from_runtime_setup().await?)
    };

    // -------------------------------------------------------------------
    // 2. Submit job
    // -------------------------------------------------------------------
    let handle = executor.submit_generate(&spec)?;
    let job_id = &handle.job_id;

    db.insert_job(
        job_id,
        "generate",
        "queued",
        &spec_json,
        target_str.trim_matches('"'),
        None,
    )?;

    // -------------------------------------------------------------------
    // 3. Event loop with progress
    // -------------------------------------------------------------------
    let rx = executor.events(job_id)?;
    db.update_job_status(job_id, "running")?;

    let pb = ProgressBar::new(spec.params.count as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} images {msg}",
        )?
        .progress_chars("█▓░"),
    );

    let mut artifact_paths: Vec<String> = Vec::new();
    let mut final_status = "completed";

    for event in rx {
        match &event.event {
            EventPayload::Progress {
                step, total_steps, ..
            } => {
                pb.set_length(*total_steps as u64);
                pb.set_position(*step as u64);
            }
            EventPayload::Artifact { path, .. } => {
                artifact_paths.push(path.clone());
            }
            EventPayload::Completed { message } => {
                pb.finish_with_message(message.as_deref().unwrap_or("done").to_string());
                break;
            }
            EventPayload::Error { code, message, .. } => {
                pb.abandon_with_message(format!("error: {code}"));
                println!("{} Generation failed: {message}", style("✗").red().bold());
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
            EventPayload::Heartbeat => {}
        }

        // Persist event
        let event_json = serde_json::to_string(&event).unwrap_or_default();
        let _ = db.insert_job_event(job_id, event.sequence, &event_json);
    }

    // -------------------------------------------------------------------
    // 4. Update status
    // -------------------------------------------------------------------
    db.update_job_status(job_id, final_status)?;

    // -------------------------------------------------------------------
    // 5. Print results
    // -------------------------------------------------------------------
    if final_status == "completed" && !artifact_paths.is_empty() {
        println!();
        println!(
            "{} Generated {} image(s):",
            style("✓").green().bold(),
            artifact_paths.len()
        );
        for path in &artifact_paths {
            println!("  {}", path);
        }

        // Register artifacts in DB
        for (i, path) in artifact_paths.iter().enumerate() {
            let artifact_id = format!("{}-img-{}", job_id, i);
            let _ = db.insert_artifact(
                &artifact_id,
                Some(job_id),
                "image",
                path,
                "", // sha256 – not computed yet
                0,  // size_bytes – not computed yet
                None,
            );
        }
    } else if artifact_paths.is_empty() && final_status == "completed" {
        println!(
            "\n{} Generation completed but no images were produced.",
            style("⚠").yellow()
        );
    }

    Ok(())
}

/// Resolve cloud provider from --provider flag or config default.
fn resolve_cloud_provider(provider: Option<CloudProvider>) -> CloudProvider {
    if let Some(p) = provider {
        return p;
    }

    // Check config for default provider
    if let Ok(config) = crate::core::config::Config::load()
        && let Some(ref cloud) = config.cloud
        && let Some(ref default) = cloud.default_provider
        && let Ok(p) = default.parse()
    {
        return p;
    }

    // Default to Modal
    CloudProvider::Modal
}
