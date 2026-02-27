use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

use crate::core::artifacts;
use crate::core::cloud::{CloudExecutor, CloudProvider};
use crate::core::dataset;
use crate::core::db::Database;
use crate::core::executor::{Executor, LocalExecutor};
use crate::core::gpu;
use crate::core::job::*;
use crate::core::presets::{self, DatasetStats, GpuContext};

/// Run the train command. Arguments are all optional; missing ones trigger
/// interactive prompts (except when --config is given).
#[allow(clippy::too_many_arguments)]
pub async fn run(
    dataset_arg: Option<&str>,
    base: Option<&str>,
    name: Option<&str>,
    trigger: Option<&str>,
    preset_arg: Option<&str>,
    steps: Option<u32>,
    config: Option<&str>,
    dry_run: bool,
    cloud: bool,
    provider: Option<&str>,
) -> Result<()> {
    // -------------------------------------------------------------------
    // Fast path: --config <yaml> loads a full spec directly
    // -------------------------------------------------------------------
    if let Some(config_path) = config {
        let yaml = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config: {config_path}"))?;
        let mut spec: TrainJobSpec =
            serde_yaml::from_str(&yaml).context("Failed to parse TrainJobSpec YAML")?;

        // Respect --cloud flag even when loading spec from file
        if cloud {
            spec.target = ExecutionTarget::Cloud;
        }

        if dry_run {
            println!("{}", serde_yaml::to_string(&spec)?);
            return Ok(());
        }

        return execute_training(spec, cloud, provider).await;
    }

    // -------------------------------------------------------------------
    // Resolve dataset
    // -------------------------------------------------------------------
    let dataset_path = match dataset_arg {
        Some(d) => resolve_dataset_path(d),
        None => {
            // Interactive: pick from managed datasets or enter path
            let datasets = dataset::list()?;
            if datasets.is_empty() {
                println!(
                    "{} No managed datasets found. Please provide a path with --dataset.",
                    style("!").yellow()
                );
                anyhow::bail!("No dataset specified");
            }

            let items: Vec<String> = datasets
                .iter()
                .map(|d| format!("{} ({} images)", d.name, d.image_count))
                .collect();

            let selection = dialoguer::Select::new()
                .with_prompt("Select dataset")
                .items(&items)
                .default(0)
                .interact()
                .context("Dataset selection cancelled")?;

            datasets[selection].path.clone()
        }
    };

    let ds_info = dataset::validate(&dataset_path)?;
    if ds_info.image_count < 5 {
        println!(
            "{} Only {} images. Consider 5-20 for good LoRA quality.",
            style("⚠").yellow(),
            ds_info.image_count
        );
    }

    // -------------------------------------------------------------------
    // Resolve base model
    // -------------------------------------------------------------------
    let base_model = match base {
        Some(b) => b.to_string(),
        None => {
            let models = &["flux-dev", "flux-schnell"];
            let selection = dialoguer::Select::new()
                .with_prompt("Base model")
                .items(models)
                .default(1)
                .interact()
                .context("Base model selection cancelled")?;
            models[selection].to_string()
        }
    };

    // -------------------------------------------------------------------
    // Resolve trigger word
    // -------------------------------------------------------------------
    let trigger_word = match trigger {
        Some(t) => t.to_string(),
        None => dialoguer::Input::<String>::new()
            .with_prompt("Trigger word")
            .default("OHWX".to_string())
            .interact_text()
            .context("Trigger word input cancelled")?,
    };

    // -------------------------------------------------------------------
    // Resolve output name
    // -------------------------------------------------------------------
    let lora_name = match name {
        Some(n) => n.to_string(),
        None => {
            let default_name = format!("{}-v1", ds_info.name);
            dialoguer::Input::<String>::new()
                .with_prompt("LoRA name")
                .default(default_name)
                .interact_text()
                .context("Name input cancelled")?
        }
    };

    // -------------------------------------------------------------------
    // Resolve preset
    // -------------------------------------------------------------------
    let preset = match preset_arg {
        Some(p) => p.parse::<Preset>()?,
        None => {
            let presets_list = &[
                "Quick (~20 min)",
                "Standard (~45 min)",
                "Advanced (edit YAML)",
            ];
            let selection = dialoguer::Select::new()
                .with_prompt("Training preset")
                .items(presets_list)
                .default(0)
                .interact()
                .context("Preset selection cancelled")?;
            match selection {
                0 => Preset::Quick,
                1 => Preset::Standard,
                _ => Preset::Advanced,
            }
        }
    };

    // -------------------------------------------------------------------
    // GPU detect + resolve params
    // -------------------------------------------------------------------
    let gpu_info = gpu::detect();
    if let Some(ref g) = gpu_info {
        println!(
            "{} Detected GPU: {} ({} MB VRAM)",
            style("→").cyan(),
            g.name,
            g.vram_mb
        );
    }

    let gpu_ctx = gpu_info.as_ref().map(|g| GpuContext { vram_mb: g.vram_mb });
    let ds_stats = DatasetStats {
        image_count: ds_info.image_count,
        caption_coverage: ds_info.caption_coverage,
    };

    let mut params = presets::resolve_params(
        preset,
        &ds_stats,
        gpu_ctx.as_ref(),
        &base_model,
        &trigger_word,
    );

    // Override steps if explicitly provided
    if let Some(s) = steps {
        params.steps = s;
    }

    // -------------------------------------------------------------------
    // Advanced preset: open $EDITOR
    // -------------------------------------------------------------------
    if preset == Preset::Advanced {
        let tmp_yaml = serde_yaml::to_string(&params)?;
        let edited = edit_in_editor(&tmp_yaml)?;
        params = serde_yaml::from_str(&edited).context("Failed to parse edited YAML")?;
    }

    // -------------------------------------------------------------------
    // Assemble TrainJobSpec
    // -------------------------------------------------------------------
    let output_dir = dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".mods")
        .join("training_output")
        .join(&lora_name);

    std::fs::create_dir_all(&output_dir)?;

    let spec = TrainJobSpec {
        dataset: DatasetRef {
            name: ds_info.name.clone(),
            path: ds_info.path.to_string_lossy().to_string(),
            image_count: ds_info.image_count,
            caption_coverage: ds_info.caption_coverage,
        },
        model: ModelRef {
            base_model_id: base_model.clone(),
            base_model_path: None,
        },
        output: OutputRef {
            lora_name: lora_name.clone(),
            destination_dir: output_dir.to_string_lossy().to_string(),
        },
        params,
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
    // Dry run: print spec and exit
    // -------------------------------------------------------------------
    if dry_run {
        println!("{} Dry run — generated spec:", style("✓").green().bold());
        println!("{}", serde_yaml::to_string(&spec)?);
        return Ok(());
    }

    execute_training(spec, cloud, provider).await
}

/// Execute training: persist job, run executor, collect artifacts.
async fn execute_training(spec: TrainJobSpec, cloud: bool, provider: Option<&str>) -> Result<()> {
    let db = Database::open()?;

    let spec_json = serde_json::to_string(&spec)?;
    let target_str = serde_json::to_string(&spec.target)?;

    // -------------------------------------------------------------------
    // 1. Bootstrap executor
    // -------------------------------------------------------------------
    let mut executor: Box<dyn Executor> = if cloud {
        let cloud_provider = resolve_cloud_provider(provider)?;
        println!(
            "{} Preparing cloud training via {}...",
            style("→").cyan(),
            style(cloud_provider.to_string()).bold()
        );
        Box::new(CloudExecutor::new(cloud_provider)?)
    } else {
        println!("{} Preparing training runtime...", style("→").cyan());
        Box::new(LocalExecutor::from_runtime_setup().await?)
    };

    // -------------------------------------------------------------------
    // 2. Submit job
    // -------------------------------------------------------------------
    let handle = executor.submit(&spec)?;
    let job_id = &handle.job_id;

    db.insert_job(
        job_id,
        "train",
        "queued",
        &spec_json,
        target_str.trim_matches('"'),
        None,
    )?;

    println!(
        "{} Training started — {}",
        style("→").cyan(),
        style(job_id).dim()
    );

    // -------------------------------------------------------------------
    // 3. Event loop with progress bar
    // -------------------------------------------------------------------
    let rx = executor.events(job_id)?;
    db.update_job_status(job_id, "running")?;

    let pb = ProgressBar::new(spec.params.steps as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} steps {msg}",
        )?
        .progress_chars("█▓░"),
    );

    let mut artifact_paths: Vec<String> = Vec::new();
    let mut final_status = "completed";

    for event in rx {
        match &event.event {
            EventPayload::Progress {
                step,
                total_steps,
                loss,
                ..
            } => {
                pb.set_length(*total_steps as u64);
                pb.set_position(*step as u64);
                if let Some(l) = loss {
                    pb.set_message(format!("loss: {l:.4}"));
                }
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
                println!("{} Training failed: {message}", style("✗").red().bold());
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

        // Persist event to DB
        let event_json = serde_json::to_string(&event).unwrap_or_default();
        let _ = db.insert_job_event(job_id, event.sequence, &event_json);
    }

    // -------------------------------------------------------------------
    // 4. Update job status
    // -------------------------------------------------------------------
    db.update_job_status(job_id, final_status)?;

    // -------------------------------------------------------------------
    // 5. Collect artifacts
    // -------------------------------------------------------------------
    if final_status == "completed" {
        let store_root = dirs::home_dir()
            .expect("Could not determine home directory")
            .join(".mods");

        for artifact_path in &artifact_paths {
            let path = PathBuf::from(artifact_path);
            if path.exists() && path.extension().is_some_and(|e| e == "safetensors") {
                match artifacts::collect_lora(
                    &path,
                    &spec.output.lora_name,
                    &spec.model.base_model_id,
                    &spec.params.trigger_word,
                    job_id,
                    &db,
                    &store_root,
                ) {
                    Ok(collected) => {
                        println!();
                        println!("{} LoRA collected!", style("✓").green().bold());
                        println!("  Name:   {}", spec.output.lora_name);
                        println!("  Path:   {}", collected.store_path.display());
                        println!("  SHA256: {}", &collected.sha256[..16]);
                        println!(
                            "  Size:   {:.1} MB",
                            collected.size_bytes as f64 / 1_048_576.0
                        );
                        for link in &collected.symlinks {
                            println!("  Link:   {}", link.display());
                        }
                    }
                    Err(e) => {
                        println!(
                            "{} Failed to collect artifact {}: {e}",
                            style("⚠").yellow(),
                            artifact_path
                        );
                    }
                }
            }
        }

        if artifact_paths.is_empty() {
            println!(
                "\n{} Training completed but no artifacts were emitted. Check output directory: {}",
                style("⚠").yellow(),
                spec.output.destination_dir
            );
        }
    }

    Ok(())
}

/// Resolve dataset name-or-path to a directory path.
fn resolve_dataset_path(name_or_path: &str) -> PathBuf {
    let path = PathBuf::from(name_or_path);
    if path.is_absolute() || name_or_path.contains('/') || name_or_path.contains('\\') {
        path
    } else {
        dirs::home_dir()
            .expect("Could not determine home directory")
            .join(".mods")
            .join("datasets")
            .join(name_or_path)
    }
}

/// Open text in $EDITOR, return edited content.
fn edit_in_editor(content: &str) -> Result<String> {
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join(format!("mods-train-{}.yaml", std::process::id()));
    std::fs::write(&tmp_path, content)?;

    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
    let status = std::process::Command::new(&editor)
        .arg(&tmp_path)
        .status()
        .with_context(|| format!("Failed to launch editor: {editor}"))?;

    if !status.success() {
        let _ = std::fs::remove_file(&tmp_path);
        anyhow::bail!("Editor exited with non-zero status");
    }

    let edited = std::fs::read_to_string(&tmp_path).context("Failed to read edited file")?;
    let _ = std::fs::remove_file(&tmp_path);
    Ok(edited)
}

/// Resolve cloud provider from --provider flag or config default.
fn resolve_cloud_provider(provider: Option<&str>) -> Result<CloudProvider> {
    if let Some(p) = provider {
        return p.parse();
    }

    // Check config for default provider
    if let Ok(config) = crate::core::config::Config::load()
        && let Some(ref cloud) = config.cloud
        && let Some(ref default) = cloud.default_provider
    {
        return default.parse();
    }

    // Default to Modal
    Ok(CloudProvider::Modal)
}
