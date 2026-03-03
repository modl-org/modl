use anyhow::{Context, Result};
use comfy_table::{ContentArrangement, Table};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

use crate::core::dataset;
use crate::core::job::{CaptionJobSpec, ResizeJobSpec, TagJobSpec};
use crate::core::registry::RegistryIndex;

#[derive(clap::Subcommand)]
pub enum DatasetCommands {
    /// Create a managed dataset from a directory of images
    Create {
        /// Name for the dataset
        name: String,
        /// Source directory containing images (jpg/jpeg/png). Supports subfolders (e.g. happy/, sad/) — each subfolder name is used as a tag prefix.
        #[arg(long)]
        from: String,
    },
    /// List all managed datasets
    Ls,
    /// Validate a dataset directory
    Validate {
        /// Dataset name or path to validate
        name_or_path: String,
    },
    /// Resize images to training resolution
    Resize {
        /// Dataset name or path
        name_or_path: String,
        /// Target resolution (max dimension in pixels)
        #[arg(long, default_value = "1024")]
        resolution: u32,
        /// Resize method: contain (fit inside, default), cover (crop to fill), squish (stretch)
        #[arg(long, default_value = "contain", value_parser = ["contain", "cover", "squish"])]
        method: String,
    },
    /// Auto-tag images with structured labels using a vision-language model
    Tag {
        /// Dataset name or path
        name_or_path: String,
        /// VL model for tagging
        #[arg(long, default_value = "florence-2", value_parser = ["florence-2", "wd-tagger"])]
        model: String,
        /// Re-tag images that already have .txt files
        #[arg(long)]
        overwrite: bool,
    },
    /// Auto-caption images using a vision-language model
    Caption {
        /// Dataset name or path
        name_or_path: String,
        /// Captioning model to use
        #[arg(long, default_value = "florence-2", value_parser = ["florence-2", "blip"])]
        model: String,
        /// Re-caption images that already have .txt files
        #[arg(long)]
        overwrite: bool,
    },
    /// Full pipeline: create → resize → tag/caption
    Prepare {
        /// Name for the dataset
        name: String,
        /// Source directory containing images
        #[arg(long)]
        from: String,
        /// Target resolution
        #[arg(long, default_value = "1024")]
        resolution: u32,
        /// VL model for tagging/captioning
        #[arg(long, default_value = "florence-2")]
        model: String,
        /// Skip image resizing
        #[arg(long)]
        no_resize: bool,
        /// Skip auto-tagging
        #[arg(long)]
        no_tag: bool,
        /// Skip auto-captioning (just tag)
        #[arg(long)]
        no_caption: bool,
    },
}

pub async fn run(command: DatasetCommands) -> Result<()> {
    match command {
        DatasetCommands::Create { name, from } => run_create(&name, &from).await.map(|_| ()),
        DatasetCommands::Ls => run_list().await,
        DatasetCommands::Validate { name_or_path } => run_validate(&name_or_path).await,
        DatasetCommands::Resize {
            name_or_path,
            resolution,
            method,
        } => run_resize(&name_or_path, resolution, &method).await,
        DatasetCommands::Tag {
            name_or_path,
            model,
            overwrite,
        } => run_tag(&name_or_path, &model, overwrite).await,
        DatasetCommands::Caption {
            name_or_path,
            model,
            overwrite,
        } => run_caption(&name_or_path, &model, overwrite).await,
        DatasetCommands::Prepare {
            name,
            from,
            resolution,
            model,
            no_resize,
            no_tag,
            no_caption,
        } => {
            run_prepare(
                &name, &from, resolution, &model, no_resize, no_tag, no_caption,
            )
            .await
        }
    }
}

async fn run_create(name: &str, from: &str) -> Result<dataset::DatasetInfo> {
    let from_path = PathBuf::from(from);
    println!(
        "{} Creating dataset '{}' from {}",
        style("→").cyan(),
        style(name).bold(),
        from_path.display()
    );

    let info = dataset::create(name, &from_path)?;

    println!("{} Dataset created", style("✓").green().bold());
    print_dataset_summary(&info);

    if info.image_count < 5 {
        println!(
            "\n{} Only {} images found. Consider adding more for better results (5-20 recommended).",
            style("⚠").yellow(),
            info.image_count
        );
    }

    if info.caption_coverage < 1.0 {
        let uncaptioned = info.image_count - info.captioned_count;
        println!(
            "{} {} images without captions. Add .txt files with the same name for better training.",
            style("ℹ").dim(),
            uncaptioned
        );
    }

    Ok(info)
}

async fn run_list() -> Result<()> {
    let datasets = dataset::list()?;

    if datasets.is_empty() {
        println!("No datasets found. Create one with:");
        println!("  modl datasets create <name> --from <dir>");
        return Ok(());
    }

    let mut table = Table::new();
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Name", "Images", "Captions", "Coverage", "Path"]);

    for ds in &datasets {
        table.add_row(vec![
            ds.name.clone(),
            ds.image_count.to_string(),
            ds.captioned_count.to_string(),
            format!("{:.0}%", ds.caption_coverage * 100.0),
            ds.path.display().to_string(),
        ]);
    }

    println!("{table}");
    Ok(())
}

async fn run_validate(name_or_path: &str) -> Result<()> {
    let path = resolve_dataset_path(name_or_path);

    println!(
        "{} Validating dataset at {}",
        style("→").cyan(),
        path.display()
    );

    let info = dataset::validate(&path)?;

    println!("{} Dataset is valid", style("✓").green().bold());
    print_dataset_summary(&info);

    if info.image_count < 5 {
        println!(
            "\n{} Only {} images. Consider 5-20 for good LoRA quality.",
            style("⚠").yellow(),
            info.image_count
        );
    }

    Ok(())
}

fn print_dataset_summary(info: &dataset::DatasetInfo) {
    println!("  Name:     {}", info.name);
    println!("  Path:     {}", info.path.display());
    println!("  Images:   {}", info.image_count);
    println!(
        "  Captions: {} / {} ({:.0}%)",
        info.captioned_count,
        info.image_count,
        info.caption_coverage * 100.0
    );
}

/// Resolve a VL model name (e.g. "florence-2", "blip") to a HuggingFace repo ID
/// by looking it up in the registry. Falls back to hardcoded IDs if the registry
/// is unavailable or doesn't contain the model.
async fn resolve_vl_model(model_name: &str) -> Option<String> {
    use crate::core::manifest::AssetType;

    // Map short CLI names → registry manifest IDs
    let registry_id = match model_name {
        "florence-2" | "florence2" | "florence" => "florence-2-large",
        "blip" | "blip-2" | "blip2" => "blip2-opt-2-7b",
        other => other,
    };

    // Try loading the registry (don't fail if unavailable — fall back gracefully)
    let index = match RegistryIndex::load_or_fetch().await {
        Ok(idx) => idx,
        Err(_) => return None,
    };

    // Find the manifest and extract huggingface_repo
    if let Some(manifest) = index.find(registry_id)
        && manifest.asset_type == AssetType::VisionLanguage
    {
        return manifest.huggingface_repo.clone();
    }

    // Fallback: hardcoded repo IDs for backwards compat
    match model_name {
        "florence-2" | "florence2" | "florence" => Some("microsoft/Florence-2-large".to_string()),
        "blip" | "blip-2" | "blip2" => Some("Salesforce/blip2-opt-2.7b".to_string()),
        _ => None,
    }
}

/// Resolve a name or path to a dataset directory.
fn resolve_dataset_path(name_or_path: &str) -> PathBuf {
    dataset::resolve_path(name_or_path)
}

// ---------------------------------------------------------------------------
// Caption
// ---------------------------------------------------------------------------

async fn run_caption(name_or_path: &str, model: &str, overwrite: bool) -> Result<()> {
    let path = resolve_dataset_path(name_or_path);

    let info = dataset::validate(&path)
        .with_context(|| format!("Could not load dataset '{name_or_path}'"))?;

    let uncaptioned = if overwrite {
        info.image_count
    } else {
        info.image_count - info.captioned_count
    };

    if uncaptioned == 0 {
        println!(
            "{} All {} images already have captions. Nothing to do.",
            style("✓").green().bold(),
            info.image_count
        );
        println!(
            "  Use {} to regenerate existing captions.",
            style("--overwrite").bold()
        );
        return Ok(());
    }

    // Show which VL model will be downloaded
    let model_id = match model {
        "florence-2" => "microsoft/Florence-2-large (~1.5GB)",
        "blip" => "Salesforce/blip2-opt-2.7b (~6GB)",
        _ => model,
    };
    println!(
        "{} Will use {} for captioning",
        style("ℹ").dim(),
        style(model_id).bold()
    );

    // Resolve the VL model from the registry to get the canonical HuggingFace repo ID.
    // This makes the registry the source of truth for model identity.
    let resolved_model_path = resolve_vl_model(model).await;
    if let Some(ref repo) = resolved_model_path {
        println!(
            "{} Resolved from registry: {}",
            style("ℹ").dim(),
            style(repo).bold()
        );
    }

    println!(
        "{} Captioning {} / {} images in '{}' using {}",
        style("→").cyan(),
        uncaptioned,
        info.image_count,
        style(&info.name).bold(),
        style(model).bold(),
    );

    let spec = CaptionJobSpec {
        dataset_path: path.to_string_lossy().to_string(),
        model: model.to_string(),
        overwrite,
        model_path: resolved_model_path,
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize caption spec")?;

    let result = spawn_dataset_worker("caption", &yaml, uncaptioned, "captioning...").await?;

    if !result.output_lines.is_empty() {
        println!("\n{}", style("Generated captions:").bold().underlined());
        for (filename, caption) in &result.output_lines {
            println!("  {} {}", style(format!("{filename}:")).cyan(), caption);
        }
    }

    if let Ok(updated) = dataset::scan(&path) {
        println!();
        print_dataset_summary(&updated);
    }

    if !result.success {
        anyhow::bail!("Caption worker exited with errors");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Shared: spawn a Python worker command and stream events
// ---------------------------------------------------------------------------

/// Result of a worker run — collected output lines and whether it succeeded.
struct WorkerResult {
    output_lines: Vec<(String, String)>, // (filename, text)
    success: bool,
}

/// Spawn a Python worker for a dataset operation and stream progress.
///
/// `worker_command` is the subcommand name (e.g. "caption", "tag", "resize").
/// `spec_yaml` is the serialized job spec.
/// `total_items` is the expected number of items to process (for the progress bar).
/// `stage_verb` is displayed in the progress bar (e.g. "captioning...", "tagging...").
async fn spawn_dataset_worker(
    worker_command: &str,
    spec_yaml: &str,
    total_items: u32,
    stage_verb: &str,
) -> Result<WorkerResult> {
    use std::process::{Command, Stdio};
    use std::sync::mpsc;
    use std::thread;

    use crate::core::executor::read_worker_stdout;
    use crate::core::job::EventPayload;
    use crate::core::runtime;
    use crate::core::training::resolve_worker_python_root;

    let runtime_root = dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".modl")
        .join("runtime");
    let jobs_dir = runtime_root.join("jobs");
    std::fs::create_dir_all(&jobs_dir)
        .with_context(|| format!("Failed to create jobs dir: {}", jobs_dir.display()))?;

    let job_id = format!(
        "{}-{}",
        worker_command,
        chrono::Utc::now().format("%Y%m%d-%H%M%S")
    );
    let spec_path = jobs_dir.join(format!("{job_id}.yaml"));
    std::fs::write(&spec_path, spec_yaml)
        .with_context(|| format!("Failed to write spec: {}", spec_path.display()))?;

    // Resolve Python
    let setup = runtime::setup_training(false).await?;
    if !setup.ready {
        anyhow::bail!("Python runtime is not ready. Run `modl train setup` first.");
    }

    let worker_root = resolve_worker_python_root()?;
    let mut py_path = worker_root.to_string_lossy().to_string();
    if let Ok(Some(aitk_dir)) = runtime::aitoolkit_path() {
        py_path = format!("{}:{}", py_path, aitk_dir.display());
    }
    if let Ok(current) = std::env::var("PYTHONPATH")
        && !current.trim().is_empty()
    {
        py_path = format!("{}:{}", py_path, current);
    }

    let mut child = Command::new(&setup.python_path)
        .arg("-m")
        .arg("modl_worker.main")
        .arg(worker_command)
        .arg("--config")
        .arg(&spec_path)
        .arg("--job-id")
        .arg(&job_id)
        .env("PYTHONPATH", py_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| {
            format!(
                "Failed to start {} worker using {}",
                worker_command,
                setup.python_path.display()
            )
        })?;

    let stdout = child
        .stdout
        .take()
        .context("Failed to capture worker stdout")?;

    let (tx, rx) = mpsc::channel();
    let job_id_clone = job_id.clone();
    thread::spawn(move || {
        read_worker_stdout(stdout, &job_id_clone, tx);
    });

    let pb = ProgressBar::new(total_items as u64);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} [{bar:30.cyan/dim}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("━╸─"),
    );
    pb.set_message("loading model...");

    let mut output_lines: Vec<(String, String)> = Vec::new();
    let mut had_error = false;

    for event in rx {
        match event.event {
            EventPayload::Progress {
                step, total_steps, ..
            } => {
                if total_steps > 0 {
                    pb.set_length(total_steps as u64);
                }
                pb.set_position(step as u64);
                pb.set_message(stage_verb.to_string());
            }
            EventPayload::Log { message, .. } => {
                // Parse structured log lines: "[N/M] filename.ext (Xs): output text"
                if let Some(start) = message.find("] ") {
                    let rest = &message[start + 2..];
                    if let Some(colon_pos) = rest.find("): ") {
                        let filename_part = &rest[..colon_pos + 1];
                        let text = &rest[colon_pos + 3..];
                        let filename = filename_part.split(" (").next().unwrap_or(filename_part);
                        output_lines.push((filename.to_string(), text.to_string()));
                        pb.set_message(filename.to_string());
                    }
                } else if message.contains("Loading")
                    || message.contains("loaded")
                    || message.contains("Downloading")
                    || message.contains("Resizing")
                {
                    pb.set_message(message.clone());
                }
            }
            EventPayload::Artifact { path, .. } => {
                pb.println(format!("  {} {}", style("✓").green(), style(&path).dim()));
            }
            EventPayload::Warning { message, .. } => {
                pb.println(format!("  {} {}", style("⚠").yellow(), message));
            }
            EventPayload::Error {
                message,
                recoverable,
                ..
            } => {
                pb.println(format!("  {} {}", style("✗").red(), message));
                if !recoverable {
                    had_error = true;
                }
            }
            EventPayload::Completed { message } => {
                pb.finish_and_clear();
                let msg = message.unwrap_or_else(|| format!("{} completed", worker_command));
                println!("{} {}", style("✓").green().bold(), msg);
            }
            _ => {}
        }
    }

    let status = child.wait().context("Failed to wait for worker")?;

    Ok(WorkerResult {
        output_lines,
        success: status.success() && !had_error,
    })
}

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------

async fn run_resize(name_or_path: &str, resolution: u32, method: &str) -> Result<()> {
    let path = resolve_dataset_path(name_or_path);

    let info = dataset::validate(&path)
        .with_context(|| format!("Could not load dataset '{name_or_path}'"))?;

    println!(
        "{} Resizing {} images in '{}' to {}px ({})",
        style("→").cyan(),
        info.image_count,
        style(&info.name).bold(),
        resolution,
        method,
    );

    let spec = ResizeJobSpec {
        dataset_path: path.to_string_lossy().to_string(),
        resolution,
        method: method.to_string(),
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize resize spec")?;

    let result = spawn_dataset_worker("resize", &yaml, info.image_count, "resizing...").await?;

    if let Ok(updated) = dataset::scan(&path) {
        println!();
        print_dataset_summary(&updated);
    }

    if !result.success {
        anyhow::bail!("Resize worker exited with errors");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tag
// ---------------------------------------------------------------------------

async fn run_tag(name_or_path: &str, model: &str, overwrite: bool) -> Result<()> {
    let path = resolve_dataset_path(name_or_path);

    let info = dataset::validate(&path)
        .with_context(|| format!("Could not load dataset '{name_or_path}'"))?;

    let to_tag = if overwrite {
        info.image_count
    } else {
        info.image_count - info.captioned_count
    };

    if to_tag == 0 {
        println!(
            "{} All {} images already have tags/captions. Use {} to regenerate.",
            style("✓").green().bold(),
            info.image_count,
            style("--overwrite").bold()
        );
        return Ok(());
    }

    // Show which VL model will be downloaded
    let model_id = match model {
        "florence-2" => "microsoft/Florence-2-large (~1.5GB)",
        "wd-tagger" => "SmilingWolf/wd-swinv2-tagger-v3 (~400MB)",
        _ => model,
    };
    println!(
        "{} Will use {} for tagging",
        style("ℹ").dim(),
        style(model_id).bold()
    );

    // Resolve VL model from registry
    let resolved_model_path = resolve_vl_model(model).await;

    println!(
        "{} Tagging {} / {} images in '{}' using {}",
        style("→").cyan(),
        to_tag,
        info.image_count,
        style(&info.name).bold(),
        style(model).bold(),
    );

    let spec = TagJobSpec {
        dataset_path: path.to_string_lossy().to_string(),
        model: model.to_string(),
        overwrite,
        model_path: resolved_model_path,
    };
    let yaml = serde_yaml::to_string(&spec).context("Failed to serialize tag spec")?;

    let result = spawn_dataset_worker("tag", &yaml, to_tag, "tagging...").await?;

    if !result.output_lines.is_empty() {
        println!("\n{}", style("Generated tags:").bold().underlined());
        for (filename, tags) in &result.output_lines {
            println!("  {} {}", style(format!("{filename}:")).cyan(), tags);
        }
    }

    if let Ok(updated) = dataset::scan(&path) {
        println!();
        print_dataset_summary(&updated);
    }

    if !result.success {
        anyhow::bail!("Tag worker exited with errors");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Prepare (full pipeline)
// ---------------------------------------------------------------------------

async fn run_prepare(
    name: &str,
    from: &str,
    resolution: u32,
    model: &str,
    no_resize: bool,
    no_tag: bool,
    no_caption: bool,
) -> Result<()> {
    // Step 1: Create
    println!(
        "\n{} {} Creating dataset",
        style("Step 1/4").bold().cyan(),
        style("→").cyan()
    );
    let info = run_create(name, from).await?;
    drop(info);

    // Step 2: Resize
    if !no_resize {
        println!(
            "\n{} {} Resizing images",
            style("Step 2/4").bold().cyan(),
            style("→").cyan()
        );
        run_resize(name, resolution, "contain").await?;
    } else {
        println!(
            "\n{} {} Skipping resize",
            style("Step 2/4").dim(),
            style("⏭").dim()
        );
    }

    // Step 3: Tag
    if !no_tag {
        println!(
            "\n{} {} Auto-tagging images",
            style("Step 3/4").bold().cyan(),
            style("→").cyan()
        );
        run_tag(name, model, true).await?;
    } else {
        println!(
            "\n{} {} Skipping tagging",
            style("Step 3/4").dim(),
            style("⏭").dim()
        );
    }

    // Step 4: Caption
    if !no_caption {
        println!(
            "\n{} {} Auto-captioning images",
            style("Step 4/4").bold().cyan(),
            style("→").cyan()
        );
        run_caption(name, model, true).await?;
    } else {
        println!(
            "\n{} {} Skipping captioning",
            style("Step 4/4").dim(),
            style("⏭").dim()
        );
    }

    // Final summary
    let final_path = resolve_dataset_path(name);
    if let Ok(final_info) = dataset::scan(&final_path) {
        println!(
            "\n{} Dataset '{}' ready for training!",
            style("✓").green().bold(),
            style(name).bold()
        );
        print_dataset_summary(&final_info);
        println!(
            "\n  Next: {} to train a LoRA",
            style(format!(
                "modl train --dataset {name} --base flux-dev --trigger <word>"
            ))
            .bold()
        );
    }

    Ok(())
}
