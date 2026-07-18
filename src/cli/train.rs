use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{IsTerminal, Write};
use std::path::PathBuf;

use crate::core::artifacts;
use crate::core::cloud::{CloudExecutor, CloudProvider};
use crate::core::dataset;
use crate::core::db::Database;
use crate::core::executor::{Executor, LocalExecutor};
use crate::core::gpu;
use crate::core::gpu_session;
use crate::core::job::*;
use crate::core::model_family;
use crate::core::preflight;
use crate::core::presets::{self, DatasetStats, GpuContext};
use crate::core::remote_executor::RemoteExecutor;
use crate::core::run_manifest;

use super::model_resolution::resolve_cloud_provider;

/// CLI overrides that take precedence over preset-resolved values.
pub struct TrainOverrides {
    pub steps: Option<u32>,
    pub rank: Option<u32>,
    pub lr: Option<f64>,
    pub batch_size: Option<u32>,
    pub resolution: Option<u32>,
    pub optimizer: Option<Optimizer>,
    pub seed: Option<u64>,
    pub repeats: Option<u32>,
    pub caption_dropout: Option<f64>,
    pub class_word: Option<String>,
    pub resume: Option<String>,
    pub sample_every: Option<u32>,
}

/// Options for BYO-pod training (`modl train --pod`).
pub struct PodArgs {
    pub gpu_type: String,
    pub max_price: f64,
    pub keep_pod: bool,
    pub yes: bool,
    /// Ignore any active pod and provision a fresh one-shot instance.
    pub fresh: bool,
}

impl PodArgs {
    fn into_options(self, label: String, disk_gb: f64) -> crate::core::pod::PodOptions {
        crate::core::pod::PodOptions {
            gpu_type: self.gpu_type,
            max_price_per_hour: self.max_price,
            disk_gb,
            yes: self.yes,
            keep_pod: self.keep_pod,
            label,
            confirm_rent: Some(super::pod::dialoguer_confirm()),
        }
    }
}

/// Run a training spec on a pod: reuse the active pod if one is up (unless
/// `--fresh`), otherwise one-shot provision → train → destroy.
async fn run_pod(spec: TrainJobSpec, args: PodArgs) -> Result<()> {
    use crate::core::{pod, pod_state};

    if spec.params.resume_from.is_some() {
        anyhow::bail!(
            "--resume isn't supported with --pod yet — the checkpoint path is local to this \
             machine and won't exist on the pod. Run locally, or drop --resume."
        );
    }

    if !args.fresh
        && let Some(rec) = pod_state::active_pod().await?
    {
        println!(
            "{} Reusing pod {} ({}, ${:.3}/hr)…",
            style("→").cyan(),
            rec.instance_id,
            rec.gpu_name,
            rec.dph_all_in()
        );

        // VRAM guard: if we can estimate the pod's VRAM and it's below the
        // model's training floor, refuse — the alternative is a slow OOM crash.
        if let (Some(pod_vram), Some(floor)) = (
            estimate_vram_gb(&rec.gpu_name),
            pod::min_vram_for_train(&spec),
        ) && pod_vram < floor
        {
            anyhow::bail!(
                "Active pod {} is a {} (~{}GB VRAM) but {} needs ≥{}GB to train.\n\
                 Replace it: modl pod rm {} — then rerun (a fresh pod will be sized correctly).",
                rec.instance_id,
                rec.gpu_name,
                pod_vram,
                spec.model.base_model_id,
                floor,
                rec.instance_id
            );
        }

        // GPU-type mismatch is a warning, not an error — the user may
        // deliberately want to reuse whatever is warm.
        if !args.gpu_type.eq_ignore_ascii_case("auto")
            && !gpu_type_matches(&rec.gpu_name, &args.gpu_type)
        {
            println!(
                "  {} you asked for {} but the active pod is {} — using the pod anyway ({} to replace).",
                style("⚠").yellow(),
                args.gpu_type,
                rec.gpu_name,
                style("--fresh").bold()
            );
        }

        let pod_obj = pod::Pod::from(rec.clone());
        pod::bootstrap(&pod_obj)?;
        pod::run_train_job(&pod_obj, &spec)?;
        println!(
            "{} Pod {} still running (${:.3}/hr) — {} when done.",
            style("⚠").yellow(),
            rec.instance_id,
            rec.dph_all_in(),
            style(format!("modl pod rm {}", rec.instance_id)).bold()
        );
        return Ok(());
    }

    // No active pod (or --fresh): today's one-shot behavior.
    let label = format!("modl-pod-{}", spec.output.lora_name);
    let disk_gb = pod::disk_gb_for_train(&spec);
    pod::run_pod_training(spec, args.into_options(label, disk_gb)).await
}

/// Human label for a LoRA type, used in guidance messages.
fn lora_type_label(t: LoraType) -> &'static str {
    match t {
        LoraType::Character => "character",
        LoraType::Object => "object",
        LoraType::Style => "style",
    }
}

/// Is the trigger a common word that carries a strong text-encoder prior
/// (so the LoRA binds to it only weakly)? Rather than guess whether an
/// arbitrary string is a real word — which false-flags invented rare tokens
/// like "mxpom" — this checks a curated set of high-risk everyday words
/// people reach for as triggers. Non-exhaustive by design: it only needs to
/// catch the obvious foot-guns (the maxi→maxi-dress case that motivated it).
fn is_common_word_trigger(trigger: &str) -> bool {
    const HIGH_RISK: &[&str] = &[
        // fashion / garments (the maxi-dress collision and friends)
        "maxi", "mini", "midi", "boho", "denim", "linen", // colours
        "red", "blue", "green", "amber", "coral", "ivory", "jade", "ruby", "rose",
        // common nouns that dominate as concepts
        "dog", "cat", "man", "woman", "girl", "boy", "car", "house", "tree", "star",
        // frequent short given names with strong priors
        "max", "leo", "luna", "bella", "coco", "milo", "lily", "ruby", "jack", "rose",
    ];
    let t = trigger.trim().to_lowercase();
    HIGH_RISK.contains(&t.as_str())
}

/// Is the class word a generic category that anchors identity but not size?
/// The class token pins scale/structure, so "dog"/"car" leave size loose while
/// the specific breed/model ("pomeranian", "911") locks it. Small curated set —
/// only the common umbrella terms people default to.
fn is_generic_class_word(class_word: &str) -> bool {
    const GENERIC: &[&str] = &[
        "dog", "cat", "bird", "horse", "animal", "pet", "car", "vehicle", "truck", "person", "man",
        "woman", "human", "thing", "object", "building", "flower", "plant",
    ];
    let c = class_word.trim().to_lowercase();
    GENERIC.contains(&c.as_str())
}

/// Best-effort VRAM (GB) for a Vast GPU name. `None` when the card is unknown
/// (we then skip the reuse VRAM guard rather than guess wrong).
fn estimate_vram_gb(gpu_name: &str) -> Option<u32> {
    let n = gpu_name.to_uppercase();
    if n.contains("H200") {
        Some(141)
    } else if n.contains("H100") {
        Some(80)
    } else if n.contains("A100") {
        if n.contains("80") { Some(80) } else { Some(40) }
    } else if n.contains("A6000") || n.contains("6000 ADA") || n.contains("L40") {
        Some(48)
    } else if n.contains("5090") {
        Some(32)
    } else if n.contains("4090") || n.contains("3090") || n.contains("A10") {
        Some(24)
    } else if n.contains("4080") {
        // 4080 and 4080 SUPER are 16GB cards — grouping them with the 24GB
        // tier let sub-floor pods pass the guard and OOM mid-train.
        Some(16)
    } else {
        None
    }
}

/// Loose match between a Vast GPU name and a requested `--pod` type.
fn gpu_type_matches(pod_gpu_name: &str, requested: &str) -> bool {
    let norm = |s: &str| s.to_uppercase().replace([' ', '_', '-'], "");
    norm(pod_gpu_name).contains(&norm(requested))
}

/// Run the train command. Arguments are all optional; missing ones trigger
/// interactive prompts (except when --config is given).
#[allow(clippy::too_many_arguments)]
pub async fn run(
    dataset_arg: Option<&str>,
    base: &str,
    name: Option<&str>,
    trigger: Option<&str>,
    lora_type: LoraType,
    preset_arg: Option<Preset>,
    overrides: TrainOverrides,
    config: Option<&str>,
    dry_run: bool,
    cloud: bool,
    provider: Option<CloudProvider>,
    attach_gpu: bool,
    gpu_type: &str,
    pod: Option<PodArgs>,
) -> Result<()> {
    // -------------------------------------------------------------------
    // Fast path: --config <yaml> loads a full spec directly
    // -------------------------------------------------------------------
    if let Some(config_path) = config {
        let yaml = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config: {config_path}"))?;
        let mut spec: TrainJobSpec =
            serde_yaml::from_str(&yaml).context("Failed to parse TrainJobSpec YAML")?;

        // Respect --cloud / --attach-gpu flag even when loading spec from file
        if attach_gpu {
            spec.target = ExecutionTarget::Remote;
        } else if cloud {
            spec.target = ExecutionTarget::Cloud;
        }

        if dry_run {
            println!("{}", serde_yaml::to_string(&spec)?);
            return Ok(());
        }

        if let Some(pod_args) = pod {
            return run_pod(spec, pod_args).await;
        }
        return execute_training(spec, cloud, provider, attach_gpu, gpu_type).await;
    }

    // -------------------------------------------------------------------
    // Resolve dataset
    // -------------------------------------------------------------------
    let dataset_path = match dataset_arg {
        Some(d) => dataset::resolve_path(d),
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
    // Base model (required CLI arg)
    // -------------------------------------------------------------------
    let base_model = base.to_string();

    // -------------------------------------------------------------------
    // Resolve trigger word
    // -------------------------------------------------------------------
    let trigger_word = match trigger {
        Some(t) => t.to_string(),
        None => dialoguer::Input::<String>::new()
            .with_prompt("Trigger word (single word, no spaces)")
            .default("OHWX".to_string())
            .interact_text()
            .context("Trigger word input cancelled")?,
    };

    // Validate and normalize trigger word
    let trigger_word = {
        let tw = trigger_word.trim().to_string();
        if tw.contains(' ') {
            let suggested = tw.replace(' ', "").to_uppercase();
            println!(
                "  {} Trigger word should be a single word with no spaces.",
                console::style("!").yellow()
            );
            println!(
                "    Multi-word triggers compete with existing vocabulary and cause poor fitting."
            );
            println!("    Suggested: {}", console::style(&suggested).bold());
            let use_suggested = dialoguer::Confirm::new()
                .with_prompt(format!("Use '{}' instead?", suggested))
                .default(true)
                .interact()
                .unwrap_or(true);
            if use_suggested {
                println!(
                    "    Using trigger word: {}",
                    console::style(&suggested).bold()
                );
                suggested
            } else {
                tw
            }
        } else {
            tw
        }
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

    // LoRA type (required CLI arg)
    // -------------------------------------------------------------------

    // -------------------------------------------------------------------
    // Resolve preset
    // -------------------------------------------------------------------
    let preset = match preset_arg {
        Some(p) => p,
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
        lora_type,
        &ds_stats,
        gpu_ctx.as_ref(),
        &base_model,
        &trigger_word,
    )
    .context("Failed to resolve training preset for base model")?;

    // -----------------------------------------------------------------
    // Apply CLI overrides (take precedence over preset values)
    // -----------------------------------------------------------------
    if let Some(s) = overrides.steps {
        params.steps = s;
    }
    if let Some(r) = overrides.rank {
        params.rank = r;
    }
    if let Some(lr) = overrides.lr {
        params.learning_rate = lr;
    }
    if let Some(bs) = overrides.batch_size {
        params.batch_size = bs; // 0 = let adapter decide per lora_type
    }
    if let Some(res) = overrides.resolution {
        params.resolution = res;
    }
    if let Some(opt) = overrides.optimizer {
        params.optimizer = opt;
    }
    if let Some(seed) = overrides.seed {
        params.seed = Some(seed);
    }
    if let Some(rep) = overrides.repeats {
        params.num_repeats = rep; // 0 = let adapter decide per lora_type
    }
    if let Some(cd) = overrides.caption_dropout {
        params.caption_dropout_rate = cd; // -1.0 = let adapter decide
    }
    if overrides.class_word.is_some() {
        params.class_word = overrides.class_word.clone();
    }

    // -------------------------------------------------------------------
    // Recipe guardrails for character/object LoRAs. Evidence (maxi Pomeranian
    // runs on krea-2-raw): (1) a bare, common-word trigger with no class word
    // collapses to the word's prior — samples never converge; (2) the class
    // token anchors SIZE/structure, so a generic class ("dog") locks colour
    // and markings but leaves size loose — the specific breed ("pomeranian")
    // pins it. Non-blocking hints.
    // -------------------------------------------------------------------
    if matches!(lora_type, LoraType::Character | LoraType::Object) {
        match params.class_word.as_deref() {
            None => eprintln!(
                "  {} No --class-word set. For {} LoRAs, anchoring the trigger to a category \
                 markedly improves convergence and makes sample prompts read '{} <class>' \
                 instead of a bare '{}' that can collapse to the word's prior. Use the MOST \
                 specific class that fits (the breed/model, e.g. --class-word pomeranian, not \
                 just 'dog') — the class token also anchors size and structure.",
                style("!").yellow(),
                lora_type_label(lora_type),
                trigger_word,
                trigger_word,
            ),
            Some(cw) if is_generic_class_word(cw) => eprintln!(
                "  {} --class-word '{}' is generic. It anchors colour and markings but leaves \
                 size/structure loose (a '{} {}' can render at the wrong size). Prefer the \
                 specific breed/model (e.g. 'pomeranian' over 'dog') to pin size too.",
                style("!").yellow(),
                cw,
                trigger_word,
                cw,
            ),
            Some(_) => {}
        }
        if is_common_word_trigger(&trigger_word) {
            eprintln!(
                "  {} Trigger '{}' is a common word — it competes with a strong existing \
                 prior in the text encoder, so the identity binds weakly to it. Prefer a rare \
                 token (e.g. an invented string) and/or pair it with --class-word.",
                style("!").yellow(),
                trigger_word,
            );
        }
    }
    if overrides.sample_every.is_some() {
        params.sample_every = overrides.sample_every;
    } else if cloud || attach_gpu {
        // Cloud/remote: skip intermediate sampling by default (saves ~10 min per round)
        params.sample_every = Some(0);
    }
    if overrides.resume.is_some() {
        params.resume_from = overrides.resume.clone();

        // When resuming, inherit steps from the original run's config so the
        // preset doesn't recompute a different value.
        if overrides.steps.is_none()
            && let Some(ref ckpt_path) = overrides.resume
            && let Some(original_steps) = read_original_steps(ckpt_path)
        {
            params.steps = original_steps;
        }
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
    let output_dir = crate::core::paths::modl_root()
        .join("training_output")
        .join(&lora_name);

    // Guard against overwriting an existing training run (skip when resuming)
    if params.resume_from.is_none() && output_dir.exists() {
        let has_safetensors = std::fs::read_dir(&output_dir)?
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"));
        if has_safetensors {
            println!(
                "{} A training run named '{}' already exists at {}",
                style("✗").red().bold(),
                style(&lora_name).bold(),
                output_dir.display()
            );
            println!(
                "  Use {} for a different name, or delete the existing run first.",
                style("--name <new-name>").bold()
            );
            anyhow::bail!("Training run '{}' already exists", lora_name);
        }
    }

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
            base_model_path: {
                // Resolve the base model to its actual store path
                let db = Database::open()?;
                db.find_installed(&base_model)?.map(|m| m.store_path)
            },
            arch_key: model_family::find_model(&base_model).map(|m| m.arch_key.to_string()),
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
    // Dry run: print spec and exit
    // -------------------------------------------------------------------
    if dry_run {
        println!("{} Dry run — generated spec:", style("✓").green().bold());
        println!("{}", serde_yaml::to_string(&spec)?);
        return Ok(());
    }

    if let Some(pod_args) = pod {
        return run_pod(spec, pod_args).await;
    }

    execute_training(spec, cloud, provider, attach_gpu, gpu_type).await
}

/// Execute training: persist job, run executor, collect artifacts.
async fn execute_training(
    spec: TrainJobSpec,
    cloud: bool,
    provider: Option<CloudProvider>,
    attach_gpu: bool,
    gpu_type: &str,
) -> Result<()> {
    // -------------------------------------------------------------------
    // 0. Pre-flight checks (fail fast with actionable hints)
    // -------------------------------------------------------------------
    if !cloud && !attach_gpu {
        preflight::for_training(&spec.model.base_model_id)?;
    }

    let db = Database::open()?;

    let spec_json = serde_json::to_string(&spec)?;
    let target_str = serde_json::to_string(&spec.target)?;

    // -------------------------------------------------------------------
    // 1. Bootstrap executor
    // -------------------------------------------------------------------
    let mut executor: Box<dyn Executor> = if attach_gpu {
        println!(
            "{} Connecting to remote GPU ({})...",
            style("→").cyan(),
            style(gpu_type).bold()
        );
        let session = gpu_session::ensure_session(
            gpu_type,
            "2h", // training sessions get a longer idle timeout
            std::slice::from_ref(&spec.model.base_model_id),
        )
        .await?;
        println!(
            "  {} Session {} ({})",
            style("✓").green(),
            style(&session.session_id).bold(),
            session.state,
        );
        Box::new(RemoteExecutor::new(session))
    } else if cloud {
        let cloud_provider = resolve_cloud_provider(provider);
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
    // 2. Clean up stale jobs + submit
    // -------------------------------------------------------------------
    // Mark any previous "running"/"queued" jobs for this LoRA as errored
    // (they crashed without updating the DB).
    if let Ok(old_jobs) = db.find_jobs_by_lora_name(&spec.output.lora_name) {
        for j in &old_jobs {
            if j.status == "running" || j.status == "queued" {
                let _ = db.update_job_status(&j.job_id, "error");
            }
        }
    }

    let handle = executor.submit(&spec)?;
    let job_id = &handle.job_id;

    db.insert_job(
        job_id,
        "train",
        "queued",
        &spec_json,
        target_str.trim_matches('"'),
        None,
        None,
    )?;

    println!(
        "{} Training started — {}",
        style("→").cyan(),
        style(job_id).dim()
    );

    if let Err(e) = run_manifest::refresh_manifest_for_spec(&spec, Some(job_id), "running") {
        eprintln!("{} Could not write run manifest: {e}", style("⚠").yellow());
    }

    // -------------------------------------------------------------------
    // 3. Event loop with progress bar
    // -------------------------------------------------------------------
    let rx = executor.events(job_id)?;
    db.update_job_status(job_id, "running")?;

    // Open a log file for the preview server to parse live progress.
    // training_status.rs reads ~/.modl/training_output/<name>.log
    let log_path = {
        let training_output = crate::core::paths::modl_root().join("training_output");
        training_output.join(format!("{}.log", spec.output.lora_name))
    };
    let mut log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .ok();

    let is_tty = std::io::stderr().is_terminal();
    let pb = if is_tty {
        let pb = ProgressBar::new(spec.params.steps as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} steps {msg}",
            )?
            .progress_chars("█▓░"),
        );
        pb.set_message("preparing...");
        pb
    } else {
        ProgressBar::hidden()
    };
    let mut got_first_step = false;
    let mut last_printed_step: u64 = 0;
    let print_interval: u64 = std::cmp::max(spec.params.steps as u64 / 20, 10);

    let mut artifact_paths: Vec<String> = Vec::new();
    let mut final_status = "completed";
    let mut recent_logs: Vec<String> = Vec::new();
    let max_recent = 20;

    for event in rx {
        match &event.event {
            EventPayload::Progress {
                stage,
                step,
                total_steps,
                loss,
                ..
            } => {
                if stage == "sample" {
                    // Sample generation — show separate message, don't
                    // overwrite training progress bar position.
                    pb.set_message(format!("generating samples ({}/{})", step, total_steps));
                    if !is_tty {
                        eprintln!("  generating samples ({step}/{total_steps})");
                    }
                } else {
                    if !got_first_step {
                        got_first_step = true;
                        pb.set_message("".to_string());
                    }
                    pb.set_length(*total_steps as u64);
                    pb.set_position(*step as u64);
                    if let Some(l) = loss {
                        pb.set_message(format!("loss: {l:.4}"));
                    }
                    // Non-TTY: print periodic progress lines
                    if !is_tty && (*step as u64) >= last_printed_step + print_interval {
                        last_printed_step = *step as u64;
                        let loss_str = loss.map(|l| format!(" loss: {l:.4}")).unwrap_or_default();
                        eprintln!("  {step}/{total_steps} steps{loss_str}");
                    }

                    // Write tqdm-style line to log file for the preview server
                    if let Some(ref mut f) = log_file {
                        let pct = if *total_steps > 0 {
                            (*step as f32 / *total_steps as f32) * 100.0
                        } else {
                            0.0
                        };
                        let loss_str = loss.map(|l| format!(" loss: {l:.3e}")).unwrap_or_default();
                        let _ = write!(
                            f,
                            "\r{name}: {pct:5.1}%| {step}/{total} [00:00<00:00,{loss_str}]",
                            name = spec.output.lora_name,
                            step = step,
                            total = total_steps,
                        );
                        let _ = f.flush();
                    }
                }
            }
            EventPayload::Artifact { path, .. } => {
                artifact_paths.push(path.clone());
            }
            EventPayload::Completed { message } => {
                let msg = message.as_deref().unwrap_or("done");
                pb.finish_with_message(msg.to_string());
                if !is_tty {
                    eprintln!("✓ Training {msg}");
                }
                break;
            }
            EventPayload::Error {
                code,
                message,
                details,
                ..
            } => {
                pb.abandon_with_message(format!("error: {code}"));
                eprintln!();
                eprintln!("{} Training failed: {message}", style("✗").red().bold());

                // Show the output tail from the error details if available
                if let Some(details_val) = details
                    && let Some(tail) = details_val.get("output_tail").and_then(|v| v.as_str())
                {
                    eprintln!();
                    eprintln!("{}", style("─── ai-toolkit output ───").dim());
                    for line in tail.lines().take(20) {
                        eprintln!("  {}", style(line).dim());
                    }
                    eprintln!("{}", style("─────────────────────────").dim());
                }
                final_status = "error";
                break;
            }
            EventPayload::Log { message, level } => {
                // Keep a rolling buffer of recent log lines for context
                recent_logs.push(message.clone());
                if recent_logs.len() > max_recent {
                    recent_logs.remove(0);
                }

                // Append to log file for preview server
                if let Some(ref mut f) = log_file {
                    let _ = writeln!(f, "{message}");
                }

                match level.as_str() {
                    "status" => {
                        // Important status updates: show prominently
                        pb.println(format!("  {} {}", style("→").cyan(), message));
                    }
                    "stderr" => {
                        // Worker stderr lines — show as warnings
                        pb.println(format!(
                            "  {} {}",
                            style("[stderr]").red().dim(),
                            style(message).dim()
                        ));
                    }
                    "info" => {
                        // Verbose info — only show if it looks important
                    }
                    _ => {}
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
            EventPayload::Result {
                result_type, data, ..
            } => {
                if result_type == "hub_registered"
                    && let Some(href) = data.get("hub_ref").and_then(|v| v.as_str())
                {
                    pb.println(format!(
                        "  {} Published to hub.modl.run/{}",
                        style("✓").green(),
                        href
                    ));
                    // Store hub ref for later pull
                    artifact_paths.push(format!("hub://{href}"));
                }
            }
        }

        // Persist event to DB
        let event_json = serde_json::to_string(&event).unwrap_or_default();
        let _ = db.insert_job_event(job_id, event.sequence, &event_json);
    }

    // If the event stream ended without a Completed or Error event,
    // it means the worker process crashed without emitting a structured error.
    if final_status == "completed" && artifact_paths.is_empty() {
        // Check if we just never got a completion event
        if !pb.is_finished() {
            pb.abandon_with_message("process exited unexpectedly".to_string());
            eprintln!();
            eprintln!(
                "{} Training process exited without reporting completion.",
                style("✗").red().bold()
            );
            if !recent_logs.is_empty() {
                eprintln!();
                eprintln!("{}", style("─── last output lines ───").dim());
                for line in recent_logs.iter().rev().take(10).rev() {
                    eprintln!("  {}", style(line).dim());
                }
                eprintln!("{}", style("─────────────────────────").dim());
            }
            final_status = "error";
        }
    }

    // -------------------------------------------------------------------
    // 4. Update job status
    // -------------------------------------------------------------------
    db.update_job_status(job_id, final_status)?;

    // -------------------------------------------------------------------
    // 5. Collect artifacts (local or cloud)
    // -------------------------------------------------------------------
    if final_status == "completed" {
        let store_root = crate::core::paths::modl_root();

        for artifact_path in &artifact_paths {
            // Hub artifacts: "hub://username/slug" — pull from hub
            if let Some(hub_ref) = artifact_path.strip_prefix("hub://") {
                println!();
                println!("{} Downloading LoRA from hub...", style("☁").cyan());
                match pull_from_hub(
                    hub_ref,
                    &spec.output.lora_name,
                    &spec.model.base_model_id,
                    &spec.params.trigger_word,
                    job_id,
                    &db,
                    &store_root,
                )
                .await
                {
                    Ok(collected) => {
                        println!("{} LoRA downloaded from cloud!", style("✓").green().bold());
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
                        println!();
                        println!(
                            "  Generate: modl generate \"a {} cat\" --lora {}",
                            spec.params.trigger_word, spec.output.lora_name
                        );
                    }
                    Err(e) => {
                        println!("{} Failed to pull from hub: {e}", style("⚠").yellow(),);
                        println!("  Pull manually: modl hub pull {hub_ref}");
                    }
                }
                continue;
            }

            let path = PathBuf::from(artifact_path);

            // Cloud artifacts come via hub:// prefix (handled above).
            // Local artifacts: collect from filesystem.
            if !path.exists() {
                // Skip non-existent paths (cloud artifacts without hub ref)
                continue;
            }
            if path.extension().is_some_and(|e| e == "safetensors") {
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

    if let Err(e) = run_manifest::refresh_manifest_for_spec(&spec, Some(job_id), final_status) {
        eprintln!(
            "{} Could not refresh run manifest at end of training: {e}",
            style("⚠").yellow()
        );
    }

    // Clean up cloud resources (destroy GPU session)
    if cloud {
        let _ = executor.cleanup();
    }

    Ok(())
}

/// Pull a LoRA from the hub and collect it locally.
async fn pull_from_hub(
    hub_ref: &str,
    lora_name: &str,
    base_model: &str,
    trigger_word: &str,
    job_id: &str,
    db: &Database,
    store_root: &std::path::Path,
) -> Result<artifacts::CollectedLora> {
    use crate::core::hub::HubClient;

    let hub = HubClient::from_config(true)?;

    let (username, slug) = hub_ref
        .split_once('/')
        .context("Invalid hub ref — expected username/slug")?;

    let pull_resp = hub.pull(username, slug, None).await?;

    // Download the LoRA file
    let bytes = reqwest::get(&pull_resp.download_url).await?.bytes().await?;

    // Write to temp, then collect
    let tmp_dir = store_root.join("tmp");
    std::fs::create_dir_all(&tmp_dir)?;
    let tmp_path = tmp_dir.join(format!("{lora_name}.safetensors"));
    std::fs::write(&tmp_path, &bytes)?;

    let collected = artifacts::collect_lora(
        &tmp_path,
        lora_name,
        base_model,
        trigger_word,
        job_id,
        db,
        store_root,
    )?;

    let _ = std::fs::remove_file(&tmp_path);
    Ok(collected)
}

/// Read the `steps` value from an existing run's config.yaml next to the
/// checkpoint file. When resuming, we want the original step budget, not
/// a freshly-computed preset value.
fn read_original_steps(checkpoint_path: &str) -> Option<u32> {
    let ckpt = std::path::Path::new(checkpoint_path);
    // Checkpoint lives at <run_dir>/<run_name>/<name>_<step>.safetensors
    // config.yaml is at <run_dir>/<run_name>/config.yaml
    let config_path = ckpt.parent()?.join("config.yaml");
    let content = std::fs::read_to_string(&config_path).ok()?;
    // Quick extraction: look for "steps: <N>" in the YAML
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("steps:") {
            let val = trimmed.trim_start_matches("steps:").trim();
            return val.parse::<u32>().ok();
        }
    }
    None
}

/// Extract the step number from a checkpoint filename like
/// `kids-art-sdxl-v2_000004800.safetensors` → `4800`.
#[allow(dead_code)]
fn step_from_checkpoint_path(path: &str) -> Option<u32> {
    let stem = std::path::Path::new(path).file_stem()?.to_str()?;
    // Pattern: <name>_<zero-padded-step>
    let step_str = stem.rsplit('_').next()?;
    step_str.parse::<u32>().ok()
}

/// Open text in $EDITOR, return edited content.
fn edit_in_editor(content: &str) -> Result<String> {
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join(format!("modl-train-{}.yaml", std::process::id()));
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

#[cfg(test)]
mod recipe_guardrail_tests {
    use super::*;

    #[test]
    fn flags_common_word_triggers() {
        assert!(is_common_word_trigger("maxi"));
        assert!(is_common_word_trigger("Red"));
        assert!(is_common_word_trigger("dog"));
        assert!(is_common_word_trigger("luna"));
    }

    #[test]
    fn does_not_flag_rare_or_invented_tokens() {
        // Invented tokens and the OHWX convention must never warn.
        assert!(!is_common_word_trigger("mxpom"));
        assert!(!is_common_word_trigger("ohwx"));
        assert!(!is_common_word_trigger("m4xi"));
        assert!(!is_common_word_trigger("sks"));
        assert!(!is_common_word_trigger("xyzzy42"));
    }

    #[test]
    fn flags_generic_class_words_but_not_specific() {
        assert!(is_generic_class_word("dog"));
        assert!(is_generic_class_word("car"));
        assert!(is_generic_class_word("Person"));
        // Specific breeds/models anchor size — not generic.
        assert!(!is_generic_class_word("pomeranian"));
        assert!(!is_generic_class_word("golden retriever"));
        assert!(!is_generic_class_word("porsche 911"));
    }
}
