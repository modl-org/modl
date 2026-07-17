use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

use crate::cli::InpaintMethod;
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

/// Known control type suffixes for auto-detection from filenames.
use crate::core::model_family::detect_control_type_from_filename;

/// Read image dimensions from a file on disk.
fn image_dimensions(path: &str) -> Result<(u32, u32)> {
    let reader = image::ImageReader::open(path)
        .with_context(|| format!("Cannot open init-image: {path}"))?
        .with_guessed_format()
        .with_context(|| format!("Cannot detect format of: {path}"))?;
    let dims = reader
        .into_dimensions()
        .with_context(|| format!("Cannot read dimensions of: {path}"))?;
    Ok(dims)
}

/// Resolve a LoRA name to its store path by looking in the DB.
fn resolve_lora(name: &str, weight: f32, db: &Database) -> Result<Option<LoraRef>> {
    let result = model_resolve::resolve_lora(name, weight, db)?;
    if result.is_none() {
        anyhow::bail!(
            "LoRA not found: {name}. Use `modl model ls --type lora` to see installed LoRAs, or provide a file path."
        );
    }
    Ok(result)
}

/// Pick the best installed generation model, falling back to "flux-schnell".
use super::model_resolution::{
    default_generation_model, resolve_cloud_provider, resolve_inpaint_model,
};

fn default_steps(base_model: &str) -> u32 {
    model_family::model_defaults(base_model).0
}

fn default_guidance(base_model: &str) -> f32 {
    model_family::model_defaults(base_model).1
}

/// All arguments for `modl generate`, used by both CLI and web UI.
pub struct GenerateArgs<'a> {
    pub prompt: &'a str,
    pub base: Option<&'a str>,
    pub lora: Option<&'a str>,
    pub lora_strength: f32,
    pub seed: Option<u64>,
    pub size: Option<&'a str>,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub count: u32,
    pub init_image: Option<&'a str>,
    pub mask: Option<&'a str>,
    pub strength: Option<f32>,
    pub inpaint: InpaintMethod,
    pub controlnet: &'a [String],
    pub cn_strength: &'a str,
    pub cn_end: &'a str,
    pub cn_type: Option<&'a str>,
    pub style_ref: &'a [String],
    pub style_strength: f32,
    pub style_type: Option<&'a str>,
    pub fast: Option<u32>,
    pub cloud: bool,
    pub provider: Option<CloudProvider>,
    pub no_worker: bool,
    pub attach_gpu: bool,
    pub gpu_type: &'a str,
    pub pod: bool,
    pub json: bool,
}

pub async fn run(args: GenerateArgs<'_>) -> Result<()> {
    // Pod execution funnels through the remote modl install — model checks
    // and the store live pod-side, so none of the local resolution below
    // applies.
    if args.pod {
        return run_on_pod(&args).await;
    }

    let db = Database::open()?;
    run_local(args, db).await
}

/// Generate on the active pod: build a one-step workflow and hand it to the
/// remote modl (`core::pod_run`). The model is pulled into the pod's store,
/// so nothing needs to be installed locally.
async fn run_on_pod(args: &GenerateArgs<'_>) -> Result<()> {
    let model = args.base.context(
        "--pod needs an explicit --base <model-id> (e.g. flux-schnell) — it's pulled into the pod's store.",
    )?;
    for (flag, set) in [("--cloud", args.cloud), ("--attach-gpu", args.attach_gpu)] {
        if set {
            anyhow::bail!("{flag} isn't supported with --pod yet — run locally or drop the flag.");
        }
    }
    // Inpaint method selection isn't a workflow field yet — only the default
    // auto behaviour ships to pods.
    if args.mask.is_some() && !matches!(args.inpaint, InpaintMethod::Auto) {
        anyhow::bail!("--inpaint isn't supported with --pod yet — run locally or drop the flag.");
    }
    // Same mode validation as local runs, against models.toml (no store needed).
    // Note: no automatic inpaint-model swap on pods — pick an inpaint-capable
    // --base explicitly; the error below lists them.
    let mode = match (args.init_image.is_some(), args.mask.is_some()) {
        (_, true) => Some("inpaint"),
        (true, false) => Some("img2img"),
        (false, false) => None,
    };
    if args.mask.is_some() && args.init_image.is_none() {
        anyhow::bail!("--mask requires --init-image (inpainting regenerates a masked region).");
    }
    if let Some(mode) = mode
        && let Err(msg) = model_family::validate_mode(model, mode)
    {
        anyhow::bail!(msg);
    }
    if args.fast.is_some() && args.lora.is_some() {
        anyhow::bail!(
            "--fast and --lora cannot be used together. \
             The --fast flag auto-applies a Lightning distillation LoRA."
        );
    }
    // Workflow `lora:` refs carry no weight — the pod applies full strength.
    if args.lora.is_some() && args.lora_strength != 1.0 {
        anyhow::bail!(
            "--lora-strength isn't supported with --pod yet — LoRAs run at full strength there."
        );
    }

    let (width, height) = match args.size {
        Some(s) => {
            let (w, h) =
                model_resolve::resolve_size_for_model(s, model_family::default_resolution(model))?;
            (Some(w), Some(h))
        }
        None => (None, None),
    };
    // count>1 maps to explicit seeds (one output per seed on the pod).
    let seeds: Vec<u64> = match (args.seed, args.count) {
        (Some(s), c) if c > 1 => (0..c as u64).map(|i| s + i).collect(),
        (Some(s), _) => vec![s],
        (None, c) if c > 1 => {
            let base = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos() as u64)
                .unwrap_or(0);
            (0..c as u64).map(|i| base + i).collect()
        }
        (None, _) => vec![],
    };

    // Build + validate ControlNet / style-ref inputs against models.toml,
    // exactly like a local run — the spec builder inlines the image files.
    let cn_inputs = build_cn_inputs(
        args.controlnet,
        args.cn_strength,
        args.cn_end,
        args.cn_type,
        model,
    )?;
    let style_inputs =
        build_style_inputs(args.style_ref, args.style_strength, args.style_type, model)?;
    for p in [args.init_image, args.mask].into_iter().flatten() {
        if !std::path::Path::new(p).exists() {
            anyhow::bail!("Image not found: {p}");
        }
    }
    let image_inputs = crate::core::pod_run::GenImageInputs {
        init_image: args.init_image.map(std::path::Path::new),
        mask: args.mask.map(std::path::Path::new),
        strength: args.strength,
        controlnet: &cn_inputs,
        style_ref: &style_inputs,
    };

    let spec = crate::core::pod_run::single_generate_spec(
        model,
        args.prompt,
        args.lora,
        args.fast,
        width,
        height,
        args.steps,
        args.guidance.map(|g| g as f64),
        &seeds,
        &image_inputs,
    )?;

    let rec = crate::core::pod_state::active_pod()
        .await?
        .context("No pod up — run `modl pod up <gpu>` first, or use --attach-gpu / --cloud.")?;
    eprintln!(
        "{} Generating on pod {} ({}, ${:.3}/hr)…",
        style("→").cyan(),
        rec.instance_id,
        rec.gpu_name,
        rec.dph_all_in()
    );
    let pod = crate::core::pod::Pod::from(rec.clone());
    let outcome =
        crate::core::pod_run::run_workflow_on_pod(&pod, &spec, None, Default::default()).await?;
    if args.json {
        println!("{}", serde_json::to_string(&pod_result_json(&outcome))?);
    }
    eprintln!(
        "{} Pod {} still running (${:.3}/hr) — {} when done.",
        style("⚠").yellow(),
        rec.instance_id,
        rec.dph_all_in(),
        style(format!("modl pod rm {}", rec.instance_id)).bold()
    );
    Ok(())
}

/// Validate and build ControlNet inputs from CLI flags. Shared between local
/// and pod execution — `effective_model` drives support validation.
fn build_cn_inputs(
    controlnet: &[String],
    cn_strength: &str,
    cn_end: &str,
    cn_type: Option<&str>,
    effective_model: &str,
) -> Result<Vec<ControlNetInput>> {
    if controlnet.is_empty() {
        return Ok(Vec::new());
    }
    if controlnet.len() > 2 {
        anyhow::bail!(
            "Maximum 2 ControlNet inputs supported. You provided {}.",
            controlnet.len()
        );
    }

    // Parse comma-separated strengths/ends
    let strengths: Vec<f32> = cn_strength
        .split(',')
        .map(|s| s.trim().parse::<f32>().unwrap_or(0.75))
        .collect();
    let ends: Vec<f32> = cn_end
        .split(',')
        .map(|s| s.trim().parse::<f32>().unwrap_or(0.8))
        .collect();

    let mut inputs = Vec::new();
    for (i, cn_path) in controlnet.iter().enumerate() {
        let path = PathBuf::from(cn_path);
        if !path.exists() {
            anyhow::bail!("ControlNet image not found: {cn_path}");
        }

        // Resolve control type: explicit flag > filename suffix > error
        let control_type = if let Some(explicit) = cn_type {
            explicit.to_string()
        } else {
            detect_control_type_from_filename(cn_path).ok_or_else(|| {
                anyhow::anyhow!(
                    "Could not auto-detect control type for '{}'. \
                     Use --cn-type to specify. Options: canny, depth, pose, softedge, scribble, hed, mlsd, gray, normal",
                    path.file_name().unwrap_or_default().to_string_lossy()
                )
            })?
        };

        // Validate this model + type combination
        if let Err(msg) = model_family::validate_controlnet(effective_model, &control_type) {
            anyhow::bail!(msg);
        }

        inputs.push(ControlNetInput {
            image: cn_path.clone(),
            control_type,
            strength: strengths.get(i).copied().unwrap_or(0.75),
            control_end: ends.get(i).copied().unwrap_or(0.8),
        });
    }
    Ok(inputs)
}

/// Validate and build style-ref inputs from CLI flags. Shared between local
/// and pod execution.
fn build_style_inputs(
    style_ref: &[String],
    style_strength: f32,
    style_type: Option<&str>,
    effective_model: &str,
) -> Result<Vec<StyleRefInput>> {
    if style_ref.is_empty() {
        return Ok(Vec::new());
    }
    if let Err(msg) = model_family::validate_style_ref(effective_model) {
        anyhow::bail!(msg);
    }

    let style_type_str = style_type.unwrap_or("style").to_string();

    style_ref
        .iter()
        .map(|path| {
            if !PathBuf::from(path).exists() {
                anyhow::bail!("Style reference image not found: {path}");
            }
            Ok(StyleRefInput {
                image: path.clone(),
                strength: style_strength,
                style_type: style_type_str.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()
}

/// JSON result for a pod run — same shape for generate/edit/run so agents
/// can parse one schema.
pub(crate) fn pod_result_json(
    outcome: &crate::core::pod_run::RemoteRunOutcome,
) -> serde_json::Value {
    serde_json::json!({
        "status": outcome.status,
        "run_id": outcome.run_id,
        "output_dir": outcome.local_dir,
        "images": outcome.artifacts,
    })
}

async fn run_local(args: GenerateArgs<'_>, db: Database) -> Result<()> {
    // Destructure for convenience
    let GenerateArgs {
        prompt,
        base,
        lora,
        lora_strength,
        seed,
        size,
        steps,
        guidance,
        count,
        init_image,
        mask,
        strength,
        inpaint,
        controlnet,
        cn_strength,
        cn_end,
        cn_type,
        style_ref,
        style_strength,
        style_type,
        fast,
        cloud,
        provider,
        no_worker,
        attach_gpu,
        gpu_type,
        pod: _,
        json,
    } = args;

    // -------------------------------------------------------------------
    // Resolve base model
    // -------------------------------------------------------------------
    let base_model = match base {
        Some(b) => b.to_string(),
        None => default_generation_model(&db),
    };

    // -------------------------------------------------------------------
    // Pre-flight checks (fail fast with actionable hints)
    // -------------------------------------------------------------------
    if !cloud && !attach_gpu {
        preflight::for_generation(&base_model)?;
    }

    let base_model_path = model_resolve::resolve_base_model_path(&base_model, &db);

    // -------------------------------------------------------------------
    // Resolve size: explicit --size wins, otherwise use init-image dims.
    // Presets scale to the model's native resolution (SD 1.5: 1:1 → 512).
    // -------------------------------------------------------------------
    let native_resolution = model_family::default_resolution(&base_model);
    let (width, height) = if let Some(s) = size {
        model_resolve::resolve_size_for_model(s, native_resolution)?
    } else if let Some(path) = init_image {
        image_dimensions(path)?
    } else {
        model_resolve::resolve_size_for_model("1:1", native_resolution)?
    };

    // -------------------------------------------------------------------
    // Resolve --fast (Lightning LoRA)
    // -------------------------------------------------------------------
    let (fast_lora, fast_steps, fast_guidance, scheduler_overrides) = if let Some(fast_steps) = fast
    {
        if lora.is_some() {
            anyhow::bail!(
                "--fast and --lora cannot be used together. \
                 The --fast flag auto-applies a Lightning distillation LoRA."
            );
        }

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

        let sched_overrides = lightning.scheduler_overrides_json();

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
    // Resolve LoRA (--fast takes priority, then --lora)
    // -------------------------------------------------------------------
    let lora_ref = if fast_lora.is_some() {
        fast_lora
    } else {
        match lora {
            Some(name) => Some(
                resolve_lora(name, lora_strength, &db)?.context("LoRA resolution returned None")?,
            ),
            None => None,
        }
    };

    // -------------------------------------------------------------------
    // Validate img2img / inpainting paths + model capabilities
    // -------------------------------------------------------------------
    if let Some(path) = init_image
        && !PathBuf::from(path).exists()
    {
        anyhow::bail!("Init image not found: {path}");
    }
    if let Some(path) = mask {
        if init_image.is_none() {
            anyhow::bail!("--mask requires --init-image");
        }
        if !PathBuf::from(path).exists() {
            anyhow::bail!("Mask image not found: {path}");
        }
    }

    // Check model supports the requested mode
    let mode = if mask.is_some() {
        "inpaint"
    } else if init_image.is_some() {
        "img2img"
    } else {
        "txt2img"
    };

    // Resolve inpainting method
    let resolved_inpaint_method = if mode == "inpaint" {
        let model_info = model_family::resolve_model(&base_model);
        let supports_lanpaint = model_info.is_some_and(|m| m.capabilities.lanpaint_inpaint);
        let supports_standard = model_info.is_some_and(|m| m.capabilities.inpaint);

        match inpaint {
            InpaintMethod::Lanpaint => {
                if !supports_lanpaint {
                    let name = model_info.map_or(&base_model as &str, |m| &m.name);
                    anyhow::bail!(
                        "{} does not support LanPaint inpainting. \
                         Models with LanPaint: z-image, z-image-turbo, flux2-klein-4b, flux2-klein-9b",
                        name
                    );
                }
                Some("lanpaint")
            }
            InpaintMethod::Auto => {
                if supports_lanpaint && !supports_standard {
                    // Model only supports LanPaint (e.g. Klein 9b)
                    Some("lanpaint")
                } else {
                    // Standard inpainting (with Flux Fill routing)
                    None
                }
            }
            InpaintMethod::Standard => {
                if !supports_standard {
                    let name = model_info.map_or(&base_model as &str, |m| &m.name);
                    anyhow::bail!(
                        "{} does not support standard inpainting. \
                         Try --inpaint lanpaint instead.",
                        name
                    );
                }
                None
            }
        }
    } else {
        None
    };

    // Smart inpaint routing: prefer dedicated fill models over generic Flux inpainting
    // Skip fill routing when using LanPaint — it uses the base model directly
    let (effective_model, effective_path) =
        if mode == "inpaint" && resolved_inpaint_method.is_none() {
            resolve_inpaint_model(&base_model, &db)
        } else {
            (base_model.clone(), base_model_path)
        };

    if let Err(msg) = model_family::validate_mode(&effective_model, mode) {
        anyhow::bail!(msg);
    }

    // -------------------------------------------------------------------
    // Validate and build ControlNet + style-ref inputs
    // -------------------------------------------------------------------
    let cn_inputs = build_cn_inputs(controlnet, cn_strength, cn_end, cn_type, &effective_model)?;
    let style_inputs = build_style_inputs(style_ref, style_strength, style_type, &effective_model)?;

    // -------------------------------------------------------------------
    // Build output directory: ~/.modl/outputs/<date>/
    // -------------------------------------------------------------------
    let date = chrono::Local::now().format("%Y-%m-%d");
    let output_dir = crate::core::paths::modl_root()
        .join("outputs")
        .join(date.to_string());
    std::fs::create_dir_all(&output_dir)?;

    // -------------------------------------------------------------------
    // Build spec
    // -------------------------------------------------------------------
    // --fast overrides defaults, but explicit --steps/--guidance wins
    let mut steps = steps
        .or(fast_steps)
        .unwrap_or_else(|| default_steps(&effective_model));
    let guidance = guidance
        .or(fast_guidance)
        .unwrap_or_else(|| default_guidance(&effective_model));

    // ControlNet step adjustment: some models need more steps with ControlNet
    let steps_adjusted = if !cn_inputs.is_empty() && args.steps.is_none() {
        if let Some(cn_support) = model_family::controlnet_support(&effective_model) {
            if steps < cn_support.recommended_min_steps {
                let old_steps = steps;
                steps = cn_support.recommended_min_steps;
                if !json {
                    println!(
                        "  {} Adjusting steps: {} → {} (ControlNet requires more steps for {})",
                        style("↳").dim(),
                        old_steps,
                        steps,
                        effective_model,
                    );
                }
                true
            } else {
                false
            }
        } else {
            false
        }
    } else if !cn_inputs.is_empty() {
        // User provided explicit --steps, warn if too low
        if let Some(cn_support) = model_family::controlnet_support(&effective_model)
            && steps < cn_support.recommended_min_steps
            && !json
        {
            println!(
                "  {} {} with ControlNet works best at {}+ steps (you set {})",
                style("⚠").yellow(),
                effective_model,
                cn_support.recommended_min_steps,
                steps,
            );
        }
        false
    } else {
        false
    };
    let _ = steps_adjusted; // used in JSON output later

    // Resolve family alias to registry manifest ID for the spec (e.g. "sdxl" → "sdxl-base-1.0").
    // This ensures remote agents and Python workers can resolve the model correctly.
    // We keep the original effective_model for local model_family lookups above.
    let spec_model_id = {
        let index = crate::core::registry::RegistryIndex::load();
        if let Ok(ref idx) = index {
            if idx.find(&effective_model).is_some() {
                effective_model.clone()
            } else if let Some(info) = model_family::resolve_model(&effective_model) {
                // Try common patterns: exact id, then {id}-base-1.0
                let candidates = [info.id.to_string(), format!("{}-base-1.0", info.id)];
                candidates
                    .into_iter()
                    .find(|c| idx.find(c).is_some())
                    .unwrap_or_else(|| effective_model.clone())
            } else {
                effective_model.clone()
            }
        } else {
            effective_model.clone()
        }
    };

    let spec = GenerateJobSpec {
        prompt: prompt.to_string(),
        model: ModelRef {
            base_model_id: spec_model_id.clone(),
            base_model_path: effective_path,
            arch_key: model_family::find_model(&spec_model_id).map(|m| m.arch_key.to_string()),
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
            init_image: init_image.map(|s| s.to_string()),
            mask: mask.map(|s| s.to_string()),
            strength,
            scheduler_overrides,
            controlnet: cn_inputs,
            style_ref: style_inputs,
            inpaint_method: resolved_inpaint_method.map(|s| s.to_string()),
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
        let mode_label = if resolved_inpaint_method == Some("lanpaint") {
            "LanPaint inpainting"
        } else if mask.is_some() {
            "inpainting"
        } else if init_image.is_some() {
            "img2img"
        } else {
            "txt2img"
        };
        println!(
            "{} Generating image(s) [{}]...",
            style("→").cyan(),
            mode_label
        );
        println!("  Prompt: {}", style(prompt).italic());
        println!("  Model:  {}", effective_model);
        if fast.is_some() {
            println!(
                "  Mode:   {}",
                style("fast (Lightning LoRA)").green().bold()
            );
        }
        if let Some(ref lr) = lora_ref
            && fast.is_none()
        {
            println!("  LoRA:   {} (strength: {:.2})", lr.name, lr.weight);
        }
        if let Some(path) = init_image {
            println!("  Init:   {}", path);
            println!("  Strength: {:.2}", strength.unwrap_or(0.75));
        }
        if let Some(path) = mask {
            println!("  Mask:   {}", path);
        }
        for (i, cn) in spec.params.controlnet.iter().enumerate() {
            let label = if spec.params.controlnet.len() > 1 {
                format!("CN {}:", i + 1)
            } else {
                "CN:".to_string()
            };
            println!(
                "  {:<6} {} (type: {}, strength: {:.2}, end: {:.1})",
                label,
                PathBuf::from(&cn.image)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                cn.control_type,
                cn.strength,
                cn.control_end,
            );
        }
        for (i, sr) in spec.params.style_ref.iter().enumerate() {
            let label = if spec.params.style_ref.len() > 1 {
                format!("Ref {}:", i + 1)
            } else {
                "Ref:".to_string()
            };
            println!(
                "  {:<6} {} (strength: {:.2})",
                label,
                PathBuf::from(&sr.image)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                sr.strength,
            );
        }
        println!("  Size:   {}×{}", width, height);
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
    execute_generate(spec, cloud, provider, no_worker, attach_gpu, gpu_type, json).await
}

async fn execute_generate(
    spec: GenerateJobSpec,
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

    // -------------------------------------------------------------------
    // 1. Bootstrap executor
    // -------------------------------------------------------------------
    let gpu_session_ref: Option<gpu_session::GpuSession>;

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
        gpu_session_ref = Some(session.clone());
        Box::new(RemoteExecutor::new(session))
    } else if cloud {
        gpu_session_ref = None;
        let cloud_provider = resolve_cloud_provider(provider);
        if !json {
            println!(
                "{} Preparing cloud generation via {}...",
                style("→").cyan(),
                style(cloud_provider.to_string()).bold()
            );
        }
        Box::new(CloudExecutor::new(cloud_provider)?)
    } else {
        gpu_session_ref = None;
        if !json {
            println!("{} Preparing runtime...", style("→").cyan());
        }
        let mut executor = LocalExecutor::for_generation().await?;
        if no_worker {
            executor.use_worker = false;
        }
        Box::new(executor)
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
        None,
    )?;

    // -------------------------------------------------------------------
    // 3. Event loop with progress
    // -------------------------------------------------------------------
    let rx = executor.events(job_id)?;
    // Drop the executor now — we only need the event receiver from here.
    // This also avoids holding a non-Send Box<dyn Executor> across .await.
    drop(executor);
    db.update_job_status(job_id, "running")?;

    let pb = if json {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(spec.params.steps as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} steps {msg}",
            )?
            .progress_chars("█▓░"),
        );
        pb
    };

    struct GeneratedArtifact {
        path: String,
        sha256: Option<String>,
        size_bytes: Option<u64>,
    }

    let mut artifacts: Vec<GeneratedArtifact> = Vec::new();
    let mut final_status = "completed";

    for event in rx {
        match &event.event {
            EventPayload::Progress {
                stage,
                step,
                total_steps,
                ..
            } => {
                if stage == "step" {
                    pb.set_length(*total_steps as u64);
                    pb.set_position(*step as u64);
                } else if stage == "generate" {
                    // Image completed — reset bar for next image
                    if *step < *total_steps {
                        pb.set_position(0);
                        pb.set_message(format!("(image {}/{})", step + 1, total_steps));
                    }
                }
            }
            EventPayload::Artifact {
                path,
                sha256,
                size_bytes,
            } => {
                artifacts.push(GeneratedArtifact {
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
            EventPayload::Heartbeat | EventPayload::Result { .. } => {}
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
    // 4b. Download remote artifacts (--attach-gpu)
    // -------------------------------------------------------------------
    if attach_gpu
        && final_status == "completed"
        && let Some(ref session) = gpu_session_ref
    {
        let client = gpu_session::GpuClient::from_session(session)?;
        let remote_artifacts = client
            .get_job_artifacts(&session.session_id, job_id)
            .await?;

        if !remote_artifacts.is_empty() {
            if !json {
                println!(
                    "{} Downloading {} artifact(s) from remote GPU...",
                    style("→").cyan(),
                    remote_artifacts.len()
                );
            }

            let output_dir = PathBuf::from(&spec.output.output_dir);
            std::fs::create_dir_all(&output_dir)?;

            artifacts.clear();
            for ra in &remote_artifacts {
                let filename = ra.filename();
                let local_path = output_dir.join(&filename);
                client
                    .download_artifact(&ra.download_url, &local_path)
                    .await
                    .with_context(|| format!("Failed to download {filename}"))?;

                artifacts.push(GeneratedArtifact {
                    path: local_path.to_string_lossy().to_string(),
                    sha256: ra.sha256.clone(),
                    size_bytes: ra.size_bytes,
                });
            }

            if !json {
                println!(
                    "  {} Downloaded {} image(s)",
                    style("✓").green(),
                    artifacts.len()
                );
            }
        }
    }

    // -------------------------------------------------------------------
    // 5. Print results
    // -------------------------------------------------------------------
    if final_status == "completed" && !artifacts.is_empty() {
        // Register artifacts in DB
        for (i, artifact) in artifacts.iter().enumerate() {
            let artifact_id = format!("{}-img-{}", job_id, i);
            let image_seed = spec.params.seed.map(|s| s + i as u64);
            let metadata = serde_json::json!({
                "generated_with": "modl.run",
                "prompt": spec.prompt,
                "base_model_id": spec.model.base_model_id,
                "base_model_path": spec.model.base_model_path,
                "lora_name": spec.lora.as_ref().map(|l| l.name.clone()),
                "lora_strength": spec.lora.as_ref().map(|l| l.weight),
                "width": spec.params.width,
                "height": spec.params.height,
                "steps": spec.params.steps,
                "guidance": spec.params.guidance,
                "seed": image_seed,
                "image_index": i,
                "count": spec.params.count,
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
                size: format!("{}x{}", spec.params.width, spec.params.height),
                lora: spec.lora.as_ref().map(|l| l.name.clone()),
                lora_strength: spec.lora.as_ref().map(|l| l.weight),
                created_at: chrono::Utc::now().to_rfc3339(),
                source: "generate".to_string(),
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
                "{} Generated {} image(s):",
                style("✓").green().bold(),
                artifact_paths.len()
            );
            for path in &artifact_paths {
                println!("  {}", path);
            }
        }
    } else if artifacts.is_empty() && final_status == "completed" {
        if json {
            println!(
                "{}",
                serde_json::json!({"status": "completed", "images": []})
            );
        } else {
            println!(
                "\n{} Generation completed but no images were produced.",
                style("⚠").yellow()
            );
        }
    } else if json {
        let artifact_paths: Vec<String> = artifacts.iter().map(|a| a.path.clone()).collect();
        println!(
            "{}",
            serde_json::json!({"status": final_status, "images": artifact_paths})
        );
    }

    if final_status == "error" {
        anyhow::bail!("Generation failed");
    }

    Ok(())
}
