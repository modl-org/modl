use anyhow::{Context, Result};
use console::style;
use image::GenericImageView;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

use crate::cli::InpaintMethod;
use crate::core::cloud::{CloudExecutor, CloudProvider};
use crate::core::db::Database;
use crate::core::executor::{Executor, LocalExecutor};
use crate::core::job::*;
use crate::core::model_family;
use crate::core::outputs::{SidecarMetadata, write_sidecar_yaml};
use crate::core::preflight;
use crate::core::runtime;

/// Known control type suffixes for auto-detection from filenames.
const CONTROL_SUFFIXES: &[(&str, &str)] = &[
    ("_canny", "canny"),
    ("_depth", "depth"),
    ("_pose", "pose"),
    ("_softedge", "softedge"),
    ("_scribble", "scribble"),
    ("_hed", "hed"),
    ("_mlsd", "mlsd"),
    ("_gray", "gray"),
    ("_normal", "normal"),
    ("_lineart", "lineart"),
];

/// Try to detect the control type from a filename suffix (e.g. "photo_depth.png" → "depth").
fn detect_control_type_from_filename(path: &str) -> Option<String> {
    let stem = PathBuf::from(path)
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();

    for &(suffix, control_type) in CONTROL_SUFFIXES {
        if stem.ends_with(suffix) {
            return Some(control_type.to_string());
        }
    }
    None
}

/// Size presets: aspect ratio → (width, height)
const SIZE_PRESETS: &[(&str, u32, u32)] = &[
    ("1:1", 1024, 1024),
    ("16:9", 1344, 768),
    ("9:16", 768, 1344),
    ("4:3", 1152, 896),
    ("3:4", 896, 1152),
];

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
fn resolve_lora(name: &str, weight: f32, db: &Database) -> Result<Option<LoraRef>> {
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
            weight,
        }));
    }

    // Look up in installed models (match by ID or display name)
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

    anyhow::bail!(
        "LoRA not found: {name}. Use `modl model ls --type lora` to see installed LoRAs, or provide a file path."
    );
}

/// Pick the best installed generation model, falling back to "flux-schnell".
fn default_generation_model(db: &Database) -> String {
    let gen_types = ["checkpoint", "diffusion_model"];
    if let Ok(installed) = db.list_installed(None) {
        let gen_models: Vec<_> = installed
            .iter()
            .filter(|m| gen_types.contains(&m.asset_type.as_str()))
            .collect();
        if gen_models.len() == 1 {
            return gen_models[0].id.clone();
        }
        // Multiple installed: prefer flux-schnell if present, else first
        if !gen_models.is_empty() {
            if let Some(m) = gen_models.iter().find(|m| m.id == "flux-schnell") {
                return m.id.clone();
            }
            return gen_models[0].id.clone();
        }
    }
    "flux-schnell".to_string()
}

/// Default inference steps based on model type.
fn default_steps(base_model: &str) -> u32 {
    model_family::model_defaults(base_model).0
}

/// Default guidance scale based on model type.
fn default_guidance(base_model: &str) -> f32 {
    model_family::model_defaults(base_model).1
}

/// Resolve base model path from installed models (match by ID, display name, or family alias).
fn resolve_base_model_path(base_model: &str, db: &Database) -> Option<String> {
    let installed = db.list_installed(None).ok()?;
    let gen_types = ["checkpoint", "diffusion_model"];

    // Exact match by ID or display name
    for model in &installed {
        if (model.name == base_model || model.id == base_model)
            && gen_types.contains(&model.asset_type.as_str())
        {
            return Some(model.store_path.clone());
        }
    }

    // Fuzzy match via model_family (e.g. "sdxl" → "sdxl-base-1.0")
    if let Some(family_info) = model_family::resolve_model(base_model) {
        let family_id = family_info.id;
        for model in &installed {
            if gen_types.contains(&model.asset_type.as_str()) {
                // Match if the installed model's ID contains the family ID or vice versa
                if model.id == family_id
                    || model.id.contains(family_id)
                    || family_id.contains(&*model.id)
                {
                    return Some(model.store_path.clone());
                }
            }
        }
    }

    None
}

/// Check if a model ID is installed in the DB.
fn is_model_installed(model_id: &str, db: &Database) -> bool {
    resolve_base_model_path(model_id, db).is_some()
}

/// Smart inpaint routing: prefer dedicated fill models over generic Flux inpainting.
///
/// When the user requests inpainting with a regular Flux 1 model (flux-dev,
/// flux-schnell), auto-route to the best installed Flux Fill model. Fill models
/// have 384 input channels and produce much cleaner inpainting results.
fn resolve_inpaint_model(base_model: &str, db: &Database) -> (String, Option<String>) {
    let info = model_family::resolve_model(base_model);
    let is_flux1 = info.is_some_and(|m| m.arch_key == "flux" || m.arch_key == "flux_schnell");
    if !is_flux1 {
        return (
            base_model.to_string(),
            resolve_base_model_path(base_model, db),
        );
    }

    for candidate in ["flux-fill-dev-onereward", "flux-fill-dev"] {
        if is_model_installed(candidate, db) {
            println!(
                "  {} Using {} for inpainting",
                style("↳").dim(),
                style(candidate).bold()
            );
            return (
                candidate.to_string(),
                resolve_base_model_path(candidate, db),
            );
        }
    }

    (
        base_model.to_string(),
        resolve_base_model_path(base_model, db),
    )
}

/// Parsed outpaint specification.
#[derive(Debug, Clone)]
struct OutpaintSpec {
    left: u32,
    right: u32,
    top: u32,
    bottom: u32,
    #[allow(dead_code)] // reserved for future feathering support
    feather: u32,
}

/// Parse an outpaint string like "right=256", "left=128,right=128,feather=40", or "all=256".
fn parse_outpaint(spec: &str) -> Result<OutpaintSpec> {
    let mut left = 0u32;
    let mut right = 0u32;
    let mut top = 0u32;
    let mut bottom = 0u32;
    let mut feather = 32u32;

    for part in spec.split(',') {
        let part = part.trim();
        if let Some((key, val)) = part.split_once('=') {
            let val: u32 = val
                .trim()
                .parse()
                .with_context(|| format!("Invalid outpaint value: {part}"))?;
            match key.trim() {
                "left" | "l" => left = val,
                "right" | "r" => right = val,
                "top" | "t" => top = val,
                "bottom" | "b" => bottom = val,
                "all" | "a" => {
                    left = val;
                    right = val;
                    top = val;
                    bottom = val;
                }
                "feather" | "f" => feather = val,
                _ => anyhow::bail!(
                    "Unknown outpaint key: {key}. Use left, right, top, bottom, all, or feather."
                ),
            }
        } else {
            // Bare number = all sides
            let val: u32 = part.parse().with_context(|| {
                format!("Invalid outpaint spec: {part}. Use 'right=256' or '256' for all sides.")
            })?;
            left = val;
            right = val;
            top = val;
            bottom = val;
        }
    }

    if left == 0 && right == 0 && top == 0 && bottom == 0 {
        anyhow::bail!(
            "Outpaint spec must extend at least one direction. Example: --outpaint 'right=256'"
        );
    }

    // Round to multiples of 8 (VAE requirement)
    left = left.div_ceil(8) * 8;
    right = right.div_ceil(8) * 8;
    top = top.div_ceil(8) * 8;
    bottom = bottom.div_ceil(8) * 8;

    Ok(OutpaintSpec {
        left,
        right,
        top,
        bottom,
        feather,
    })
}

/// Pad an image for outpainting by extending edge pixels with a gradient blur.
///
/// Strategy: fill padded areas by repeating the nearest edge pixel, then
/// apply a progressive gaussian blur so content fades smoothly away from
/// the original edges. This avoids the mirror-symmetry artifacts that
/// reflected padding can produce.
fn pad_image_for_outpaint(
    init_image_path: &str,
    spec: &OutpaintSpec,
) -> Result<(String, u32, u32)> {
    let img = image::open(init_image_path)
        .with_context(|| format!("Cannot open init-image: {init_image_path}"))?;
    let (orig_w, orig_h) = img.dimensions();

    let new_w = orig_w + spec.left + spec.right;
    let new_h = orig_h + spec.top + spec.bottom;

    let mut padded = image::RgbImage::new(new_w, new_h);
    let rgb = img.to_rgb8();

    // Place original image
    for y in 0..orig_h {
        for x in 0..orig_w {
            padded.put_pixel(x + spec.left, y + spec.top, *rgb.get_pixel(x, y));
        }
    }

    // Fill padded areas with nearest edge pixel (edge-extend / clamp)
    for y in 0..new_h {
        for x in 0..new_w {
            if x >= spec.left && x < spec.left + orig_w && y >= spec.top && y < spec.top + orig_h {
                continue;
            }

            let src_x = if x < spec.left {
                0
            } else if x >= spec.left + orig_w {
                orig_w - 1
            } else {
                x - spec.left
            };

            let src_y = if y < spec.top {
                0
            } else if y >= spec.top + orig_h {
                orig_h - 1
            } else {
                y - spec.top
            };

            padded.put_pixel(x, y, *rgb.get_pixel(src_x, src_y));
        }
    }

    // Apply gaussian blur to the padded image, then paste original back on top.
    // This creates a smooth, blurry transition in the padded areas while keeping
    // the original pixel-perfect.
    let max_pad = spec.left.max(spec.right).max(spec.top).max(spec.bottom);
    let blur_sigma = (max_pad as f32 / 4.0).max(8.0);
    let blurred = image::DynamicImage::ImageRgb8(padded).blur(blur_sigma);
    let mut result = blurred.to_rgb8();

    // Paste original back (sharp, untouched)
    for y in 0..orig_h {
        for x in 0..orig_w {
            result.put_pixel(x + spec.left, y + spec.top, *rgb.get_pixel(x, y));
        }
    }

    // Save to tmp
    let tmp_dir = crate::core::paths::modl_root().join("tmp");
    std::fs::create_dir_all(&tmp_dir)?;
    let filename = format!("outpaint_{}.png", chrono::Utc::now().timestamp_millis());
    let dest = tmp_dir.join(&filename);
    result
        .save(&dest)
        .with_context(|| format!("Failed to save padded image to {}", dest.display()))?;

    Ok((dest.to_string_lossy().to_string(), new_w, new_h))
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
    pub outpaint: Option<&'a str>,
    pub fast: bool,
    pub cloud: bool,
    pub provider: Option<CloudProvider>,
    pub no_worker: bool,
    pub json: bool,
}

pub async fn run(args: GenerateArgs<'_>) -> Result<()> {
    let db = Database::open()?;

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
        outpaint,
        fast,
        cloud,
        provider,
        no_worker,
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
    if !cloud {
        preflight::for_generation(&base_model)?;
    }

    let base_model_path = resolve_base_model_path(&base_model, &db);

    // -------------------------------------------------------------------
    // Resolve size: explicit --size wins, otherwise use init-image dims
    // -------------------------------------------------------------------
    let (width, height) = if let Some(s) = size {
        resolve_size(s)?
    } else if let Some(path) = init_image {
        image_dimensions(path)?
    } else {
        resolve_size("1:1")?
    };

    // -------------------------------------------------------------------
    // Resolve --fast (Lightning LoRA)
    // -------------------------------------------------------------------
    let (fast_lora, fast_steps, fast_guidance) = if fast {
        if lora.is_some() {
            anyhow::bail!(
                "--fast and --lora cannot be used together. \
                 The --fast flag auto-applies a Lightning distillation LoRA."
            );
        }

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

    // -------------------------------------------------------------------
    // Outpaint: pad image and route to edit pipeline
    // -------------------------------------------------------------------
    if let Some(outpaint_spec) = outpaint {
        let init_path =
            init_image.ok_or_else(|| anyhow::anyhow!("--outpaint requires --init-image"))?;

        // Check model supports edit mode
        let model_info = model_family::resolve_model(&base_model);
        let supports_edit = model_info.is_some_and(|m| m.capabilities.edit);
        if !supports_edit {
            let name = model_info.map_or(&base_model as &str, |m| m.name);
            anyhow::bail!(
                "{} does not support outpainting (no edit capability). \
                 Models with outpaint support: flux2-klein-9b, flux2-klein-4b, qwen-image-edit",
                name
            );
        }

        let outpaint = parse_outpaint(outpaint_spec)?;

        // Pad the image
        let (padded_path, new_w, new_h) = pad_image_for_outpaint(init_path, &outpaint)?;

        // Resolve defaults
        let (default_steps, default_guidance) = model_family::model_defaults(&base_model);
        let steps = steps.or(fast_steps).unwrap_or(default_steps);
        let guidance = guidance.or(fast_guidance).unwrap_or(default_guidance);

        if !json {
            let dirs: Vec<String> = [
                (outpaint.left, "left"),
                (outpaint.right, "right"),
                (outpaint.top, "top"),
                (outpaint.bottom, "bottom"),
            ]
            .iter()
            .filter(|(v, _)| *v > 0)
            .map(|(v, d)| format!("{d}={v}"))
            .collect();
            println!(
                "{} Outpainting via edit pipeline [{}]...",
                style("→").cyan(),
                dirs.join(", ")
            );
            println!("  Prompt: {}", style(prompt).italic());
            println!("  Model:  {}", base_model);
            if fast {
                println!(
                    "  Mode:   {}",
                    style("fast (Lightning LoRA)").green().bold()
                );
            }
            if let Some(ref lr) = lora_ref
                && !fast
            {
                println!("  LoRA:   {} (strength: {:.2})", lr.name, lr.weight);
            }
            println!(
                "  Source:  {} ({}×{})",
                init_path,
                new_w - outpaint.left - outpaint.right,
                new_h - outpaint.top - outpaint.bottom
            );
            println!("  Output:  {}×{}", new_w, new_h);
            println!("  Steps:  {}", steps);
            if let Some(s) = seed {
                println!("  Seed:   {}", s);
            }
            if count > 1 {
                println!("  Count:  {}", count);
            }
        }

        // Build an EditJobSpec and route to the edit pipeline
        let date = chrono::Local::now().format("%Y-%m-%d");
        let output_dir = crate::core::paths::modl_root()
            .join("outputs")
            .join(date.to_string());
        std::fs::create_dir_all(&output_dir)?;

        let spec = EditJobSpec {
            prompt: prompt.to_string(),
            model: ModelRef {
                base_model_id: base_model.clone(),
                base_model_path: base_model_path.clone(),
            },
            lora: lora_ref,
            output: GenerateOutputRef {
                output_dir: output_dir.to_string_lossy().to_string(),
            },
            params: EditParams {
                image_paths: vec![padded_path],
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

        return crate::cli::edit::execute_edit(spec, cloud, provider, no_worker, json).await;
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
                    let name = model_info.map_or(&base_model as &str, |m| m.name);
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
                    let name = model_info.map_or(&base_model as &str, |m| m.name);
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
    // Validate and build ControlNet inputs
    // -------------------------------------------------------------------
    let cn_inputs = if !controlnet.is_empty() {
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
            if let Err(msg) = model_family::validate_controlnet(&effective_model, &control_type) {
                anyhow::bail!(msg);
            }

            inputs.push(ControlNetInput {
                image: cn_path.clone(),
                control_type,
                strength: strengths.get(i).copied().unwrap_or(0.75),
                control_end: ends.get(i).copied().unwrap_or(0.8),
            });
        }
        inputs
    } else {
        Vec::new()
    };

    // -------------------------------------------------------------------
    // Validate and build style-ref inputs
    // -------------------------------------------------------------------
    let style_inputs: Vec<StyleRefInput> = if !style_ref.is_empty() {
        // Validate model supports style-ref
        if let Err(msg) = model_family::validate_style_ref(&effective_model) {
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
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };

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

    let spec = GenerateJobSpec {
        prompt: prompt.to_string(),
        model: ModelRef {
            base_model_id: effective_model.clone(),
            base_model_path: effective_path,
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
            controlnet: cn_inputs,
            style_ref: style_inputs,
            inpaint_method: resolved_inpaint_method.map(|s| s.to_string()),
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
        if fast {
            println!(
                "  Mode:   {}",
                style("fast (Lightning LoRA)").green().bold()
            );
        }
        if let Some(ref lr) = lora_ref
            && !fast
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
    execute_generate(spec, cloud, provider, no_worker, json).await
}

async fn execute_generate(
    spec: GenerateJobSpec,
    cloud: bool,
    provider: Option<CloudProvider>,
    no_worker: bool,
    json: bool,
) -> Result<()> {
    let db = Database::open()?;
    let spec_json = serde_json::to_string(&spec)?;
    let target_str = serde_json::to_string(&spec.target)?;

    // -------------------------------------------------------------------
    // 1. Bootstrap executor
    // -------------------------------------------------------------------
    let mut executor: Box<dyn Executor> = if cloud {
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
    )?;

    // -------------------------------------------------------------------
    // 3. Event loop with progress
    // -------------------------------------------------------------------
    let rx = executor.events(job_id)?;
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
