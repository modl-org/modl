//! Workflow spec parser and validator for `modl run`.
//!
//! A workflow is a sequentially-executed list of generate/edit jobs with shared
//! model/lora defaults. See `docs/plans/workflow-run.md` for the design.
//!
//! This module only parses and validates. Materialization into
//! `GenerateJobSpec` / `EditJobSpec` happens at execution time (Phase B).

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine as _;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public types (materialized after parse + validate)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Workflow {
    pub name: String,
    pub model: String,
    pub lora: Option<String>,
    pub steps: Vec<Step>,
}

#[derive(Debug, Clone)]
pub struct Step {
    pub id: String,
    pub kind: StepKind,
}

#[derive(Debug, Clone)]
pub enum StepKind {
    Generate(GenerateStep),
    Edit(EditStep),
}

#[derive(Debug, Clone)]
pub struct GenerateStep {
    pub prompt: String,
    /// Per-step model override. If set, this step uses this model instead of
    /// the workflow-level default. Only validated structurally at parse time —
    /// existence in the local store is checked at plan-build time.
    pub model: Option<String>,
    /// Per-step LoRA override. Semantics when unset:
    ///
    /// - model not overridden → inherit workflow-level `lora`
    /// - model overridden → no LoRA (auto-disabled; the workflow-level LoRA
    ///   probably belongs to a different model family)
    pub lora: Option<String>,
    /// Lightning fast mode: target step count (4 or 8). Resolves to the
    /// model's Lightning distillation LoRA + scheduler overrides at plan
    /// time. Mutually exclusive with `lora` on the same step; overrides an
    /// inherited workflow-level `lora`.
    pub fast: Option<u32>,
    pub seed: Option<u64>,
    /// Explicit seed list for variation exploration — one output per seed.
    /// Mutually exclusive with `seed` + `count`. Empty list is rejected at parse time.
    pub seeds: Option<Vec<u64>>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub count: Option<u32>,
    /// img2img source image. Presence switches the step to img2img mode
    /// (or inpaint when `mask` is also set).
    pub init_image: Option<ImageRef>,
    /// Inpaint mask (white = regenerate region). Requires `init_image`.
    pub mask: Option<ImageRef>,
    /// Denoising strength for img2img (0.0–1.0). Requires `init_image`.
    pub strength: Option<f32>,
    /// ControlNet inputs (max 2, same cap as `modl generate --controlnet`).
    pub controlnet: Vec<ControlNetRef>,
    /// Style reference inputs (IP-Adapter).
    pub style_ref: Vec<StyleRefRef>,
}

#[derive(Debug, Clone)]
pub struct ControlNetRef {
    pub image: ImageRef,
    /// Control type (canny, depth, pose, …). Resolved at parse time from the
    /// explicit `type:` field or, for plain file paths, the filename suffix.
    pub control_type: String,
    pub strength: Option<f32>,
    pub end: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct StyleRefRef {
    pub image: ImageRef,
    pub strength: Option<f32>,
    pub style_type: Option<String>,
}

/// Mask on an edit step: an image ref, or one of the worker's sentinel modes.
#[derive(Debug, Clone)]
pub enum EditMask {
    /// Worker derives the mask (reserved sentinel, `"auto"`).
    Auto,
    /// Derive from the first source image's alpha channel (`"from-alpha"`).
    FromAlpha,
    Image(ImageRef),
}

#[derive(Debug, Clone)]
pub struct EditStep {
    /// Source images (1 or more). Multi-image edits feed every image to the
    /// model as a reference (Qwen Image Edit 2511, Flux 2 Klein).
    pub sources: Vec<ImageRef>,
    pub prompt: String,
    /// Mask: image ref or `auto` / `from-alpha` sentinel.
    pub mask: Option<EditMask>,
    /// Blend mode for masked edits (`pixel` default, `latent` = native inpaint).
    pub blend: Option<crate::core::job::BlendMode>,
    /// Per-step model override. See `GenerateStep::model`.
    pub model: Option<String>,
    /// Per-step LoRA override. See `GenerateStep::lora`.
    pub lora: Option<String>,
    /// Lightning fast mode. See `GenerateStep::fast`.
    pub fast: Option<u32>,
    pub seed: Option<u64>,
    /// Explicit seed list for variation exploration — one output per seed.
    /// Mutually exclusive with `seed` + `count`. Empty list is rejected at parse time.
    pub seeds: Option<Vec<u64>>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub count: Option<u32>,
}

/// Reference to a source image: either a local file (uploaded at submit time
/// when running `--cloud`) or an output of a prior step (resolved at runtime
/// from disk).
///
/// The output index cannot be bounds-checked statically: a `generate` step's
/// `count` determines its output cardinality at execution time, so `[N]`
/// out-of-bounds surfaces as a runtime error during the referencing step's
/// image resolution — not at parse time.
#[derive(Debug, Clone)]
pub enum ImageRef {
    Local(PathBuf),
    StepOutput { step_id: String, index: usize },
}

impl Step {
    /// Every image ref this step consumes, across all input slots. Used for
    /// dependency scheduling — any slot may reference an earlier step's output.
    pub fn image_refs(&self) -> Vec<&ImageRef> {
        let mut refs = Vec::new();
        match &self.kind {
            StepKind::Generate(g) => {
                refs.extend(g.init_image.as_ref());
                refs.extend(g.mask.as_ref());
                refs.extend(g.controlnet.iter().map(|c| &c.image));
                refs.extend(g.style_ref.iter().map(|s| &s.image));
            }
            StepKind::Edit(e) => {
                refs.extend(e.sources.iter());
                if let Some(EditMask::Image(r)) = &e.mask {
                    refs.push(r);
                }
            }
        }
        refs
    }
}

// ---------------------------------------------------------------------------
// Raw YAML types (deserialization only)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RawWorkflow {
    name: String,
    model: String,
    #[serde(default)]
    lora: Option<String>,
    #[serde(default)]
    defaults: StepDefaults,
    /// Named image variables — defined once, referenced in any edit step as `$name`.
    /// Values are either `data:image/...;base64,...` URIs or server-side file paths.
    #[serde(default)]
    images: HashMap<String, String>,
    steps: Vec<RawStep>,
}

#[derive(Debug, Default, Deserialize)]
struct StepDefaults {
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    steps: Option<u32>,
    #[serde(default)]
    guidance: Option<f32>,
    #[serde(default)]
    count: Option<u32>,
}

/// A YAML value that is either a single string or a list of strings.
/// Lets `edit:` accept one source (`edit: "$a"`) or many (`edit: ["$a", "$b"]`).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

impl OneOrMany {
    fn as_slice(&self) -> &[String] {
        match self {
            OneOrMany::One(s) => std::slice::from_ref(s),
            OneOrMany::Many(v) => v.as_slice(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawControlNet {
    image: String,
    #[serde(default, rename = "type")]
    control_type: Option<String>,
    #[serde(default)]
    strength: Option<f32>,
    #[serde(default)]
    end: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RawStyleRef {
    image: String,
    #[serde(default)]
    strength: Option<f32>,
    #[serde(default, rename = "type")]
    style_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawStep {
    id: String,
    #[serde(default)]
    generate: Option<String>,
    #[serde(default)]
    edit: Option<OneOrMany>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    init_image: Option<String>,
    /// Mask: image ref on generate steps; ref or `auto`/`from-alpha` on edit steps.
    #[serde(default)]
    mask: Option<String>,
    #[serde(default)]
    strength: Option<f32>,
    #[serde(default)]
    blend: Option<crate::core::job::BlendMode>,
    #[serde(default)]
    controlnet: Option<Vec<RawControlNet>>,
    #[serde(default)]
    style_ref: Option<Vec<RawStyleRef>>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    lora: Option<String>,
    #[serde(default)]
    fast: Option<u32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    seeds: Option<Vec<u64>>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    steps: Option<u32>,
    #[serde(default)]
    guidance: Option<f32>,
    #[serde(default)]
    count: Option<u32>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn parse_file(path: &Path) -> Result<Workflow> {
    let yaml = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read workflow file: {}", path.display()))?;
    let base_dir = path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    parse_str(&yaml, &base_dir).with_context(|| format!("In workflow file: {}", path.display()))
}

pub fn parse_str(yaml: &str, base_dir: &Path) -> Result<Workflow> {
    let raw: RawWorkflow = serde_yaml::from_str(yaml).context("Failed to parse workflow YAML")?;

    if raw.name.trim().is_empty() {
        bail!("workflow `name` is required");
    }
    if raw.model.trim().is_empty() {
        bail!("workflow `model` is required");
    }
    if raw.steps.is_empty() {
        bail!("workflow must have at least one step");
    }

    // Resolve named image variables before processing steps. Each value is either
    // a base64 data URI or a server-side path; both materialise to a PathBuf.
    let image_vars = resolve_image_vars(&raw.images, base_dir)?;

    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut materialized: Vec<Step> = Vec::with_capacity(raw.steps.len());

    for (idx, raw_step) in raw.steps.iter().enumerate() {
        // --- id validation
        if raw_step.id.trim().is_empty() {
            bail!("step {idx}: `id` is required");
        }
        if !is_valid_id(&raw_step.id) {
            bail!(
                "step {idx}: id `{}` must contain only letters, digits, `-`, `_`",
                raw_step.id
            );
        }
        if !seen_ids.insert(raw_step.id.clone()) {
            bail!(
                "duplicate step id `{}` at step {idx} (ids must be unique)",
                raw_step.id
            );
        }

        // --- kind validation: exactly one of generate/edit
        let kind_count = raw_step.generate.is_some() as u8 + raw_step.edit.is_some() as u8;
        if kind_count == 0 {
            bail!(
                "step `{}`: must have exactly one of `generate:` or `edit:`",
                raw_step.id
            );
        }
        if kind_count > 1 {
            bail!(
                "step `{}`: cannot have both `generate:` and `edit:`",
                raw_step.id
            );
        }

        // --- seeds validation (shared between generate + edit kinds)
        validate_seeds(&raw_step.id, &raw_step.seeds, raw_step.seed, raw_step.count)?;

        // --- per-step model/lora non-empty check (set but empty string is a user error)
        if let Some(ref m) = raw_step.model
            && m.trim().is_empty()
        {
            bail!(
                "step `{}`: `model` is set but empty — remove the field or provide a model id",
                raw_step.id
            );
        }
        if let Some(ref l) = raw_step.lora
            && l.trim().is_empty()
        {
            bail!(
                "step `{}`: `lora` is set but empty — remove the field or provide a LoRA id",
                raw_step.id
            );
        }
        if raw_step.fast.is_some() && raw_step.lora.is_some() {
            bail!(
                "step `{}`: cannot set both `fast` and `lora` — `fast` auto-applies the model's Lightning LoRA",
                raw_step.id
            );
        }

        // --- image-input fields are kind-specific
        if raw_step.generate.is_some() {
            if raw_step.blend.is_some() {
                bail!(
                    "step `{}`: `blend` is only valid on edit steps",
                    raw_step.id
                );
            }
        } else {
            for (field, set) in [
                ("init_image", raw_step.init_image.is_some()),
                ("strength", raw_step.strength.is_some()),
                ("controlnet", raw_step.controlnet.is_some()),
                ("style_ref", raw_step.style_ref.is_some()),
            ] {
                if set {
                    bail!(
                        "step `{}`: `{field}` is only valid on generate steps",
                        raw_step.id
                    );
                }
            }
        }

        // When a step overrides the model, don't inherit workflow-level
        // defaults for steps/guidance — those are model-dependent and the
        // runner will fall back to the correct model-specific defaults from
        // models.toml instead. Width/height/count/seed are model-independent
        // and always inherit.
        let has_model_override = raw_step.model.is_some();
        let default_steps = if has_model_override {
            None
        } else {
            raw.defaults.steps
        };
        let default_guidance = if has_model_override {
            None
        } else {
            raw.defaults.guidance
        };

        let kind = if let Some(prompt) = &raw_step.generate {
            // img2img / inpaint inputs
            let init_image = raw_step
                .init_image
                .as_deref()
                .map(|s| parse_image_ref(s, base_dir, &seen_ids, &image_vars, &raw_step.id, "init"))
                .transpose()?;
            if init_image.is_none() {
                if raw_step.mask.is_some() {
                    bail!(
                        "step `{}`: `mask` requires `init_image` on generate steps (inpainting regenerates a masked region of the init image)",
                        raw_step.id
                    );
                }
                if raw_step.strength.is_some() {
                    bail!(
                        "step `{}`: `strength` requires `init_image` (it is the img2img denoising strength)",
                        raw_step.id
                    );
                }
            }
            let mask = raw_step
                .mask
                .as_deref()
                .map(|s| parse_image_ref(s, base_dir, &seen_ids, &image_vars, &raw_step.id, "mask"))
                .transpose()?;

            // ControlNet inputs
            let raw_cn = raw_step.controlnet.as_deref().unwrap_or_default();
            if raw_cn.len() > 2 {
                bail!(
                    "step `{}`: maximum 2 `controlnet` inputs supported (got {})",
                    raw_step.id,
                    raw_cn.len()
                );
            }
            let mut controlnet = Vec::with_capacity(raw_cn.len());
            for (i, cn) in raw_cn.iter().enumerate() {
                let slot = format!("cn-{}", i + 1);
                let image = parse_image_ref(
                    &cn.image,
                    base_dir,
                    &seen_ids,
                    &image_vars,
                    &raw_step.id,
                    &slot,
                )?;
                // Explicit `type:` wins; plain file paths fall back to filename
                // detection. `$var`/`$step.outputs[N]`/data-URI refs have no
                // meaningful filename, so `type:` is required there.
                let control_type = match &cn.control_type {
                    Some(t) => t.clone(),
                    None if !cn.image.starts_with('$') && !cn.image.starts_with("data:") => {
                        crate::core::models::detect_control_type_from_filename(&cn.image)
                            .ok_or_else(|| anyhow!(
                                "step `{}`: controlnet[{i}]: cannot auto-detect control type from `{}` — add `type:` (canny, depth, pose, softedge, scribble, hed, mlsd, gray, normal)",
                                raw_step.id, cn.image
                            ))?
                    }
                    None => bail!(
                        "step `{}`: controlnet[{i}]: `type:` is required for `$name`, step-output, and inline image refs (canny, depth, pose, …)",
                        raw_step.id
                    ),
                };
                controlnet.push(ControlNetRef {
                    image,
                    control_type,
                    strength: cn.strength,
                    end: cn.end,
                });
            }

            // Style reference inputs
            let mut style_ref = Vec::new();
            for (i, sr) in raw_step
                .style_ref
                .as_deref()
                .unwrap_or_default()
                .iter()
                .enumerate()
            {
                let slot = format!("style-{}", i + 1);
                let image = parse_image_ref(
                    &sr.image,
                    base_dir,
                    &seen_ids,
                    &image_vars,
                    &raw_step.id,
                    &slot,
                )?;
                style_ref.push(StyleRefRef {
                    image,
                    strength: sr.strength,
                    style_type: sr.style_type.clone(),
                });
            }

            StepKind::Generate(GenerateStep {
                prompt: prompt.clone(),
                model: raw_step.model.clone(),
                lora: raw_step.lora.clone(),
                fast: raw_step.fast,
                seed: raw_step.seed.or(raw.defaults.seed),
                seeds: raw_step.seeds.clone(),
                width: raw_step.width.or(raw.defaults.width),
                height: raw_step.height.or(raw.defaults.height),
                steps: raw_step.steps.or(default_steps),
                guidance: raw_step.guidance.or(default_guidance),
                count: raw_step.count.or(raw.defaults.count),
                init_image,
                mask,
                strength: raw_step.strength,
                controlnet,
                style_ref,
            })
        } else {
            let source_strs = raw_step.edit.as_ref().ok_or_else(|| {
                anyhow!(
                    "step `{}`: expected `generate:` or `edit:` field",
                    raw_step.id
                )
            })?;
            let edit_prompt = raw_step.prompt.as_ref().ok_or_else(|| {
                anyhow!(
                    "step `{}`: edit steps require a `prompt:` field",
                    raw_step.id
                )
            })?;
            let source_strs = source_strs.as_slice();
            if source_strs.is_empty() {
                bail!(
                    "step `{}`: `edit: []` is empty — provide at least one image ref",
                    raw_step.id
                );
            }
            let mut sources = Vec::with_capacity(source_strs.len());
            for (i, s) in source_strs.iter().enumerate() {
                let slot = if i == 0 {
                    "source".to_string()
                } else {
                    format!("source-{}", i + 1)
                };
                sources.push(parse_image_ref(
                    s,
                    base_dir,
                    &seen_ids,
                    &image_vars,
                    &raw_step.id,
                    &slot,
                )?);
            }
            let mask = match raw_step.mask.as_deref() {
                None => None,
                Some("auto") => Some(EditMask::Auto),
                Some("from-alpha") => Some(EditMask::FromAlpha),
                Some(s) => Some(EditMask::Image(parse_image_ref(
                    s,
                    base_dir,
                    &seen_ids,
                    &image_vars,
                    &raw_step.id,
                    "mask",
                )?)),
            };
            StepKind::Edit(EditStep {
                sources,
                prompt: edit_prompt.clone(),
                mask,
                blend: raw_step.blend,
                model: raw_step.model.clone(),
                lora: raw_step.lora.clone(),
                fast: raw_step.fast,
                seed: raw_step.seed.or(raw.defaults.seed),
                seeds: raw_step.seeds.clone(),
                width: raw_step.width.or(raw.defaults.width),
                height: raw_step.height.or(raw.defaults.height),
                steps: raw_step.steps.or(default_steps),
                guidance: raw_step.guidance.or(default_guidance),
                count: raw_step.count.or(raw.defaults.count),
            })
        };

        materialized.push(Step {
            id: raw_step.id.clone(),
            kind,
        });
    }

    Ok(Workflow {
        name: raw.name,
        model: raw.model,
        lora: raw.lora,
        steps: materialized,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_valid_id(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Validate seed-related fields on a single step.
///
/// Rules:
/// - `seeds: []` (empty list) is rejected — must have at least one seed.
/// - `seeds` + `seed` both set is rejected — ambiguous.
/// - `seeds` + `count` both set is rejected — `seeds.len()` is the count.
/// - `seed` alone or `seed` + `count` is allowed (existing worker behavior).
/// - Neither is allowed (model/defaults take over).
fn validate_seeds(
    step_id: &str,
    seeds: &Option<Vec<u64>>,
    seed: Option<u64>,
    count: Option<u32>,
) -> Result<()> {
    let Some(seeds) = seeds else {
        return Ok(());
    };
    if seeds.is_empty() {
        bail!(
            "step `{step_id}`: `seeds: []` is empty — provide at least one seed, or remove the field"
        );
    }
    if seed.is_some() {
        bail!(
            "step `{step_id}`: cannot set both `seed` and `seeds` — pick one (use `seeds` for variation exploration, `seed` + `count` for noise variation at a fixed seed)"
        );
    }
    if count.is_some() {
        bail!(
            "step `{step_id}`: cannot set both `seeds` and `count` — the length of `seeds` is the count"
        );
    }
    Ok(())
}

/// Pre-process the top-level `images:` map: decode/resolve each value and write
/// it to `base_dir/{name}-ref.{ext}` so every step can reference it by name.
fn resolve_image_vars(
    images: &HashMap<String, String>,
    base_dir: &Path,
) -> Result<HashMap<String, PathBuf>> {
    let mut vars = HashMap::with_capacity(images.len());
    for (name, value) in images {
        if !is_valid_id(name) {
            bail!("images: variable name `{name}` must contain only letters, digits, `-`, `_`");
        }
        let path = if value.starts_with("data:image/") {
            let (header, data) = value
                .split_once(',')
                .ok_or_else(|| anyhow!("images.{name}: data URI is missing the comma separator"))?;
            let ext = header
                .trim_start_matches("data:image/")
                .split_once(';')
                .map(|(t, _)| t)
                .unwrap_or("png");
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data)
                .with_context(|| format!("images.{name}: failed to decode base64 data"))?;
            let out_path = base_dir.join(format!("{name}-ref.{ext}"));
            std::fs::write(&out_path, &bytes).with_context(|| {
                format!("images.{name}: failed to write to `{}`", out_path.display())
            })?;
            out_path
        } else {
            let p = if Path::new(value).is_absolute() {
                PathBuf::from(value)
            } else {
                base_dir.join(value)
            };
            if !p.exists() {
                bail!("images.{name}: file `{}` does not exist", p.display());
            }
            p
        };
        vars.insert(name.clone(), path);
    }
    Ok(vars)
}

/// Parse an image reference (edit source, init image, mask, controlnet or
/// style-ref input).
///
/// Four forms:
/// - `$name` (no `.outputs[`) → named image variable from the top-level `images:` map
/// - `$step-id.outputs[N]` → resolved at runtime to the Nth output of an earlier step
/// - `data:image/...;base64,...` → decoded and written to `base_dir/{step}-{slot}.{ext}`
/// - Anything else → filesystem path relative to `base_dir`, must exist
///
/// `slot` names the input position (`source`, `init`, `mask`, `cn-1`, …) so
/// inline data URIs in different slots of one step get distinct filenames.
fn parse_image_ref(
    s: &str,
    base_dir: &Path,
    earlier_step_ids: &HashSet<String>,
    image_vars: &HashMap<String, PathBuf>,
    current_step_id: &str,
    slot: &str,
) -> Result<ImageRef> {
    if let Some(rest) = s.strip_prefix('$') {
        // `$name` with no dot → image variable ref.
        // `$step-id.outputs[N]` → step-output ref.
        // Anything else (e.g. `$a.outputs` missing `[N]`) → malformed, caught below.
        if !rest.contains('.') {
            let path = image_vars.get(rest).ok_or_else(|| {
                anyhow!(
                    "step `{current_step_id}`: image variable `{s}` is not defined — add `{rest}:` to the top-level `images:` map"
                )
            })?;
            return Ok(ImageRef::Local(path.clone()));
        }

        let (step_id, bracket_part) = rest.split_once(".outputs[").ok_or_else(|| {
            anyhow!(
                "step `{current_step_id}`: invalid step-output ref `{s}` — expected `$step-id.outputs[N]`"
            )
        })?;
        let index_str = bracket_part.strip_suffix(']').ok_or_else(|| {
            anyhow!("step `{current_step_id}`: invalid step-output ref `{s}` — missing closing `]`")
        })?;
        let index: usize = index_str.parse().map_err(|_| {
            anyhow!(
                "step `{current_step_id}`: invalid step-output ref `{s}` — `{index_str}` is not a valid index"
            )
        })?;

        if step_id == current_step_id {
            bail!("step `{current_step_id}`: cannot reference its own outputs via `{s}`");
        }
        // Note: earlier_step_ids is built as we iterate, so this naturally
        // rejects forward references and self-references.
        if !earlier_step_ids.contains(step_id) {
            bail!(
                "step `{current_step_id}`: step-output ref `{s}` points to unknown or later step `{step_id}` (only earlier steps can be referenced)"
            );
        }
        Ok(ImageRef::StepOutput {
            step_id: step_id.to_string(),
            index,
        })
    } else if s.starts_with("data:image/") {
        // Base64 inline image — decode and materialise to disk next to the spec file.
        // Enables passing images from a remote client through the MCP run_workflow tool.
        let (header, data) = s.split_once(',').ok_or_else(|| {
            anyhow!(
                "step `{current_step_id}`: inline image `data:...` is missing the comma separator"
            )
        })?;
        let ext = header
            .trim_start_matches("data:image/")
            .split_once(';')
            .map(|(t, _)| t)
            .unwrap_or("png");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .with_context(|| {
                format!("step `{current_step_id}`: failed to decode base64 inline image")
            })?;
        let out_path = base_dir.join(format!("{current_step_id}-{slot}.{ext}"));
        std::fs::write(&out_path, &bytes).with_context(|| {
            format!(
                "step `{current_step_id}`: failed to write inline image to `{}`",
                out_path.display()
            )
        })?;
        Ok(ImageRef::Local(out_path))
    } else {
        let path = if Path::new(s).is_absolute() {
            PathBuf::from(s)
        } else {
            base_dir.join(s)
        };
        if !path.exists() {
            bail!(
                "step `{current_step_id}`: local image `{}` does not exist",
                path.display()
            );
        }
        Ok(ImageRef::Local(path))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn parse(yaml: &str) -> Result<Workflow> {
        parse_str(yaml, Path::new("."))
    }

    #[test]
    fn parses_minimal_workflow() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "a cat"
"#;
        let wf = parse(yaml).unwrap();
        assert_eq!(wf.name, "test");
        assert_eq!(wf.model, "flux-dev");
        assert_eq!(wf.steps.len(), 1);
        assert_eq!(wf.steps[0].id, "a");
        match &wf.steps[0].kind {
            StepKind::Generate(g) => assert_eq!(g.prompt, "a cat"),
            _ => panic!("expected generate"),
        }
    }

    #[test]
    fn defaults_inherited_by_steps() {
        let yaml = r#"
name: test
model: flux-dev
defaults:
  seed: 42
  width: 512
  height: 512
  steps: 28
  guidance: 3.5
  count: 2
steps:
  - id: a
    generate: "a cat"
"#;
        let wf = parse(yaml).unwrap();
        match &wf.steps[0].kind {
            StepKind::Generate(g) => {
                assert_eq!(g.seed, Some(42));
                assert_eq!(g.width, Some(512));
                assert_eq!(g.height, Some(512));
                assert_eq!(g.steps, Some(28));
                assert_eq!(g.guidance, Some(3.5));
                assert_eq!(g.count, Some(2));
            }
            _ => panic!("expected generate"),
        }
    }

    #[test]
    fn step_overrides_default() {
        let yaml = r#"
name: test
model: flux-dev
defaults:
  seed: 42
steps:
  - id: a
    generate: "a cat"
    seed: 99
"#;
        let wf = parse(yaml).unwrap();
        match &wf.steps[0].kind {
            StepKind::Generate(g) => assert_eq!(g.seed, Some(99)),
            _ => panic!("expected generate"),
        }
    }

    #[test]
    fn model_override_skips_workflow_defaults_for_steps_and_guidance() {
        let yaml = r#"
name: test
model: flux-dev
defaults:
  steps: 28
  guidance: 3.5
  width: 512
  height: 512
steps:
  - id: a
    generate: "a cat"
  - id: b
    model: z-image-turbo
    generate: "a dog"
"#;
        let wf = parse(yaml).unwrap();
        // Step without model override inherits steps/guidance from defaults
        match &wf.steps[0].kind {
            StepKind::Generate(g) => {
                assert_eq!(g.steps, Some(28));
                assert_eq!(g.guidance, Some(3.5));
                assert_eq!(g.width, Some(512));
            }
            _ => panic!("expected generate"),
        }
        // Step with model override does NOT inherit steps/guidance from defaults
        // (runner will fall back to model-specific defaults from models.toml)
        // but still inherits model-independent fields like width/height.
        match &wf.steps[1].kind {
            StepKind::Generate(g) => {
                assert_eq!(g.steps, None);
                assert_eq!(g.guidance, None);
                assert_eq!(g.width, Some(512));
                assert_eq!(g.height, Some(512));
            }
            _ => panic!("expected generate"),
        }
    }

    #[test]
    fn duplicate_step_ids_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "cat"
  - id: a
    generate: "dog"
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("duplicate step id"), "got: {err}");
    }

    #[test]
    fn missing_model_rejected() {
        let yaml = r#"
name: test
model: ""
steps:
  - id: a
    generate: "cat"
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("model"), "got: {err}");
    }

    #[test]
    fn empty_steps_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps: []
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("at least one step"), "got: {err}");
    }

    #[test]
    fn both_generate_and_edit_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "cat"
    edit: "./input.png"
    prompt: "add rain"
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("both"), "got: {err}");
    }

    #[test]
    fn neither_generate_nor_edit_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    prompt: "hi"
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(
            err.contains("generate") || err.contains("edit"),
            "got: {err}"
        );
    }

    #[test]
    fn edit_without_prompt_rejected() {
        let tmp = TempDir::new().unwrap();
        let img = tmp.path().join("input.png");
        std::fs::write(&img, b"fake").unwrap();
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    edit: "input.png"
"#;
        let err = parse_str(yaml, tmp.path()).unwrap_err().to_string();
        assert!(err.contains("prompt"), "got: {err}");
    }

    #[test]
    fn step_ref_to_earlier_step_works() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: scene
    generate: "cat"
  - id: rain
    edit: "$scene.outputs[0]"
    prompt: "add rain"
"#;
        let wf = parse(yaml).unwrap();
        match &wf.steps[1].kind {
            StepKind::Edit(e) => match &e.sources[0] {
                ImageRef::StepOutput { step_id, index } => {
                    assert_eq!(step_id, "scene");
                    assert_eq!(*index, 0);
                }
                _ => panic!("expected step output ref"),
            },
            _ => panic!("expected edit"),
        }
    }

    #[test]
    fn step_ref_to_unknown_step_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    edit: "$ghost.outputs[0]"
    prompt: "hi"
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("unknown or later"), "got: {err}");
    }

    #[test]
    fn step_ref_to_later_step_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    edit: "$b.outputs[0]"
    prompt: "hi"
  - id: b
    generate: "cat"
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("unknown or later"), "got: {err}");
    }

    #[test]
    fn step_ref_to_self_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    edit: "$a.outputs[0]"
    prompt: "hi"
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(
            err.contains("own outputs") || err.contains("unknown or later"),
            "got: {err}"
        );
    }

    #[test]
    fn local_path_that_doesnt_exist_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    edit: "./nope.png"
    prompt: "hi"
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("does not exist"), "got: {err}");
    }

    #[test]
    fn local_path_that_exists_resolved_to_absolute() {
        let tmp = TempDir::new().unwrap();
        let img = tmp.path().join("input.png");
        std::fs::write(&img, b"fake").unwrap();
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    edit: "input.png"
    prompt: "hi"
"#;
        let wf = parse_str(yaml, tmp.path()).unwrap();
        match &wf.steps[0].kind {
            StepKind::Edit(e) => match &e.sources[0] {
                ImageRef::Local(p) => assert!(p.ends_with("input.png")),
                _ => panic!("expected local ref"),
            },
            _ => panic!("expected edit"),
        }
    }

    #[test]
    fn invalid_id_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: "bad id with spaces"
    generate: "cat"
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("id"), "got: {err}");
    }

    #[test]
    fn seeds_list_parsed() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "cat"
    seeds: [42, 7, 99]
"#;
        let wf = parse(yaml).unwrap();
        match &wf.steps[0].kind {
            StepKind::Generate(g) => {
                assert_eq!(g.seeds, Some(vec![42, 7, 99]));
                assert_eq!(g.seed, None);
                assert_eq!(g.count, None);
            }
            _ => panic!("expected generate"),
        }
    }

    #[test]
    fn seeds_empty_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "cat"
    seeds: []
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn seeds_plus_seed_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "cat"
    seed: 42
    seeds: [7, 99]
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("both `seed` and `seeds`"), "got: {err}");
    }

    #[test]
    fn seeds_plus_count_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "cat"
    count: 4
    seeds: [7, 99]
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("both `seeds` and `count`"), "got: {err}");
    }

    #[test]
    fn seeds_on_edit_step_allowed() {
        let tmp = TempDir::new().unwrap();
        let img = tmp.path().join("input.png");
        std::fs::write(&img, b"fake").unwrap();
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    edit: "input.png"
    prompt: "add rain"
    seeds: [1, 2, 3]
"#;
        let wf = parse_str(yaml, tmp.path()).unwrap();
        match &wf.steps[0].kind {
            StepKind::Edit(e) => assert_eq!(e.seeds, Some(vec![1, 2, 3])),
            _ => panic!("expected edit"),
        }
    }

    #[test]
    fn per_step_model_parsed() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "cat"
  - id: b
    model: qwen-image
    generate: "dog"
"#;
        let wf = parse(yaml).unwrap();
        match &wf.steps[0].kind {
            StepKind::Generate(g) => assert_eq!(g.model, None),
            _ => panic!("expected generate"),
        }
        match &wf.steps[1].kind {
            StepKind::Generate(g) => assert_eq!(g.model.as_deref(), Some("qwen-image")),
            _ => panic!("expected generate"),
        }
    }

    #[test]
    fn per_step_lora_parsed() {
        let yaml = r#"
name: test
model: flux-dev
lora: default-lora
steps:
  - id: a
    generate: "cat"
  - id: b
    lora: override-lora
    generate: "dog"
"#;
        let wf = parse(yaml).unwrap();
        match &wf.steps[0].kind {
            StepKind::Generate(g) => assert_eq!(g.lora, None),
            _ => panic!(),
        }
        match &wf.steps[1].kind {
            StepKind::Generate(g) => assert_eq!(g.lora.as_deref(), Some("override-lora")),
            _ => panic!(),
        }
    }

    #[test]
    fn per_step_fast_parsed() {
        let yaml = r#"
name: test
model: qwen-image
steps:
  - id: a
    generate: "cat"
    fast: 4
  - id: b
    generate: "dog"
"#;
        let wf = parse(yaml).unwrap();
        match &wf.steps[0].kind {
            StepKind::Generate(g) => assert_eq!(g.fast, Some(4)),
            _ => panic!(),
        }
        match &wf.steps[1].kind {
            StepKind::Generate(g) => assert_eq!(g.fast, None),
            _ => panic!(),
        }
    }

    #[test]
    fn fast_and_lora_on_same_step_rejected() {
        let yaml = r#"
name: test
model: qwen-image
steps:
  - id: a
    generate: "cat"
    fast: 4
    lora: my-lora
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(
            err.contains("cannot set both `fast` and `lora`"),
            "got: {err}"
        );
    }

    #[test]
    fn empty_step_model_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    model: ""
    generate: "cat"
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("`model` is set but empty"), "got: {err}");
    }

    #[test]
    fn empty_step_lora_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    lora: ""
    generate: "cat"
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("`lora` is set but empty"), "got: {err}");
    }

    #[test]
    fn per_step_model_on_edit() {
        let tmp = TempDir::new().unwrap();
        let img = tmp.path().join("input.png");
        std::fs::write(&img, b"fake").unwrap();
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    model: qwen-image-edit-2511
    edit: "input.png"
    prompt: "add rain"
"#;
        let wf = parse_str(yaml, tmp.path()).unwrap();
        match &wf.steps[0].kind {
            StepKind::Edit(e) => {
                assert_eq!(e.model.as_deref(), Some("qwen-image-edit-2511"));
                assert_eq!(e.lora, None);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn malformed_step_ref_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "cat"
  - id: b
    edit: "$a.outputs"
    prompt: "hi"
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("step-output ref"), "got: {err}");
    }

    #[test]
    fn inline_base64_image_materialised() {
        use std::io::Read;
        // 1×1 red pixel PNG, base64-encoded.
        let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI6QAAAABJRU5ErkJggg==";
        let tmp = tempfile::tempdir().unwrap();
        let yaml = format!(
            "name: test\nmodel: flux-dev\nsteps:\n  - id: retouch\n    edit: \"data:image/png;base64,{b64}\"\n    prompt: \"fix it\"\n"
        );
        let wf = parse_str(&yaml, tmp.path()).unwrap();
        match &wf.steps[0].kind {
            StepKind::Edit(e) => match &e.sources[0] {
                ImageRef::Local(p) => {
                    assert!(p.exists(), "inline image should be written to disk");
                    assert!(p.to_string_lossy().ends_with("retouch-source.png"));
                    let mut bytes = Vec::new();
                    std::fs::File::open(p)
                        .unwrap()
                        .read_to_end(&mut bytes)
                        .unwrap();
                    assert!(!bytes.is_empty());
                }
                _ => panic!("expected Local after inline decode"),
            },
            _ => panic!("expected edit step"),
        }
    }

    // 1×1 red pixel PNG used across image-var tests.
    const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI6QAAAABJRU5ErkJggg==";

    #[test]
    fn image_var_base64_resolved_by_name() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = format!(
            "name: test\nmodel: flux-dev\nimages:\n  alice: \"data:image/png;base64,{RED_PNG_B64}\"\nsteps:\n  - id: s1\n    edit: \"$alice\"\n    prompt: \"fix it\"\n"
        );
        let wf = parse_str(&yaml, tmp.path()).unwrap();
        match &wf.steps[0].kind {
            StepKind::Edit(e) => match &e.sources[0] {
                ImageRef::Local(p) => {
                    assert!(p.exists());
                    assert!(p.to_string_lossy().ends_with("alice-ref.png"));
                }
                _ => panic!("expected Local"),
            },
            _ => panic!("expected edit"),
        }
    }

    #[test]
    fn image_var_reused_across_steps_same_path() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = format!(
            "name: test\nmodel: flux-dev\nimages:\n  hero: \"data:image/png;base64,{RED_PNG_B64}\"\nsteps:\n  - id: s1\n    edit: \"$hero\"\n    prompt: \"scene 1\"\n  - id: s2\n    edit: \"$hero\"\n    prompt: \"scene 2\"\n"
        );
        let wf = parse_str(&yaml, tmp.path()).unwrap();
        let path0 = match &wf.steps[0].kind {
            StepKind::Edit(e) => match &e.sources[0] {
                ImageRef::Local(p) => p.clone(),
                _ => panic!(),
            },
            _ => panic!(),
        };
        let path1 = match &wf.steps[1].kind {
            StepKind::Edit(e) => match &e.sources[0] {
                ImageRef::Local(p) => p.clone(),
                _ => panic!(),
            },
            _ => panic!(),
        };
        assert_eq!(path0, path1, "both steps should point to the same file");
    }

    #[test]
    fn multi_image_edit_parses() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("a.png"), b"fake").unwrap();
        let yaml = format!(
            "name: test\nmodel: klein-9b\nimages:\n  style: \"data:image/png;base64,{RED_PNG_B64}\"\nsteps:\n  - id: base\n    generate: \"a cat\"\n  - id: combine\n    edit: [\"$base.outputs[0]\", \"$style\", \"a.png\"]\n    prompt: \"blend\"\n"
        );
        let wf = parse_str(&yaml, tmp.path()).unwrap();
        match &wf.steps[1].kind {
            StepKind::Edit(e) => {
                assert_eq!(e.sources.len(), 3);
                assert!(
                    matches!(&e.sources[0], ImageRef::StepOutput { step_id, index: 0 } if step_id == "base")
                );
                assert!(
                    matches!(&e.sources[1], ImageRef::Local(p) if p.ends_with("style-ref.png"))
                );
                assert!(matches!(&e.sources[2], ImageRef::Local(p) if p.ends_with("a.png")));
            }
            _ => panic!("expected edit"),
        }
    }

    #[test]
    fn empty_edit_list_rejected() {
        let yaml = r#"
name: test
model: klein-9b
steps:
  - id: a
    edit: []
    prompt: "hi"
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("at least one image"), "got: {err}");
    }

    #[test]
    fn edit_mask_sentinels_and_blend_parsed() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("in.png"), b"fake").unwrap();
        std::fs::write(tmp.path().join("m.png"), b"fake").unwrap();
        let yaml = r#"
name: test
model: klein-9b
steps:
  - id: a
    edit: "in.png"
    prompt: "hi"
    mask: "from-alpha"
    blend: latent
  - id: b
    edit: "in.png"
    prompt: "hi"
    mask: "m.png"
"#;
        let wf = parse_str(yaml, tmp.path()).unwrap();
        match &wf.steps[0].kind {
            StepKind::Edit(e) => {
                assert!(matches!(e.mask, Some(EditMask::FromAlpha)));
                assert_eq!(e.blend, Some(crate::core::job::BlendMode::Latent));
            }
            _ => panic!(),
        }
        match &wf.steps[1].kind {
            StepKind::Edit(e) => {
                assert!(
                    matches!(&e.mask, Some(EditMask::Image(ImageRef::Local(p))) if p.ends_with("m.png"))
                );
                assert_eq!(e.blend, None);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn generate_init_image_mask_strength_parsed() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("photo.png"), b"fake").unwrap();
        std::fs::write(tmp.path().join("m.png"), b"fake").unwrap();
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "a cat"
    init_image: "photo.png"
    mask: "m.png"
    strength: 0.6
"#;
        let wf = parse_str(yaml, tmp.path()).unwrap();
        match &wf.steps[0].kind {
            StepKind::Generate(g) => {
                assert!(
                    matches!(&g.init_image, Some(ImageRef::Local(p)) if p.ends_with("photo.png"))
                );
                assert!(matches!(&g.mask, Some(ImageRef::Local(p)) if p.ends_with("m.png")));
                assert_eq!(g.strength, Some(0.6));
            }
            _ => panic!("expected generate"),
        }
    }

    #[test]
    fn generate_init_image_from_step_output() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: base
    generate: "a cat"
  - id: refine
    generate: "a detailed cat"
    init_image: "$base.outputs[0]"
    strength: 0.5
"#;
        let wf = parse(yaml).unwrap();
        match &wf.steps[1].kind {
            StepKind::Generate(g) => {
                assert!(
                    matches!(&g.init_image, Some(ImageRef::StepOutput { step_id, index: 0 }) if step_id == "base")
                );
            }
            _ => panic!("expected generate"),
        }
        // init_image contributes to image_refs (dependency scheduling)
        assert_eq!(wf.steps[1].image_refs().len(), 1);
    }

    #[test]
    fn generate_mask_without_init_image_rejected() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("m.png"), b"fake").unwrap();
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "a cat"
    mask: "m.png"
"#;
        let err = parse_str(yaml, tmp.path()).unwrap_err().to_string();
        assert!(err.contains("requires `init_image`"), "got: {err}");
    }

    #[test]
    fn generate_strength_without_init_image_rejected() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "a cat"
    strength: 0.5
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("requires `init_image`"), "got: {err}");
    }

    #[test]
    fn controlnet_explicit_type_and_filename_detection() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("ref_pose.png"), b"fake").unwrap();
        std::fs::write(tmp.path().join("edges.png"), b"fake").unwrap();
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "a cat"
    controlnet:
      - image: "ref_pose.png"
      - image: "edges.png"
        type: canny
        strength: 0.9
        end: 0.7
"#;
        let wf = parse_str(yaml, tmp.path()).unwrap();
        match &wf.steps[0].kind {
            StepKind::Generate(g) => {
                assert_eq!(g.controlnet.len(), 2);
                assert_eq!(g.controlnet[0].control_type, "pose");
                assert_eq!(g.controlnet[1].control_type, "canny");
                assert_eq!(g.controlnet[1].strength, Some(0.9));
                assert_eq!(g.controlnet[1].end, Some(0.7));
            }
            _ => panic!("expected generate"),
        }
    }

    #[test]
    fn controlnet_var_ref_requires_explicit_type() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = format!(
            "name: test\nmodel: flux-dev\nimages:\n  pose: \"data:image/png;base64,{RED_PNG_B64}\"\nsteps:\n  - id: a\n    generate: \"a cat\"\n    controlnet:\n      - image: \"$pose\"\n"
        );
        let err = parse_str(&yaml, tmp.path()).unwrap_err().to_string();
        assert!(err.contains("`type:` is required"), "got: {err}");
    }

    #[test]
    fn controlnet_undetectable_filename_rejected() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("photo.png"), b"fake").unwrap();
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "a cat"
    controlnet:
      - image: "photo.png"
"#;
        let err = parse_str(yaml, tmp.path()).unwrap_err().to_string();
        assert!(err.contains("cannot auto-detect"), "got: {err}");
    }

    #[test]
    fn more_than_two_controlnets_rejected() {
        let tmp = TempDir::new().unwrap();
        for n in ["a_pose.png", "b_canny.png", "c_depth.png"] {
            std::fs::write(tmp.path().join(n), b"fake").unwrap();
        }
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "a cat"
    controlnet:
      - image: "a_pose.png"
      - image: "b_canny.png"
      - image: "c_depth.png"
"#;
        let err = parse_str(yaml, tmp.path()).unwrap_err().to_string();
        assert!(err.contains("maximum 2"), "got: {err}");
    }

    #[test]
    fn style_ref_parsed_with_var() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = format!(
            "name: test\nmodel: sdxl\nimages:\n  mood: \"data:image/png;base64,{RED_PNG_B64}\"\nsteps:\n  - id: a\n    generate: \"a cat\"\n    style_ref:\n      - image: \"$mood\"\n        strength: 0.8\n"
        );
        let wf = parse_str(&yaml, tmp.path()).unwrap();
        match &wf.steps[0].kind {
            StepKind::Generate(g) => {
                assert_eq!(g.style_ref.len(), 1);
                assert!(
                    matches!(&g.style_ref[0].image, ImageRef::Local(p) if p.ends_with("mood-ref.png"))
                );
                assert_eq!(g.style_ref[0].strength, Some(0.8));
            }
            _ => panic!("expected generate"),
        }
    }

    #[test]
    fn generate_only_fields_rejected_on_edit() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("in.png"), b"fake").unwrap();
        std::fs::write(tmp.path().join("i.png"), b"fake").unwrap();
        let yaml = r#"
name: test
model: klein-9b
steps:
  - id: a
    edit: "in.png"
    prompt: "hi"
    init_image: "i.png"
"#;
        let err = parse_str(yaml, tmp.path()).unwrap_err().to_string();
        assert!(err.contains("only valid on generate steps"), "got: {err}");
    }

    #[test]
    fn blend_rejected_on_generate() {
        let yaml = r#"
name: test
model: flux-dev
steps:
  - id: a
    generate: "a cat"
    blend: latent
"#;
        let err = parse(yaml).unwrap_err().to_string();
        assert!(err.contains("only valid on edit steps"), "got: {err}");
    }

    #[test]
    fn inline_data_uri_slots_get_distinct_filenames() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = format!(
            "name: test\nmodel: klein-9b\nsteps:\n  - id: mix\n    edit: [\"data:image/png;base64,{RED_PNG_B64}\", \"data:image/png;base64,{RED_PNG_B64}\"]\n    prompt: \"hi\"\n    mask: \"data:image/png;base64,{RED_PNG_B64}\"\n"
        );
        let wf = parse_str(&yaml, tmp.path()).unwrap();
        match &wf.steps[0].kind {
            StepKind::Edit(e) => {
                assert!(
                    matches!(&e.sources[0], ImageRef::Local(p) if p.ends_with("mix-source.png"))
                );
                assert!(
                    matches!(&e.sources[1], ImageRef::Local(p) if p.ends_with("mix-source-2.png"))
                );
                assert!(
                    matches!(&e.mask, Some(EditMask::Image(ImageRef::Local(p))) if p.ends_with("mix-mask.png"))
                );
            }
            _ => panic!("expected edit"),
        }
    }

    #[test]
    fn image_var_undefined_gives_clear_error() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = "name: test\nmodel: flux-dev\nsteps:\n  - id: s1\n    edit: \"$nobody\"\n    prompt: \"hi\"\n";
        let err = parse_str(yaml, tmp.path()).unwrap_err().to_string();
        assert!(err.contains("nobody"), "got: {err}");
        assert!(err.contains("images:"), "got: {err}");
    }
}
