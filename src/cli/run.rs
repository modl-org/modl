//! `modl run <workflow.yaml>` — execute a workflow spec locally.
//!
//! Phase B of the workflow-run plan. See `docs/plans/workflow-run.md`.
//!
//! Sequential execution: parse + validate spec, resolve model/lora once upfront,
//! then iterate steps, materializing each into a GenerateJobSpec/EditJobSpec and
//! handing it to LocalExecutor. Outputs of step N are available as inputs to
//! step N+1 via the `$step-id.outputs[N]` reference.

use anyhow::{Context, Result, anyhow, bail};
use console::style;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use crate::core::db::Database;
use crate::core::executor::{Executor, LocalExecutor};
use crate::core::job::{
    EditJobSpec, EditParams, EventPayload, ExecutionTarget, GenerateJobSpec, GenerateOutputRef,
    GenerateParams, JobEvent, LoraRef, ModelRef, RuntimeRef,
};
use crate::core::workflow::{EditStep, GenerateStep, ImageRef, StepKind, parse_file};
use crate::core::{model_family, model_resolve, paths, registry, runtime};

pub async fn run(spec_path: &str, _auto_pull: bool) -> Result<()> {
    // -------------------------------------------------------------------
    // 1. Parse + validate
    // -------------------------------------------------------------------
    let wf = parse_file(Path::new(spec_path))
        .with_context(|| format!("Failed to load workflow from {spec_path}"))?;

    println!(
        "{} Running workflow {} ({} step{})",
        style("→").cyan(),
        style(&wf.name).bold(),
        wf.steps.len(),
        if wf.steps.len() == 1 { "" } else { "s" }
    );
    println!("  Model: {}", wf.model);
    if let Some(ref l) = wf.lora {
        println!("  LoRA:  {}", l);
    }

    // -------------------------------------------------------------------
    // 2. Resolve model + lora once upfront (fail fast before any step runs)
    // -------------------------------------------------------------------
    let db = Database::open()?;
    let spec_model_id = resolve_registry_model_id(&wf.model);
    let base_model_path = model_resolve::resolve_base_model_path(&spec_model_id, &db);
    if base_model_path.is_none() {
        bail!(
            "Model `{}` is not installed locally. Run `modl pull {}`.",
            spec_model_id,
            spec_model_id
        );
    }
    let arch_key = model_family::find_model(&spec_model_id).map(|m| m.arch_key.to_string());

    let lora_ref = match wf.lora.as_deref() {
        Some(name) => Some(
            model_resolve::resolve_lora(name, 1.0, &db)?
                .ok_or_else(|| anyhow!("LoRA `{name}` not found in local store"))?,
        ),
        None => None,
    };

    // -------------------------------------------------------------------
    // 3. Output directory — flat under today's date so `modl serve` sees
    // workflow outputs alongside regular `modl generate` outputs. Per-step
    // grouping is preserved in DB metadata via job labels, not the filesystem.
    // -------------------------------------------------------------------
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let output_dir = paths::modl_root().join("outputs").join(&date);
    std::fs::create_dir_all(&output_dir)?;

    // A stable run id used only for grouping jobs in the DB (labels).
    let run_id = format!(
        "{}-{}",
        chrono::Local::now().format("%Y%m%d-%H%M%S"),
        sanitize_for_path(&wf.name)
    );
    println!("  Output: {}", output_dir.display());
    println!("  Run ID: {}\n", run_id);

    // -------------------------------------------------------------------
    // 4. Bootstrap executor (shared across all steps to reuse the worker)
    // -------------------------------------------------------------------
    println!("{} Preparing runtime...", style("→").cyan());
    let mut executor = LocalExecutor::for_generation().await?;

    // -------------------------------------------------------------------
    // 5. Execute steps sequentially
    // -------------------------------------------------------------------
    let mut step_outputs: HashMap<String, Vec<PathBuf>> = HashMap::new();
    let total_steps = wf.steps.len();

    for (idx, step) in wf.steps.iter().enumerate() {
        println!(
            "\n{} [{}/{}] {} ({})",
            style("▸").cyan().bold(),
            idx + 1,
            total_steps,
            style(&step.id).bold(),
            step_kind_label(&step.kind),
        );

        let step_labels = build_step_labels(&wf.name, &run_id, &step.id);

        let mut step_artifacts: Vec<PathBuf> = Vec::new();

        match &step.kind {
            StepKind::Generate(g) => {
                let sub_jobs = expand_seeds(g.seed, g.count, &g.seeds);
                print_generate_preview(g, sub_jobs.len());
                for (sub_idx, (seed, count)) in sub_jobs.iter().enumerate() {
                    if sub_jobs.len() > 1 {
                        println!(
                            "  {} [{}/{}] seed={}",
                            style("·").dim(),
                            sub_idx + 1,
                            sub_jobs.len(),
                            seed.map(|s| s.to_string())
                                .unwrap_or_else(|| "?".to_string()),
                        );
                    }
                    let spec = build_generate_spec(
                        &wf.model,
                        &spec_model_id,
                        base_model_path.clone(),
                        arch_key.clone(),
                        lora_ref.clone(),
                        g,
                        *seed,
                        *count,
                        &output_dir,
                        step_labels.clone(),
                    );
                    let artifacts = execute_generate_step(&mut executor, &spec, &step.id, &db)?;
                    step_artifacts.extend(artifacts);
                }
            }
            StepKind::Edit(e) => {
                let source_path = resolve_image_ref(&e.source, &step_outputs, &step.id)?;
                let sub_jobs = expand_seeds(e.seed, e.count, &e.seeds);
                print_edit_preview(e, &source_path, sub_jobs.len());
                for (sub_idx, (seed, count)) in sub_jobs.iter().enumerate() {
                    if sub_jobs.len() > 1 {
                        println!(
                            "  {} [{}/{}] seed={}",
                            style("·").dim(),
                            sub_idx + 1,
                            sub_jobs.len(),
                            seed.map(|s| s.to_string())
                                .unwrap_or_else(|| "?".to_string()),
                        );
                    }
                    let spec = build_edit_spec(
                        &wf.model,
                        &spec_model_id,
                        base_model_path.clone(),
                        arch_key.clone(),
                        lora_ref.clone(),
                        e,
                        &source_path,
                        *seed,
                        *count,
                        &output_dir,
                        step_labels.clone(),
                    );
                    let artifacts = execute_edit_step(&mut executor, &spec, &step.id, &db)?;
                    step_artifacts.extend(artifacts);
                }
            }
        };

        if step_artifacts.is_empty() {
            bail!("step `{}` produced no artifacts", step.id);
        }

        println!(
            "  {} {} artifact{}",
            style("✓").green(),
            step_artifacts.len(),
            if step_artifacts.len() == 1 { "" } else { "s" }
        );
        step_outputs.insert(step.id.clone(), step_artifacts);
    }

    // -------------------------------------------------------------------
    // 6. Summary
    // -------------------------------------------------------------------
    let total_artifacts: usize = step_outputs.values().map(|v| v.len()).sum();
    println!(
        "\n{} workflow {} complete: {} step{}, {} artifact{}",
        style("✓").green().bold(),
        style(&wf.name).bold(),
        total_steps,
        if total_steps == 1 { "" } else { "s" },
        total_artifacts,
        if total_artifacts == 1 { "" } else { "s" },
    );
    println!("  {}", output_dir.display());

    Ok(())
}

/// Labels attached to every step's job row so the UI can group workflow outputs.
fn build_step_labels(workflow_name: &str, run_id: &str, step_id: &str) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    labels.insert("workflow".to_string(), workflow_name.to_string());
    labels.insert("workflow_run".to_string(), run_id.to_string());
    labels.insert("workflow_step".to_string(), step_id.to_string());
    labels
}

/// Expand a step's seed/count/seeds fields into a list of concrete
/// (seed, count) sub-jobs. When `seeds` is set, one sub-job per seed with
/// count=1. Otherwise a single sub-job that delegates count to the worker
/// (existing behavior — worker varies internal noise at a fixed seed).
fn expand_seeds(
    seed: Option<u64>,
    count: Option<u32>,
    seeds: &Option<Vec<u64>>,
) -> Vec<(Option<u64>, u32)> {
    if let Some(seeds) = seeds {
        seeds.iter().map(|s| (Some(*s), 1u32)).collect()
    } else {
        vec![(seed, count.unwrap_or(1))]
    }
}

// ---------------------------------------------------------------------------
// Spec builders
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_generate_spec(
    wf_model: &str,
    spec_model_id: &str,
    base_model_path: Option<String>,
    arch_key: Option<String>,
    lora: Option<LoraRef>,
    step: &GenerateStep,
    seed: Option<u64>,
    count: u32,
    output_dir: &Path,
    labels: HashMap<String, String>,
) -> GenerateJobSpec {
    let (default_steps, default_guidance) = model_family::model_defaults(wf_model);
    let steps = step.steps.unwrap_or(default_steps);
    let guidance = step.guidance.unwrap_or(default_guidance);
    let width = step.width.unwrap_or(1024);
    let height = step.height.unwrap_or(1024);

    GenerateJobSpec {
        prompt: step.prompt.clone(),
        model: ModelRef {
            base_model_id: spec_model_id.to_string(),
            base_model_path: base_model_path.clone(),
            arch_key: arch_key.clone(),
        },
        lora,
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
            init_image: None,
            mask: None,
            strength: None,
            scheduler_overrides: HashMap::new(),
            controlnet: Vec::new(),
            style_ref: Vec::new(),
            inpaint_method: None,
        },
        runtime: RuntimeRef {
            profile: runtime::resolved_generation_profile().to_string(),
            python_version: Some("3.11.12".to_string()),
        },
        target: ExecutionTarget::Local,
        labels,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_edit_spec(
    wf_model: &str,
    spec_model_id: &str,
    base_model_path: Option<String>,
    arch_key: Option<String>,
    lora: Option<LoraRef>,
    step: &EditStep,
    source_path: &Path,
    seed: Option<u64>,
    count: u32,
    output_dir: &Path,
    labels: HashMap<String, String>,
) -> EditJobSpec {
    let (default_steps, default_guidance) = model_family::model_defaults(wf_model);
    let steps = step.steps.unwrap_or(default_steps);
    let guidance = step.guidance.unwrap_or(default_guidance);

    EditJobSpec {
        prompt: step.prompt.clone(),
        model: ModelRef {
            base_model_id: spec_model_id.to_string(),
            base_model_path: base_model_path.clone(),
            arch_key: arch_key.clone(),
        },
        lora,
        output: GenerateOutputRef {
            output_dir: output_dir.to_string_lossy().to_string(),
        },
        params: EditParams {
            image_paths: vec![source_path.to_string_lossy().to_string()],
            steps,
            guidance,
            seed,
            count,
            width: step.width,
            height: step.height,
            scheduler_overrides: HashMap::new(),
        },
        runtime: RuntimeRef {
            profile: runtime::resolved_generation_profile().to_string(),
            python_version: Some("3.11.12".to_string()),
        },
        target: ExecutionTarget::Local,
        labels,
    }
}

// ---------------------------------------------------------------------------
// Execution + event consumption
// ---------------------------------------------------------------------------

fn execute_generate_step(
    executor: &mut LocalExecutor,
    spec: &GenerateJobSpec,
    step_id: &str,
    db: &Database,
) -> Result<Vec<PathBuf>> {
    let handle = executor.submit_generate(spec)?;
    let job_id = handle.job_id.clone();
    register_job(db, &job_id, "generate", spec)?;
    let rx = executor.events(&job_id)?;
    let result = consume_events(rx, step_id, db, &job_id);
    match &result {
        Ok(_) => {
            let _ = db.update_job_status(&job_id, "completed");
        }
        Err(_) => {
            let _ = db.update_job_status(&job_id, "error");
        }
    }
    result
}

fn execute_edit_step(
    executor: &mut LocalExecutor,
    spec: &EditJobSpec,
    step_id: &str,
    db: &Database,
) -> Result<Vec<PathBuf>> {
    let handle = executor.submit_edit(spec)?;
    let job_id = handle.job_id.clone();
    register_job(db, &job_id, "edit", spec)?;
    let rx = executor.events(&job_id)?;
    let result = consume_events(rx, step_id, db, &job_id);
    match &result {
        Ok(_) => {
            let _ = db.update_job_status(&job_id, "completed");
        }
        Err(_) => {
            let _ = db.update_job_status(&job_id, "error");
        }
    }
    result
}

/// Insert a job row so the UI can discover workflow outputs alongside
/// regular `modl generate` / `modl edit` results. Mirrors what those CLI
/// handlers do directly.
fn register_job<S: serde::Serialize>(
    db: &Database,
    job_id: &str,
    kind: &str,
    spec: &S,
) -> Result<()> {
    let spec_json = serde_json::to_string(spec).unwrap_or_else(|_| "{}".to_string());
    db.insert_job(job_id, kind, "running", &spec_json, "local", None)?;
    Ok(())
}

/// Drain the worker's event stream for a single step. Returns the list of
/// artifact paths produced, or an error on worker failure. Persists every
/// event to the job_events table so `modl serve` can reconstruct step details.
fn consume_events(
    rx: mpsc::Receiver<JobEvent>,
    step_id: &str,
    db: &Database,
    job_id: &str,
) -> Result<Vec<PathBuf>> {
    let mut artifacts: Vec<PathBuf> = Vec::new();
    let mut last_progress_len: usize = 0;

    for event in rx {
        // Persist every event so the UI can rehydrate step details.
        let event_json = serde_json::to_string(&event).unwrap_or_default();
        let _ = db.insert_job_event(job_id, event.sequence, &event_json);

        match &event.event {
            EventPayload::Progress {
                stage,
                step: cur,
                total_steps,
                ..
            } => {
                if stage == "step" {
                    let msg = format!("  step {}/{}", cur, total_steps);
                    // Overwrite previous progress line
                    eprint!(
                        "\r{}{}",
                        msg,
                        " ".repeat(last_progress_len.saturating_sub(msg.len()))
                    );
                    last_progress_len = msg.len();
                }
            }
            EventPayload::Artifact { path, .. } => {
                artifacts.push(PathBuf::from(path));
            }
            EventPayload::Completed { .. } => {
                if last_progress_len > 0 {
                    eprint!("\r{}\r", " ".repeat(last_progress_len));
                }
                break;
            }
            EventPayload::Error { message, code, .. } => {
                if last_progress_len > 0 {
                    eprint!("\r{}\r", " ".repeat(last_progress_len));
                }
                bail!("step `{step_id}` failed ({code}): {message}");
            }
            EventPayload::Cancelled => {
                if last_progress_len > 0 {
                    eprint!("\r{}\r", " ".repeat(last_progress_len));
                }
                bail!("step `{step_id}` cancelled");
            }
            EventPayload::Log { message, level } if level == "warning" || level == "error" => {
                eprintln!("  [{level}] {message}");
            }
            _ => {}
        }
    }

    Ok(artifacts)
}

// ---------------------------------------------------------------------------
// Image ref resolution
// ---------------------------------------------------------------------------

/// Resolve an `ImageRef` to a local filesystem path. Handles runtime bounds
/// checking for `$step.outputs[N]` references since cardinality isn't known
/// until the referenced step has executed.
fn resolve_image_ref(
    r: &ImageRef,
    step_outputs: &HashMap<String, Vec<PathBuf>>,
    current_step_id: &str,
) -> Result<PathBuf> {
    match r {
        ImageRef::Local(p) => Ok(p.clone()),
        ImageRef::StepOutput { step_id, index } => {
            let outputs = step_outputs.get(step_id).ok_or_else(|| {
                anyhow!(
                    "step `{current_step_id}`: referenced step `{step_id}` produced no outputs (bug — should have been caught at parse time)"
                )
            })?;
            outputs.get(*index).cloned().ok_or_else(|| {
                anyhow!(
                    "step `{current_step_id}`: step `{step_id}` produced {} output(s), index [{}] is out of bounds",
                    outputs.len(),
                    index
                )
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve a family-alias model name (e.g. `"sdxl"`) to the canonical registry
/// ID (e.g. `"sdxl-base-1.0"`). Mirrors the logic in `cli::generate` so specs
/// submitted via `modl run` use the same naming Python workers understand.
fn resolve_registry_model_id(effective_model: &str) -> String {
    let index = registry::RegistryIndex::load();
    if let Ok(ref idx) = index {
        if idx.find(effective_model).is_some() {
            return effective_model.to_string();
        }
        if let Some(info) = model_family::resolve_model(effective_model) {
            let candidates = [info.id.to_string(), format!("{}-base-1.0", info.id)];
            if let Some(c) = candidates.into_iter().find(|c| idx.find(c).is_some()) {
                return c;
            }
        }
    }
    effective_model.to_string()
}

fn sanitize_for_path(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

fn step_kind_label(kind: &StepKind) -> &'static str {
    match kind {
        StepKind::Generate(_) => "generate",
        StepKind::Edit(_) => "edit",
    }
}

fn print_generate_preview(step: &GenerateStep, sub_job_count: usize) {
    println!("  Prompt: {}", style(&step.prompt).italic());
    let width = step.width.unwrap_or(1024);
    let height = step.height.unwrap_or(1024);
    println!("  Size:   {}×{}", width, height);
    if let Some(ref seeds) = step.seeds {
        println!(
            "  Seeds:  {:?} ({} variation{})",
            seeds,
            seeds.len(),
            if seeds.len() == 1 { "" } else { "s" }
        );
    } else if let Some(seed) = step.seed {
        println!("  Seed:   {}  Count: {}", seed, step.count.unwrap_or(1));
    } else {
        println!("  Count:  {}", step.count.unwrap_or(1));
    }
    let _ = sub_job_count; // accepted for symmetry with edit preview; no current use
}

fn print_edit_preview(step: &EditStep, source_path: &Path, sub_job_count: usize) {
    println!("  Source: {}", source_path.display());
    println!("  Prompt: {}", style(&step.prompt).italic());
    if let Some(ref seeds) = step.seeds {
        println!(
            "  Seeds:  {:?} ({} variation{})",
            seeds,
            seeds.len(),
            if seeds.len() == 1 { "" } else { "s" }
        );
    } else if let Some(seed) = step.seed {
        println!("  Seed:   {}  Count: {}", seed, step.count.unwrap_or(1));
    } else {
        println!("  Count:  {}", step.count.unwrap_or(1));
    }
    let _ = sub_job_count;
}
