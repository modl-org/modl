//! `modl run <workflow.yaml>` — execute a workflow spec locally.
//!
//! Phase B + B.1.5 of the workflow-run plan. See `docs/plans/workflow-run.md`.
//!
//! Two-phase flow:
//!   1. `build_plan()` — pure: parse, validate, resolve model/lora, expand seeds.
//!      No side effects. Returns a `Plan` or a structured `PlanError`.
//!   2. `execute_plan()` — impure: bootstrap executor, run each step, stream events,
//!      insert DB jobs. Takes a `Plan` and runs it.
//!
//! `--dry-run` stops after phase 1 and prints the plan (human or JSON). Agents
//! can depend on the JSON schema to validate/iterate workflows without burning
//! GPU cycles.

use anyhow::{Result, anyhow, bail};
use console::style;
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use thiserror::Error;

use crate::core::db::Database;
use crate::core::executor::{Executor, LocalExecutor};
use crate::core::job::{
    EditJobSpec, EditParams, EventPayload, ExecutionTarget, GenerateJobSpec, GenerateOutputRef,
    GenerateParams, JobEvent, LoraRef, ModelRef, RuntimeRef,
};
use crate::core::workflow::{EditStep, GenerateStep, ImageRef, StepKind, Workflow, parse_file};
use crate::core::{model_family, model_resolve, paths, registry, runtime};

// ---------------------------------------------------------------------------
// Plan — the fully-validated + resolved workflow, ready for execution or
// inspection via --dry-run.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Plan {
    pub workflow: Workflow,
    /// Default model resolution from the workflow's top-level `model:`.
    /// Individual steps may override via `PlannedStep::resolved_model`.
    pub default_model: ResolvedModel,
    /// Default LoRA resolution from the workflow's top-level `lora:`.
    pub default_lora: Option<LoraRef>,
    pub run_id: String,
    pub output_dir: PathBuf,
    pub planned_steps: Vec<PlannedStep>,
    pub total_artifacts: usize,
    /// Optimized execution order computed from the step DAG + model grouping.
    /// See `schedule_steps` for the algorithm.
    pub schedule: Schedule,
}

/// Result of the scheduler: the execution order (as indices into
/// `workflow.steps`) plus statistics about model switches saved vs
/// YAML-declared order.
///
/// `switches_in_order` is what naive sequential execution would cost.
/// `switches_scheduled` is what the greedy scheduler found. The difference
/// is the win — each switch is one model reload on the Python worker,
/// typically 30-90 seconds depending on model size.
#[derive(Debug, Clone)]
pub struct Schedule {
    /// Step indices in execution order. Always length == workflow.steps.len().
    pub order: Vec<usize>,
    /// Model switches if steps ran in YAML declaration order.
    pub switches_in_order: usize,
    /// Model switches in `order`. Always <= `switches_in_order`.
    pub switches_scheduled: usize,
}

impl Schedule {
    /// True when the scheduler found a better execution order than the YAML
    /// declaration order. When false, YAML order is already optimal (or
    /// forced by dependencies) and the scheduler made no changes.
    pub fn reordered(&self) -> bool {
        self.switches_scheduled < self.switches_in_order
    }

    /// Number of model switches the scheduler avoided.
    pub fn switches_saved(&self) -> usize {
        self.switches_in_order
            .saturating_sub(self.switches_scheduled)
    }
}

/// Model fully resolved against the local store: canonical registry id,
/// filesystem path to the weights directory, optional Python worker arch key.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub id: String,
    pub base_path: String,
    pub arch_key: Option<String>,
}

#[derive(Debug)]
pub struct PlannedStep {
    /// Copy of the step id; kept for symmetry with `Workflow.steps` and for
    /// future queries that don't zip with the original workflow.
    #[allow(dead_code)]
    pub id: String,
    pub sub_jobs: Vec<(Option<u64>, u32)>,
    /// Per-step effective model after override resolution. If the step
    /// didn't override, this equals `Plan::default_model`.
    pub resolved_model: ResolvedModel,
    /// Per-step effective LoRA. `None` can mean either "inherited no lora"
    /// or "auto-disabled because step overrides model". See the comment on
    /// `build_plan` for the rules.
    pub resolved_lora: Option<LoraRef>,
    /// True when this step's effective model differs from `Plan::default_model`.
    /// Used for display ("show the override in the plan print") and for
    /// warm-worker accounting (model switches cost a reload).
    pub model_overridden: bool,
}

impl PlannedStep {
    pub fn expected_artifacts(&self) -> usize {
        self.sub_jobs.iter().map(|(_, c)| *c as usize).sum()
    }
}

// ---------------------------------------------------------------------------
// PlanError — structured errors for JSON dry-run output. Converts to
// anyhow::Error for the normal execution path so callers don't have to know
// about this type.
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum PlanError {
    #[error("Failed to parse workflow spec: {0}")]
    Parse(String),

    #[error("Model `{model}` is not installed locally. Run `modl pull {model}`.")]
    ModelNotInstalled { model: String },

    #[error("LoRA `{lora}` not found in local store")]
    LoraNotFound { lora: String },

    #[error("{0}")]
    Other(String),
}

impl PlanError {
    /// Stable error kind string for JSON consumers. Agents can branch on this.
    fn kind(&self) -> &'static str {
        match self {
            PlanError::Parse(_) => "parse_error",
            PlanError::ModelNotInstalled { .. } => "model_not_installed",
            PlanError::LoraNotFound { .. } => "lora_not_found",
            PlanError::Other(_) => "other",
        }
    }

    /// Human-actionable fix suggestion for JSON consumers.
    fn fix(&self) -> Option<String> {
        match self {
            PlanError::ModelNotInstalled { model } => Some(format!("modl pull {model}")),
            PlanError::LoraNotFound { lora } => {
                Some(format!("modl pull {lora} or train it with `modl train`"))
            }
            _ => None,
        }
    }
}

// (No manual `From<PlanError> for anyhow::Error` — `thiserror`'s derived
// `std::error::Error` impl + anyhow's blanket `From<E>` already covers it.)

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn run(
    spec_path: &str,
    _auto_pull: bool,
    dry_run: bool,
    json: bool,
    in_order: bool,
) -> Result<()> {
    let db = Database::open()?;

    let plan_result = build_plan(spec_path, &db);

    if dry_run {
        return run_dry_run(plan_result, json, in_order);
    }

    let plan = plan_result?;
    execute_plan(plan, &db, in_order).await
}

// ---------------------------------------------------------------------------
// Dry-run path
// ---------------------------------------------------------------------------

fn run_dry_run(plan_result: Result<Plan, PlanError>, json: bool, in_order: bool) -> Result<()> {
    match plan_result {
        Ok(plan) => {
            if json {
                print_plan_json(&plan, in_order)?;
            } else {
                print_plan_human(&plan, in_order);
            }
            Ok(())
        }
        Err(e) => {
            if json {
                // In JSON mode we exit 0 regardless so agents can reliably
                // parse the result — validity is signalled by `"valid": false`.
                print_error_json(&e)?;
                Ok(())
            } else {
                // In human mode, bubble up so the shell exit code is nonzero.
                Err(e.into())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// build_plan — the pure validation + resolution phase. No side effects.
// Used by both dry-run and normal execution.
// ---------------------------------------------------------------------------

/// Build the full execution plan for a workflow. Pure — no GPU work, no DB
/// writes, no filesystem writes.
///
/// **Per-step model / LoRA resolution rules:**
///
/// 1. Workflow-level `model:` is always resolved to a `ResolvedModel`
///    (`Plan::default_model`). This is the baseline.
/// 2. Workflow-level `lora:` is always resolved if set (`Plan::default_lora`).
/// 3. For each step:
///    - If `step.model` is set, resolve it to its own `ResolvedModel` and
///      flag `model_overridden = true`. Otherwise use `Plan::default_model`.
///    - For LoRA:
///      - If `step.lora` is set, resolve it and use it.
///      - Else if `model_overridden`, use `None` (auto-disable — the
///        workflow-level LoRA probably belongs to a different model).
///      - Else inherit `Plan::default_lora`.
///
/// Auto-disabling LoRA on model override is a safe default, not a
/// restriction: if the user wants to force a LoRA through anyway, they
/// can set `lora:` explicitly on the step.
pub fn build_plan(spec_path: &str, db: &Database) -> Result<Plan, PlanError> {
    // 1. Parse + validate YAML
    let wf = parse_file(Path::new(spec_path)).map_err(|e| PlanError::Parse(format!("{e:#}")))?;

    // 2. Resolve workflow-level model (the default used by unoverridden steps)
    let default_model = resolve_model(&wf.model, db)?;

    // 3. Resolve workflow-level LoRA if set
    let default_lora = match wf.lora.as_deref() {
        Some(name) => Some(resolve_lora(name, db)?),
        None => None,
    };

    // 4. Compute run id + output dir (dry-run uses these only for display)
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let output_dir = paths::modl_root().join("outputs").join(&date);
    let run_id = format!(
        "{}-{}",
        chrono::Local::now().format("%Y%m%d-%H%M%S"),
        sanitize_for_path(&wf.name)
    );

    // 5. Per-step resolution: model override, lora override, seed expansion.
    // We cache resolved models by ID so a workflow that uses the same model
    // in N steps only does the DB lookup once.
    let mut resolved_model_cache: std::collections::HashMap<String, ResolvedModel> =
        std::collections::HashMap::new();
    resolved_model_cache.insert(default_model.id.clone(), default_model.clone());

    let mut planned_steps = Vec::with_capacity(wf.steps.len());
    let mut total_artifacts = 0usize;

    for step in &wf.steps {
        let (step_model_override, step_lora_override, sub_jobs) = match &step.kind {
            StepKind::Generate(g) => (
                g.model.clone(),
                g.lora.clone(),
                expand_seeds(g.seed, g.count, &g.seeds),
            ),
            StepKind::Edit(e) => (
                e.model.clone(),
                e.lora.clone(),
                expand_seeds(e.seed, e.count, &e.seeds),
            ),
        };

        // Resolve effective model for this step
        let (resolved_model, model_overridden) = match step_model_override {
            Some(ref id) => {
                let canonical = resolve_registry_model_id(id);
                let resolved = if let Some(cached) = resolved_model_cache.get(&canonical) {
                    cached.clone()
                } else {
                    let r = resolve_model(id, db)?;
                    resolved_model_cache.insert(r.id.clone(), r.clone());
                    r
                };
                let overridden = resolved.id != default_model.id;
                (resolved, overridden)
            }
            None => (default_model.clone(), false),
        };

        // Resolve effective LoRA for this step
        let resolved_lora = match step_lora_override {
            Some(ref name) => Some(resolve_lora(name, db)?),
            None => {
                if model_overridden {
                    // Auto-disable inherited LoRA on model override
                    None
                } else {
                    default_lora.clone()
                }
            }
        };

        total_artifacts += sub_jobs.iter().map(|(_, c)| *c as usize).sum::<usize>();

        planned_steps.push(PlannedStep {
            id: step.id.clone(),
            sub_jobs,
            resolved_model,
            resolved_lora,
            model_overridden,
        });
    }

    // 6. Compute execution schedule (DAG + greedy model grouping)
    let schedule = schedule_steps(&wf, &planned_steps);

    Ok(Plan {
        workflow: wf,
        default_model,
        default_lora,
        run_id,
        output_dir,
        planned_steps,
        total_artifacts,
        schedule,
    })
}

/// Compute an execution schedule for a workflow's steps that:
///
/// 1. Respects the step dependency DAG (`$step-id.outputs[N]` edges — a step
///    can only run after every step it references has run).
/// 2. Minimizes model switches greedily, preferring ready steps whose
///    effective model matches the currently-loaded model.
///
/// The scheduler is a greedy topological sort with a tiebreaker on
/// "currently loaded model". Not provably optimal for arbitrary DAGs
/// (model-switch minimization is NP-hard in general), but close to optimal
/// for real workflow shapes and O(V + E) to compute.
///
/// Deterministic: among equally-preferred steps, always picks the one with
/// the earliest YAML-declared index. This keeps behavior stable across runs
/// and makes the tests predictable.
fn schedule_steps(wf: &Workflow, planned: &[PlannedStep]) -> Schedule {
    use std::collections::{BTreeSet, HashMap, HashSet};

    let n = wf.steps.len();

    // 1. Build id → index map
    let id_to_idx: HashMap<&str, usize> = wf
        .steps
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id.as_str(), i))
        .collect();

    // 2. Build dependency graph: deps[i] = set of step indices i depends on.
    // Only edit steps can have deps (via `ImageRef::StepOutput`).
    let mut deps: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut reverse_deps: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, step) in wf.steps.iter().enumerate() {
        let StepKind::Edit(e) = &step.kind else {
            continue;
        };
        let ImageRef::StepOutput { step_id, .. } = &e.source else {
            continue;
        };
        let Some(&dep_idx) = id_to_idx.get(step_id.as_str()) else {
            continue;
        };
        if deps[i].insert(dep_idx) {
            reverse_deps[dep_idx].push(i);
        }
    }

    // 3. Baseline: switch count in declared order
    let in_order: Vec<usize> = (0..n).collect();
    let switches_in_order = count_model_switches(planned, &in_order);

    // 4. Greedy topological sort preferring current loaded model.
    // `ready` is sorted by step index so tiebreaking picks earliest declared.
    let mut remaining_deps: Vec<HashSet<usize>> = deps.clone();
    let mut ready: BTreeSet<usize> = (0..n).filter(|i| deps[*i].is_empty()).collect();
    let mut order = Vec::with_capacity(n);
    let mut current_model: Option<String> = None;

    while !ready.is_empty() {
        // Prefer a ready step matching the currently loaded model.
        let pick: usize = current_model
            .as_ref()
            .and_then(|m| {
                ready
                    .iter()
                    .find(|&&i| planned[i].resolved_model.id == *m)
                    .copied()
            })
            .unwrap_or_else(|| {
                // No match (or no model loaded yet) — pick the earliest
                // declared ready step. BTreeSet iteration is sorted.
                *ready.iter().next().expect("ready non-empty")
            });

        ready.remove(&pick);
        order.push(pick);
        current_model = Some(planned[pick].resolved_model.id.clone());

        // Unlock any dependents whose last dep just got scheduled.
        for &dependent in &reverse_deps[pick] {
            remaining_deps[dependent].remove(&pick);
            if remaining_deps[dependent].is_empty() {
                ready.insert(dependent);
            }
        }
    }

    debug_assert_eq!(
        order.len(),
        n,
        "scheduler didn't visit every step — bug in DAG construction"
    );

    let switches_scheduled = count_model_switches(planned, &order);

    // Sanity: the greedy sort should never be worse than declaration order.
    debug_assert!(
        switches_scheduled <= switches_in_order,
        "scheduler produced a worse order than YAML — bug"
    );

    Schedule {
        order,
        switches_in_order,
        switches_scheduled,
    }
}

/// Count model switches in a given execution order. A "switch" happens when
/// consecutive steps use different effective models. The first step is free.
fn count_model_switches(planned: &[PlannedStep], order: &[usize]) -> usize {
    let mut count = 0usize;
    let mut prev_model: Option<&str> = None;
    for &idx in order {
        let model = planned[idx].resolved_model.id.as_str();
        if let Some(p) = prev_model
            && p != model
        {
            count += 1;
        }
        prev_model = Some(model);
    }
    count
}

/// Resolve a model id to a `ResolvedModel` (canonical registry id + local
/// path + arch key). Returns a `PlanError` if the model isn't installed
/// locally.
fn resolve_model(id: &str, db: &Database) -> Result<ResolvedModel, PlanError> {
    let canonical = resolve_registry_model_id(id);
    let base_path = model_resolve::resolve_base_model_path(&canonical, db).ok_or_else(|| {
        PlanError::ModelNotInstalled {
            model: canonical.clone(),
        }
    })?;
    let arch_key = model_family::find_model(&canonical).map(|m| m.arch_key.to_string());
    Ok(ResolvedModel {
        id: canonical,
        base_path,
        arch_key,
    })
}

/// Resolve a LoRA name to a `LoraRef`. Returns a `PlanError` if not in the
/// local store.
fn resolve_lora(name: &str, db: &Database) -> Result<LoraRef, PlanError> {
    model_resolve::resolve_lora(name, 1.0, db)
        .map_err(|e| PlanError::Other(format!("LoRA resolution error: {e}")))?
        .ok_or_else(|| PlanError::LoraNotFound {
            lora: name.to_string(),
        })
}

// ---------------------------------------------------------------------------
// execute_plan — the GPU work. Runs every step, streams events, registers
// each step as a DB job row with workflow labels for UI filtering.
// ---------------------------------------------------------------------------

pub async fn execute_plan(plan: Plan, db: &Database, in_order: bool) -> Result<()> {
    let wf = &plan.workflow;

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

    // Pick execution order: scheduled (default) or YAML-declared (--in-order).
    let exec_order: Vec<usize> = if in_order {
        (0..wf.steps.len()).collect()
    } else {
        plan.schedule.order.clone()
    };

    // Model switch accounting for the chosen order
    let effective_switches = count_model_switches(&plan.planned_steps, &exec_order);

    if !in_order && plan.schedule.reordered() {
        println!(
            "  {} Scheduler reordered {} steps to reduce model switches ({} → {}, saved {})",
            style("↻").cyan(),
            wf.steps.len(),
            plan.schedule.switches_in_order,
            plan.schedule.switches_scheduled,
            plan.schedule.switches_saved(),
        );
    }
    if effective_switches > 0 {
        println!(
            "  {} {} model switch{} during execution (~worker reload per switch)",
            style("ℹ").cyan(),
            effective_switches,
            if effective_switches == 1 { "" } else { "es" }
        );
    }

    std::fs::create_dir_all(&plan.output_dir)?;
    println!("  Output: {}", plan.output_dir.display());
    println!("  Run ID: {}\n", plan.run_id);

    println!("{} Preparing runtime...", style("→").cyan());
    let mut executor = LocalExecutor::for_generation().await?;

    let mut step_outputs: HashMap<String, Vec<PathBuf>> = HashMap::new();
    let total_steps = wf.steps.len();

    for (exec_idx, &declared_idx) in exec_order.iter().enumerate() {
        let step = &wf.steps[declared_idx];
        let planned = &plan.planned_steps[declared_idx];

        // When execution order differs from declaration order, show both
        // indices in the header so the user can map back to their YAML.
        let reordered = exec_idx != declared_idx;
        let header_position = if reordered {
            format!(
                "[exec {}/{}, yaml {}/{}]",
                exec_idx + 1,
                total_steps,
                declared_idx + 1,
                total_steps
            )
        } else {
            format!("[{}/{}]", exec_idx + 1, total_steps)
        };
        let header_model_note = if planned.model_overridden {
            format!(" [model={}]", planned.resolved_model.id)
        } else {
            String::new()
        };
        println!(
            "\n{} {} {} ({}){}",
            style("▸").cyan().bold(),
            header_position,
            style(&step.id).bold(),
            step_kind_label(&step.kind),
            style(header_model_note).dim(),
        );

        let step_labels = build_step_labels(&wf.name, &plan.run_id, &step.id);
        let sub_jobs = &planned.sub_jobs;
        let resolved_model = &planned.resolved_model;
        let resolved_lora = planned.resolved_lora.clone();
        // Source key for `model_family::model_defaults` lookup: use the step's
        // effective model, not the workflow default, so per-step overrides
        // fall back to the right model-specific defaults.
        let effective_model_key = resolved_model.id.as_str();

        let mut step_artifacts: Vec<PathBuf> = Vec::new();

        match &step.kind {
            StepKind::Generate(g) => {
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
                        effective_model_key,
                        &resolved_model.id,
                        Some(resolved_model.base_path.clone()),
                        resolved_model.arch_key.clone(),
                        resolved_lora.clone(),
                        g,
                        *seed,
                        *count,
                        &plan.output_dir,
                        step_labels.clone(),
                    );
                    let artifacts = execute_generate_step(&mut executor, &spec, &step.id, db)?;
                    step_artifacts.extend(artifacts);
                }
            }
            StepKind::Edit(e) => {
                let source_path = resolve_image_ref(&e.source, &step_outputs, &step.id)?;
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
                        effective_model_key,
                        &resolved_model.id,
                        Some(resolved_model.base_path.clone()),
                        resolved_model.arch_key.clone(),
                        resolved_lora.clone(),
                        e,
                        &source_path,
                        *seed,
                        *count,
                        &plan.output_dir,
                        step_labels.clone(),
                    );
                    let artifacts = execute_edit_step(&mut executor, &spec, &step.id, db)?;
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
    // Summary
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
    println!("  {}", plan.output_dir.display());

    Ok(())
}

// ---------------------------------------------------------------------------
// Plan printers — human + JSON output for `--dry-run`.
// ---------------------------------------------------------------------------

fn print_plan_human(plan: &Plan, in_order: bool) {
    let wf = &plan.workflow;
    println!("Workflow: {}", style(&wf.name).bold());
    println!(
        "  Model: {} ({})",
        plan.default_model.id,
        style("installed ✓").green()
    );
    if let Some(ref lora) = plan.default_lora {
        println!("  LoRA:  {} ({})", lora.name, style("installed ✓").green());
    }

    // Count + list unique per-step model overrides
    let overrides: Vec<&str> = plan
        .planned_steps
        .iter()
        .filter(|p| p.model_overridden)
        .map(|p| p.resolved_model.id.as_str())
        .collect();
    if !overrides.is_empty() {
        let mut unique: Vec<&str> = overrides.clone();
        unique.sort();
        unique.dedup();
        println!(
            "  {} {} step{} override the model: {}",
            style("ℹ").cyan(),
            overrides.len(),
            if overrides.len() == 1 { "" } else { "s" },
            unique.join(", ")
        );
    }

    println!(
        "  Total planned artifacts: {}",
        style(plan.total_artifacts.to_string()).bold()
    );

    // Scheduler summary
    let exec_order: Vec<usize> = if in_order {
        (0..wf.steps.len()).collect()
    } else {
        plan.schedule.order.clone()
    };
    let effective_switches = count_model_switches(&plan.planned_steps, &exec_order);
    if plan.schedule.switches_in_order > 0 || effective_switches > 0 {
        if in_order {
            println!(
                "  Execution order: {} (--in-order)  |  Model switches: {}",
                style("YAML declared").dim(),
                effective_switches
            );
        } else if plan.schedule.reordered() {
            println!(
                "  Execution order: {}  |  Model switches: {} → {} (saved {})",
                style("optimized").green().bold(),
                plan.schedule.switches_in_order,
                plan.schedule.switches_scheduled,
                plan.schedule.switches_saved()
            );
        } else {
            println!(
                "  Execution order: {}  |  Model switches: {}",
                style("YAML order (already optimal)").dim(),
                effective_switches
            );
        }
    }

    // Figure out which indices to display and in what order
    println!("\nSteps ({}):", wf.steps.len());
    for (exec_idx, &declared_idx) in exec_order.iter().enumerate() {
        let step = &wf.steps[declared_idx];
        let planned = &plan.planned_steps[declared_idx];
        let kind_label = step_kind_label(&step.kind);
        let expected = planned.expected_artifacts();
        let i = exec_idx; // used for the leading index column
        let reordered = !in_order && exec_idx != declared_idx;
        // Use the step's *effective* model for the defaults lookup, so per-step
        // model overrides fall back to the overridden model's defaults.
        let (default_steps, default_guidance) =
            model_family::model_defaults(&planned.resolved_model.id);

        let model_annotation = if planned.model_overridden {
            format!(" [model={}]", planned.resolved_model.id)
        } else {
            String::new()
        };
        let lora_annotation = match &planned.resolved_lora {
            Some(l) if planned.model_overridden => format!(" [lora={}]", l.name),
            _ => String::new(),
        };
        let reorder_annotation = if reordered {
            format!(" (yaml #{})", declared_idx + 1)
        } else {
            String::new()
        };

        match &step.kind {
            StepKind::Generate(g) => {
                let width = g.width.unwrap_or(1024);
                let height = g.height.unwrap_or(1024);
                let steps = g.steps.unwrap_or(default_steps);
                let guidance = g.guidance.unwrap_or(default_guidance);
                let seed_desc = if let Some(ref seeds) = g.seeds {
                    format!("{} seeds {:?}", seeds.len(), seeds)
                } else if let Some(s) = g.seed {
                    format!("seed={} count={}", s, g.count.unwrap_or(1))
                } else {
                    format!("count={}", g.count.unwrap_or(1))
                };
                println!(
                    "  [{}] {:<12} {:<8} {}  {}×{}  {} steps  g={}{}{}{}",
                    i + 1,
                    style(&step.id).cyan(),
                    kind_label,
                    seed_desc,
                    width,
                    height,
                    steps,
                    guidance,
                    style(&model_annotation).yellow(),
                    style(&lora_annotation).magenta(),
                    style(&reorder_annotation).dim(),
                );
                println!("      {}", style(truncate(&g.prompt, 80)).italic());
            }
            StepKind::Edit(e) => {
                let steps = e.steps.unwrap_or(default_steps);
                let guidance = e.guidance.unwrap_or(default_guidance);
                let source = match &e.source {
                    ImageRef::Local(p) => format!("local: {}", p.display()),
                    ImageRef::StepOutput { step_id, index } => {
                        format!("${step_id}.outputs[{index}]")
                    }
                };
                let seed_desc = if let Some(ref seeds) = e.seeds {
                    format!("{} seeds {:?}", seeds.len(), seeds)
                } else if let Some(s) = e.seed {
                    format!("seed={}", s)
                } else {
                    format!("count={}", e.count.unwrap_or(1))
                };
                println!(
                    "  [{}] {:<12} {:<8} {}  {} steps  g={}{}{}{}",
                    i + 1,
                    style(&step.id).cyan(),
                    kind_label,
                    seed_desc,
                    steps,
                    guidance,
                    style(&model_annotation).yellow(),
                    style(&lora_annotation).magenta(),
                    style(&reorder_annotation).dim(),
                );
                println!("      source: {}", source);
                println!("      {}", style(truncate(&e.prompt, 80)).italic());
            }
        }
        let _ = expected;
    }

    println!(
        "\n{} Workflow is valid. Run without --dry-run to execute.",
        style("✓").green().bold()
    );
}

fn print_plan_json(plan: &Plan, in_order: bool) -> Result<()> {
    let wf = &plan.workflow;

    // Build steps in YAML declaration order. Agents reason about step ids,
    // not positions, so the array is always in YAML order and execution
    // order is conveyed separately via `execution.order`.
    let steps: Vec<StepJson> = wf
        .steps
        .iter()
        .zip(plan.planned_steps.iter())
        .map(|(step, planned)| {
            let (default_steps, default_guidance) =
                model_family::model_defaults(&planned.resolved_model.id);
            let model_json = if planned.model_overridden {
                Some(planned.resolved_model.id.clone())
            } else {
                None
            };
            let lora_json = if planned.model_overridden {
                planned.resolved_lora.as_ref().map(|l| l.name.clone())
            } else {
                None
            };

            match &step.kind {
                StepKind::Generate(g) => StepJson {
                    id: step.id.clone(),
                    kind: "generate",
                    prompt: g.prompt.clone(),
                    source_ref: None,
                    model: model_json,
                    lora: lora_json,
                    seed: g.seed,
                    seeds: g.seeds.clone(),
                    effective: EffectiveJson {
                        width: g.width.unwrap_or(1024),
                        height: g.height.unwrap_or(1024),
                        steps: g.steps.unwrap_or(default_steps),
                        guidance: g.guidance.unwrap_or(default_guidance),
                        count: g.count.unwrap_or(1),
                    },
                    expected_artifacts: planned.expected_artifacts(),
                },
                StepKind::Edit(e) => StepJson {
                    id: step.id.clone(),
                    kind: "edit",
                    prompt: e.prompt.clone(),
                    source_ref: Some(match &e.source {
                        ImageRef::Local(p) => p.to_string_lossy().into_owned(),
                        ImageRef::StepOutput { step_id, index } => {
                            format!("${step_id}.outputs[{index}]")
                        }
                    }),
                    model: model_json,
                    lora: lora_json,
                    seed: e.seed,
                    seeds: e.seeds.clone(),
                    effective: EffectiveJson {
                        width: e.width.unwrap_or(0),
                        height: e.height.unwrap_or(0),
                        steps: e.steps.unwrap_or(default_steps),
                        guidance: e.guidance.unwrap_or(default_guidance),
                        count: e.count.unwrap_or(1),
                    },
                    expected_artifacts: planned.expected_artifacts(),
                },
            }
        })
        .collect();

    // Execution plan: scheduled order + switch counts.
    let exec_order_indices: Vec<usize> = if in_order {
        (0..wf.steps.len()).collect()
    } else {
        plan.schedule.order.clone()
    };
    let exec_order_ids: Vec<String> = exec_order_indices
        .iter()
        .map(|&i| wf.steps[i].id.clone())
        .collect();
    let effective_switches = count_model_switches(&plan.planned_steps, &exec_order_indices);

    let execution = ExecutionJson {
        mode: if in_order { "yaml_order" } else { "optimized" },
        order: exec_order_ids,
        model_switches: effective_switches,
        switches_in_yaml_order: plan.schedule.switches_in_order,
        switches_saved: if in_order {
            0
        } else {
            plan.schedule.switches_saved()
        },
    };

    let output = PlanJson {
        valid: true,
        workflow: WorkflowJson {
            name: wf.name.clone(),
            model: plan.default_model.id.clone(),
            lora: plan.default_lora.as_ref().map(|l| l.name.clone()),
        },
        total_planned_artifacts: plan.total_artifacts,
        execution,
        steps,
    };

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn print_error_json(err: &PlanError) -> Result<()> {
    let output = ErrorJson {
        valid: false,
        error: ErrorInfoJson {
            kind: err.kind(),
            message: format!("{err}"),
            fix: err.fix(),
        },
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}

// ---------------------------------------------------------------------------
// JSON output types — stable schema for agent consumption.
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct PlanJson {
    valid: bool,
    workflow: WorkflowJson,
    total_planned_artifacts: usize,
    /// Execution plan (order, model switches). Steps in the `steps` array are
    /// always in YAML declaration order; `execution.order` is the list of
    /// step ids in execution order.
    execution: ExecutionJson,
    steps: Vec<StepJson>,
}

#[derive(Serialize)]
struct ExecutionJson {
    /// Either `"optimized"` (scheduler reordered for fewer model switches) or
    /// `"yaml_order"` (forced via `--in-order`).
    mode: &'static str,
    /// Step ids in execution order.
    order: Vec<String>,
    /// Number of model switches in this execution order.
    model_switches: usize,
    /// Number of model switches if run in YAML declaration order. Useful
    /// for agents to reason about the optimization headroom.
    switches_in_yaml_order: usize,
    /// Number of switches saved by reordering. Zero in `yaml_order` mode.
    switches_saved: usize,
}

#[derive(Serialize)]
struct WorkflowJson {
    name: String,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    lora: Option<String>,
}

#[derive(Serialize)]
struct StepJson {
    id: String,
    kind: &'static str,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_ref: Option<String>,
    /// Present only when the step overrides the workflow-level model.
    /// Absent means "uses workflow default" (see `WorkflowJson::model`).
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    /// Present only when the step overrides the workflow-level LoRA
    /// (explicit step-level lora on an overridden-model step).
    #[serde(skip_serializing_if = "Option::is_none")]
    lora: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seeds: Option<Vec<u64>>,
    effective: EffectiveJson,
    expected_artifacts: usize,
}

#[derive(Serialize)]
struct EffectiveJson {
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    count: u32,
}

#[derive(Serialize)]
struct ErrorJson {
    valid: bool,
    error: ErrorInfoJson,
}

#[derive(Serialize)]
struct ErrorInfoJson {
    kind: &'static str,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    fix: Option<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tests (scheduler only — parser + validation tests live in core::workflow)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::workflow::parse_str;

    fn fake_resolved(id: &str) -> ResolvedModel {
        ResolvedModel {
            id: id.to_string(),
            base_path: format!("/fake/{id}"),
            arch_key: None,
        }
    }

    /// Build a `PlannedStep` for each workflow step with a supplied effective
    /// model id per step. `seeds: 1 count=1` for simplicity.
    fn build_planned(wf: &Workflow, models: &[&str]) -> Vec<PlannedStep> {
        assert_eq!(wf.steps.len(), models.len());
        wf.steps
            .iter()
            .zip(models.iter())
            .map(|(s, m)| PlannedStep {
                id: s.id.clone(),
                sub_jobs: vec![(None, 1)],
                resolved_model: fake_resolved(m),
                resolved_lora: None,
                model_overridden: false,
            })
            .collect()
    }

    #[test]
    fn single_model_workflow_is_identity_order() {
        let yaml = r#"
name: single
model: a
steps:
  - id: s1
    generate: "p1"
  - id: s2
    generate: "p2"
  - id: s3
    generate: "p3"
"#;
        let wf = parse_str(yaml, std::path::Path::new(".")).unwrap();
        let planned = build_planned(&wf, &["a", "a", "a"]);
        let sched = schedule_steps(&wf, &planned);
        assert_eq!(sched.order, vec![0, 1, 2]);
        assert_eq!(sched.switches_in_order, 0);
        assert_eq!(sched.switches_scheduled, 0);
        assert!(!sched.reordered());
    }

    #[test]
    fn linear_dep_chain_is_forced_in_order() {
        // Each step depends on the previous — no reordering possible
        // even though models alternate.
        let yaml = r#"
name: linear
model: a
steps:
  - id: s1
    generate: "p1"
  - id: s2
    edit: "$s1.outputs[0]"
    prompt: "x"
  - id: s3
    edit: "$s2.outputs[0]"
    prompt: "y"
  - id: s4
    edit: "$s3.outputs[0]"
    prompt: "z"
"#;
        let wf = parse_str(yaml, std::path::Path::new(".")).unwrap();
        // Alternating models — would be optimal to group but deps force the order
        let planned = build_planned(&wf, &["a", "b", "a", "b"]);
        let sched = schedule_steps(&wf, &planned);
        assert_eq!(sched.order, vec![0, 1, 2, 3]);
        assert_eq!(sched.switches_in_order, 3);
        assert_eq!(sched.switches_scheduled, 3);
        assert!(!sched.reordered());
    }

    #[test]
    fn interleaved_independent_steps_get_grouped_by_model() {
        // 4 independent generate steps with alternating models.
        // YAML order: a b a b → 3 switches.
        // Optimized:  a a b b → 1 switch.
        let yaml = r#"
name: interleave
model: a
steps:
  - id: s1
    generate: "p1"
  - id: s2
    generate: "p2"
  - id: s3
    generate: "p3"
  - id: s4
    generate: "p4"
"#;
        let wf = parse_str(yaml, std::path::Path::new(".")).unwrap();
        let planned = build_planned(&wf, &["a", "b", "a", "b"]);
        let sched = schedule_steps(&wf, &planned);
        assert_eq!(sched.switches_in_order, 3);
        assert_eq!(sched.switches_scheduled, 1);
        assert!(sched.reordered());
        // Greedy scheduler: first pick s1 (a), then s3 (also a, earliest unscheduled),
        // then s2 (b), then s4 (b). Order: [0, 2, 1, 3].
        assert_eq!(sched.order, vec![0, 2, 1, 3]);
    }

    #[test]
    fn scenes_then_refines_pattern() {
        // The motivating book-chapter pattern:
        //   scene-1 (flux) → refine-1 (qwen) → scene-2 (flux) → refine-2 (qwen)
        //     with refine-N depending on scene-N.
        // YAML order: flux qwen flux qwen → 3 switches
        // Optimized:  flux flux qwen qwen → 1 switch
        let yaml = r#"
name: scenes
model: flux
steps:
  - id: scene-1
    generate: "s1"
  - id: refine-1
    model: qwen
    edit: "$scene-1.outputs[0]"
    prompt: "r1"
  - id: scene-2
    generate: "s2"
  - id: refine-2
    model: qwen
    edit: "$scene-2.outputs[0]"
    prompt: "r2"
"#;
        let wf = parse_str(yaml, std::path::Path::new(".")).unwrap();
        let planned = build_planned(&wf, &["flux", "qwen", "flux", "qwen"]);
        let sched = schedule_steps(&wf, &planned);
        assert_eq!(sched.switches_in_order, 3);
        assert_eq!(sched.switches_scheduled, 1);
        assert_eq!(sched.order, vec![0, 2, 1, 3]);
        assert_eq!(sched.switches_saved(), 2);
    }

    #[test]
    fn diamond_dag_respects_dependencies() {
        // a (flux)  ─┐
        // b (flux)  ─┼→ d (flux)
        // c (qwen)  ─┘
        // d depends on a AND b AND c.
        // Optimal order respecting deps: a, b, c, d (or b, a, c, d — tiebreak by declared index).
        // Switches: flux→qwen→flux = 2.
        let yaml = r#"
name: diamond
model: flux
steps:
  - id: a
    generate: "a"
  - id: b
    generate: "b"
  - id: c
    model: qwen
    generate: "c"
  - id: d
    edit: "$c.outputs[0]"
    prompt: "d"
"#;
        let wf = parse_str(yaml, std::path::Path::new(".")).unwrap();
        let planned = build_planned(&wf, &["flux", "flux", "qwen", "flux"]);
        let sched = schedule_steps(&wf, &planned);
        // Greedy: starts with a (flux), then b (flux, same), then c (qwen, switch),
        // then d (flux, switch). 2 switches total.
        // Note: d only actually depends on c in this YAML (the other deps are
        // commented out above for clarity). The scheduler just has to order c before d.
        assert_eq!(sched.order, vec![0, 1, 2, 3]);
        assert_eq!(sched.switches_scheduled, 2);
    }

    #[test]
    fn deterministic_tiebreak_earliest_declared() {
        // Two independent ready steps with the same model — scheduler should
        // pick the earliest declared.
        let yaml = r#"
name: tie
model: a
steps:
  - id: s1
    generate: "p1"
  - id: s2
    generate: "p2"
"#;
        let wf = parse_str(yaml, std::path::Path::new(".")).unwrap();
        let planned = build_planned(&wf, &["a", "a"]);
        let sched = schedule_steps(&wf, &planned);
        assert_eq!(sched.order, vec![0, 1]);
    }

    #[test]
    fn switches_saved_computes_correctly() {
        // Ensure switches_saved() reflects the difference even on a trivially
        // optimal-as-declared workflow.
        let yaml = r#"
name: already_optimal
model: a
steps:
  - id: s1
    generate: "p1"
  - id: s2
    generate: "p2"
"#;
        let wf = parse_str(yaml, std::path::Path::new(".")).unwrap();
        let planned = build_planned(&wf, &["a", "a"]);
        let sched = schedule_steps(&wf, &planned);
        assert_eq!(sched.switches_saved(), 0);
        assert!(!sched.reordered());
    }
}
