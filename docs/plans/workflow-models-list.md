# Spec: `models:` list in workflow YAML

> **Status:** Proposed
> **Author:** Pedro Alonso
> **Date:** 2026-04-16

## Problem

Cross-model comparison (same prompt, multiple models) requires copy-pasting
every step N times with only `model:` and `id:` changing. A 5-scene x 3-model
workflow is 15 hand-written steps with duplicated prompts. Adding a 4th model
means 5 more copy-pasted steps.

This came up building an Atlantis kids book workflow: 5 scenes across
z-image-turbo, ernie-image-turbo, and flux2-klein-9b meant 15 nearly-identical
steps where only the model and ID suffix differed.

## Design

**New top-level field:** `models: [model-id, ...]`

When `models:` is set (instead of `model:`), every step is expanded into N
copies — one per model. Step IDs are auto-suffixed with the model name:
`gates` becomes `gates-z-image-turbo`, `gates-ernie-image-turbo`,
`gates-flux2-klein-9b`.

### Before (current, 15 steps)

```yaml
name: atlantis-kids-book
model: z-image-turbo
steps:
  - id: gates-zimage
    generate: "the grand golden gates..."
    seed: 101
  - id: gates-ernie
    model: ernie-image-turbo
    generate: "the grand golden gates..."
    seed: 101
  - id: gates-klein
    model: flux2-klein-9b
    generate: "the grand golden gates..."
    seed: 101
  # ... 12 more steps
```

### After (proposed, 5 steps)

```yaml
name: atlantis-kids-book
models: [z-image-turbo, ernie-image-turbo, flux2-klein-9b]
steps:
  - id: gates
    generate: "the grand golden gates..."
    seed: 101
  # ... 4 more steps
```

## Spec details

### 1. `models:` vs `model:` — mutually exclusive

The two fields are mutually exclusive at the top level. If both are set, emit
a parse error. `models:` must have at least 2 entries (use `model:` for a
single model).

### 2. Step expansion

At parse time, each step is cloned N times. The expanded step gets:

- **ID:** `{original-id}-{model-id}` (e.g. `gates-z-image-turbo`)
- **Model:** set to the respective model from the list
- All other fields inherited from the original step unchanged

Expansion happens in `parse_str()` after initial YAML deserialization but
before the ID uniqueness check, so the uniqueness check runs against
expanded IDs.

### 3. Per-step `model:` override skips expansion

If a step already has `model:` set, it is NOT expanded. It runs once with
that specific model. This lets you mix: most steps fan out across all models,
but an edit step targets one specific model's output.

```yaml
models: [z-image-turbo, ernie-image-turbo, flux2-klein-9b]
steps:
  - id: gates
    generate: "the grand golden gates..."
    seed: 101
    # Expanded to: gates-z-image-turbo, gates-ernie-image-turbo, gates-flux2-klein-9b

  - id: refine-winner
    model: qwen-image-edit-2511
    edit: "$gates-z-image-turbo.outputs[0]"
    prompt: "add golden light rays"
    # NOT expanded — runs once with qwen-image-edit-2511
```

### 4. Cross-step references

`$gates.outputs[0]` is ambiguous when `gates` expanded to 3 steps. Resolution:

- `$gates-z-image-turbo.outputs[0]` — explicit expanded ID (always works)
- `$gates.outputs[0]` — parse error, must use expanded ID

**Decision:** require expanded IDs in refs. Simple, unambiguous, no magic.

### 5. LoRA handling

Workflow-level `lora:` applies only to steps using the workflow's first model
(or per the existing auto-disable logic in `src/core/workflow.rs`). Each model
in the `models:` list may need its own LoRA. Options considered:

| Option | Approach | Complexity |
|--------|----------|------------|
| A (v1) | Disallow workflow-level `lora:` with `models:` | Lowest — parse error |
| B (future) | Per-model LoRA map: `loras: { z-image-turbo: lora-a, flux2-klein-9b: lora-b }` | Medium |
| C | Inherit workflow `lora:` to all models (probably wrong) | Risky |

**Decision for v1:** disallow workflow-level `lora:` when `models:` is set.
Per-step `lora:` still works for individual steps that need it.

### 6. Scheduler

No changes needed. The expanded steps are regular steps with per-step model
overrides. The existing scheduler (`docs/guides/workflows.md` "Execution order
and the scheduler" section) groups same-model steps automatically via its
greedy topological sort with model affinity.

### 7. Dry-run output (human)

Shows expanded steps (the user sees 15 steps from 5 declarations). Add a
summary line:

```
info: 5 steps expanded across 3 models (15 total)
```

### 8. Dry-run output (JSON)

Add an `expansion` object to the JSON dry-run response:

```json
{
  "expansion": {
    "models": ["z-image-turbo", "ernie-image-turbo", "flux2-klein-9b"],
    "declared_steps": 5,
    "expanded_steps": 15
  }
}
```

This is an additive field — does not bump `schema_version`.

## Implementation plan

### 1. Parser (`src/core/workflow.rs`)

- Add `models: Option<Vec<String>>` to `RawWorkflow`
- Make `model` optional (`Option<String>`) in `RawWorkflow`
- Validate mutual exclusivity: exactly one of `model:` or `models:` must be set
- Validate `models:` has >= 2 entries
- Validate `models:` + `lora:` is an error (v1)
- Expand steps in `parse_str()` after initial YAML deserialization, before
  the ID uniqueness check loop
- Auto-suffix IDs with model name, set per-step `model` on each expanded copy
- Steps with per-step `model:` already set are passed through unexpanded

The public `Workflow` struct keeps its existing `model: String` field. When
`models:` is used, set `Workflow.model` to the first model in the list (for
backward compat with code that reads it). Add a new field
`Workflow.models: Option<Vec<String>>` so callers can distinguish.

### 2. Plan builder (`src/cli/run.rs`)

- Detect `workflow.models.is_some()` and add expansion info to dry-run output
  (both human-readable and JSON modes)
- No scheduler changes needed — expanded steps are regular steps

### 3. Docs (`docs/guides/workflows.md`)

- Add "Multi-model comparison" subsection under the existing "Multi-model
  workflows" section
- Document `models:` field in the spec reference table
- Add example workflow showing before/after

### 4. Tests

| Test | What it verifies |
|------|------------------|
| `models_expands_correctly` | 2 steps x 3 models = 6 expanded steps with correct IDs and per-step models |
| `models_and_model_both_set_rejected` | Parse error when both `model:` and `models:` present |
| `models_single_entry_rejected` | Parse error when `models:` has only 1 entry |
| `per_step_model_override_skips_expansion` | Step with `model:` set is not cloned |
| `models_plus_lora_rejected` | Parse error when `models:` + top-level `lora:` (v1) |
| `cross_step_refs_with_expanded_ids` | `$gates-z-image-turbo.outputs[0]` resolves correctly |
| `cross_step_ref_to_unexpanded_id_rejected` | `$gates.outputs[0]` fails when `gates` was expanded |
| `dry_run_json_includes_expansion` | JSON output contains `expansion` object with correct counts |

## Not in v1

- **Per-model LoRA map** (`loras: { model-a: lora-x, model-b: lora-y }`)
- **`$base-id.outputs[0]` shorthand** (must use expanded ID for now)
- **Per-step `models:` override** (expand one step differently from the rest)
- **Model-specific defaults** (e.g. different step counts per model)
- **Filtering expansion** (e.g. `models: [a, b, c]` but step X only expands
  across `[a, b]`)
