# modl: The Opinionated Image Generation Toolkit

> Last updated: 2026-02-27 — audited against `feat/train-command` branch

1. CLI-first (like rails/cargo/git)
2. Opinionated defaults with escape hatches
3. Covers the *full lifecycle*: models → datasets → training → inference → outputs
4. Local-first, cloud-burst capable
5. Actually maintained for modern models (Flux, Z-Image, Qwen, etc.)

---

## Current Status Summary

The CLI is on the `feat/train-command` branch. **All 55 unit tests pass.**
The full Rust+Python pipeline exists for both training and generation.
The key missing pieces are E2E testing on a real GPU and output management UX.

| Area | Status | Notes |
|------|--------|-------|
| Model pull/install | ✅ Done | Registry, HF download, content-addressed store, dep resolution, symlinks |
| Model ls/info/search | ✅ Done | Filtering by type, detailed info, popular/trending |
| Model link (ComfyUI/A1111) | ✅ Done | Auto-detect layouts, bidirectional symlinks |
| Config (YAML) | ✅ Done | `modl config`, ~/.modl/config.yaml, targets, storage |
| GPU detection | ✅ Done | NVML + nvidia-smi fallback, variant auto-selection |
| Auth (HF, CivitAI) | ✅ Done | Token prompting and storage |
| Dataset create/ls/validate | ✅ Done | Copy images, pair captions, scan managed datasets |
| Dataset caption | ✅ Done | Florence-2/BLIP auto-captioning via Python adapter |
| Training presets | ✅ Done | Quick/Standard/Advanced with full test coverage |
| TrainJobSpec + events | ✅ Done | All types, serde roundtrips, event protocol |
| Executor trait + LocalExecutor | ✅ Done | submit, events (mpsc), cancel, stdout→JobEvent parsing |
| CLI train (interactive + flags) | ✅ Done | Dialoguer prompts, dry-run, $EDITOR for Advanced |
| Artifact collection | ✅ Done | Hash, store, register in DB, symlink to ~/.modl/loras/ |
| Job tracking (DB) | ✅ Done | jobs/job_events/artifacts tables, full CRUD |
| Python train adapter | ✅ Done | spec→ai-toolkit YAML, stdout progress parsing, artifact scan |
| CLI generate | ✅ Done | Prompt, --lora, --seed, --size presets, --count, progress bar |
| Python gen adapter | ✅ Done | FluxPipeline/SDXL/SD1.5, LoRA loading, artifact emission |
| Runtime management | ✅ Done | Python venv bootstrap, ai-toolkit install, setup command |
| Doctor/GC/Export/Import | ✅ Done | Health checks, garbage collection, lockfile round-trip |
| `modl upgrade` | ✅ Done | Self-update from GitHub releases |
| Output management CLI | ✅ Done | `modl outputs` list/show/open/search |
| E2E GPU validation | ❌ Blocked | Need real GPU test of full train→generate flow |
| Batch generation | ❌ Not started | `modl generate --batch prompts.txt` |
| `--cloud` flag | 🟡 Stubbed | CloudExecutor struct + provider enum + cred resolution done, submit not-implemented |
| Cloud training | ❌ Not started | API service + Modal backend + CloudExecutor.submit() |
| Web UI (`modl serve`) | ⏸️ Deferred | CLI-first. UI reads same DB, build independently later |

---

## Architecture

```
CLI layer (interactive prompts, progress display)
    │
    ├── presets::resolve_params()  ── pure logic, no I/O
    ├── dataset::validate()        ── filesystem scan
    ├── gpu::detect()              ── NVML + nvidia-smi
    │
    ▼
TrainJobSpec / GenerateJobSpec  ◄── the serialization boundary
    │
    ▼
┌─────────────────────┐
│  dyn Executor       │  ◄── trait: submit / submit_generate / events / cancel
├─────────────────────┤
│  LocalExecutor      │  ◄── Implemented ✅
│  CloudExecutor      │  ◄── Future (just impl the trait)
└─────────────────────┘
    │
    ▼
artifacts::collect_lora()  ── hash, store, register, symlink
```

The job spec is the contract. Same `TrainJobSpec` struct gets built by presets,
persisted to DB, and handed to whichever executor runs it. Adding `--cloud`
means implementing one new struct, not refactoring the pipeline.

### File Map (key modules, ~10500 LOC total)

| File | LOC | Purpose |
|------|-----|---------|
| `src/core/runtime.rs` | 803 | Python venv bootstrap, ai-toolkit install, profile management |
| `src/core/executor.rs` | 627 | Executor trait + LocalExecutor (train + generate) |
| `src/cli/mod.rs` | 508 | CLI arg definitions, command dispatch |
| `src/core/db.rs` | 448 | SQLite: installed, symlinks, deps, jobs, events, artifacts |
| `src/cli/train.rs` | 478 | Interactive prompts, executor dispatch, progress display |
| `src/cli/install.rs` | 430 | `modl model pull` — download, verify, register |
| `src/core/job.rs` | 382 | TrainJobSpec, GenerateJobSpec, JobEvent, EventPayload |
| `src/cli/generate.rs` | 370 | Generate command with LoRA resolution, size presets |
| `src/core/dataset.rs` | 322 | Dataset create/scan/validate/list |
| `src/core/presets.rs` | 298 | Quick/Standard/Advanced param resolution |
| `src/core/cloud.rs` | 238 | CloudExecutor stub + provider enum + credential resolution |
| `src/core/artifacts.rs` | 217 | LoRA collection: hash, store, register, symlink |
| `python/modl_worker/adapters/gen_adapter.py` | 250 | Diffusers pipeline loading + inference |
| `python/modl_worker/adapters/caption_adapter.py` | 240 | Florence-2/BLIP auto-captioning adapter |
| `python/modl_worker/adapters/train_adapter.py` | 222 | ai-toolkit config translation + process orchestration |
| `python/modl_worker/protocol.py` | 99 | EventEmitter: JSON-line protocol over stdout |

---

## The MVP: What modl Actually Does

### Philosophy: CLI is the truth, UI is a window

```
modl model pull flux-dev        # download from HF, deps auto-resolved
modl model ls                   # list installed (checkpoints, loras, vaes)
modl model ls --type lora       # filter by type

modl dataset create myface --from ~/photos/headshots/
modl dataset ls                 # table: name, images, captions, coverage
modl dataset validate myface    # checks image count, warns if < 5

modl train                      # interactive: pick dataset, model, preset
modl train --dataset myface --base flux-schnell --name myface-v1
modl train --config custom.yml  # escape hatch: full TrainJobSpec YAML
modl train --dry-run            # print generated spec without running

modl generate "a photo of OHWX on marble countertop"
modl generate "a photo of OHWX" --lora myface-v1 --seed 42
modl generate "a cat" --base flux-schnell --size 16:9 --count 4
```

### The Three Layers

```
Layer 1: modl CLI (Rust, single binary)
├── Model manager (download, dep resolution, content-addressed store)
├── Dataset manager (create, validate, scan)
├── Training orchestrator (presets → spec → executor → artifacts)
├── Generation orchestrator (spec → executor → images)
├── Job tracker (SQLite: jobs, events, artifacts)
└── Tooling (doctor, gc, export/import, upgrade, init)

Layer 2: modl Python runtime (managed by CLI)
├── ai-toolkit (training, managed as dependency)
├── diffusers (inference: Flux, SDXL, SD1.5 pipelines)
└── LoRA loading + fusion

Layer 3: modl web UI (future, `modl serve`)
├── Reads from same SQLite + filesystem
└── Generation playground + training dashboard
```

### Why Rust CLI + Python runtime?

- Rust CLI: fast startup, single binary distribution, file I/O, cross-platform
- Python runtime: ai-toolkit and diffusers are Python. No point fighting this.
- The CLI orchestrates Python processes. Like how `cargo` doesn't compile Rust itself — it calls `rustc`.

---

## What's Left to Ship: Prioritized

### Priority 1: E2E Validation (real GPU test)

Everything is wired. The critical next step is running the full flow on a machine
with a GPU to shake out integration issues:

```bash
modl dataset create test --from ./some-images/
modl train --dataset test --base flux-schnell --name test-v1 --preset quick
modl generate "a photo of OHWX in a park" --lora test-v1
```

Likely issues to fix:
- ai-toolkit config field mapping (model names, paths)
- Runtime bootstrap edge cases (torch version, CUDA compatibility)
- Diffusers pipeline loading (from_pretrained vs from_single_file logic)

**This blocks everything below. Nothing else matters until generate produces an image.**

### Priority 2: Output Management (mini DAM)

Surface what's already in the DB. The `jobs`, `artifacts`, and `job_events` tables
already store full specs, file paths, and metadata. This is mostly CLI presentation.

```
[x] modl outputs                  # list recent generations (table: id, prompt, model, lora, time)
[x] modl outputs show <id>        # full metadata (prompt, seed, model, loras, params, paths)
[x] modl outputs open <id>        # open image in system viewer
[x] modl outputs search <query>   # search by prompt text, model, or lora name
```

Why before batch/cloud: without a way to find and review what you generated,
every additional feature just creates more noise.

### Priority 3: Batch Generation

```
[ ] modl generate --batch prompts.txt
    - One prompt per line
    - Sequential generation (VRAM limited to one at a time)
    - Each image tracked as separate artifact in DB
```

### Priority 4: Reproducible Export

```
[ ] modl outputs export <id>      # dump full JobSpec as YAML (all params, model refs, versions)
```

The spec_json already contains everything needed. This is a one-command feature
that makes any generation reproducible.

### Priority 5: Cloud Training (`--cloud`, training only)

Cloud inference deferred — cold start + keep_warm economics are brutal for
interactive use. Cloud training is the real monetization.

See [cloud/plan.md](cloud/plan.md) for full architecture.

```
[ ] modl API service (auth, job dispatch, S3)
[ ] Modal backend (train_fn, model volumes)
[ ] CloudExecutor.submit() implementation
[ ] modl cloud login / modl cloud status
```

### Deferred (not blocking v1 UX)

| Feature | Why deferred |
|---------|-------------|
| Cloud inference | Cold start UX + keep_warm cost. Generate locally with downloaded LoRA. |
| Web UI (`modl serve`) | CLI-first. UI reads same DB, can be built independently. |
| Runs/sessions | Build if grep-search over outputs isn't enough. Nullable `run_id` FK on jobs table. |
| Tags/notes on outputs | Nice-to-have. Prompt search covers 90% of cases. |
| Video generation | Architecture supports it (`output_kind` enum on GenerateJobSpec), but adapter not built. |
| Bundle export (zip) | `modl outputs export` as YAML covers reproducibility. Zip packaging is a later packaging problem. |

### Done ✅

| Feature | Notes |
|---------|-------|
| Dataset captioning | `modl dataset caption <name>` — Florence-2/BLIP, --model, --overwrite |
| CloudExecutor stub | Provider enum + credential resolution. submit() returns not-implemented. |
| Doctor/GC/Export/Import | Health checks, garbage collection, lockfile round-trip |
| `modl upgrade` | Self-update from GitHub releases |

---

## What modl is NOT

- **Not a node editor.** No graphs. If you want ComfyUI, use ComfyUI.
- **Not a marketplace.** CivitAI exists. modl can *pull from* CivitAI.
- **Not a hosted service.** You run it. On your machine or your cloud account.
- **Not infinitely configurable.** Three training presets. Advanced gives you full YAML. That's it.

---

## Verification Checklist

1. **Unit tests** (all passing ✅ — 55 tests): Preset scaling, dataset scanning, spec roundtrips, DB CRUD, event parsing, artifact collection, cloud provider parsing
2. **Integration test** (TODO): `modl dataset create` → `modl train --dry-run` → verify spec YAML
3. **E2E with GPU** (TODO — Priority 1): Full training + generation flow on real hardware
4. **Output management** (TODO — Priority 2): `modl outputs` list/show/open/search
5. **Cloud training** (TODO — Priority 5): Implement CloudExecutor.submit(), test with Modal