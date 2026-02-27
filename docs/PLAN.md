# mods: The Opinionated Image Generation Toolkit

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
The key missing pieces are E2E testing on a real GPU and a few UX gaps.

| Area | Status | Notes |
|------|--------|-------|
| Model pull/install | ✅ Done | Registry, HF download, content-addressed store, dep resolution, symlinks |
| Model ls/info/search | ✅ Done | Filtering by type, detailed info, popular/trending |
| Model link (ComfyUI/A1111) | ✅ Done | Auto-detect layouts, bidirectional symlinks |
| Config (YAML) | ✅ Done | `mods config`, ~/.mods/config.yaml, targets, storage |
| GPU detection | ✅ Done | NVML + nvidia-smi fallback, variant auto-selection |
| Auth (HF, CivitAI) | ✅ Done | Token prompting and storage |
| Dataset create/ls/validate | ✅ Done | Copy images, pair captions, scan managed datasets |
| Training presets | ✅ Done | Quick/Standard/Advanced with full test coverage |
| TrainJobSpec + events | ✅ Done | All types, serde roundtrips, event protocol |
| Executor trait + LocalExecutor | ✅ Done | submit, events (mpsc), cancel, stdout→JobEvent parsing |
| CLI train (interactive + flags) | ✅ Done | Dialoguer prompts, dry-run, $EDITOR for Advanced |
| Artifact collection | ✅ Done | Hash, store, register in DB, symlink to ~/.mods/loras/ |
| Job tracking (DB) | ✅ Done | jobs/job_events/artifacts tables, full CRUD |
| Python train adapter | ✅ Done | spec→ai-toolkit YAML, stdout progress parsing, artifact scan |
| CLI generate | ✅ Done | Prompt, --lora, --seed, --size presets, --count, progress bar |
| Python gen adapter | ✅ Done | FluxPipeline/SDXL/SD1.5, LoRA loading, artifact emission |
| Runtime management | ✅ Done | Python venv bootstrap, ai-toolkit install, setup command |
| Doctor/GC/Export/Import | ✅ Done | Health checks, garbage collection, lockfile round-trip |
| `mods upgrade` | ✅ Done | Self-update from GitHub releases |
| Dataset caption | ✅ Implemented | Florence-2/BLIP auto-captioning |
| Batch generation | ❌ Not started | `mods generate --batch prompts.txt` |
| Output management CLI | ❌ Not started | `mods outputs`, search, open |
| `--cloud` flag | 🟡 Stubbed | CloudExecutor struct + provider enum + cred resolution done, submit returns not-implemented |
| Web UI (`mods serve`) | ❌ Not started | Deferred to later phase |

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
| `src/cli/install.rs` | 430 | `mods model pull` — download, verify, register |
| `src/core/job.rs` | 382 | TrainJobSpec, GenerateJobSpec, JobEvent, EventPayload |
| `src/cli/generate.rs` | 370 | Generate command with LoRA resolution, size presets |
| `src/core/dataset.rs` | 322 | Dataset create/scan/validate/list |
| `src/core/presets.rs` | 298 | Quick/Standard/Advanced param resolution |
| `src/core/cloud.rs` | 238 | CloudExecutor stub + provider enum + credential resolution |
| `src/core/artifacts.rs` | 217 | LoRA collection: hash, store, register, symlink |
| `python/mods_worker/adapters/gen_adapter.py` | 250 | Diffusers pipeline loading + inference |
| `python/mods_worker/adapters/caption_adapter.py` | 240 | Florence-2/BLIP auto-captioning adapter |
| `python/mods_worker/adapters/train_adapter.py` | 222 | ai-toolkit config translation + process orchestration |
| `python/mods_worker/protocol.py` | 99 | EventEmitter: JSON-line protocol over stdout |

---

## The MVP: What mods Actually Does

### Philosophy: CLI is the truth, UI is a window

```
mods model pull flux-dev        # download from HF, deps auto-resolved
mods model ls                   # list installed (checkpoints, loras, vaes)
mods model ls --type lora       # filter by type

mods dataset create myface --from ~/photos/headshots/
mods dataset ls                 # table: name, images, captions, coverage
mods dataset validate myface    # checks image count, warns if < 5

mods train                      # interactive: pick dataset, model, preset
mods train --dataset myface --base flux-schnell --name myface-v1
mods train --config custom.yml  # escape hatch: full TrainJobSpec YAML
mods train --dry-run            # print generated spec without running

mods generate "a photo of OHWX on marble countertop"
mods generate "a photo of OHWX" --lora myface-v1 --seed 42
mods generate "a cat" --base flux-schnell --size 16:9 --count 4
```

### The Three Layers

```
Layer 1: mods CLI (Rust, single binary)
├── Model manager (download, dep resolution, content-addressed store)
├── Dataset manager (create, validate, scan)
├── Training orchestrator (presets → spec → executor → artifacts)
├── Generation orchestrator (spec → executor → images)
├── Job tracker (SQLite: jobs, events, artifacts)
└── Tooling (doctor, gc, export/import, upgrade, init)

Layer 2: mods Python runtime (managed by CLI)
├── ai-toolkit (training, managed as dependency)
├── diffusers (inference: Flux, SDXL, SD1.5 pipelines)
└── LoRA loading + fusion

Layer 3: mods web UI (future, `mods serve`)
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
mods dataset create test --from ./some-images/
mods train --dataset test --base flux-schnell --name test-v1 --preset quick
mods generate "a photo of OHWX in a park" --lora test-v1
```

Likely issues to fix:
- ai-toolkit config field mapping (model names, paths)
- Runtime bootstrap edge cases (torch version, CUDA compatibility)
- Diffusers pipeline loading (from_pretrained vs from_single_file logic)

### Priority 2: Dataset Captioning ✅

```
[x] mods dataset caption <name>
    - Run Florence-2 or BLIP on uncaptioned images (--model flag)
    - Write .txt files alongside images
    - Show captions for review in terminal
    - --overwrite flag to re-caption existing
    - Progress bar with per-image status
```

Implemented via `caption_adapter.py` Python adapter + CLI subcommand.
CLI: `mods dataset caption <name> [--model florence-2|blip] [--overwrite]`

### Priority 3: Output Management

```
[ ] mods outputs                 # list recent generations
[ ] mods outputs search <query>  # search by prompt text
[ ] mods outputs open <id>       # open in system viewer
```

The DB already tracks artifacts. This is mostly CLI presentation.

### Priority 4: Batch Generation

```
[ ] mods generate --batch prompts.txt
    - One prompt per line
    - Sequential generation (VRAM limited to one at a time)
```

### Priority 5: Cloud Executor (`--cloud`)

The architecture is ready. Adding `--cloud` means:

```
New code:
  src/core/cloud_executor.rs  — implements Executor trait
    - Dataset upload to cloud storage
    - API calls to provider (Modal, Replicate, RunPod)
    - Event polling → same mpsc channel
    - Artifact download on completion

  Add --cloud / --provider flags to Commands::Train and Commands::Generate

Untouched code (everything else):
  - TrainJobSpec — same struct serialized as JSON in API call
  - presets.rs — same preset logic
  - dataset.rs — same validation (cloud executor handles upload)
  - db.rs — same tables
  - artifacts.rs — same collection (cloud executor downloads artifact first)
  - cli/train.rs — same flow, one branch:

    let executor: Box<dyn Executor> = if cloud {
        Box::new(CloudExecutor::new(provider)?)
    } else {
        Box::new(LocalExecutor::from_runtime_setup().await?)
    };
```

### Priority 6: Web UI (`mods serve`)

Deferred. Everything reads from the same `~/.mods/` directory and SQLite DB,
so the UI can be built independently whenever it makes sense.

---

## What mods is NOT

- **Not a node editor.** No graphs. If you want ComfyUI, use ComfyUI.
- **Not a marketplace.** CivitAI exists. mods can *pull from* CivitAI.
- **Not a hosted service.** You run it. On your machine or your cloud account.
- **Not infinitely configurable.** Three training presets. Advanced gives you full YAML. That's it.

---

## Verification Checklist

1. **Unit tests** (all passing ✅ — 55 tests): Preset scaling, dataset scanning, spec roundtrips, DB CRUD, event parsing, artifact collection, cloud provider parsing
2. **Integration test** (TODO): `mods dataset create` → `mods train --dry-run` → verify spec YAML
3. **E2E with GPU** (TODO): Full training + generation flow on real hardware
4. **Cloud executor** (TODO): Implement trait methods, test with one provider