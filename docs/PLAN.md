# modl — Plan & Status

> Last updated: 2026-03-04 (storage layer + business model revision)
> Single source of truth. Everything else in `docs/` is reference material.

## What modl is

The antidote to ComfyUI config hell and 2-hour YouTube tutorials.

You want to train a LoRA on your art style. You don't want to spend an afternoon
reading ai-toolkit docs, tuning 40 parameters, writing YAML, and praying the
venv works. You want: `modl train --dataset my-art --base flux-dev`. Done.

1. **Opinionated model installer** — `modl pull flux-dev`, deps handled, no thinking
2. **Opinionated LoRA trainer** — presets (quick/standard/advanced), sensible defaults
3. **Opinionated generator** — simple flags, seeds, batch, size presets, LoRA stacking
4. **Works anywhere** — your GPU box, your laptop, cloud — same tool, same data

CLI-first. Rust binary + managed Python runtime. Local-first, cloud when needed.

---

## Current Status

**~15K Rust LOC, ~2K Python LOC. 55 unit tests passing. Active branch: `feat/lora-strength`.**

| Area | Status | Notes |
|------|--------|-------|
| Model pull/ls/rm/search/info | ✅ Solid | Registry (68 models), HF direct pulls (`hf:` prefix), content-addressed store |
| Model link (ComfyUI/A1111) | ✅ Solid | Auto-detect layouts, cross-device fallback |
| HuggingFace integration | ✅ Done | `modl pull hf:owner/repo`, HF fallback in search |
| Config / Auth / GPU detect | ✅ Done | YAML config, HF/CivitAI tokens, NVML + nvidia-smi |
| Dataset create/validate/caption | ✅ Done | Florence-2/BLIP auto-captioning, tag, resize |
| Training (presets + executor) | ✅ Working | SDXL LoRA trained successfully, previews generated |
| Generation (CLI) | ✅ Built | Flux/SDXL/SD1.5 via diffusers, LoRA loading, `--json` output flag |
| Output management | ✅ Done | `modl outputs` list/show/open/search |
| Runtime bootstrap | ✅ Done | Python venv + ai-toolkit install |
| Doctor/GC/Export/Import | ✅ Done | Orphan detection, repair, lockfile round-trip |
| `modl upgrade` | ✅ Done | Self-update from GitHub releases |
| Preview UI | 🔄 Active | Training evolution + Outputs gallery + metadata panel; Generate tab incoming |

---

## Architecture

```
modl CLI (Rust, single binary)
    │
    ├── presets::resolve_params()   ── pure logic, no I/O
    ├── dataset::validate()         ── filesystem scan
    ├── gpu::detect()               ── NVML + nvidia-smi
    │
    ▼
TrainJobSpec / GenerateJobSpec      ◄── serialization boundary
    │
    ├──────────────────┬─────────────────────┐
    ▼                  ▼                     ▼
LocalExecutor     CloudExecutor         SyncClient
(spawn Python)    (Modal API)           (R2 via presigned URLs)
    │                  │                     │
    ▼                  │                     ▼
Python runtime         │              Cloudflare R2
(modl_worker/)         │              └── users/{id}/
├── train_adapter.py   │                  ├── datasets/   (content-addressed)
├── gen_adapter.py     │                  ├── loras/      (trained outputs)
├── caption_adapter.py │                  └── outputs/    (generated images)
├── llm_adapter.py     │
└── protocol.py        │
    (JSONL events)      │
                        ▼
                  modl API (Rust / Axum)
                  ├── auth (JWT, device tokens)
                  ├── job dispatch → Modal
                  ├── R2 presigned URL broker
                  └── artifact sync registry
```

**Storage layer**: Cloudflare R2 (S3-compatible, zero egress cost). The modl API
never touches the data — it issues short-lived presigned URLs, client uploads/downloads
direct. Datasets are content-addressed (hash first, skip upload if exists). LoRAs and
outputs sync back automatically after cloud jobs complete. `modl sync` reconciles
local DB with cloud state.

---

## GPU Resource Management

Training, generation, captioning, and prompt enhancement all compete for the
same GPU. These run as separate CLI processes, so in-process mutexes don't work.
Users have wildly different GPUs (8GB–48GB+), so static rules are too coarse.

See [specs/gpu-resource-management.md](specs/gpu-resource-management.md) for
full design. Summary:

**Core mechanism**: File-based activity lock (`~/.modl/gpu.lock`) with PID-based
stale detection and RAII guards. Every GPU-touching command acquires a lock
before spawning Python, releases on completion/crash.

**Rules:**
1. Training is exclusive — blocks everything else
2. One diffusion pipeline at a time (generation XOR captioning)
3. Small models (LLM enhance ≤4GB) can coexist with a pipeline on 24GB+ GPUs
4. Zero-GPU tasks (builtin enhancer, remote API) bypass the lock entirely
5. `--force` overrides all checks for power users

**VRAM budget**: Each task declares an estimated VRAM footprint from a static
table (registry already has `vram_required_mb` per variant). Decision is
`vram_free ≥ task_estimate`, not precise accounting. ±20% is fine — the goal
is preventing obvious conflicts, not a GPU scheduler.

**Enhance graceful degradation**: The enhance feature has a natural fallback
chain — GPU LLM → CPU LLM → builtin rules → remote API. The ✨ button always
works; it just gets faster with more resources.

```
$ modl generate "a cat"
✗ GPU busy: training sdxl (PID 12345, started 2h 14m ago)
  Use --force to override, --cloud for cloud GPU, or --wait.

$ modl gpu
GPU: NVIDIA RTX 4090 (24576 MB total, 2564 MB free)
Active: training sdxl  PID 12345  22012 MB est.  2h 14m
```

---

## Phases

### Phase 1 — Bulletproof install + setup ✅

*Done.* One `curl | sh` install. `modl doctor` catches broken symlinks, missing
deps, corrupt files. `modl pull` resolves dependencies. GPU auto-detection picks
the right variant.

Remaining polish:
- [ ] CUDA compatibility edge cases (test on more machines)
- [ ] `modl init` wizard for first-time setup

### Phase 2 — Validate full train→generate flow 🔄

The pipeline exists end-to-end. SDXL LoRA training works with preview generation.

- [x] SDXL LoRA training (confirmed working)
- [x] Training previews / samples
- [x] Dataset annotation
- [x] Style-mode captioning (`--style` strips medium/technique references)
- [ ] Upgrade captioner to Qwen3-VL-8B-Instruct (vision-language model,
      instruction-following: system prompt controls exactly what to describe.
      No regex post-processing — tell it "describe visual content only, ignore
      medium and artistic style" and it obeys. ~4.5GB VRAM in fp8, fits easily
      on a 4090 alongside other work. Add as `--model qwen` option,
      keep Florence-2 as fast default for non-style captioning)
- [ ] Flux training E2E validation
- [x] Generation E2E validation (`modl generate` → image on disk, metadata written)
- [x] `--json` flag on `modl generate` — machine-readable output for UI and scripting
- [ ] Fix integration issues that surface on GPU

### Phase 3 — Multi-arch training support

Curated ai-toolkit configs for models people actually train on.
See [multi-arch-training-plan.md](multi-arch-training-plan.md) for full gap analysis.

Priority order (by real-world demand):

| # | Model | Status | Effort |
|---|-------|--------|--------|
| 1 | flux-dev / flux-schnell | ✅ Config ready | — |
| 2 | sdxl / sd1.5 | ✅ Working | — |
| 3 | z-image-turbo | 🟡 Config ready | E2E test |
| 4 | chroma | 🟡 Config ready | E2E test |
| 5 | Flux Kontext | ❌ Needs paired dataset support | ~1h |
| 6 | FLUX.2 | ❌ Needs arch entry | ~30min |
| 7 | Qwen-Image | ❌ Needs qtype + quant for 24GB | ~1h |

### Phase 4 — Polish, batch & GPU resource management

- [ ] `GpuLock` file-based lock (`~/.modl/gpu.lock`) + `GpuGuard` RAII drop guard
- [ ] Gate `modl generate`, `modl train`, `modl dataset caption` behind GPU lock
- [ ] `modl gpu` command — show GPU info, active tasks, release stale locks
- [ ] `--force` flag to override GPU lock on all GPU commands
- [ ] `modl doctor` GPU diagnostics (CUDA check, driver version, stale locks)
- [ ] Batch generation (`modl generate --batch prompts.txt`)
- [ ] Reproducible export (`modl outputs export <id>` → full YAML spec)
- [ ] Registry curation: core tier (flux-dev, flux-schnell, sdxl) vs experimental
- [ ] VRAM-aware config tuning (auto quantize/offload based on GPU)

### Phase 5 — Persistent worker (performance)

Eliminate 20-45s cold start on repeated `modl generate` calls.
Python daemon with LRU model cache, LoRA hot-swap.
See [specs/persistent-worker.md](specs/persistent-worker.md) for full spec.

- [ ] Python serve mode (`modl_worker serve` on Unix socket)
- [ ] Rust worker management (auto-spawn, health check, idle timeout)
- [ ] Model cache with LRU eviction (explicit VRAM tracking per loaded model)
- [ ] Worker owns GPU lock while alive — individual requests don't re-acquire

### Phase 6 — Web UI (`modl serve`) 🔄 Current sprint

Prompt-first generate page. Training dashboard. Gallery.
The preview UI (`modl serve`) is the foundation — Axum server + single HTML
file already baked into the binary via `include_str!()`. The train→generate→iterate
loop is the immediate target.

**UI stack: React (Vite, TSX) compiled to `dist/index.html`, replaced in binary.**
Rationale: React over Svelte because familiarity = velocity. Both compile to a
self-contained bundle; the binary size difference (~20KB) is irrelevant at this
scale. Same `include_str!()` embed, same Axum API — only the authoring changes.

**`--json` flag strategy**: CLI commands that the UI drives (`generate`, and
eventually `train`, `dataset caption`) emit JSONL events on stdout when `--json`
is passed. The UI calls these via `/api/generate` (which spawns the CLI internally)
or directly polls the existing JSONL job stream. This keeps the CLI as the
canonical interface — the UI is a thin shell around it, not a parallel path.

**Generate tab spec (10 controls + Advanced):**
- Prompt + negative prompt (collapsed)
- Model picker (filtered to installed checkpoints)
- LoRA stack (add/remove rows: name + strength slider)
- Size preset (square/portrait/landscape + custom)
- Seed (randomize + lock toggle)
- Steps, Guidance, Batch count
- Generate button + live progress (SSE from job stream)
- Advanced accordion: sampler, clip skip, raw kwargs override
- **Prompt field designed as textarea with programmatic fill support** (required by
  "Open as recipe" and future LLM enhance — do this right from day one)
- Enhance button slot (hidden/disabled until Phase 9 — hook is there, cost is zero)

**Gallery upgrades:**
- Search (client-side, across prompt/model/lora)
- Filter chips: model, LoRA, date
- "Open as recipe" — loads image metadata back into Generate form (the killer feature)
- Copy prompt / Copy CLI command
- Re-generate with new seed

**GPU contention strategy**: Replaced the frontend-only `training_active` check
with the cross-process GPU lock from Phase 4. `GET /api/gpu` now returns lock
activity (task type, PID, VRAM estimate, elapsed time), not just a boolean.
Server-side `/api/generate` checks `GpuLock::try_acquire()` and returns
structured error with activity details. No queue system — just a hard lock with
clear messaging. Full queue deferred to Phase 5 persistent worker.
See [specs/gpu-resource-management.md](specs/gpu-resource-management.md).

- [x] Outputs gallery with metadata lightbox and delete
- [x] Training evolution viewer with live status banner
- [x] Dataset viewer
- [x] `POST /api/generate` scaffolding + `--json` flag
- [ ] `GET /api/gpu` endpoint — return GPU lock activity, VRAM, active tasks
- [ ] Replace `AtomicBool` in server.rs with `GpuLock::try_acquire()`
- [ ] `GET /api/models` endpoint (installed checkpoints + LoRAs from DB)
- [x] React (Vite) project under `src/ui/web/`, builds to `dist/index.html`
- [x] Generate page — 10-control form + GPU lock state
- [x] LoRA stack component (add/remove rows, strength slider per row)
- [ ] SSE progress stream during generation
- [ ] "Open as recipe" — fills Generate form from image metadata
- [ ] Gallery search + filter chips (client-side, no new API)

### Phase 7a — Identity + storage sync (`modl auth login` / `modl sync`)

**Goal: your `~/.modl/` follows you to any machine.**
This is the foundation everything cloud-related builds on. No teams, no orgs —
just you, logged in, with your data available wherever you sit down.

**Why R2 over Modal Volumes**: Modal Volumes are ephemeral, job-scoped, tied to
Modal's infrastructure. A trained LoRA must outlive the GPU job and be pullable
to any machine forever. R2 is $0.015/GB/month, zero egress fees, globally accessible.
At 1000 users with 5GB each: ~$75/month storage cost, irrelevant.

**Data model**: Three artifact classes, each synced differently:
| Class | Direction | Strategy |
|-------|-----------|----------|
| Datasets | Local → Cloud | Content-addressed upload, skip if hash exists |
| LoRAs | Cloud → Local (after train) + bidirectional push/pull | Keyed by name+version |
| Outputs | Cloud → Local (after generate) + optional push | Date-grouped, prunable |

Base models (flux-dev, sdxl, etc.) are **never synced** — Modal pulls them from
HuggingFace directly into a persistent Modal Volume. Users don't pay egress for 11GB files.

- [ ] `modl auth login` — device flow, JWT stored in `~/.modl/config.yaml`
- [ ] `modl auth whoami` / `modl auth logout`
- [ ] modl API service: auth endpoint + R2 presigned URL broker
- [ ] DB schema: add `sync_state` (local-only/synced/cloud-only) + `owner_id` to artifacts
- [ ] `modl sync` — reconcile local DB with cloud artifact registry
- [ ] `modl push <lora-id>` / `modl pull me/<lora-id>` — explicit sync for LoRAs
- [ ] Auto-sync: datasets before cloud job, LoRAs + outputs after job completes

### Phase 7b — Cloud execution (`--cloud`)

`modl train --cloud` and `modl generate --cloud` submit to Modal via the modl API.
Builds directly on 7a — auth + R2 are prerequisites.
See [archive/cloud-plan.md](archive/cloud-plan.md) for architecture detail.

**Execution flow**:
1. Hash dataset → check R2 → upload only if not cached
2. POST job spec to modl API → dispatches Modal function
3. Modal pulls dataset from R2, pulls base model from HF (cached in Modal Volume)
4. Job runs, streams JSONL events back via SSE
5. Modal pushes LoRA / outputs to R2
6. Client auto-downloads artifacts, updates local DB

- [ ] Modal GPU backend (train + generate functions)
- [ ] `CloudExecutor.submit()` implementation
- [ ] SSE job stream proxied through modl API
- [ ] Billing: usage tracking per job, Stripe integration
- [ ] Pricing: per-workflow flat rate (not GPU seconds shown to user)

### Phase 9 — LLM integration (local-first, agentic assist)

Optional local LLM for prompt intelligence. Same adapter pattern as generation —
`llm_adapter.py` speaks the same JSONL protocol, `/api/enhance` is the single
new endpoint. The UI hook is already present from Phase 6 (disabled button).

**Guiding principle**: local-first, opt-in, never required. Users without a GPU
large enough for the LLM skip this entirely — the tool works the same without it.

**Model fit**: Qwen3.5-4B (~3.5GB VRAM in Q4 — fits alongside a running Flux fp8
pipeline on a 4090). Qwen3.5-9B also fits but leaves less headroom. Alternatively
route to any OpenAI-compatible API (Ollama, cloud) via config. One adapter,
multiple backends.

**Use cases in priority order:**
1. **Prompt enhance** — short phrase → rich detailed prompt. Single call, instant value.
2. **Story → batch prompts** — "10 scenes from a fantasy quest" → 10 prompts →
   feed directly into batch generation. Turns a feature into a workflow.
3. **Caption critique** — review training captions, suggest improvements.
   Direct quality feedback on dataset preparation before training.
4. **LoRA naming / tagging assist** — suggest trigger words from dataset samples.

**What NOT to build**: a chat interface, a general assistant, anything that
requires internet by default. This is a diffusion tool with a smart prompt field,
not an AI chatbot that happens to generate images.

- [ ] `llm_adapter.py` — local Qwen3.5-4B or OpenAI-compatible API
- [x] `POST /api/enhance` — endpoint implemented (builtin rule-based backend)
- [x] `modl enhance` CLI command with `--model`, `--intensity`, `--json` flags
- [x] `PromptEnhancer` trait + pluggable backend architecture (`src/core/enhance.rs`)
- [x] Enhance button active in Generate form (✨ with intensity selector)
- [ ] Enhance auto-negotiation: GPU LLM → CPU LLM → builtin → remote API
- [ ] `modl pull qwen3.5-4b-instruct` registry entry
- [ ] Story → prompts mode (textarea with line-per-prompt output)
- [ ] Caption critique in dataset viewer

### Implicit preference signals (instrument from day one)

Every time a user generates from a specific checkpoint — not the final one, just any checkpoint mid-run — that's a revealed preference signal. No ratings, no surveys. The signal is: "they stopped here and used it."

**What to record per training run** (local DB, synced to cloud):
- `base_model`, `lora_type`, `dataset_image_count`, `total_steps`
- `first_generated_at_step` — the checkpoint step the user first generated from (nullable until they do)
- `marked_final_step` — if they explicitly pick a "best" checkpoint

**What this becomes at scale**: aggregate across users → "style LoRAs on SDXL with 40–60 images →
users first generate around 6–8k steps" becomes a data-backed default, not a gut feeling. Surface this
in the UI as a subtle indicator: "most users like this around step X" during eval.

**Cost to implement**: one nullable column (`first_generated_from_step INTEGER`) on the `training_runs`
table. Write it when `/api/generate` is called and the request references a checkpoint artifact. Zero
overhead to collect, compound value over time.

- [ ] Add `first_generated_from_step` + `marked_final_step` columns to training_runs DB
- [ ] Write `first_generated_from_step` on first generate call referencing a checkpoint
- [ ] Include in cloud sync (Phase 7a) — aggregate is only useful cross-user

---

### Phase 8 — Native app (Tauri)

Wrap the Phase 6 React UI in a Tauri native app for Mac/Windows distribution.
Same UI, same Rust core, native webview instead of `localhost` in a browser.
Targets the **paying user segment**: creative professionals on laptops with no
GPU who use cloud training (Phase 7).

- [ ] Tauri shell around existing Axum + React stack
- [ ] Native file pickers, drag-and-drop images, dock/taskbar icon
- [ ] Code signing + DMG/MSI distribution
- [ ] Auto-update via Tauri updater

### Backlog — Not planned

| Feature | Why not now |
|---------|------------|
| Video model training (Wan, LTX) | Different pipeline, defer until image is solid |
| CivitAI direct pulls | Need API key setup, lower priority than HF |
| Cloud inference | Cold start + keep_warm economics are brutal |
| DAM / tagging / collections | Filesystem + `modl outputs search` is enough |
| Node/graph editor | ComfyUI owns this, don't compete |
| Multi-provider cloud (RunPod, etc.) | Get one provider working first |

---

## Business model

**CLI is free, open source. Cloud execution + artifact storage is the paid product.**

### The actual value proposition

Not "GPU time for people without GPUs." That's a commodity — Modal, RunPod, and
Lambda all sell it cheaper. The real product is **the workflow**: dataset prep →
captioning → training → generation → iterate, with zero configuration overhead.
The person who was spending a weekend reading ai-toolkit docs now spends 10 minutes.
That time savings is what people pay for. Price by outcome, not by GPU second.

### Pricing structure (don't show users GPU meters)

| Action | Price | What's included |
|--------|-------|------------------|
| Train style LoRA | ~$4–6 | Captioning + training + LoRA stored in cloud |
| Train character LoRA | ~$8–12 | Longer run, more steps |
| Generate batch (10 images) | ~$0.50–1 | With loaded LoRA, outputs stored |
| Storage | Free up to 5GB, then $2/10GB | Datasets + LoRAs + outputs |

Users never see "$0.0023/GPU-sec." They see "Train LoRA — $5." This is a
creative tool, not a cloud provider console.

### User segments

| Segment | How they use it | Revenue |
|---------|-----------------|----------|
| GPU box users (Linux) | CLI + `modl serve`, free local usage | Free tier, occasional cloud for heavy models |
| Any-machine users | CLI + cloud, data synced everywhere | Recurring — they train and generate regularly |
| Tauri app users (Mac/Windows) | Native app + cloud, no GPU needed | Highest LTV — all compute is cloud |

### Where this is going (honest take)

The endgame isn't a better ComfyUI. It's closer to what **Vercel did for deployment**:
took something that required deep expertise (servers, nginx, CI, certs) and made
it a single command. modl is doing the same for diffusion training. The open-source
CLI is the credibility layer — it proves the tool is serious and attracts the
people who would otherwise spend weekends in config files. The cloud product
monetizes the much larger group who want the output without the process.

If traction happens: the natural expansion is **shared LoRA libraries within
a team/studio** (one person trains the house style, everyone generates with it).
But don't build that now — let real users ask for it. The individual "works
anywhere" story is the one to ship first and validate.

---

## Reference docs

| Doc | What it covers | Status |
|-----|---------------|--------|
| [multi-arch-training-plan.md](multi-arch-training-plan.md) | ai-toolkit arch configs, per-model gaps | Active — Phase 3 guide |
| [specs/aitoolkit-mapping.md](specs/aitoolkit-mapping.md) | TrainJobSpec → ai-toolkit YAML field mapping | Implemented, canonical |
| [specs/jobs-schema-v1.md](specs/jobs-schema-v1.md) | Job/event/artifact JSON schemas | Implemented, canonical |
| [specs/worker-protocol.md](specs/worker-protocol.md) | JSONL protocol between Rust and Python | Implemented, canonical |
| [specs/execution-target.md](specs/execution-target.md) | Executor trait contract | Implemented (local), stubbed (cloud) |
| [specs/gpu-resource-management.md](specs/gpu-resource-management.md) | Cross-process GPU lock, VRAM budget, enhance fallback | Phase 4 spec |
| [specs/persistent-worker.md](specs/persistent-worker.md) | Daemon architecture for fast generation | Phase 5 spec |
| [archive/ui-architecture.md](archive/ui-architecture.md) | Web UI product spec | Phase 6 reference |
| [archive/cloud-plan.md](archive/cloud-plan.md) | Cloud platform architecture + pricing | Phase 7 reference |
| [archive/runtime-architecture.md](archive/runtime-architecture.md) | ComfyUI sidecar / YAML workflow vision | Aspirational, not planned |
| [archive/capability-model.md](archive/capability-model.md) | Cloud auth/quota gating | Phase 7 detail |
| [archive/runtime-profiles.md](archive/runtime-profiles.md) | Reproducible runtime manifests | Over-engineered, current venv works |

---

## What modl is NOT

- Not a node editor (use ComfyUI)
- Not a marketplace (use CivitAI)
- Not infinitely configurable (three presets, Advanced gives full YAML)
- Not a DAM (filesystem + metadata JSON is enough)
- Not an Electron app (Tauri = native webview, Rust backend, no Chromium bloat)
