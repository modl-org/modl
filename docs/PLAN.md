# modl — Plan & Status

> Last updated: 2026-03-06
> Single source of truth. Everything else in `docs/` is reference material.

## What modl is

The antidote to ComfyUI config hell and 2-hour YouTube tutorials.

Two paths to the same product:

1. **Local (free)** — Linux + GPU. `modl pull flux-dev && modl train`. Everything runs on your hardware. CLI + web UI (`modl serve`). Power users, hobbyists, tinkerers.

2. **Cloud (paid)** — Mac / Windows / any machine. Tauri desktop app, cloud subscription. Same UI, same workflows, zero GPU needed. Creative professionals who want results, not infrastructure.

Both paths use the same Rust binary, same React UI, same agent. The only difference is which `Executor` and `LlmBackend` implementation runs behind the trait.

---

## Current State

**~17K Rust LOC, ~2K Python LOC. Active branch: `feat/serve-ui`.**

| Area | Status |
|------|--------|
| Model pull/ls/rm/search/info | Done — registry (68 models), HF direct pulls, content-addressed store |
| Model link (ComfyUI/A1111) | Done — auto-detect layouts, cross-device fallback |
| Config / Auth / GPU detect | Done — YAML config, HF/CivitAI tokens, NVML + nvidia-smi |
| Dataset create/validate/caption | Done — Florence-2/BLIP auto-captioning, tag, resize |
| Training (presets + executor) | Done — SDXL LoRA tested & working, z-image-turbo setup ready to test, Flux config ready |
| Generation (CLI + UI) | Done — Flux/SDXL/SD1.5, LoRA stacking, `--json` output |
| Output management | Done — list/show/open/search, gallery with metadata lightbox |
| Runtime bootstrap | Done — Python venv + ai-toolkit install |
| Doctor/GC/Export/Import | Done — orphan detection, repair, lockfile round-trip |
| `modl upgrade` | Done — self-update from GitHub releases |
| Web UI (`modl serve`) | Done — React/Vite, Generate + Outputs + Datasets + Training tabs |
| Prompt enhance | Done — `PromptEnhancer` trait, builtin rules + LLM backends |
| LLM runtime (`LlmBackend` trait) | Done — Cloud + Builtin backends working. Local backend stubbed (needs llama-cpp-2 wiring) |
| Agent framework | Done — tool-use loop, 7 tools, session management |
| Studio backend (Axum) | Done — session CRUD, image upload, SSE streaming |
| Studio UI (React) | Done — upload zone, intent input, agent timeline, result gallery |
| llama-cpp-2 actual inference | Not started |
| GPU resource lock | Not started |
| Persistent Python worker | Not started |
| Cloud execution (Modal) | Not started |
| Auth + sync (R2) | Not started |
| Tauri desktop app | Not started |
| Billing (Stripe) | Not started |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      React UI (Vite, TSX)                    │
│  Generate │ Outputs │ Datasets │ Training │ Studio           │
└──────────────────────────┬──────────────────────────────────┘
                           │
              Axum HTTP + SSE (same binary)
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                      modl (Rust binary)                      │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ LlmBackend  │  │   Executor   │  │  PromptEnhancer   │   │
│  │   (trait)    │  │   (trait)    │  │     (trait)        │   │
│  ├─────────────┤  ├──────────────┤  ├───────────────────┤   │
│  │ Local (llama│  │ Local (spawn │  │ LLM-based         │   │
│  │   -cpp-2)   │  │   Python)    │  │ Builtin rules     │   │
│  │ Cloud (HTTP)│  │ Cloud (Modal │  │ Cloud API         │   │
│  │ Builtin     │  │   via API)   │  │                   │   │
│  └─────────────┘  └──────────────┘  └───────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              Agent (tool-use loop)                   │     │
│  │  analyze → dataset → caption → train → generate     │     │
│  │  Uses LlmBackend + Executor traits, backend-agnostic│     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  Core: presets, preflight, db, registry, store, gpu, outputs │
│                                                              │
│  Python runtime (ai-toolkit): train, generate, caption       │
└──────────────────────────────────────────────────────────────┘
```

**Trait pattern**: `LlmBackend`, `Executor`, and `PromptEnhancer` are all Rust traits with pluggable implementations. Local user gets local impls. Cloud user gets HTTP impls. The agent and UI don't know the difference.

---

## Phases

### Phase 1 — Core CLI (done)

Install, setup, model management, dataset prep, training, generation, output management. Web UI with Generate + Outputs + Datasets + Training tabs. Prompt enhance with builtin rules.

### Phase 2 — Studio + LLM agent (done)

`LlmBackend` trait with Local/Cloud/Builtin backends. Agent framework with 7 tools. Studio UI with upload → intent → timeline → results flow. SSE streaming for real-time progress.

Remaining:
- [ ] Wire `llama-cpp-2` crate into `LocalLlmBackend` (text + VL inference)
- [ ] Test VL inference with Qwen3-VL GGUF
- [ ] Connect Studio agent to actual executor for real training/generation

### Phase 3 — GPU resource management + persistent worker

Cross-process GPU lock so training, generation, captioning, and LLM don't collide. See [specs/gpu-resource-management.md](specs/gpu-resource-management.md).

- [ ] `GpuLock` file-based lock + `GpuGuard` RAII + `modl gpu` command
- [ ] Gate GPU commands behind lock
- [ ] Persistent Python worker (daemon on Unix socket, LRU model cache)
- [ ] Replace server `AtomicBool` with `GpuLock`
- [ ] `/api/gpu` endpoint with lock activity + VRAM info

### Phase 4 — Cloud platform (auth + execution + sync)

Everything needed for `--cloud` flag to work end-to-end.

**Auth + storage:**
- [ ] `modl auth login` — device flow, JWT in `~/.modl/config.yaml`
- [ ] modl API service (Rust/Axum): auth, R2 presigned URL broker
- [ ] `modl sync` — reconcile local DB with cloud artifact registry
- [ ] R2 storage: datasets (content-addressed upload), LoRAs (bidirectional), outputs (pull after generate)

**Cloud execution:**
- [ ] `CloudExecutor` — submit train/generate jobs to Modal via modl API
- [ ] Modal GPU functions (train + generate)
- [ ] SSE job stream proxied through modl API
- [ ] Auto-sync: datasets before cloud job, LoRAs + outputs after completion

**Cloud LLM** (already works — `CloudLlmBackend` calls modl API):
- [ ] Deploy LLM inference endpoint (OpenAI-compatible `/v1/chat/completions` + `/v1/vision`)

### Phase 5 — Tauri desktop app

Wrap the React UI in a Tauri v2 native app for Mac/Windows distribution. Same Rust core, native webview.

- [ ] Tauri shell around existing Axum + React stack
- [ ] Native file pickers, drag-and-drop, dock/taskbar icon
- [ ] Code signing + DMG / MSI distribution
- [ ] Auto-update via Tauri updater
- [ ] Cloud-first defaults (no local GPU assumed)
- [ ] Onboarding flow: sign up → subscribe → start creating

### Phase 6 — Billing + launch

- [ ] Stripe integration — subscription management
- [ ] Usage tracking per job
- [ ] Landing page + download links
- [ ] Pricing enforced server-side (API gate checks subscription tier)

---

## Business model

**CLI is free, open source. Cloud execution is the paid product.**

### Pricing (subscription, not per-job)

| Tier | Price | What's included |
|------|-------|-----------------|
| **Free** | $0 | CLI + local execution. Full feature set on your own GPU. |
| **Pro** | $20/mo | 10 training runs/mo, 500 generations/mo, 10GB cloud storage |
| **Max** | $100/mo | Unlimited training, unlimited generation, 100GB cloud storage, priority GPU |

Users never see GPU seconds or compute meters. They see "Train LoRA" and it works. Price by outcome, not by infrastructure.

### Unit economics (from cloud-plan.md analysis)

| Operation | Cloud cost (Modal) | Margin at Pro price |
|-----------|--------------------|---------------------|
| Train character LoRA (1000 steps, A100) | ~$2.50 | ~60% at 10 runs/mo |
| Train style LoRA (500 steps, A100) | ~$1.25 | ~75% |
| Generate 10 images (Flux, A100) | ~$0.15 | High margin |
| Cloud storage (R2, 10GB) | ~$0.15/mo | Negligible |

### User segments

| Segment | Setup | Revenue |
|---------|-------|---------|
| GPU box users (Linux) | CLI + `modl serve`, free | Free tier, occasional cloud for heavy models |
| Any-machine users | CLI + `--cloud` flag | Pro/Max subscribers |
| Tauri app users (Mac/Windows) | Native app, no GPU | Highest LTV — all compute is cloud |

### Where this is going

Closer to what **Vercel did for deployment**: took something that required deep expertise and made it a single command. The open-source CLI is the credibility layer. The cloud product monetizes the much larger group who want the output without the process.

Natural expansion: shared LoRA libraries within a team/studio. Don't build that now.

---

## Key files

| File | Role |
|------|------|
| `src/core/llm.rs` | `LlmBackend` trait, Cloud/Local/Builtin backends, resolution logic |
| `src/core/agent.rs` | Agent loop, session state, tool definitions, system prompt |
| `src/core/agent_tools.rs` | Tool implementations wrapping existing modl services |
| `src/core/enhance.rs` | `PromptEnhancer` trait, builtin enhancer |
| `src/core/presets.rs` | Training presets per model family |
| `src/core/preflight.rs` | Pre-train checks (runtime, base model, deps, auth) |
| `src/core/job.rs` | Job spec types, LoRA types |
| `src/core/db.rs` | SQLite database (state.db) |
| `src/ui/server.rs` | Axum server — all API endpoints, SSE streaming |
| `src/ui/web/` | React frontend (Vite, TSX) |
| `src/cli/` | CLI commands (one file per subcommand) |
| `python/modl_worker/` | Python runtime (ai-toolkit adapters) |

## Reference docs

| Doc | Status |
|-----|--------|
| [specs/llm-runtime.md](specs/llm-runtime.md) | Current — Rust-native LlmBackend trait architecture |
| [specs/gpu-resource-management.md](specs/gpu-resource-management.md) | Spec — Phase 3, not yet implemented |
| [specs/persistent-worker.md](specs/persistent-worker.md) | Spec — Phase 3, not yet implemented |
| [specs/jobs-schema-v1.md](specs/jobs-schema-v1.md) | Implemented, canonical |
| [specs/worker-protocol.md](specs/worker-protocol.md) | Implemented, canonical |
| [specs/execution-target.md](specs/execution-target.md) | Implemented (local), stubbed (cloud) |

Private docs (modl-org/docs/plans/):
| Doc | Status |
|-----|--------|
| cloud-plan.md | Valuable — business model, pricing, Modal architecture, unit economics |
| multi-arch-training-plan.md | Active — per-model training gap analysis |
| ui-architecture.md | Partially stale — product vision still relevant, tech stack section wrong (says vanilla JS, actually React/Vite/TSX from the start) |
| capability-model.md | Future — cloud auth/quota gating, not yet implemented |
| runtime-architecture.md | Historical — aspirational ComfyUI sidecar vision, path not taken |
| runtime-profiles.md | Over-engineered — simpler approach implemented |

---

## What modl is NOT

- Not a node editor (use ComfyUI)
- Not a marketplace (use CivitAI)
- Not infinitely configurable (three presets, Advanced gives full control)
- Not an Electron app (Tauri = native webview, Rust backend, no Chromium bloat)
- Not a general chatbot — the LLM is a tool for the diffusion workflow, not a chat interface
