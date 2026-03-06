# CLAUDE.md — Modl

## What is Modl?

Modl is a local-first image generation app. One binary. Pull models from a registry, generate images through a curated web UI, train LoRAs, organize outputs. Like a self-hosted Midjourney alternative with the simplicity of Fooocus and the model ecosystem of Ollama.

```
curl -fsSL https://modl.run/install.sh | bash
modl pull flux-schnell
modl serve
# -> browser opens, type prompt, get image
```

The core loop: **pull models -> generate images -> train LoRAs -> manage outputs**. Everything through CLI or web UI, backed by a single Rust binary + Python worker for GPU inference.

## Product Vision

Modl occupies the "Easy to use + Full control" quadrant: Fooocus simplicity + Ollama model management + LoRA training. Local-first, single binary, curated UX.

The 80/20 features that cover ~95% of daily usage:

| Feature | Priority | Status |
|---------|----------|--------|
| txt2img | P0 | Built (CLI + UI) |
| img2img | P0 | Built (UI) |
| LoRA application | P0 | Built (CLI + UI) |
| LoRA training | P0 | Built (CLI + UI) |
| Model registry + pull | P0 | Built |
| Gallery / DAM | P1 | Built |
| Inpainting | P1 | Not built |
| Upscale | P1 | Not built |
| Video gen | P2 | Not built |

What to deliberately NOT build: node/graph editor, ControlNet UI, regional prompting, custom scheduler picker, textual inversions, multi-model pipelines. ComfyUI owns that space.

### Studio / Agent Workflow (Experimental)

The Studio is modl's agentic mode: the user provides an intent ("train a LoRA on these product photos and generate lifestyle shots") plus a folder of images, and an LLM-driven agent orchestrates the entire pipeline end-to-end.

The agent loop (`core/agent.rs`) runs as a tool-use cycle:

1. **Analyze** uploaded images (VL model via `LlmBackend::vision()`)
2. **Create + caption** a dataset (`core/dataset`, per-image captioning via VL)
3. **Select base model** from installed models (`core/db` + `core/registry`)
4. **Train** a LoRA (existing training pipeline via `core/executor`)
5. **Generate** output images with the trained LoRA

Each step emits `AgentEvent`s (Thinking, ToolStart, ToolProgress, ToolComplete, OutputReady, Error) streamed to the web UI via SSE. The UI shows a real-time timeline of agent decisions and progress.

The architecture is intentionally decoupled:
- `core/agent.rs` -- Event-driven tool-use loop. Calls `LlmBackend` in a cycle, dispatches tool calls, emits events. No terminal output, no direct DB access.
- `core/agent_tools.rs` -- Tool implementations that wrap existing modl services (dataset create, train, generate). The agent doesn't have its own inference code; it calls the same functions the CLI uses.
- `core/llm.rs` -- `LlmBackend` trait with pluggable implementations: `BuiltinLlmBackend` (rule-based, zero deps), `CloudLlmBackend` (API), `LocalLlmBackend` (llama-cpp, pending).

Studio sessions are persisted in SQLite (intent, status, events, input/output images) and accessible via both the web UI (`/api/studio/sessions/*`) and the database directly.

This is experimental but first-class -- the agentic "intent in, results out" UX is the direction modl is heading. The manual CLI/UI workflow remains the primary interface for now.

**GitHub org:** [github.com/modl-org](https://github.com/modl-org)

## Tech Stack

- **Language:** Rust (stable toolchain)
- **CLI:** `clap` v4 with derive macros
- **Terminal UX:** `indicatif` (progress bars), `console` (colors/styling), `dialoguer` (interactive prompts), `comfy-table` (tables)
- **Serialization:** `serde` + `serde_yaml` + `serde_json`
- **HTTP:** `reqwest` with `stream` + `rustls-tls` features, `tokio` async runtime
- **Web server:** `axum` 0.8 with SSE (Server-Sent Events) for real-time progress
- **Database:** `rusqlite` with `bundled` feature (SQLite compiled into binary)
- **Hashing:** `sha2` crate (SHA256)
- **GPU detection:** `nvml-wrapper` with fallback to parsing `nvidia-smi` output
- **Frontend:** React 19 + TypeScript + Tailwind CSS + Radix UI + TanStack Query + Vite (compiled to `src/ui/dist/`, embedded in binary)
- **Dirs:** `dirs` crate for platform-specific paths

### Two-Process Architecture

| Process | Language | Lifecycle | Role |
|---------|----------|-----------|------|
| **modl** (Rust) | Rust | CLI or long-running (`modl serve`) | HTTP API, web UI, CLI, job orchestration, DB |
| **Worker** | Python | Auto-spawned, idle timeout | Model loading, inference, training, VRAM management |

The Rust binary handles everything except GPU compute. The Python worker handles inference and training via ai-toolkit/diffusers. Communication is via subprocess stdout (JSON events) or Unix socket (persistent worker).

## Architecture Overview

```
+-----------------------------------------------------------------+
|                     modl (Rust binary)                           |
|                                                                 |
|  CLI Layer              API Layer           Core Layer           |
|  +-- model pull/ls      +-- GET /models     +-- store.rs        |
|  +-- generate           +-- POST /generate  +-- executor.rs     |
|  +-- train              +-- POST /train     +-- dataset.rs      |
|  +-- dataset ...        +-- GET /outputs    +-- db.rs           |
|  +-- outputs            +-- GET /gpu        +-- cloud.rs        |
|  +-- worker start/stop  +-- SSE /stream     +-- artifacts.rs    |
|  +-- serve              +-- Static UI       +-- outputs.rs      |
|       |                                                         |
|       +-- starts API server + manages persistent worker         |
+-----------------------------------------------------------------+
              | REST/SSE          | Unix socket / subprocess
              |                   |
    +---------+----+    +---------+-----------+
    |  Browser     |    | Persistent Worker   |
    |  localhost   |    | (Python daemon)     |
    |  or remote   |    | model cache, VRAM   |
    +--------------+    | LoRA hot-swap       |
                        +---------------------+
```

### Data Flow

```
+-----------------------+
|  Modl Registry        |  <- Git repo of YAML manifests (separate repo)
|  (GitHub repo)        |     Community contributes via PRs
+-----------+-----------+
            |
  modl update (fetches compiled index.json)
            |
+-----------+-----------+
|   Modl CLI + Server   |  <- This repo. Single Rust binary.
|   + Local DB (SQLite) |     Tracks installed state, jobs, artifacts
+-----+----------+------+
      |          |
   downloads     |  symlinks
      |          |
+-----+------+  ++-----------+
| ~/modl/    |  | ComfyUI/   |
| store/     |--| A1111/     |
| (content   |  | Invoke/    |
| addressed) |  | (linked)   |
+------------+  +------------+
```

## Source Code Organization

```
src/
+-- main.rs              Entry point, tokio runtime, background update check
+-- cli/                  One file per command (30+ handlers)
|   +-- mod.rs            Clap definitions, command dispatch
|   +-- install.rs        modl pull / modl model pull
|   +-- generate.rs       modl generate
|   +-- train.rs          modl train
|   +-- datasets.rs       modl dataset *
|   +-- outputs.rs        modl outputs *
|   +-- serve.rs          modl serve
|   +-- worker.rs         modl worker *
|   +-- ...
+-- core/                 Business logic (27 modules). No terminal output.
|   +-- config.rs         Load/save ~/.modl/config.yaml
|   +-- db.rs             SQLite (installed models, jobs, artifacts, favorites)
|   +-- store.rs          Content-addressed storage paths + hash verification
|   +-- manifest.rs       Registry manifest type definitions
|   +-- registry.rs       Load/search registry index.json
|   +-- resolver.rs       Dependency resolution algorithm
|   +-- download.rs       Resilient HTTP streaming downloads
|   +-- executor.rs       Executor trait + LocalExecutor (subprocess management)
|   +-- cloud.rs          CloudExecutor stub (future)
|   +-- job.rs            TrainJobSpec, GenerateJobSpec (serializable specs)
|   +-- presets.rs         Training parameter resolution (preset + GPU + model)
|   +-- preflight.rs      Pre-flight checks before expensive operations
|   +-- runtime.rs        Managed Python runtime (venv, pip, ai-toolkit)
|   +-- dataset.rs        Dataset listing, validation, path resolution
|   +-- outputs.rs        Service layer for generated images (list, delete, fav)
|   +-- training.rs       Worker path resolution
|   +-- training_status.rs Training progress parsing
|   +-- artifacts.rs      Training artifact registration
|   +-- gpu.rs            GPU detection (NVML / nvidia-smi fallback)
|   +-- symlink.rs        Symlink management for tool folders
|   +-- huggingface.rs    HuggingFace API integration
|   +-- llm.rs            LlmBackend trait (local/cloud/builtin)
|   +-- agent.rs          Tool-use agent loop (event-driven)
|   +-- agent_tools.rs    Agent tool implementations
|   +-- enhance.rs        Prompt enhancement
|   +-- update_check.rs   Background CLI update check
+-- auth/                 Auth provider implementations
|   +-- huggingface.rs    HF token management
|   +-- civitai.rs        Civitai API key management
+-- compat/               Tool-specific folder layouts
|   +-- layouts.rs        ComfyUI, A1111, InvokeAI path mappings
+-- ui/                   Web UI
    +-- mod.rs            UI module
    +-- server.rs         Axum web server, SSE, all API endpoints (1600 LOC)
    +-- web/              React frontend (TypeScript + Tailwind + Radix)
    |   +-- src/
    |   |   +-- App.tsx
    |   |   +-- api.ts
    |   |   +-- components/
    |   |       +-- generate/     Generate page components
    |   |       +-- studio/       Agent/studio components
    |   |       +-- ui/           Shared UI primitives (shadcn/radix)
    |   |       +-- OutputsGallery.tsx
    |   |       +-- TrainingRuns.tsx
    |   |       +-- DatasetViewer.tsx
    |   |       +-- ...
    |   +-- package.json          React 19, TanStack Query, Radix, Tailwind
    +-- dist/             Built frontend assets (embedded in binary)
```

## Key Concepts

### Content-Addressed Storage

Models are stored by SHA256 hash in `~/modl/store/`. Symlinks with human-readable names point into the store. Benefits:
- Same model referenced by multiple manifests = one file on disk
- Hash verification is built-in (corrupted downloads caught automatically)
- `modl gc` can safely identify and remove unreferenced files

### Configurable Folder Layout

Different tools expect models in different places. Modl supports multiple layouts:

```yaml
# ~/.modl/config.yaml
storage:
  root: ~/modl

targets:
  - path: ~/ComfyUI
    type: comfyui
    symlink: true
  - path: ~/stable-diffusion-webui
    type: a1111
    symlink: true
```

Layouts define where each asset type goes for each tool:
- **ComfyUI:** `models/checkpoints/`, `models/loras/`, `models/vae/`, etc.
- **A1111:** `models/Stable-diffusion/`, `models/Lora/`, `models/VAE/`, etc.

`modl init` auto-detects installed tools and configures this.

### Dependency Resolution

Manifests declare dependencies. Installing a checkpoint automatically installs its required VAE and text encoders:

```yaml
requires:
  - id: flux-vae
    type: vae
  - id: t5-xxl-fp16
    type: text_encoder
  - id: clip-l
    type: text_encoder
```

`modl pull flux-dev` installs all 4 items. The resolver handles transitive dependencies, already-installed items (skip), variant matching, and circular dependency detection.

### Variant Selection

Models come in variants (fp16, fp8, GGUF quantizations). Modl auto-selects based on detected GPU VRAM:

| VRAM | flux-dev variant | Notes |
|------|-----------------|-------|
| 24GB+ | fp16 (23.8GB) | Full quality |
| 12-23GB | fp8 (11.9GB) | Slight quality reduction |
| 8-11GB | gguf-q4 (6.8GB) | Quantized, needs GGUF loader |
| <8GB | gguf-q2 (4.2GB) | Lower quality, functional |

User can always override: `modl pull flux-dev --variant fp8`

### Gated Models (Authentication)

Models like Flux Dev require accepting terms on HuggingFace. Modl handles this:

1. Manifest declares `auth.provider: huggingface` and `auth.gated: true`
2. On install, CLI detects gating and guides user to accept terms + provide token
3. Token stored in `~/.modl/auth.yaml`
4. Subsequent downloads use the token automatically

Supports: HuggingFace (`hf_...` tokens), Civitai (API keys).

### Executor Trait (Local vs Cloud)

The `Executor` trait abstracts job submission:
- `LocalExecutor` — Spawns Python worker subprocess, streams events via stdout
- `CloudExecutor` — Stub for future Modal.com integration (same interface)

Both CLI and web UI use the same executor. Jobs are submitted as serializable specs (`TrainJobSpec`, `GenerateJobSpec`) written to `~/.modl/runtime/jobs/`.

### Event Streaming for Long Tasks

Training and generation emit structured `JobEvent` objects (JSON via stdout). These are consumed by:
- CLI: Progress bars via `indicatif`
- Web UI: Broadcast channel -> SSE to browser

Standard interface: `mpsc::Receiver<JobEvent>` (sync) or broadcast channel (async).

## CLI Commands

### System

| Command | Description |
|---------|-------------|
| `modl init` | First-time setup — detect tools, configure storage |
| `modl doctor` | Check symlinks, hashes, deps, runtime health |
| `modl config [key] [value]` | View or update configuration |
| `modl auth <provider>` | Configure auth (HuggingFace, Civitai) |
| `modl upgrade` | Self-update modl CLI |
| `modl serve [--port] [--no-open]` | Launch web UI (axum server, opens browser) |

### Models

| Command | Description |
|---------|-------------|
| `modl pull <id>` | Download model with all dependencies (alias: `modl model pull`) |
| `modl rm <id>` | Remove an installed model |
| `modl ls [--type <type>]` | List installed models |
| `modl info <id>` | Show detailed model info |
| `modl search <query>` | Search the registry + HuggingFace |
| `modl popular` | Show trending models |
| `modl link [--comfyui <path>]` | Adopt existing model folders |
| `modl update` | Fetch latest registry index |
| `modl space` | Show disk usage breakdown |
| `modl gc` | Remove unreferenced store files |
| `modl export` / `import` | Lock files for reproducible setups |

### Generation

| Command | Description |
|---------|-------------|
| `modl generate "prompt"` | Generate images (flags: `--base`, `--lora`, `--count`, `--size`, `--steps`, `--guidance`, `--seed`, `--cloud`) |
| `modl enhance "prompt"` | AI-enhanced prompt expansion |

### Training

| Command | Description |
|---------|-------------|
| `modl train` | Train a LoRA (interactive or with flags: `--base`, `--dataset`, `--lora-type`, `--preset`, `--rank`, `--lr`, `--steps`) |
| `modl train setup` | Install training dependencies (ai-toolkit + PyTorch) |
| `modl train status [--watch]` | Monitor live training progress |

### Datasets

| Command | Description |
|---------|-------------|
| `modl dataset create <name> --from <dir>` | Create managed dataset from images |
| `modl dataset ls` | List all managed datasets |
| `modl dataset rm <name>` | Delete a dataset |
| `modl dataset validate <name>` | Validate dataset for training |
| `modl dataset resize <name> --resolution <px>` | Resize images |
| `modl dataset tag <name>` | Auto-tag images (VL model) |
| `modl dataset caption <name>` | Auto-caption images (VL model) |
| `modl dataset prepare <name> --from <dir>` | Full pipeline (create + resize + caption) |

### Outputs

| Command | Description |
|---------|-------------|
| `modl outputs ls [--limit] [--favorites]` | List generated images |
| `modl outputs show <id>` | Show image metadata |
| `modl outputs open <id>` | Open image in viewer |
| `modl outputs search <query>` | Search by prompt text |
| `modl outputs fav <id>` / `unfav <id>` | Toggle favorites |
| `modl outputs rm <id>` | Delete output |

### Worker & Runtime

| Command | Description |
|---------|-------------|
| `modl worker start` / `stop` / `status` | Manage persistent GPU worker |
| `modl runtime install` | Install managed Python runtime |
| `modl runtime status` / `doctor` | Check runtime health |
| `modl runtime upgrade` / `reset` | Manage runtime lifecycle |

### LLM (experimental)

| Command | Description |
|---------|-------------|
| `modl llm pull <model>` | Download GGUF models |
| `modl llm chat <prompt>` | Local LLM inference |
| `modl llm ls` | List installed LLMs |

## Web UI

The web UI (`modl serve`) is the primary user experience. Built with React 19 + TypeScript + Tailwind + Radix UI.

### Pages

1. **Generate** — Prompt input, model/LoRA selection, aspect ratio, sampling params, img2img upload, output feed
2. **Gallery** — Grid of all outputs, search, filter, metadata panel, favorites, delete
3. **Training** — Training run list, live progress bars, sample image viewer, new training form
4. **Studio** — Agent-driven workflow (experimental): intent input, timeline, result gallery
5. **Datasets** — Dataset viewer

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/gpu` | GPU status |
| `GET /api/models` | List installed models |
| `POST /api/generate` | Submit generation job |
| `GET /api/generate/stream` | SSE stream of generation progress |
| `GET /api/generate/queue` | Queue status |
| `GET /api/training` | List training runs |
| `POST /api/train` | Submit training job |
| `POST /api/enhance` | Enhance prompt |
| `GET /api/datasets` | List datasets |
| `GET /api/outputs` | List generated images |
| `DELETE /api/outputs/:id` | Delete output |
| `POST /api/outputs/:id/fav` | Toggle favorite |
| `GET /files/*` | Serve generated files |

## Manifest Schema

Every item in the registry is a YAML file. Here's the complete schema:

### Checkpoint

```yaml
id: flux-dev
name: "FLUX.1 Dev"
type: checkpoint
architecture: flux
author: black-forest-labs
license: flux-1-dev-non-commercial
homepage: https://huggingface.co/black-forest-labs/FLUX.1-dev
description: |
  High-quality text-to-image model. Best with 20-30 steps, CFG 3.5-4.

variants:
  - id: fp16
    file: flux1-dev.safetensors
    url: https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors
    sha256: "a1b2c3d4..."
    size: 23800000000
    format: safetensors
    precision: fp16
    vram_required: 24576
    vram_recommended: 24576
  - id: fp8
    file: flux1-dev-fp8-e4m3fn.safetensors
    url: https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8-e4m3fn.safetensors
    sha256: "e5f6g7h8..."
    size: 11900000000
    format: safetensors
    precision: fp8-e4m3fn
    vram_required: 12288
    vram_recommended: 16384

requires:
  - id: flux-vae
    type: vae
  - id: t5-xxl-fp16
    type: text_encoder
    optional_variant: t5-xxl-fp8
  - id: clip-l
    type: text_encoder

auth:
  provider: huggingface
  terms_url: https://huggingface.co/black-forest-labs/FLUX.1-dev
  gated: true

defaults:
  steps: 20
  cfg: 3.5
  sampler: euler
  scheduler: normal

tags: [flux, text-to-image, image-to-image, high-quality]
added: 2024-08-01
updated: 2025-01-15
```

### LoRA

```yaml
id: realistic-skin-v3
name: "Realistic Skin Texture v3"
type: lora
author: civitai-user-xyz
license: cc-by-nc-4.0
base_models: [flux-dev, flux-schnell]

file:
  url: https://civitai.com/api/download/models/123456
  sha256: "m3n4o5p6..."
  size: 186000000
  format: safetensors

auth:
  provider: civitai
  gated: false

trigger_words: ["realistic skin texture"]
recommended_weight: 0.7
weight_range: [0.4, 1.0]
tags: [portrait, skin, realistic, photography]
rating: 4.8
downloads: 12400
added: 2025-01-10
```

### Other asset types
- **vae:** Single variant, no deps
- **text_encoder:** Has `architecture` field (t5, clip, etc.)
- **controlnet:** Has `preprocessor` field and `base_models`
- **upscaler:** Has `scale_factor` field
- **embedding:** Has `base_models` and `trigger_words`
- **ipadapter:** Has `clip_vision_model` dependency

## Config Files

### ~/.modl/config.yaml
```yaml
storage:
  root: ~/modl

targets:
  - path: ~/ComfyUI
    type: comfyui
    symlink: true

# gpu:
#   vram_mb: 24576
```

### ~/.modl/auth.yaml
```yaml
huggingface:
  token: "hf_..."
civitai:
  api_key: "..."
```

### modl.lock
```yaml
generated: 2026-02-22T14:30:00Z
modl_version: 0.1.0

items:
  - id: flux-dev
    type: checkpoint
    variant: fp16
    sha256: "a1b2c3d4..."
  - id: flux-vae
    type: vae
    sha256: "x1y2z3..."
```

## Design Principles

### CLI is the master for all operations

Every user-facing operation (install, delete, favourite, generate, etc.) **must** be available as a CLI command first. The web UI is a convenience layer that calls the same underlying logic -- it must never be the only way to do something.

When adding a new feature:
1. Implement the core logic in `src/core/` (database, filesystem, etc.)
2. Expose it as a CLI subcommand in `src/cli/`
3. Wire it up in the web UI server (`src/ui/server.rs`) and frontend (`src/ui/web/`)

The CLI and web UI should have **full feature parity**. If the UI can do it, the CLI can do it, and vice versa.

### Service layer abstraction (`src/core/`)

The web UI server and CLI must **never** talk to the database or filesystem directly for domain operations. All mutations (delete, favourite, etc.) go through a service layer in `src/core/`.

```
  CLI handler (src/cli/)           UI handler (src/ui/server.rs)
         |                                  |
         +----------+          +------------+
                    v          v
            Service layer (src/core/outputs.rs, etc.)
                    |
                    v
            DB + Filesystem
```

**Example:** `core::outputs` handles `list_outputs()`, `delete_output()`, `toggle_favorite()`. Both `cli::outputs` and `ui::server` call these functions instead of touching `Database` or `std::fs` directly.

### Pluggable backends (traits)

- **Executor trait:** `LocalExecutor` vs `CloudExecutor` (same interface for job submission)
- **LlmBackend trait:** `BuiltinLlmBackend` vs `CloudLlmBackend` vs `LocalLlmBackend`

Swap implementations at instantiation; calling code doesn't care which backend runs.

### Specification-driven execution

`TrainJobSpec` and `GenerateJobSpec` are serializable (YAML/JSON). Specs are:
- Written to disk before execution
- Stored in DB for provenance
- Immutable after submission
- Enable reproducibility, async submission, and audit trail

## Code Style & Conventions

### Rust
- Use `clap` derive macros for CLI definition (not builder pattern)
- Use `anyhow` for error handling in CLI layer, `thiserror` for library errors
- Use `tokio` async runtime for downloads (parallel download support)
- Async only where needed (downloads, HTTP, web server). Keep file I/O synchronous.
- Prefer `reqwest` streaming for large file downloads (don't buffer 24GB in memory)
- `serde` derive on all manifest/config structs
- SQLite for local state -- NOT JSON files
- All paths handled via `std::path::PathBuf`, cross-platform aware

### CLI handlers should be thin
- Parse args, call `core/` service, format output
- Business logic belongs in `src/core/`, not in CLI handlers
- Terminal output (progress bars, styled text) is the only CLI-specific code

### Error Handling
- CLI layer: use `anyhow::Result` with context. Print user-friendly errors.
- Core/library: use `thiserror` enums. Be specific about what went wrong.
- Never panic in normal operation. Downloads fail gracefully with retry.
- Always clean up partial downloads on failure.

### Testing
- Unit tests: inline `#[cfg(test)]` modules in each file
- Integration tests: in `tests/` directory
- Mock HTTP responses for download tests
- Test manifest parsing extensively
- Test dependency resolution with various graph shapes
- Test symlink creation on the actual filesystem (use temp dirs)

## Important Implementation Notes

### Download Resilience
- Resume partial downloads (HTTP Range headers)
- Retry on failure (3 attempts with exponential backoff)
- Verify SHA256 after download, delete and retry on mismatch
- Clean up partial files on ctrl-c / crash (handle signals)
- Show speed, ETA, and total progress for multi-file installs

### Symlink Strategy
- Modl store: `~/modl/store/<type>/<hash>/<filename>`
- Symlinks: `~/ComfyUI/models/checkpoints/flux1-dev.safetensors` -> store
- If the target already has a real file with matching hash, register it but don't replace it
- Cross-device symlinks may not work -- detect and warn

### GPU Detection
- Primary: NVML (nvidia-smi programmatic API via `nvml-wrapper`)
- Fallback: Parse `nvidia-smi` CLI output
- Fallback: No GPU detected -- default to smallest variant with a note
- Cache GPU info in config after first detection

### Port killing (SSH safety)
When killing processes on a port (e.g. server restart), **always** use `lsof -sTCP:LISTEN` to match only listeners. Plain `lsof -ti :PORT` also returns PIDs of processes with client connections to that port -- including VS Code Remote SSH port-forwarding. Killing those drops the SSH session. See `kill_existing_on_port()` in `src/ui/server.rs`.

### Training runs via SSH
`modl train` runs the worker as a direct child process. If the SSH session drops, SIGHUP cascades and kills training. Users should run long training jobs inside `tmux` or `screen`.

### Frontend build
The web frontend is a React app in `src/ui/web/`. Build with `cd src/ui/web && npm run build`, which outputs to `src/ui/dist/`. The dist files are embedded in the Rust binary. Don't commit `node_modules/`.

## Registry (Separate Repo: modl-registry)

```
modl-registry/
  manifests/
    checkpoints/
      flux-dev.yaml
      flux-schnell.yaml
      ...
    loras/
    vae/
    text_encoders/
    controlnet/
    upscalers/
    embeddings/
    ipadapters/
  schemas/
    checkpoint.json
    lora.json
    ...
  scripts/
    build_index.py
    validate_manifests.py
  index.json                  # Auto-generated, don't edit
```

CI on the registry repo:
- On every PR: validate manifest schema
- On merge to main: regenerate index.json, publish as GitHub Release asset
- The CLI fetches `index.json` from the latest release on `modl update`

## What is NOT built yet

- **Persistent worker warm path** -- Worker restarts per job; no VRAM caching between generations yet
- **CloudExecutor** -- Stub only. Modal.com integration is Phase 4.
- **Local LLM inference** -- llama-cpp-2 integration pending
- **Inpainting / upscale** -- Not implemented
- **Video generation** -- Phase 2
