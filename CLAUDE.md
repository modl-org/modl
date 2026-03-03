# CLAUDE.md — Modl: Model Manager

## What is Modl?

Modl is a CLI model manager for the AI image generation ecosystem. It handles downloading, dependency resolution, variant selection, and folder placement for models, LoRAs, VAEs, text encoders, ControlNets, and other assets used by tools like ComfyUI, A1111, and InvokeAI.

Think of it as **npm/Homebrew for image gen models**. `modl install flux-dev` downloads the model, its required VAE, its text encoders — everything — to the right folders, with verified hashes and compatibility checking.

## Project Vision

Modl is the foundational piece of a larger **modl** platform — a next-generation alternative to ComfyUI and similar tools. The full vision:

1. **modl** (this repo) — CLI model manager. Independently useful, well-crafted, and complete on its own.
2. **modl-registry** — Community-contributed model manifests (separate repo).
3. **Pipelines (future)** — LLM-first pipeline authoring via natural language instead of node graphs. Text in, pipeline out.
4. **Deploy (future)** — One-click deployment to Modal.com, Replicate, RunPod, and others.
5. **AI UX (future)** — LLM with tools/context to assist with parameter tuning, workflow sharing, and end-to-end generation.

The key differentiator vs ComfyUI: **LLM-native, text-first UX** instead of node-based visual programming. Better params input, easier sharing, easier deployment. But **this repo is strictly the model manager**. It must stand on its own before anything else gets built.

**GitHub org:** [github.com/modl](https://github.com/modl)

## Tech Stack

- **Language:** Rust (stable toolchain)
- **CLI:** `clap` v4 with derive macros
- **Terminal UX:** `indicatif` (progress bars), `console` (colors/styling), `dialoguer` (interactive prompts), `comfy-table` (tables)
- **Serialization:** `serde` + `serde_yaml` + `serde_json`
- **HTTP:** `reqwest` with `stream` + `rustls-tls` features, `tokio` async runtime
- **Database:** `rusqlite` with `bundled` feature (SQLite compiled into binary)
- **Hashing:** `sha2` crate (SHA256)
- **GPU detection:** `nvml-wrapper` with fallback to parsing `nvidia-smi` output
- **Dirs:** `dirs` crate for platform-specific paths

### Why Rust

Modl downloads and manages files. It does NOT do ML inference. No PyTorch, no CUDA, no Python runtime needed. Rust gives us:
- Single static binary (~10-15MB) — no runtime dependencies for users
- Distribution via `brew install`, `cargo install`, curl one-liner, or direct download
- Fast SHA256 verification on 24GB model files (seconds, not minutes)
- Cross-platform: Linux, macOS, Windows from one codebase

## Architecture Overview

```
┌───────────────────┐
│  Modl Registry    │  ← Git repo of YAML manifests (separate repo)
│  (GitHub repo)    │    Community contributes via PRs
└────────┬──────────┘
         │
  modl update (fetches compiled index.json)
         │
┌────────┴──────────┐
│   Modl CLI        │  ← This repo. Single Rust binary.
│   + Local DB      │    SQLite for installed state
└────┬─────────┬────┘
     │         │
  downloads    │  symlinks
     │         │
┌────┴─────┐  ┌┴──────────┐
│ ~/modl/  │  │ ComfyUI/  │
│ store/   │──│ A1111/    │
│ (content │  │ Invoke/   │
│ addressed│  │ (linked)  │
└──────────┘  └───────────┘
```

### Two Repos

1. **modl** (`modl/modl`) — The CLI tool. Rust. This repo.
2. **modl-registry** (`modl/modl-registry`) — YAML manifest files + CI that compiles `index.json`. Community contributes manifests here.

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
  root: ~/modl           # Where modl keeps its store

targets:
  - path: ~/ComfyUI
    type: comfyui
    symlink: true         # Symlink from store into ComfyUI folders

  # Can target multiple installations simultaneously
  - path: ~/stable-diffusion-webui
    type: a1111
    symlink: true
```

Layouts define where each asset type goes for each tool:
- **ComfyUI:** `models/checkpoints/`, `models/loras/`, `models/vae/`, etc.
- **A1111:** `models/Stable-diffusion/`, `models/Lora/`, `models/VAE/`, etc.
- **InvokeAI:** Uses its own model management but we can integrate
- **Custom:** User defines arbitrary paths per asset type

`modl init` auto-detects installed tools and configures this.

### Dependency Resolution

Manifests declare dependencies. Installing a checkpoint automatically installs its required VAE and text encoders:

```yaml
# flux-dev manifest declares:
requires:
  - id: flux-vae
    type: vae
  - id: t5-xxl-fp16
    type: text_encoder
  - id: clip-l
    type: text_encoder
```

`modl install flux-dev` installs all 4 items. The resolver handles:
- Transitive dependencies
- Already-installed items (skip)
- Variant matching (if user requests fp8, also get fp8 text encoders if available)
- Circular dependency detection (shouldn't happen, but be safe)

### Variant Selection

Models come in variants (fp16, fp8, GGUF quantizations). Modl auto-selects based on detected GPU VRAM:

| VRAM | flux-dev variant | Notes |
|------|-----------------|-------|
| 24GB+ | fp16 (23.8GB) | Full quality |
| 12-23GB | fp8 (11.9GB) | Slight quality reduction |
| 8-11GB | gguf-q4 (6.8GB) | Quantized, needs GGUF loader |
| <8GB | gguf-q2 (4.2GB) | Lower quality, functional |

User can always override: `modl install flux-dev --variant fp8`

### Gated Models (Authentication)

Models like Flux Dev require accepting terms on HuggingFace. Modl handles this:

1. Manifest declares `auth.provider: huggingface` and `auth.gated: true`
2. On install, CLI detects gating and guides user to accept terms + provide token
3. Token stored in `~/.modl/auth.yaml`
4. Subsequent downloads use the token automatically

Supports: HuggingFace (`hf_...` tokens), Civitai (API keys).

## Manifest Schema

Every item in the registry is a YAML file. Here's the complete schema:

### Checkpoint

```yaml
id: flux-dev                    # Unique identifier
name: "FLUX.1 Dev"              # Human-readable name
type: checkpoint                # Asset type
architecture: flux              # Model architecture family
author: black-forest-labs
license: flux-1-dev-non-commercial
homepage: https://huggingface.co/black-forest-labs/FLUX.1-dev
description: |
  High-quality text-to-image model. Best with 20-30 steps, CFG 3.5-4.

variants:
  - id: fp16
    file: flux1-dev.safetensors
    url: https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors
    sha256: "a1b2c3d4..."       # REQUIRED for verification
    size: 23800000000            # Size in bytes
    format: safetensors
    precision: fp16
    vram_required: 24576         # MB
    vram_recommended: 24576      # MB
  - id: fp8
    file: flux1-dev-fp8-e4m3fn.safetensors
    url: https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8-e4m3fn.safetensors
    sha256: "e5f6g7h8..."
    size: 11900000000
    format: safetensors
    precision: fp8-e4m3fn
    vram_required: 12288
    vram_recommended: 16384
    note: "Quantized. Slight quality reduction vs fp16."

requires:                       # Dependencies — installed automatically
  - id: flux-vae
    type: vae
    reason: "Flux models require the Flux-specific VAE"
  - id: t5-xxl-fp16
    type: text_encoder
    reason: "T5-XXL for prompt processing"
    optional_variant: t5-xxl-fp8  # Suggested for low-VRAM
  - id: clip-l
    type: text_encoder
    reason: "CLIP-L for secondary encoding"

auth:
  provider: huggingface
  terms_url: https://huggingface.co/black-forest-labs/FLUX.1-dev
  gated: true

defaults:                       # Sensible defaults (informational)
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

base_models: [flux-dev, flux-schnell]   # Compatible base models

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

preview_images:
  - https://image.civitai.com/...

tags: [portrait, skin, realistic, photography]
rating: 4.8
downloads: 12400
added: 2025-01-10
```

### Other types follow the same pattern:
- **vae:** Similar to checkpoint but simpler (usually single variant, no deps)
- **text_encoder:** Has `architecture` field (t5, clip, etc.)
- **controlnet:** Has `preprocessor` field and `base_models` compatibility
- **upscaler:** Has `scale_factor` field
- **embedding:** Has `base_models` and `trigger_words`
- **ipadapter:** Has `clip_vision_model` dependency

## CLI Commands

### modl init
Interactive first-run setup:
- Auto-detect ComfyUI / A1111 / Invoke installations
- Ask user which to target (can target multiple)
- Choose symlink mode (recommended) or direct mode
- Scan existing model files, match by hash to registry entries
- Generate `~/.modl/config.yaml`

### modl install <id> [--variant <v>] [--dry-run]
- Resolve dependencies (full install tree)
- Auto-select variant based on GPU VRAM (unless --variant specified)
- Check auth requirements, guide user if needed
- Download with progress bars (indicatif)
- Verify SHA256 hash
- Store in content-addressed store
- Create symlinks to configured targets
- Update local SQLite database
- `--dry-run` shows what would be installed without doing it

### modl uninstall <id>
- Check if other installed items depend on this
- Warn user if so, require --force to proceed
- Remove symlinks
- Mark as uninstalled in DB
- Actual store file removed on next `modl gc` (safe)

### modl list [--type <type>]
- Table output: Name, Type, Variant, Size, Location
- Filter by type: checkpoint, lora, vae, text_encoder, controlnet, upscaler, embedding, ipadapter
- Show total disk usage at bottom

### modl info <id>
- Full details: all variants, VRAM requirements, dependencies, description, tags
- If installed: show location, installed variant, disk usage
- If not installed: show download sizes, auth requirements

### modl search <query> [--type <t>] [--for <base_model>] [--tag <tag>] [--min-rating <r>]
- Search the registry index
- Filterable by type, compatible base model, tag, minimum rating
- Results: name, type, rating, downloads, size

### modl space
- Tree view of disk usage by type and by model
- Show total store size
- Suggest cleanup candidates (old versions, unused items)

### modl doctor
- Check for broken symlinks
- Verify hashes of installed files (detect corruption)
- Check LoRA/base model compatibility
- Check for missing dependencies
- Report any issues with suggested fixes

### modl gc
- Remove files in store not referenced by any installed entry
- Show space recovered
- Require confirmation

### modl link --comfyui <path> | --a1111 <path>
- Scan the tool's model folders
- Hash each file, match against registry
- Register matched files in modl' database (no copy/move)
- Set up symlink configuration for future installs

### modl auth <provider>
- Interactive: prompt for token/key
- Store in `~/.modl/auth.yaml`
- Verify token works (test API call)
- Providers: huggingface, civitai

### modl update
- Fetch latest registry index.json
- Show if any installed items have newer versions available
- Does NOT auto-upgrade (user runs `modl upgrade` for that)

### modl export
- Generate `modl.lock` file listing all installed items with exact versions and hashes
- Shareable — anyone can reproduce the environment

### modl import <modl.lock>
- Install everything from a lock file
- Respects exact variants and versions specified

### modl popular [type] [--for <base_model>] [--period <day|week|month>]
- Show trending items from registry
- Useful for discovery

## Code Style & Conventions

### Rust
- Use `clap` derive macros for CLI definition (not builder pattern)
- Use `anyhow` for error handling in CLI layer, `thiserror` for library errors
- Use `tokio` async runtime for downloads (parallel download support)
- Async only where needed (downloads, HTTP). Keep file I/O synchronous — it's simpler and fast enough
- Prefer `reqwest` streaming for large file downloads (don't buffer 24GB in memory)
- Use `indicatif::MultiProgress` for parallel download progress bars
- `serde` derive on all manifest/config structs — YAML in, structs out
- SQLite for local state (installed items, hashes, paths) — NOT JSON files
- All paths handled via `std::path::PathBuf`, cross-platform aware
- Symlinks: use `std::os::unix::fs::symlink` on Unix, `std::os::windows::fs::symlink_file` on Windows

### File Organization
- `src/cli/` — One file per command. Each exposes a function that `main.rs` calls.
- `src/core/` — Business logic. No terminal output. Returns Results.
- `src/auth/` — Auth provider implementations.
- `src/compat/` — Tool-specific folder layouts and scanning.
- `tests/` — Integration tests. Unit tests are inline (`#[cfg(test)]` modules).

### Error Handling
- CLI layer: use `anyhow::Result` with context. Print user-friendly errors.
- Core/library: use `thiserror` enums. Be specific about what went wrong.
- Never panic in normal operation. Downloads fail gracefully with retry.
- Always clean up partial downloads on failure (don't leave half-downloaded 12GB files).

### Testing
- Unit tests: inline `#[cfg(test)]` modules in each file
- Integration tests: in `tests/` directory
- Mock HTTP responses for download tests (don't hit real servers in CI)
- Test manifest parsing extensively (it's the most likely source of bugs)
- Test dependency resolution with various graph shapes
- Test symlink creation on the actual filesystem (use temp dirs)

## Registry (Separate Repo: forge-registry)

```
modl-registry/
  manifests/
    checkpoints/
      flux-dev.yaml
      flux-schnell.yaml
      sdxl-base.yaml
      ...
    loras/
      realistic-skin-v3.yaml
      ...
    vae/
    text_encoders/
    controlnet/
    upscalers/
    embeddings/
    ipadapters/
  schemas/
    checkpoint.json           # JSON Schema for validation
    lora.json
    ...
  scripts/
    build_index.py            # CI script: compile all manifests → index.json
    validate_manifests.py     # CI script: validate all YAML against schema
  index.json                  # Compiled index (auto-generated, don't edit)
  CONTRIBUTING.md
```

### CI on the registry repo:
- On every PR: validate manifest schema, check URLs are accessible (optional, can be slow)
- On merge to main: regenerate index.json, publish as GitHub Release asset
- The CLI fetches `index.json` from the latest release on `modl update`

### Initial catalog to create manually:
**Checkpoints:** flux-dev, flux-schnell, sdxl-base-1.0, sd-3.5-large, sd-3.5-medium, sd-1.5, playground-v2.5
**VAEs:** flux-vae, sdxl-vae-fp16, sd-vae-ft-mse
**Text Encoders:** t5-xxl-fp16, t5-xxl-fp8, clip-l, clip-g, clip-vit-large
**ControlNets:** depth (flux), canny (flux), depth (sdxl), canny (sdxl), openpose (sdxl)
**LoRAs:** Top 30 from Civitai by downloads for Flux + SDXL
**Upscalers:** 4x-UltraSharp, RealESRGAN-x4plus, SwinIR-4x
**IP-Adapters:** ip-adapter-plus (sdxl), ip-adapter-faceid

~80 manifests total covering 90% of what users actually download.

## Config Files

### ~/.modl/config.yaml
```yaml
# Created by `modl init`, editable by user
storage:
  root: ~/modl

targets:
  - path: ~/ComfyUI
    type: comfyui
    symlink: true
  # Can add more targets

# GPU override (auto-detected if not set)
# gpu:
#   vram_mb: 24576
```

### ~/.modl/auth.yaml
```yaml
# Created by `modl auth`
huggingface:
  token: "hf_..."
civitai:
  api_key: "..."
```

### modl.lock
```yaml
# Generated by `modl export`
# Machine-readable, reproducible environment specification
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
  - id: t5-xxl-fp16
    type: text_encoder
    sha256: "..."
  - id: clip-l
    type: text_encoder
    sha256: "..."
  - id: realistic-skin-v3
    type: lora
    sha256: "..."
```

## Important Implementation Notes

### Download Resilience
- Resume partial downloads (HTTP Range headers)
- Retry on failure (3 attempts with exponential backoff)
- Verify SHA256 after download, delete and retry on mismatch
- Clean up partial files on ctrl-c / crash (handle signals)
- Show speed, ETA, and total progress for multi-file installs

### Symlink Strategy
- Modl store: `~/modl/store/<type>/<hash>/<filename>`
- Symlinks: `~/ComfyUI/models/checkpoints/flux1-dev.safetensors` → `~/modl/store/checkpoints/<hash>/flux1-dev.safetensors`
- If the target already has a real file (not symlink) with matching hash, register it but don't replace it
- Cross-device symlinks may not work — detect and warn, suggest same-device storage

### GPU Detection
- Primary: NVML (nvidia-smi programmatic API via `nvml-wrapper`)
- Fallback: Parse `nvidia-smi` CLI output
- Fallback: No GPU detected — skip variant auto-selection, default to smallest variant with a note
- Cache GPU info in config after first detection

### First Run Experience
If no `~/.modl/config.yaml` exists, `modl install` (or any command) should suggest running `modl init` first, but still work with sensible defaults (store in `~/modl/`, no symlinks).

### Port killing (SSH safety)
When killing processes on a port (e.g. preview server restart), **always** use `lsof -sTCP:LISTEN` to match only listeners. Plain `lsof -ti :PORT` also returns PIDs of processes with client connections to that port — including VS Code Remote SSH port-forwarding. Killing those drops the SSH session. See `kill_existing_on_port()` in `src/ui/server.rs`.

### Training runs via SSH
`modl train` runs the worker as a direct child process. If the SSH session drops, SIGHUP cascades and kills training. Users should run long training jobs inside `tmux` or `screen`. A future improvement would be to daemonize the worker, but that requires the worker to write events directly to the DB instead of stdout.

## Non-Goals (for now)
- Custom node management (ComfyUI Manager's domain)
- Pipeline definitions / execution
- LLM agent / copilot
- Web dashboard / GUI
- Cloud deployment
- Model training
- Image generation / inference

These are future layers in the larger **modl** platform. This repo is the model manager only.
