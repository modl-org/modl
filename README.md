# mods

**CLI model manager for the AI image generation ecosystem.**

`mods install flux-dev` downloads the model, its required VAE, its text encoders — everything — to the right folders, with verified hashes and compatibility checking.

Think of it as **npm/Homebrew for image gen models**.

## Quick Start

```bash
# Install mods
cargo install --path .

# First-time setup (auto-detects ComfyUI, A1111, etc.)
mods init

# Install a model (auto-selects variant for your GPU)
mods install flux-dev

# See what's installed
mods list

# Search for LoRAs
mods search "realistic" --type lora
```

## Features

- **Dependency resolution** — installs required VAE, text encoders automatically
- **GPU-aware variant selection** — picks fp16/fp8/GGUF based on your VRAM
- **Content-addressed storage** — deduplicated, hash-verified downloads
- **Multi-tool support** — symlinks into ComfyUI, A1111, InvokeAI simultaneously
- **Resumable downloads** — partial downloads resume automatically
- **Lock files** — `mods export` / `mods import` for reproducible environments

## Part of the modshq Platform

Mods is the foundation of a larger platform:

| Layer | Repo | Status |
|-------|------|--------|
| Model Manager | `modshq/mods` | **This repo** — in progress |
| Registry | `modshq/mods-registry` | Coming soon |
| Pipeline Authoring | TBD | Future — LLM-first pipeline creation |
| Deploy | TBD | Future — one-click to Modal/Replicate/RunPod |

## License

MIT
