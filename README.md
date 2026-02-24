# mods

**CLI model manager for the AI image generation ecosystem.**

`mods install flux-dev` downloads the model, its required VAE, its text encoders — everything — to the right folders, with verified hashes and compatibility checking.

Think of it as **npm/Homebrew for image gen models**.

**[Website](https://mods.pedroalonso.net)** · **[Docs](https://mods.pedroalonso.net/docs)** · **[Model Registry](https://github.com/modshq-org/mods-registry)** · **[Issues](https://github.com/modshq-org/mods/issues)**

## Quick Start

```bash
# Install mods
curl -fsSL https://raw.githubusercontent.com/modshq-org/mods/main/install.sh | sh

# Or build from source
# git clone https://github.com/modshq-org/mods && cd mods && cargo install --path .

# First-time setup (auto-detects ComfyUI, A1111, etc.)
mods init

# Install a model (auto-selects variant for your GPU)
mods install flux-dev

# See what's installed
mods list

# Search for LoRAs
mods search "realistic" --type lora
```

## How It Works

Mods keeps **one copy** of every model in a content-addressed store (`~/mods/store/`). Your tools see symlinks that point into the store.

```
~/mods/store/checkpoint/a1b2c3.../flux1-dev.safetensors   ← single file on disk
    ↑                       ↑
    │                       └── ~/A1111/models/Stable-diffusion/flux1-dev.safetensors (symlink)
    └── ~/ComfyUI/models/checkpoints/flux1-dev.safetensors (symlink)
```

Install once, use everywhere. No duplicate 24GB files across tools.

## Already Have Models?

If you already have models downloaded in ComfyUI or A1111, `mods link` adopts the ones it recognizes — moves them into the store and replaces them with symlinks. Your tools keep working, nothing breaks.

```bash
# Adopt existing ComfyUI models
mods link --comfyui ~/ComfyUI

# Or A1111
mods link --a1111 ~/stable-diffusion-webui
```

What happens:
- Mods scans your model folders and hashes each file
- Files that match the registry are **moved** to `~/mods/store/` and replaced with symlinks
- Files mods doesn't recognize are **left untouched** (custom merges, community models, etc.)
- Your tools don't notice the difference — symlinks are transparent

After linking, `mods install` will automatically symlink new models into all your configured tools.

## Features

- **Dependency resolution** — installs required VAE, text encoders automatically
- **GPU-aware variant selection** — picks fp16/fp8/GGUF based on your VRAM
- **Content-addressed storage** — deduplicated, hash-verified downloads
- **Multi-tool support** — symlinks into ComfyUI, A1111, and more (InvokeAI planned)
- **Adopt existing models** — `mods link` migrates your current library without re-downloading
- **Resumable downloads** — partial downloads resume automatically
- **Lock files** — `mods export` / `mods import` for reproducible environments

## Commands

| Command | Description |
|---------|-------------|
| `mods init` | First-time setup — detect tools, configure storage |
| `mods install <id>` | Install a model with all dependencies |
| `mods uninstall <id>` | Remove an installed model |
| `mods list` | List installed models |
| `mods info <id>` | Show detailed info about a model |
| `mods search <query>` | Search the registry |
| `mods link --comfyui <path>` | Adopt existing models into mods |
| `mods doctor` | Check for broken symlinks and missing files |
| `mods space` | Show disk usage breakdown |
| `mods gc` | Remove unreferenced files from the store |
| `mods update` | Fetch latest registry index |
| `mods export` / `mods import` | Shareable lock files for reproducible setups |

## Variant Selection

Models come in multiple variants. Mods picks the best one for your GPU automatically:

| VRAM | Variant | Notes |
|------|---------|-------|
| 24GB+ | fp16 | Full quality |
| 12-23GB | fp8 | Slight quality reduction |
| 8-11GB | gguf-q4 | Quantized, needs GGUF loader |
| <8GB | gguf-q2 | Lower quality, functional |

Override with `mods install flux-dev --variant fp8`.

## Part of the modshq Platform

Mods is the foundation of a larger platform:

| Layer | Repo | Status |
|-------|------|--------|
| Website | [mods.pedroalonso.net](https://mods.pedroalonso.net) | Live |
| Model Manager | [`modshq/mods`](https://github.com/modshq-org/mods) | **This repo** — in progress |
| Registry | [`modshq/mods-registry`](https://github.com/modshq-org/mods-registry) | In progress |
| Pipeline Authoring | TBD | Future — LLM-first pipeline creation |
| Deploy | TBD | Future — one-click to Modal/Replicate/RunPod |

## License

MIT
