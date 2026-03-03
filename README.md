# modl

**The opinionated toolkit for AI image generation.** Models, training, inference — one CLI.

`modl model pull flux-dev` downloads the model, its required VAE, its text encoders — everything — to the right folders, with verified hashes and compatibility checking. Then `modl train` fine-tunes a LoRA on your photos. Then `modl generate` creates images.

**[Website](https://modl.run)** · **[Docs](https://modl.run/docs)** · **[Model Registry](https://github.com/modl-org/modl-registry)** · **[Issues](https://github.com/modl-org/modl/issues)**

## Quick Start

```bash
# Install modl
curl -fsSL https://raw.githubusercontent.com/modl-org/modl/main/install.sh | sh

# Or build from source
# git clone https://github.com/modl-org/modl && cd modl && cargo install --path .

# First-time setup (auto-detects ComfyUI, A1111, etc.)
modl init

# Pull a model (auto-selects variant for your GPU)
modl model pull flux-dev

# See what's installed
modl model ls

# Search for LoRAs
modl model search "realistic" --type lora
```

## The Full Journey

```bash
# 1. Pull a base model
modl model pull flux-schnell

# 2. Prepare a training dataset
modl dataset create products --from ~/photos/my-products/

# 3. Train a LoRA
modl train --dataset products --base flux-schnell --name product-v1

# 4. Generate images (coming soon)
modl generate "a photo of OHWX on marble countertop" --lora product-v1
```

## How It Works

Modl keeps **one copy** of every model in a content-addressed store (`~/modl/store/`). Your tools see symlinks that point into the store.

```
~/modl/store/checkpoint/a1b2c3.../flux1-dev.safetensors   ← single file on disk
    ↑                       ↑
    │                       └── ~/A1111/models/Stable-diffusion/flux1-dev.safetensors (symlink)
    └── ~/ComfyUI/models/checkpoints/flux1-dev.safetensors (symlink)
```

Install once, use everywhere. No duplicate 24GB files across tools.

## Already Have Models?

If you already have models downloaded in ComfyUI or A1111, `modl model link` adopts the ones it recognizes — moves them into the store and replaces them with symlinks. Your tools keep working, nothing breaks.

```bash
# Adopt existing ComfyUI models
modl model link --comfyui ~/ComfyUI

# Or A1111
modl model link --a1111 ~/stable-diffusion-webui
```

What happens:
- Modl scans your model folders and hashes each file
- Files that match the registry are **moved** to `~/modl/store/` and replaced with symlinks
- Files modl doesn't recognize are **left untouched** (custom merges, community models, etc.)
- Your tools don't notice the difference — symlinks are transparent

After linking, `modl model pull` will automatically symlink new models into all your configured tools.

## Features

- **Dependency resolution** — `modl model pull flux-dev` installs required VAE, text encoders automatically
- **GPU-aware variant selection** — picks fp16/fp8/GGUF based on your VRAM
- **Content-addressed storage** — deduplicated, hash-verified downloads
- **Multi-tool support** — symlinks into ComfyUI, A1111, and more (InvokeAI planned)
- **Adopt existing models** — `modl model link` migrates your current library without re-downloading
- **Resumable downloads** — partial downloads resume automatically
- **Lock files** — `modl model export` / `modl model import` for reproducible environments
- **LoRA training** — opinionated presets (Quick/Standard/Advanced) powered by ai-toolkit
- **Managed runtime** — auto-installs Python, PyTorch, ai-toolkit — no conda/venv juggling
- **Dataset management** — organize, validate, and caption training images

## Commands

### System

| Command | Description |
|---------|-------------|
| `modl init` | First-time setup — detect tools, configure storage |
| `modl doctor` | Check for broken symlinks, missing deps, corrupt files |
| `modl config [key] [value]` | View or update configuration |
| `modl auth <provider>` | Configure authentication (HuggingFace, Civitai) |
| `modl upgrade` | Update modl CLI to the latest release |

### Models (`modl model`)

| Command | Description |
|---------|-------------|
| `modl model pull <id>` | Download a model with all dependencies |
| `modl model rm <id>` | Remove an installed model |
| `modl model ls` | List installed models |
| `modl model info <id>` | Show detailed info about a model |
| `modl model search <query>` | Search the registry |
| `modl model popular` | Show trending models |
| `modl model link` | Adopt existing tool model folders |
| `modl model update` | Fetch latest registry index |
| `modl model space` | Show disk usage breakdown |
| `modl model gc` | Remove unreferenced files from the store |
| `modl model export` / `import` | Shareable lock files for reproducible setups |

### Training (`modl train`)

| Command | Description |
|---------|-------------|
| `modl train` | Train a LoRA (interactive or with flags) |
| `modl train setup` | Install training dependencies (ai-toolkit + PyTorch) |

### Datasets (`modl dataset`)

| Command | Description |
|---------|-------------|
| `modl dataset create <name> --from <dir>` | Create a managed dataset from images |
| `modl dataset ls` | List all managed datasets |
| `modl dataset validate <name>` | Validate a dataset for training |

### Runtime (`modl runtime`)

| Command | Description |
|---------|-------------|
| `modl runtime install` | Install managed Python runtime |
| `modl runtime status` | Show runtime installation status |
| `modl runtime doctor` | Run runtime health checks |
| `modl runtime bootstrap` | Bootstrap environment and install deps |
| `modl runtime upgrade` | Upgrade runtime to latest version |
| `modl runtime reset` | Reset runtime state |

## Variant Selection

Models come in multiple variants. Modl picks the best one for your GPU automatically:

| VRAM | Variant | Notes |
|------|---------|-------|
| 24GB+ | fp16 | Full quality |
| 12-23GB | fp8 | Slight quality reduction |
| 8-11GB | gguf-q4 | Quantized, needs GGUF loader |
| <8GB | gguf-q2 | Lower quality, functional |

Override with `modl model pull flux-dev --variant fp8`.

## License

MIT
