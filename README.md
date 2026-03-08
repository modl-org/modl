# modl

**Local-first AI image generation toolkit.** Pull models, train LoRAs, generate images. One CLI, no glue code.

```bash
modl pull flux-schnell          # download model + all dependencies
modl generate "a cat on mars"   # generate an image
modl train --dataset ./photos --base flux-schnell --name my-v1   # train a LoRA
```

**[Website](https://modl.run)** · **[Docs](https://modl.run/docs)** · **[Model Registry](https://github.com/modl-org/modl-registry)** · **[Issues](https://github.com/modl-org/modl/issues)**

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/modl-org/modl/main/install.sh | sh
```

Or build from source:

```bash
git clone https://github.com/modl-org/modl && cd modl && cargo install --path .
```

## Quick Start

```bash
# First-time setup (auto-detects ComfyUI, A1111, etc.)
modl init

# Pull a model (auto-selects variant for your GPU)
modl pull flux-dev

# Generate
modl generate "a photo of a mountain lake at sunset" --base flux-dev
```

## The Full Journey

```bash
# 1. Pull a base model
modl pull flux-schnell

# 2. Prepare a training dataset
modl dataset create products --from ~/photos/my-products/

# 3. Train a LoRA
modl train --dataset products --base flux-schnell --name product-v1 --lora-type object

# 4. Generate with your LoRA
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

```bash
modl link --comfyui ~/ComfyUI
modl link --a1111 ~/stable-diffusion-webui
```

Modl scans your model folders, hashes each file, and moves recognized models into the store — replacing them with symlinks. Your tools keep working, nothing breaks. Unrecognized files are left untouched.

## Commands

<!-- BEGIN AUTO-GENERATED (scripts/generate-cli-reference.sh) -->

Run `modl <command> --help` for full usage details.

| Command | Description |
|---------|-------------|
| `modl pull <id>` | Download a model, LoRA, VAE, or other asset |
| `modl rm <id>` | Remove an installed model |
| `modl ls` | List installed models |
| `modl info <id>` | Show detailed info about a model |
| `modl search [query]` | Search the registry |
| `modl update` | Fetch latest registry index |
| `modl link [path]` | Link a tool's model folder (ComfyUI, A1111) |
| `modl gc` | Remove unreferenced files from the store |
| `modl generate <prompt>` | Generate images using diffusers |
| `modl train` | Train a LoRA with managed runtime |
| `modl train status [name]` | Show live training progress |
| `modl train ls` | List training runs |
| `modl dataset create <name>` | Create a managed dataset from images |
| `modl dataset caption <name>` | Auto-caption images using a VL model |
| `modl dataset prepare <name>` | Full pipeline: create, resize, caption |
| `modl dataset ls` | List all managed datasets |
| `modl outputs ls` | List recent generation outputs |
| `modl outputs search <query>` | Search outputs by prompt, model, or LoRA |
| `modl score <paths>` | Score image aesthetic quality |
| `modl detect <paths>` | Detect faces in images |
| `modl segment <image>` | Generate segmentation mask for inpainting |
| `modl face-restore <paths>` | Restore faces using CodeFormer |
| `modl upscale <paths>` | Upscale images using Real-ESRGAN |
| `modl remove-bg <paths>` | Remove image background (transparent PNG) |
| `modl enhance <prompt>` | AI-enhanced prompt expansion |
| `modl doctor` | Check for broken symlinks, missing deps, corrupt files |
| `modl serve` | Launch the web UI |
| `modl worker start/stop/status` | Manage persistent GPU worker |
| `modl upgrade` | Update modl CLI to the latest release |

Full reference: `modl --help` or run `scripts/generate-cli-reference.sh` to regenerate this table.

<!-- END AUTO-GENERATED -->

## Variant Selection

Models come in multiple variants. Modl picks the best one for your GPU:

| VRAM | Variant | Notes |
|------|---------|-------|
| 24GB+ | fp16 | Full quality |
| 12-23GB | fp8 | Slight quality reduction |
| 8-11GB | gguf-q4 | Quantized |
| <8GB | gguf-q2 | Lower quality, functional |

Override: `modl pull flux-dev --variant fp8`

## Architecture

Rust CLI for speed and single-binary distribution. Managed Python runtime for GPU compute. SQLite tracks everything.

```
modl (Rust binary)          Python Worker
├── CLI commands            ├── Inference (diffusers)
├── Web UI (axum)           ├── Training (ai-toolkit)
├── Model registry          ├── Analysis (CLIP, SAM, etc.)
├── Content store           └── VRAM management
└── SQLite DB
```

See [CLAUDE.md](CLAUDE.md) for full architecture docs.

## License

MIT
