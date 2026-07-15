---
name: modl
description: >
  AI image generation toolkit — generate images, edit photos, train LoRAs, manage models,
  analyze images, upscale, remove backgrounds, and more via the modl CLI.
triggers:
  # Direct invocation
  - /modl
  # Generation
  - generate an image
  - generate a photo
  - create an image
  - make an image
  - txt2img
  - img2img
  - inpaint
  - image generation
  # Editing
  - edit an image
  - edit this photo
  - image editing
  # Training
  - train a lora
  - train a model
  - lora training
  - fine-tune
  - fine tune
  # Model management
  - pull a model
  - download a model
  - install a model
  - list models
  - search models
  - what models
  # Analysis
  - describe this image
  - caption this image
  - score this image
  - image quality
  - detect faces
  - ground
  # Processing
  - upscale
  - remove background
  - segment
  - compose images
  - composite
  - layer images
  # Prompt
  - enhance prompt
  - improve prompt
  # Workflows
  - modl generate
  - modl edit
  - modl train
  - modl pull
  - modl search
  - modl vision
  - modl process
  - modl serve
  - modl dataset
  - modl gpu
  - modl run
  - modl status
  - run a workflow
  - batch generate
  - multi-step workflow
  - generate then edit
  - fire and forget
invocable: true
argument-hint: "[action] [args...]"
---

# modl — AI Image Generation Toolkit

modl is a local-first CLI for AI image generation: pull models, generate images, edit photos, train LoRAs, manage outputs. Like Ollama for image gen.

## Agent Rules

1. **Always use `--json` output** when available. Parse JSON for file paths, status, and structured data. Fall back to raw stdout only when `--json` isn't supported.
2. **Check installed models first** — run `modl ls` before generating. If the user's requested model isn't installed, offer to `modl pull` it.
3. **Pick the right model for the task** — see the Model Guide below. Don't default to the biggest model; match speed/quality to the user's need.
4. **Long-running commands need monitoring** — `modl generate`, `modl edit`, `modl train`, and `modl pull` can take seconds to hours. For training, use `modl train status --watch` to monitor.
5. **File paths are absolute** — modl outputs go to `~/.modl/outputs/` by default. Always return full paths to the user.
6. **Never delete outputs** — `~/.modl/outputs/` is the user's generation library, not a cache. Never remove files from it.
7. **Prompt each model its way** — run `modl info <model>` for the canonical prompting guide before writing prompts (prose vs tags, negative-prompt support, text-in-quotes). See Per-Model Prompting below.
8. **Batch work goes through workflows** — for more than ~2 related generations/edits, write a YAML spec and use `modl run` (with `--dry-run` first to validate) instead of looping individual commands. Use `seeds: [...]` for variations, then `modl status <run-id>` to collect results.

## Quick Reference

| Task | Command |
|------|---------|
| Generate image | `modl generate "prompt" --base flux-schnell --json` |
| Generate with LoRA | `modl generate "prompt" --lora my-style --json` |
| img2img | `modl generate "prompt" --init-image photo.jpg --strength 0.7 --json` |
| Inpaint | `modl generate "prompt" --init-image photo.jpg --mask mask.png --json` |
| Edit image | `modl edit "make the sky sunset" --image photo.jpg --json` |
| Fast generation | `modl generate "prompt" --fast --json` |
| Train LoRA | `modl train --base flux-dev --lora-type style --dataset my-photos --name my-style` |
| Check training | `modl train status --json` |
| Pull model | `modl pull flux-dev` |
| List installed | `modl ls` |
| Search registry | `modl search "anime" --json` |
| Search CivitAI | `modl search "realistic" --civitai --json` |
| Describe image | `modl vision describe photo.jpg --json` |
| Score quality | `modl vision score photo.jpg --json` |
| Detect faces | `modl vision detect photo.jpg --json` |
| Ground objects | `modl vision ground "the red car" photo.jpg --json` |
| Upscale 4x | `modl process upscale photo.jpg --json` |
| Remove background | `modl process remove-bg photo.jpg --json` |
| Segment object | `modl process segment photo.jpg --bbox 100,100,400,400 --json` |
| Compose layers | `modl process compose --background bg.png --layer subject.png --position 0.5,0.7 --scale 0.3` |
| Enhance prompt | `modl enhance "a cat" --json` |
| Serve web UI | `modl serve` |
| GPU status | `modl worker status` |

## Model Guide

### Choosing a Model

**For quick iterations / drafts:**
- `flux-schnell` — 4 steps, fast, good quality. The default.
- `z-image-turbo` — 8 steps, good for stylized images.
- `ernie-image-turbo` — 8 steps, best for structured layouts and text-heavy images.

**For high quality finals:**
- `flux-dev` — 28 steps, excellent quality, most LoRA support.
- `z-image` — 20 steps, strong on realistic/artistic styles.
- `chroma` — 20 steps, Apache 2.0 Flux fork, supports negative prompts.
- `qwen-image` — 25 steps, best text rendering in images.
- `ernie-image` — 50 steps, best for complex structured content (posters, infographics, multi-panel, character sheets). Needs long, detailed prompts.

**For editing (instruction-based):**
- `qwen-image-edit-2511` — Best edit quality, 40 steps, needs 30GB+ VRAM. The default for `modl edit`.
- `klein-4b` — Fast edits, 4 steps, fits in 10GB VRAM.
- `klein-9b` — Better edit quality, 4 steps, needs 16GB VRAM.

**For inpainting (mask-based):**
- `flux-fill-dev` — Purpose-built for inpainting, best quality, 50 steps.
- `flux-dev`, `flux-schnell`, `z-image`, `sdxl` — All support inpainting via `--mask`.

**For training LoRAs:**
- `flux-dev` — Best ecosystem, most LoRA recipes.
- `flux-schnell` — Fast iteration on LoRA experiments.
- `z-image` — Good alternative for stylized LoRAs.
- `sdxl` — Lower VRAM, mature ecosystem.

**Low VRAM (< 12GB):**
- `sdxl` (5-7 GB), `sd-1.5` (3-4 GB), `klein-4b` (10 GB fp8)

### Model Capabilities Matrix

| Model | Steps | Guidance | VRAM (fp8) | txt2img | img2img | inpaint | edit | train |
|-------|------:|--------:|-----------:|:-------:|:-------:|:-------:|:----:|:-----:|
| flux-schnell | 4 | 1.0 | 20 GB | Y | Y | Y | | Y |
| flux-dev | 28 | 3.5 | 20 GB | Y | Y | Y | | Y |
| flux-fill-dev | 50 | 30.0 | 20 GB | | | Y | | |
| flux2-dev | 28 | 4.0 | 35 GB | Y | | | | Y |
| klein-4b | 4 | 1.0 | 10 GB | Y | | | Y | Y |
| klein-9b | 4 | 1.0 | 16 GB | Y | | | Y | Y |
| chroma | 20 | 4.0 | 16 GB | Y | Y | Y | | Y |
| z-image | 20 | 4.0 | 14 GB | Y | Y | Y | | Y |
| z-image-turbo | 8 | 0.0 | 14 GB | Y | Y | Y | | Y |
| qwen-image | 25 | 3.0 | 30 GB | Y | | | | Y |
| qwen-image-edit-2511 | 40 | 4.0 | 30 GB | | | | Y | |
| ernie-image | 50 | 4.0 | 14 GB | Y | Y | Y | | |
| ernie-image-turbo | 8 | 1.0 | 14 GB | Y | Y | Y | | |
| sdxl | 30 | 7.5 | 5 GB | Y | Y | Y | | Y |
| sd-1.5 | 30 | 7.5 | 3 GB | Y | Y | Y | | Y |

Default steps and guidance are shown — override with `--steps` and `--guidance`.

### Size Presets

| Preset | Resolution |
|--------|-----------|
| `1:1` | 1024x1024 (default) |
| `16:9` | 1344x768 |
| `9:16` | 768x1344 |
| `4:3` | 1152x896 |
| `3:4` | 896x1152 |
| Custom | `WxH` (e.g. `1920x1080`) |

### Per-Model Prompting

Each model wants a different prompting style. **`modl info <model>` prints the canonical prompting guide** for any model (and `list_models` over MCP includes them) — check it before writing prompts for a model you haven't used in this session. Summary:

| Model | Style | Negative prompts |
|-------|-------|:----------------:|
| flux-dev, flux2-dev | Long descriptive prose: subject, setting, lighting, camera, style | no effect |
| flux-schnell, klein-4b/9b | Short focused prose, subject first (4-step distilled) | no effect |
| chroma | Descriptive prose + aesthetic keywords | **works — use them** |
| z-image | Prose with style/lighting detail, EN/ZH | works |
| z-image-turbo | Short-to-medium prose (guidance 0.0) | no effect |
| qwen-image, ernie-image | Long structured prompts; quote exact text to render, EN/ZH | works |
| sdxl, sd-1.5 | Comma-separated tags, key tokens first; SD 1.5 truncates at ~75 tokens | **essential** |

**Editing models (klein-4b/9b, qwen-image-edit-2511):** state the change as one imperative command per edit ("change the background to a white studio"), chain multiple edit steps in a workflow instead of stacking changes in one prompt, and say what to preserve ("keep the bracelet exactly identical: same texture, same pattern, same proportions"). For text edits with qwen-image-edit-2511, quote the exact replacement text.

### ERNIE Image Prompting Guide

ERNIE-Image (`ernie-image`, `ernie-image-turbo`) is an 8B DiT model from Baidu. #1 open-weights model on GenEval, #2 on LongTextBench. Requires a different prompting style than Flux or Z-Image — long, structured prompts (200-600 words) produce dramatically better results than short ones.

**When to choose ERNIE over other models:**
- Infographics, flowcharts, posters with embedded readable text
- Multi-panel compositions (sticker sheets, expression grids, contact sheets, triptychs)
- Character design sheets with annotated callouts and color palettes
- Anime storyboards and manga pages with panel organization
- Technical diagrams with leader lines and dimension annotations
- Images requiring accurate text rendering (titles, labels, hex codes)
- Structured recipe cards, tutorials, step-by-step visual guides

**When NOT to use ERNIE:**
- Quick single-image generations with short prompts (use flux-schnell or z-image-turbo)
- LoRA-based character consistency (no LoRA ecosystem yet)
- Speed-critical workflows (50 steps base, 8 turbo)

**ERNIE prompting rules (critical — these differ from other models):**

1. **Write long, structured prompts.** ERNIE thrives on 200-600 word prompts that read like detailed art direction briefs. Short prompts produce mediocre results. Describe composition, spatial layout, materials, lighting, and typography explicitly.

2. **Specify spatial layout explicitly.** Use phrases like "upper-left corner contains X", "the lower-center shows Y", "right side features Z". ERNIE places elements where told.

3. **Embed exact text strings in quotes.** For text rendering, include literal strings: `reading 'CAFE & ROASTERY'`, `with the title 'ECO-CITY HORIZON'`, `labeled '#C84B31'`. ERNIE renders quoted text faithfully.

4. **Use specific technique vocabulary.** Not "anime style" but "high-quality cel-shading flat coloring with clean outlines and crisp shadow boundaries." Not "old painting" but "gongbi brushwork with iron-wire line drawing (tiexianmiao)."

5. **Use camera/lens language as precise instructions.** "85mm lens", "f/2.8 macro", "tilt-shift", "direct frontal flash" — ERNIE treats these as structural directives, not vibes.

6. **State exclusions explicitly.** "No gradients, no 3D effects, no atmospheric perspective" — ERNIE respects negative constraints better than most models.

7. **Describe each panel/section for multi-panel work.** For sticker sheets, expression grids, or multi-panel compositions, describe every panel individually: "First sticker: character waves hello with text 'Hi!' beside her. Second sticker: character sits crying with text 'Uwaaa'."

**Example — ERNIE-style prompt for a character expression sheet:**
```
modl generate "An illustration presenting a collection of 8 expression stickers arranged in a 2-row by 4-column grid layout. The background is light yellow. The protagonist of all stickers is the same chibi-style girl with pink twin-tail hair, large emerald-green eyes, light blue sailor uniform with a red bow. Each sticker has a thick white outline and faint gray drop shadow. First row from left to right: (1) bright smile with eyes closed, waving, yellow stars around head, pink text 'Hi!'; (2) sitting on ground crying, fists rubbing eyes, blue text 'Uwaaa'; (3) arms crossed, cheeks puffed in anger, red vein mark, red text 'Hmph!'; (4) heart gesture with both hands, blushing, pink hearts floating, text 'Love'. Second row: (5) shocked expression, white dot eyes, hands clutching head, yellow lightning bolts, text 'What?!'; (6) winking, thumbs up, green text 'Good!'; (7) lying face down sleeping, sleep bubble with 'Zzz'; (8) chin on hand thinking, furrowed brows, black question mark above head. Bright saturated colors, cute art style, smooth rounded linework, flat coloring, even lighting." --base ernie-image-turbo --size 16:9
```

## Command Reference

### modl generate

Generate images from text prompts. Supports txt2img, img2img, inpainting, ControlNet, style transfer.

```
modl generate <prompt> [OPTIONS]

Core:
  --base <MODEL>              Base model (default: flux-schnell)
  --seed <SEED>               Random seed for reproducibility
  --size <PRESET|WxH>         Output size (default: 1:1)
  --steps <N>                 Inference steps (model default if omitted)
  --guidance <SCALE>          Guidance scale (model default if omitted)
  --count <N>                 Number of images (default: 1)
  --json                      JSON output with file paths

LoRA:
  --lora <NAME_OR_PATH>       LoRA to apply
  --lora-strength <0.0-1.0>   LoRA weight (default: 1.0)

img2img:
  --init-image <PATH>         Source image
  --strength <0.0-1.0>        Denoising strength (default: 0.75)

Inpainting:
  --init-image <PATH>         Source image
  --mask <PATH>               Mask (white=regenerate, black=keep)
  --inpaint <METHOD>          auto, lanpaint, standard (default: auto)

ControlNet:
  --controlnet <PATH>         Control image (can use 2x)
  --cn-type <TYPE>            canny, depth, pose, softedge, scribble, hed, mlsd, gray, normal
  --cn-strength <0.0-1.0>     Conditioning strength (default: 0.75)
  --cn-end <0.0-1.0>          Stop at fraction of steps (default: 0.8)

Style Reference:
  --style-ref <PATH>          Style reference image (can repeat)
  --style-strength <0.0-1.0>  Style weight (default: 0.6)
  --style-type <TYPE>         style, face, content

Speed:
  --fast [STEPS]              Lightning LoRA (4 or 8 steps, default 4)
  --no-worker                 Skip persistent worker, use subprocess

Cloud:
  --cloud                     Run on cloud GPU
  --provider <PROVIDER>       modal, replicate, runpod
```

**JSON output format:**
```json
{"status": "completed", "images": ["/home/user/.modl/outputs/2026-03-25/img_001.png"]}
```

### modl edit

Edit images using natural language instructions. Different from inpainting — no mask needed.

```
modl edit <prompt> --image <PATH> [OPTIONS]

  --image <PATH>              Source image (required, repeatable)
  --base <MODEL>              Edit model (default: qwen-image-edit-2511)
  --seed <SEED>               Random seed
  --steps <N>                 Inference steps
  --guidance <SCALE>          Guidance scale
  --count <N>                 Number of outputs (default: 1)
  --size <PRESET|WxH>         Output size (for outpainting: larger than source)
  --fast [STEPS]              Lightning LoRA
  --json                      JSON output
```

**Edit vs Inpainting:**
- `modl edit` — instruction-based ("make the sky sunset"), no mask
- `modl generate --mask` — region-based (paint white where you want changes)

### modl train

Train LoRA adapters on your images.

```
modl train [OPTIONS]

Required:
  --base <MODEL>              Base model to train on
  --lora-type <TYPE>          style, character, or object

Dataset:
  --dataset <PATH>            Dataset name or directory
  --name <NAME>               Output LoRA name
  --trigger <WORD>            Trigger word for activation

Hyperparameters:
  --preset <PRESET>           quick (500 steps), standard (1500), advanced (3000)
  --steps <N>                 Override step count
  --rank <N>                  LoRA rank/dimensionality
  --lr <RATE>                 Learning rate (e.g. 1e-4)
  --batch-size <N>            Batch size
  --resolution <PX>           Training resolution
  --optimizer <TYPE>          adamw8bit, prodigy, adamw, adafactor, sgd
  --repeats <N>               Dataset repetitions per epoch
  --caption-dropout <0-1>     Caption dropout rate
  --class-word <WORD>         Class word (e.g. "man", "woman", "dog")
  --resume <PATH>             Resume from checkpoint

Control:
  --dry-run                   Preview spec without executing
  --config <YAML>             Load full TrainJobSpec from file
  --cloud                     Train on cloud GPU

Subcommands:
  modl train status [NAME]    Show training progress (--watch for live, --json)
  modl train ls               List training runs
  modl train rm <NAME>        Delete a training run
  modl train setup            Install training dependencies (--reinstall)
```

**Training workflow:**
1. `modl dataset prepare my-photos --from ./photos/` — resize + auto-caption
2. `modl train --base flux-dev --lora-type character --dataset my-photos --name my-face --trigger "ohwx"`
3. `modl train status --watch` — monitor progress
4. `modl generate "ohwx person in a garden" --lora my-face --json`

### modl pull / ls / rm / search / info

Model management commands.

```
# Download
modl pull <ID> [--variant fp16|fp8|bf16|gguf-q4] [--dry-run] [--force]
  ID formats: flux-dev, hf:owner/model, civitai:12345, user/slug

# List installed
modl ls [-t checkpoint|lora|vae|text_encoder|...] [--summary] [-a]

# Remove
modl rm <ID> [--force]

# Search registry
modl search [QUERY] [-t TYPE] [--for MODEL] [--popular] [--civitai] [--json]

# Show details
modl info <ID>
```

### modl vision

Image understanding and analysis.

```
# Caption/describe
modl vision describe <PATHS> [--detail brief|detailed|verbose] [--model qwen3-vl-2b|qwen3-vl-8b] [--json]

# Quality scoring
modl vision score <PATHS> [--json]

# Face/object detection
modl vision detect <PATHS> [--type face] [--embeddings] [--json]

# Visual grounding (find objects by description)
modl vision ground <QUERY> <PATHS> [--threshold VALUE] [--model qwen3-vl-2b|qwen3-vl-8b] [--json]

# Image similarity
modl vision compare <PATHS> [--reference PATH] [--json]
```

### modl process

Image processing and enhancement.

```
# Super-resolution
modl process upscale <PATHS> [--scale 2|4] [-o DIR] [--json]

# Background removal (outputs transparent PNG)
modl process remove-bg <PATHS> [-o DIR] [--json]

# Segmentation (create masks)
modl process segment <IMAGE> [--method bbox|background|sam] [--bbox X1,Y1,X2,Y2] [--point X,Y] [--expand PX] [-o PATH] [--json]

# Extract control images
modl process preprocess <METHOD> <IMAGE>
  Methods: canny, depth, pose, softedge, scribble, hed, mlsd, lineart

# Compose layers onto a canvas (CPU-only, no GPU needed)
modl process compose --background <PATH|transparent|white|black> --layer <PATH> [OPTIONS]
  --position <X,Y>          Fractional 0.0-1.0 coordinates (center of layer, default: 0.5,0.5)
  --scale <SCALE>           Layer scale (>0, default: 1.0)
  --opacity <0.0-1.0>       Layer opacity (default: 1.0)
  --canvas-size <WxH>       Required for solid color backgrounds
  -o <DIR>                  Output directory
  --json                    JSON output
  # Repeat --layer/--position/--scale/--opacity for multiple layers (order = bottom to top)
```

### modl dataset

Prepare training datasets.

```
modl dataset prepare <NAME> --from <DIR>    Auto pipeline: create + resize + caption
modl dataset create <NAME> --from <DIR>     Import images only
modl dataset caption <NAME>                 Auto-caption with VL model
modl dataset validate <NAME>                Check for issues
modl dataset resize <NAME> [--resolution PX]
modl dataset list
modl dataset rm <NAME>
```

### modl enhance

Improve prompts for better generation results.

```
modl enhance <PROMPT> [--model sdxl|flux|sd3] [--intensity subtle|moderate|aggressive] [--json]
```

### modl serve / worker

Web UI and GPU worker management.

```
# Web UI
modl serve [--port 3939] [--no-open] [--foreground]
modl serve --install-service    Install as system service
modl serve --remove-service

# Persistent GPU worker (keeps models in VRAM)
modl worker start [--timeout 600]
modl worker stop
modl worker status
```

### modl gpu

Remote GPU session management.

```
modl gpu attach <SPEC> [--idle 30m|1h]    Attach remote GPU
modl gpu detach                            Shut down GPU instance
modl gpu status                            Show running instances
modl gpu ssh                               Shell into GPU
```

### modl system

Maintenance and utilities.

```
modl doctor [--verify-hashes] [--repair]   Check installation health
modl system gc                              Clean unreferenced store files
modl system update                          Fetch latest registry
modl system link --comfyui <PATH>           Symlink models for ComfyUI/A1111
modl config [KEY] [VALUE]                   View/set configuration
modl upgrade                                Update modl to latest
modl init [--defaults]                      First-run setup
```

### modl auth

Authentication for model sources and hub.

```
modl auth login                   Login to modl hub
modl auth logout
modl auth whoami
modl auth add huggingface         Configure HuggingFace token
modl auth add civitai             Configure CivitAI token
```

### modl push

Share LoRAs and datasets.

```
modl push lora <FILE> --name <SLUG> [--visibility public|private] [--base MODEL] [--trigger WORD]
modl push dataset <DIR> --name <SLUG> [--visibility public|private]
```

### modl export / import

Backup and restore.

```
modl export <output.tar.zst> [--no-outputs] [--since YYYY-MM-DD]
modl import <backup.tar.zst> [--dry-run] [--overwrite]
```

### modl run

Run a multi-step workflow from a YAML spec.

```
modl run <SPEC.yaml>                    # execute
modl run a.yaml b.yaml c.yaml           # sequential multi-spec
modl run spec.yaml --dry-run            # validate + show plan, no execution
modl run spec.yaml --dry-run --json     # machine-readable plan
modl run spec.yaml --skip-existing      # resume — skip already-generated sub-jobs
```

**YAML format** (flat fields — `generate:` takes the prompt directly; `edit:` takes the image ref with a separate `prompt:`):

```yaml
name: my-workflow
model: flux2-klein-9b      # workflow-level default (required)
lora: my-lora              # optional, inherited by all steps

defaults:                  # optional, applied to every step
  width: 1024
  height: 1024

# Named image variables — define once, reference anywhere as $name.
# Values: base64 data URI (works over remote MCP) or server-side path.
images:
  product: "/server/refs/product.png"
  style: "data:image/png;base64,iVBORw0..."

steps:
  - id: base
    generate: "a woven gold bracelet on white studio background"
    seeds: [42, 7, 99]         # one output per seed (variation exploration)

  - id: refine
    edit: "$base.outputs[0]"   # step-output ref: $step-id.outputs[N]
    prompt: "warm morning daylight, soft shadows"

  - id: hero
    edit: "$product"           # image variable ref: $name (no dot)
    prompt: "place on travertine stone surface"
    model: qwen-image-edit-2511  # per-step model override (disables workflow lora)
```

Image refs in `edit:` — three forms: `$step-id.outputs[N]` (prior step output), `$name` (from `images:` map), or a bare server-side path.

Run IDs look like `{timestamp}-{workflow-name}`. Logs: `~/.modl/run-logs/<run-id>.log`.

### modl status

Check aggregate state of a workflow run.

```
modl status <run_id> [--json]
MODL_BASE_URL=http://server:3939 modl status <run_id> --json   # artifact HTTP URLs
```

Aggregate values: `pending`, `running`, `completed`, `partial_failure`, `cancelled`. `partial_failure` = all steps terminated, at least one errored.

### modl outputs export

```
modl outputs export <run_id> --dest ./out                        # copy local artifacts
modl outputs export <run_id> --server http://s:3939 --dest ./out # fetch ZIP remotely
```

### MCP Server

modl exposes 15 tools over stdio JSON-RPC 2.0. Activate via `modl mcp` (configured in Claude Code's MCP settings).

**Workflow tools (fire-and-forget pattern):**
- `run_workflow` — submit YAML spec, returns `run_id` immediately
- `job_status` — poll aggregate status for a run
- `list_run_outputs` — fetch artifact paths/URLs for a completed run

Submit, then poll `job_status` later — the run survives client disconnects. Set `MODL_BASE_URL=http://server:3939` on the server so returned URLs are reachable from the client.

**Image refs over remote MCP:** client-side file paths don't exist on the server. Use the workflow `images:` map with base64 data URIs — encode each reference image once (`base64 -i photo.png | tr -d '\n'` on Mac), define it under `images:`, reference it as `$name` in any edit step. `$step-id.outputs[N]` chain refs and pre-copied server paths also work.

## Common Workflows

### Generate + Refine Pipeline

```bash
# Draft with fast model
modl generate "portrait of a woman in golden light" --base flux-schnell --json

# Upscale the best result
modl process upscale /path/to/output.png --scale 4 --json

# Enhance faces if needed
modl process face-restore /path/to/upscaled.png --json
```

### Inpainting Workflow

```bash
# Create a mask from bounding box
modl process segment photo.jpg --bbox 100,50,300,250 --json

# Inpaint the masked region
modl generate "a beautiful garden" --init-image photo.jpg --mask photo_mask.png --base flux-fill-dev --json
```

### LoRA Training Pipeline

```bash
# 1. Prepare dataset (resize + caption)
modl dataset prepare my-character --from ./photos/

# 2. Validate
modl dataset validate my-character

# 3. Train
modl train --base flux-dev --lora-type character --dataset my-character \
  --name my-face --trigger "ohwx" --preset standard

# 4. Monitor
modl train status my-face --watch

# 5. Use the trained LoRA
modl generate "ohwx person as an astronaut" --lora my-face --base flux-dev --json
```

### Compose → Edit Pipeline

```bash
# 1. Remove background from subject
modl process remove-bg subject.jpg --json

# 2. Compose subject onto a background with precise placement
modl process compose --background forest.png --layer subject_nobg.png \
  --position 0.5,0.7 --scale 0.3 --json

# 3. Edit for photorealistic integration (blends lighting, shadows, edges)
modl edit "photorealistic integration, unified lighting, preserve subject" \
  --image composite.png --base klein-9b --json
```

### Multi-Layer Composition

```bash
# Build a scene with multiple subjects on a transparent canvas
modl process compose --background transparent --canvas-size 1024x1024 \
  --layer cup.png --position 0.3,0.5 --scale 0.35 \
  --layer book.png --position 0.7,0.55 --scale 0.4 --json
```

### Fire-and-Forget Batch (workflow)

```bash
# 1. Write the spec (see modl run YAML format), validate it
modl run batch.yaml --dry-run

# 2. Execute — returns a run_id; the run is tracked in SQLite
modl run batch.yaml

# 3. Check later (survives disconnects; resume partial runs with --skip-existing)
modl status <run-id> --json

# 4. Collect artifacts
modl outputs export <run-id> --dest ./results
```

### Image Analysis

```bash
# Score a batch of images
modl vision score ./outputs/*.png --json

# Get detailed captions
modl vision describe photo.jpg --detail verbose --json

# Find specific objects
modl vision ground "the red car on the left" photo.jpg --json
```

## Decision Trees

### "Generate an image" — which model?

1. User wants fast/draft? → `flux-schnell` (4 steps)
2. User wants high quality? → `flux-dev` (28 steps)
3. User wants text in image? → `qwen-image` (25 steps)
4. User wants negative prompts? → `chroma` (40 steps)
5. User has < 12GB VRAM? → `sdxl` (5 GB) or `sd-1.5` (3 GB)
6. User wants to edit an existing image? → `modl edit` with `klein-9b` (fast) or `qwen-image-edit-2511` (best)
7. User wants to inpaint a specific region? → `modl generate --mask` with `flux-fill-dev`

### "Train a LoRA" — which settings?

1. **Character/face**: `--lora-type character --trigger "ohwx" --class-word "man"` (or woman/person)
2. **Art style**: `--lora-type style --trigger "in the style of sks"`
3. **Object**: `--lora-type object --trigger "sks object" --class-word "object"`
4. Quick experiment? → `--preset quick` (500 steps)
5. Production quality? → `--preset standard` (1500 steps)
6. Best base for LoRAs? → `flux-dev` (most ecosystem support)

## Error Handling

- If `modl generate` fails with "model not found", run `modl pull <model>` first
- If VRAM errors occur, try `--base` with a smaller model or use `--fast` mode
- If training crashes, check `modl train status --json` for error details
- Run `modl doctor` to diagnose installation issues
- Use `modl worker status` to check if GPU worker is running and what's loaded
