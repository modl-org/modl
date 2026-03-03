# modl — Architecture & Product Plan

> **Last updated:** 2026-03-02  
> **Status:** Active — building toward Reddit launch  
> **Domain:** modl.run (to register)

---

## What modl is

A local-first image and video generation app. One binary. Pull models from a registry, generate images/video through a curated UI, train LoRAs, organize outputs. Like a self-hosted Midjourney alternative with the simplicity of Fooocus and the model ecosystem of Ollama.

```
curl -fsSL https://modl.run/install.sh | bash
modl pull flux-schnell
modl serve
# → browser opens, type prompt, get image
```

---

## Vision: the 80/20 of image gen

### What Fooocus got right (40k+ stars)

1. **Prompt-first.** Open it → type prompt → get image. No nodes, no graphs, no setup.
2. **Smart defaults.** GPT-2 prompt expansion behind the scenes. Users type "house in garden", get a beautiful image. No need to learn prompt engineering.
3. **< 3 clicks from download to first image.** Models auto-download on first run.
4. **Presets, not configs.** General / Anime / Realistic — pick one, it handles model + settings.
5. **Progressive disclosure.** Simple view by default. "Advanced" toggle for power users. Most users never open it.
6. **4GB VRAM minimum.** Ran on laptop 3060s. Accessible.

### What killed Fooocus

- SDXL-only. Refused to support Flux and newer architectures. Now in "maintenance mode."
- Single developer (lllyasviel) moved on to other projects.
- No model management — you manually download from CivitAI and put files in folders.
- No registry. No training. No video. No DAM.

### What InvokeAI got wrong

- **Too complex.** Canvas + node editor + layers + control panels = feels like Photoshop, not a quick gen tool.
- **Heavy install.** Python + CUDA + pip dependencies. Fragile, slow to start.
- **Tried to be ComfyUI.** Nodes are powerful but hostile to casual users. You can't compete with ComfyUI on flexibility — it has 10k+ custom nodes. Don't try.
- **Feature creep.** Kept adding pro features nobody asked for instead of nailing the simple flow.

### The 80/20 — what to build

These features cover ~95% of what people actually do daily:

| Feature | Priority | Why |
|---------|----------|-----|
| **txt2img** | P0 — launch | 80% of all usage. Prompt → image. |
| **img2img** | P0 — launch | Upload reference image → variation / style transfer |
| **Inpainting** | P1 — soon after | Paint over a region → regenerate. Fix faces, change elements. |
| **Upscale** | P1 — soon after | Make it bigger/better. One-click with a good model. |
| **LoRA application** | P0 — launch | Pick from installed LoRAs, set weight, generate. Key differentiator. |
| **Video gen** | P2 — growth | Wan 2.1, CogVideo, etc. Greenfield — nobody has clean UX for this. |

### What to deliberately NOT build

| Feature | Why skip |
|---------|----------|
| Node/graph editor | ComfyUI owns this. Can't compete, shouldn't try. |
| ControlNet UI | Complex, niche. Maybe later as "advanced" toggle. |
| Regional prompting | Power-user feature. 5% use it. |
| Custom scheduler/sampler picker | Pick good defaults (DPM++ 2M Karras for SDXL, Euler for Flux). Hide it. |
| Textual inversions / hypernetworks | Replaced by LoRAs. Dead tech. |
| Refiner model selection | Models are good enough without refiners now. |
| Custom VAE picker | Auto-detect. Nobody should think about VAEs. |
| Multi-model pipelines | Node graph territory. Stay out. |

### UX design rules

1. **First-run magic.** `modl serve` with no models installed → prompt user: "No models installed. Pull flux-schnell (fast, 12GB) or flux-dev (quality, 24GB)?" → auto-pull → ready to generate.
2. **One-screen generation.** Prompt box + Generate button visible without scrolling. Model/LoRA selectors compact. No tabs within the generate page.
3. **Smart defaults per model.** Flux-schnell: 4 steps, guidance 0. Flux-dev: 28 steps, guidance 3.5. SDXL: 30 steps, guidance 7. User never picks these unless they want to.
4. **Presets for aspect ratio.** Square (1:1), Landscape (16:9), Portrait (9:16), Phone (9:19.5). Not pixel dimensions.
5. **Results as a feed.** Each generation appends to a scrollable output feed (newest on top). Not one image at a time — see your iteration history.
6. **Click to iterate.** Click any output → "Vary (Subtle)" / "Vary (Strong)" / "Upscale" / "Use as img2img". Like Midjourney's U/V buttons.
7. **LoRAs are first-class.** Not buried in settings. Visible LoRA chips above the prompt, click to add, slider for weight. Like tags.

---

## Product positioning

```
                        Easy to use
                            ↑
                            |
                  modl ←────|────→ Midjourney
                  (local)   |     (cloud, $30/mo)
                            |
    ←───────────────────────+───────────────────────→
    Full control                          No control
    (local, own data)                     (cloud, their rules)
                            |
                            |
           ComfyUI ←────────|────→ DALL-E
           (complex)        |     (simple but limited)
                            |
                            ↓
                       Hard to use
```

**modl's quadrant:** Easy to use + Full control. That's the gap.

- Midjourney users who want privacy / no subscription / NSFW freedom → modl
- ComfyUI users who want quick results without node graphs → modl
- Fooocus users left behind when it stopped updating → modl
- A1111 users tired of extension hell → modl
- Artists who want to train character/style LoRAs without being technical → modl

---

## Design Principles

1. **The UI is the product.** `modl serve` is what people use daily. The CLI is plumbing for power users and automation.
2. **API-first.** REST + WebSocket API powers both the UI and CLI. Third-party tools can integrate.
3. **Single binary.** UI compiled in via `include_str!`. `modl serve` just works. No npm, no build step.
4. **Remote-first.** Works over Tailscale/LAN. Mac → GPU server is a primary use case.
5. **Models stay hot.** Persistent worker keeps models in VRAM. Second image in ~3-8s, not 40s.
6. **Assets are local.** Images, models, LoRAs — all on your disk. Cloud is only for GPU compute, never storage.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     modl (Rust binary)                       │
│                                                              │
│  CLI Layer              API Layer           Core Layer        │
│  ├── model pull/ls      ├── GET /models     ├── store.rs      │
│  ├── generate           ├── POST /generate  ├── executor.rs   │
│  ├── train              ├── POST /train     ├── dataset.rs    │
│  ├── dataset ...        ├── GET /outputs    ├── db.rs         │
│  ├── worker start/stop  ├── GET /gpu        ├── cloud.rs      │
│  └── serve              ├── WS /events      └── artifacts.rs  │
│       │                 └── Static UI                         │
│       └─ starts API server + manages persistent worker        │
└─────────────┬──────────────────┬────────────────────────────┘
              │ REST/WS          │ Unix socket
              │                  │
    ┌─────────┴────┐    ┌───────┴──────────┐
    │  Browser     │    │ Persistent Worker │
    │  localhost   │    │ (Python daemon)   │
    │  or remote   │    │ model cache, VRAM │
    └──────────────┘    │ LoRA hot-swap     │
                        └──────────────────┘
```

### Two processes, one command

| Process | Language | Lifecycle | Role |
|---------|----------|-----------|------|
| **modl serve** | Rust | Long-running | HTTP API, WebSocket, UI, job orchestration |
| **Persistent worker** | Python | Auto-spawned, idle timeout | Model loading, inference, VRAM management |

`modl serve` auto-spawns the worker on first generate. Worker stays alive with models in VRAM. If idle for 10 min, self-terminates to free VRAM. If it crashes, serve respawns it.

---

## User Scenarios

| User | Setup | Generates | Trains | UI Access |
|------|-------|-----------|--------|-----------|
| **Artist on Mac** | No GPU | `--cloud` | `--cloud` | `localhost:3333` |
| **Hobbyist, Windows, RTX 3060** | Weak GPU | Local (SDXL) or cloud (Flux) | `--cloud` | `localhost:3333` |
| **Power user, Linux, RTX 4090** | Strong GPU | Local | Local | `localhost:3333` or remote |
| **Pedro** | Mac + Pop!_OS/4090 via Tailscale | Local | Local | `pop-os:3333` from Mac |

---

## Current State (2026-03-02)

### What works

| Component | Status |
|-----------|--------|
| `modl model pull/ls/info/search` | ✅ Working |
| `modl train` (interactive + flags) | ✅ Working |
| `modl dataset create/caption/tag/resize` | ✅ Working |
| `modl generate` (one-shot subprocess) | ✅ Built, untested on GPU |
| Python worker (train/generate/caption/resize/tag) | ✅ Working |
| `gen_adapter.py` (Flux, SDXL, SD1.5 + LoRA) | ✅ Built |
| `modl preview` (training sample viewer) | ✅ Working |
| SQLite DB (models, jobs, artifacts) | ✅ Working |
| Registry (model search, pull, versioning) | ✅ Working |

### What's not built yet

| Component | Effort | Blocks |
|-----------|--------|--------|
| Smoke-test generate on GPU | ~1 hour | Everything below |
| Persistent worker (Python: serve.py, model cache) | ~3-4 days | Fast generation |
| Persistent worker (Rust: worker.rs, socket executor) | ~2 days | Fast generation |
| `modl serve` (rename preview + API endpoints) | ~2 days | UI |
| WebSocket (worker events → browser) | ~1 day | Live progress |
| Generate UI page | ~3 days | Launch |
| Gallery / DAM page | ~2 days | Launch |
| Training UI page (launch + live progress) | ~2 days | Launch |
| img2img support in gen_adapter | ~1 day | img2img UI |
| Inpainting support | ~2 days | Inpaint UI |
| Upscaler integration | ~1 day | Upscale button |

---

## UI Pages

### Page 1: Generate (the main screen)

```
┌─────────────────────────────────────────────────────────┐
│  modl                          [Generate] [Gallery] [⚙] │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─ Model ─────────────┐  ┌─ LoRAs ──────────────────┐  │
│  │ flux-schnell      ▼  │  │ [art-style ×] [+Add]    │  │
│  └──────────────────────┘  └──────────────────────────┘  │
│                                                          │
│  ┌─ Prompt ──────────────────────────────────────────┐   │
│  │ a cat sitting on a windowsill at sunset           │   │
│  │                                                    │   │
│  └────────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─ Negative ────────────────────────────────────────┐   │
│  │ blurry, low quality                               │   │
│  └────────────────────────────────────────────────────┘   │
│                                                          │
│  [1:1] [16:9] [9:16] [4:3]             [▶ Generate]     │
│                                                          │
│  ▸ Advanced (steps, guidance, seed)                       │
│                                                          │
├──────────────────────────────────────────────────────────┤
│  Output Feed                                             │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐          │
│  │      │ │      │ │      │ │      │ │      │          │
│  │ img5 │ │ img4 │ │ img3 │ │ img2 │ │ img1 │          │
│  │      │ │      │ │      │ │      │ │      │          │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘          │
│  Click: [Vary Subtle] [Vary Strong] [Upscale] [img2img] │
└──────────────────────────────────────────────────────────┘
```

**Key UX decisions:**
- Model + LoRA selectors are always visible, compact (not buried in tabs)
- Aspect ratio buttons, not width/height inputs
- Advanced settings collapsed by default (steps, guidance, seed, sampler)
- Smart defaults per model — user never needs to touch Advanced
- Output feed shows iteration history, not just last image
- Click any output for quick actions (vary, upscale, use as input)

### Page 2: Gallery / DAM

```
┌─────────────────────────────────────────────────────────┐
│  modl                          [Generate] [Gallery] [⚙] │
├─────────────────────────────────────────────────────────┤
│  Search: [___________________]  Filter: [All ▼] [★ Fav] │
│                                                          │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │
│  │      │ │      │ │      │ │      │ │      │ │      │ │
│  │      │ │      │ │      │ │      │ │      │ │      │ │
│  │      │ │      │ │      │ │      │ │      │ │      │ │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │
│  │      │ │      │ │      │ │      │ │      │ │      │ │
│  │      │ │      │ │      │ │      │ │      │ │      │ │
│  │      │ │      │ │      │ │      │ │      │ │      │ │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ │
│                                                          │
│  ┌─ Selected Image ──────────────────────────────────┐   │
│  │ Prompt: a cat sitting on a windowsill at sunset   │   │
│  │ Model: flux-schnell  LoRA: art-style (0.8)        │   │
│  │ Steps: 4  Guidance: 0  Seed: 42  Size: 1024x1024  │   │
│  │ [★ Favorite] [📋 Copy Prompt] [🗑 Delete] [↗ Open] │   │
│  └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

**Key UX decisions:**
- Search by prompt text (full-text search on SQLite)
- Filter by model, LoRA, date, favorites
- Click image → metadata panel slides in (not a modal)
- Copy prompt to regenerate with different settings
- Favorite to keep, delete to discard
- All metadata stored in SQLite, searchable

### Page 3: Train (secondary, but differentiating)

```
┌─────────────────────────────────────────────────────────┐
│  modl                    [Generate] [Gallery] [Train] [⚙]│
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─ Training Runs ──────────────────────────────────┐   │
│  │ ✅ art-style-v3    SDXL    2h 14m    3000 steps  │   │
│  │ 🔄 art-style-v4    SDXL    1h 02m    1200/10000  │   │
│  │ ✅ my-character    Flux    4h 30m    5000 steps  │   │
│  └──────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─ art-style-v4 (training) ─────────────────────────┐   │
│  │ Progress: ████████░░░░░░░░░░░░  12%  1200/10000   │   │
│  │                                                    │   │
│  │ Sample images:                                     │   │
│  │ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐              │   │
│  │ │step 0│ │ 2000 │ │ 4000 │ │ 6000 │              │   │
│  │ └──────┘ └──────┘ └──────┘ └──────┘              │   │
│  └────────────────────────────────────────────────────┘   │
│                                                          │
│  [+ New Training Run]                                    │
│  Dataset: [Select... ▼]  Trigger: [________]            │
│  Base model: [SDXL ▼]   Preset: [Standard ▼]           │
│  [▶ Start Training]                                      │
└──────────────────────────────────────────────────────────┘
```

**Key UX decisions:**
- Training is a secondary page, not competing for attention with Generate
- Show running/completed training runs as a list
- Click a run → sample images at different steps (already built in preview)
- "New Training Run" is minimal: pick dataset, type trigger word, go
- Presets abstract away the 60+ params (Quick/Standard/Quality)
- Results (trained LoRAs) automatically appear in Generate's LoRA picker

### Settings (gear icon, slide-out panel)

- GPU info (VRAM, utilization, temperature)
- Worker status (running/stopped, loaded models)
- Default model selection
- Output directory
- Cloud toggle + API key (future)
- Theme (dark/light)

NOT a full settings page — a slide-out panel. Most users never open it.

---

## The LoRA story

Training is difficult today. That's exactly why it's valuable in modl.

### Current state of LoRA training (for a normal person)

1. Find ai-toolkit or kohya_ss on GitHub
2. Install Python, deal with CUDA version conflicts
3. Edit a 60-parameter YAML config
4. Figure out dataset format, naming, captions
5. Google "good learning rate for SDXL LoRA"
6. Wait, hope, check outputs
7. Manually copy .safetensors somewhere
8. Figure out how to load it in your gen tool

### What modl does

```
modl dataset create art-photos --from ./photos
modl dataset caption art-photos
modl train --dataset art-photos --trigger ARTSTYLE
# ... wait for training ...
modl generate "a cityscape in ARTSTYLE"
```

Or in the UI: click "Train" tab → pick dataset → type trigger → click Start → LoRA appears in Generate's LoRA picker when done.

**If modl makes LoRA training 10x easier, more people will train LoRAs.** Character LoRAs (train on 15-20 photos of a person → consistent character in any scene), style LoRAs (train on an artist's work → apply that style to anything), object LoRAs (train on product photos → generate product in any context). These are killer use cases that today require technical skill.

Keep it, but don't lead with it. Lead with "generate images easily." Training is the hook that makes modl sticky — once you've trained a LoRA in modl, you're not switching to another tool.

---

## Build Order

### Step 1: Smoke-test `modl generate` on GPU (~1 hour)
> **Prereq:** GPU free (after current LoRA finishes)

Run on the 4090. Validate full flow: prompt → Python subprocess → image → artifact in DB.

### Step 2: Persistent worker — Python side (~3-4 days)
> **Prereq:** Step 1

`python/modl_worker/serve.py` — Unix socket daemon, ModelCache, LoRA hot-swap, idle timeout. See `persistent-worker.md`.

### Step 3: Persistent worker — Rust side (~2 days)
> **Prereq:** Step 2

`src/cli/worker.rs` — start/stop/status. Socket mode in executor.rs. Auto-spawn on first generate.

After this: second image in ~3-8s (model cached in VRAM).

### Step 4: `modl serve` — API + UI foundation (~2 days)
> **Prereq:** Step 3

Rename `preview` → `serve`. Add `--host`. API endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /` | Web UI |
| `GET /api/models` | List installed models + LoRAs |
| `POST /api/generate` | Submit generate job |
| `GET /api/jobs/:id` | Job status + result |
| `GET /api/outputs` | List generated images with metadata |
| `GET /api/gpu` | GPU info, worker status |
| `WS /api/events` | Real-time progress |
| `GET /api/runs` | Training runs (existing) |
| `GET /api/datasets` | Datasets (existing) |
| `POST /api/train` | Start training job |
| `GET /files/*` | Serve images (existing) |

### Step 5: Generate UI page (~3 days)
> **Prereq:** Step 4

The main screen. Prompt → image, model/LoRA pickers, aspect ratio buttons, output feed with iteration actions. This is the screenshot for Reddit.

### Step 6: Gallery / DAM page (~2 days)
> **Prereq:** Step 5

Grid of all outputs. Search, filter, metadata panel, favorites, delete. SQLite-backed.

### Step 7: Training UI page (~2 days)
> **Prereq:** Step 4

Training run list, live progress, sample viewer, "New Training" form. Trained LoRAs auto-appear in Generate.

### Step 8: img2img + inpainting (~2-3 days)
> **Prereq:** Step 5

Add image upload to generate page. "Use as input" action on gallery images. Inpaint mask canvas (simple brush, not Photoshop).

### Step 9: Upscaler (~1 day)
> **Prereq:** Step 5

One-click upscale button on any output. Pull upscaler model from registry.

---

## Reddit Launch Checklist

Minimum for a compelling post:

- [x] `modl model pull flux-schnell` — one command installs model + runtime
- [x] `modl train --dataset ./photos --trigger MYSTYLE` — LoRA training
- [ ] `modl generate "a cat in MYSTYLE"` — generate with LoRA (Step 1)
- [ ] `modl serve` → browser opens → type prompt → image in seconds (Steps 4-5)
- [ ] Fast iteration: ~5s per image (persistent worker, Steps 2-3)
- [ ] Clean generate UI screenshot with output feed
- [ ] Gallery page with metadata
- [x] Single binary install via `curl` or `cargo install`

### Skip for launch
- img2img / inpainting (Step 8, after launch)
- Cloud toggle in UI (CLI flag works)
- Video generation (Phase 2)
- Tauri desktop app (Phase 3)
- Settings UI beyond basics
- Model management UI (CLI works)

---

## Timeline

| Step | What | Days | Running Total |
|------|------|------|---------------|
| 1 | Smoke-test generate | 0.5 | 0.5 |
| 2 | Persistent worker (Python) | 3-4 | 4 |
| 3 | Persistent worker (Rust) | 2 | 6 |
| 4 | `modl serve` API | 2 | 8 |
| 5 | Generate UI | 3 | 11 |
| 6 | Gallery/DAM | 2 | 13 |
| 7 | Training UI | 2 | 15 |

**~3 weeks** to a launchable product. Then iterate based on feedback.

---

## Future Phases

### Phase 2: Expand capabilities
- img2img + inpainting
- Upscaler
- Video generation (Wan 2.1, CogVideo) — **timing play, greenfield UX**
- Batch generation / prompt queues
- `torch.compile()` for 20-30% faster repeat inference
- Prompt templates / styles (like Fooocus presets)

### Phase 3: Polish & distribution
- PWA manifest (installable from browser)
- Auto-start on boot (systemd / launchd)
- Tauri desktop wrapper (native window, system tray)
- `.dmg` / `.exe` / `.AppImage` installers
- Auto-updater

### Phase 4: Cloud & revenue
- `--cloud` dispatches to Modal GPUs
- Per-job pricing (~$0.01-0.02/image, ~$2-3/LoRA training)
- No subscription — pay for what you use
- User accounts, API keys, usage dashboard
- Registry: private models, team sharing

---

## Tech Stack

| Layer | Tech | Notes |
|-------|------|-------|
| CLI + binary | Rust, clap v4 | Working |
| API server | axum 0.8 | Working (preview) |
| Real-time | axum WebSocket | To build |
| Worker protocol | Unix socket + JSONL | To build |
| Frontend | Vanilla HTML/CSS/JS | Compiled in, no build step |
| Database | SQLite (rusqlite) | Working |
| Inference | diffusers (Python) | Working via gen_adapter |
| Training | ai-toolkit | Working via train_adapter |
| Desktop wrapper | Tauri v2 | Future, Phase 3 |

### Why vanilla JS?
- The UI is 3 pages. Vanilla works.
- No build step = `include_str!("index.html")` and done.
- When it outgrows vanilla (10+ components, shared state), migrate to Svelte.
- Svelte compiles to vanilla JS — carries into Tauri later.

---

## Competitive landscape

| Tool | Strengths | Weakness modl exploits |
|------|-----------|----------------------|
| **Midjourney** | Best quality, easy UX | Cloud-only, $30/mo, no training, Discord-bound |
| **ComfyUI** | Infinite flexibility, huge ecosystem | Hostile UX, steep learning curve |
| **A1111** | Established, many extensions | Stagnating, Python install hell, dated UI |
| **Fooocus** | Simple, fast, beautiful defaults | Dead (SDXL only, LTS mode), no registry, no training |
| **InvokeAI** | Polished, canvas mode | Heavy, complex install, too many features |
| **Draw Things** | Great Mac app | Mac/iOS only, no training, no registry |
| **CivitAI** | Massive model library, on-site gen | Browser-only, no local, no CLI, ad-heavy |

**modl's position:** Fooocus simplicity + Ollama model management + LoRA training. Local-first, single binary, curated UX.

### The unfiltered advantage

modl is software that runs on your machine. There are no content policies, no banned prompts, no NSFW filters. What you generate is your business — like how VLC doesn't police what videos you watch.

This matters commercially:
- **Midjourney:** Strict content policy, bans NSFW, bans public figures, bans "gore"
- **DALL-E:** Even stricter. Won't generate anything remotely edgy.
- **CivitAI:** Allows NSFW on-platform but increasingly adding restrictions under legal pressure. Their most popular models are NSFW — users are nervous about losing access.
- **Cloud GPU providers:** Some (RunPod, Lambda) have acceptable use policies that technically prohibit certain content.
- **modl:** It's your GPU, your models, your images. No telemetry, no content scanning, no terms of service for what you create locally.

This isn't something to lead marketing with (attracts the wrong attention and scares off mainstream users). But it's a quiet, powerful draw for artists who want creative freedom — and that's a large, underserved, paying audience. The `--cloud` tier would need basic legal-CYA terms, but local use is completely unrestricted by design.
