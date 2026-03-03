# Cloud LoRA Training Plan

> Last updated: 2026-02-27
>
> **STATUS:** Phase 1 (Executor trait + CloudExecutor stub) done. Phases 2-5 (API, Modal backend, CloudExecutor impl, polish) not started.
> Related specs: [execution-target.md](../specs/execution-target.md), [capability-model.md](capability-model.md)

## Business Model: modl cloud as a Product Layer

**Key decision: users never interact with Modal directly.** `modl --cloud` is a paid
modl feature, not a passthrough to a GPU provider.

```
User → modl CLI → modl API (api.modl.run) → Modal (backend, invisible to user)
```

### Why This Matters

If users bring their own Modal key, modl is a free tool with zero revenue.
By owning the API layer, modl controls:

- **Pricing** — per-job or subscription, with margin on compute
- **UX** — one login (`modl cloud login`), no cloud provider setup
- **Portability** — swap Modal for RunPod/bare-metal without user impact
- **Quota/limits** — enforce per-account, not per-provider
- **Model hosting** — modl manages model volumes, users don't think about it

### Pricing Model (MVP)

Inspired by Ollama's approach: local is always free and unlimited. Cloud is a
subscription with qualitative usage tiers, not per-unit pricing. Users should
never hesitate before generating an image because they're counting credits.

| Tier | Price | What you get |
|------|-------|-------------|
| **Free** | $0 | Full CLI, run everything locally on your hardware. Unlimited. |
| **Pro** | $20/mo | Cloud training + inference. Day-to-day usage. 3 private LoRAs on cloud. |
| **Max** | $100/mo | Heavy cloud usage. 5x more than Pro. 10 private LoRAs. Priority GPUs. |

#### What "usage limits" means (internal, not on pricing page)

Soft limits designed to prevent abuse, not slow down work:

| | Free | Pro | Max |
|--|------|-----|-----|
| Cloud training | — | ~8-10 jobs/mo | ~40-50 jobs/mo |
| Cloud inference | — | ~300-500 images/mo | ~1,500-2,500 images/mo |
| Cloud LoRAs stored | — | 3 | 10 |
| Concurrency | — | 1 job at a time | 3 jobs at a time |
| GPU tier | — | A100 80GB (train), A10 (inference) | A100 80GB (both) |

These are **internal enforcement limits**, not numbers on the pricing page.
The page says "day-to-day work" and "heavy, sustained usage" — same as Ollama.
If a user hits a limit, friendly message: "You've hit your Pro usage limit for
this month. Upgrade to Max or wait for next cycle."

#### Why No Per-Unit / PAYG

- Per-unit pricing kills creative flow. "$0.06/image" makes people think twice.
- PAYG creates billing anxiety ("what if I forget and run up a bill?")
- Subscriptions are predictable for both user and us
- Easier to explain: "Pro is $20/mo" vs a pricing calculator
- Ollama, Cursor, GitHub Copilot all prove this works for dev tools

#### Internal Unit Economics (for us, not users)

Modal GPU pricing (2026-02):

| GPU | $/hr |
|-----|------|
| A100 80GB | $2.50 |
| A10 | $1.10 |

Realistic cost per operation (including cold start, S3 transfer, overhead):

| Operation | GPU time | Cost |
|-----------|----------|------|
| Quick LoRA training | ~15-17 min | ~$0.65-$0.70 |
| Standard LoRA training | ~32-40 min | ~$1.35-$1.65 |
| Inference (cold start) | ~90-120s | ~$0.03-$0.04 |
| Inference (warm) | ~8-12s | ~$0.003 |
| Inference (amortized avg) | varies | ~$0.02 |

Overhead breakdown:
- Cold start (container spin-up + volume mount): ~30-60s
- Dataset download from S3 (5-20 images, <50MB): ~10-20s
- Base model loading from volume into VRAM: ~60-120s (model-dependent)
- LoRA artifact save + S3 upload: ~15-30s

#### Margin Analysis (per tier at average utilization ~50%)

| Tier | Revenue | Avg monthly cost | Margin |
|------|---------|-----------------|--------|
| Pro $20/mo | $20 | ~$5 train + ~$4 inf = **~$9** | 2.2x |
| Max $100/mo | $100 | ~$25 train + ~$20 inf = **~$45** | 2.2x |

At 100% utilization (worst case):

| Tier | Revenue | Max monthly cost | Margin |
|------|---------|-----------------|--------|
| Pro $20/mo | $20 | ~$10 train + ~$8 inf = **~$18** | 1.1x |
| Max $100/mo | $100 | ~$50 train + ~$40 inf = **~$90** | 1.1x |

Thin at max but survivable. Most users won't max out — Ollama and every SaaS
subscription bet on this. Adjust soft limits if needed.

#### keep_warm Cost

Keeping 1 A10 warm 24/7: $1.10/hr × 24 × 30 = **~$792/mo**

Strategy:
- No keep_warm at launch. Accept 60-90s cold start.
- Users learn to batch generations (natural UX anyway).
- Revisit when paying user base can amortize the cost.
- Could do keep_warm during peak hours only to save ~50%.

---

## Supported Models (Launch)

Start with 4 base models that cover the current landscape:

| Model | ID | Size | LoRA Training Time (A100) | Notes |
|-------|----|------|---------------------------|-------|
| Flux Schnell | `flux-schnell` | ~12GB | ~10 min (quick) | Fast, popular, good default |
| Flux Dev | `flux-dev` | ~24GB | ~25 min (standard) | Higher quality, more VRAM |
| Z-Image Turbo | `z-image-turbo` | ~12GB | ~10 min | New fast model |
| Qwen Image | `qwen-image` | ~15GB | ~15 min | Multi-modal, emerging |

All 4 supported by ai-toolkit for LoRA. Validation in `TrainJobSpec` rejects
unknown `base_model_id` values on cloud (local can do anything).

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────┐
│  User's machine                                     │
│                                                     │
│  modl CLI (Rust)                                    │
│    ├── modl cloud login        → get modl API key   │
│    ├── modl cloud status       → check quota/usage  │
│    ├── modl train --cloud      → submit to API      │
│    └── modl generate --cloud   → submit to API      │
│                                                     │
│  Dataset upload: CLI zips + uploads to presigned S3  │
│  Artifact download: CLI pulls .safetensors from S3   │
└─────────────────┬───────────────────────────────────┘
                  │ HTTPS (JSON)
                  ▼
┌─────────────────────────────────────────────────────┐
│  modl API (api.modl.run)                         │
│                                                     │
│  - Auth: modl API keys (not Modal tokens)           │
│  - Billing: Stripe, usage metering                  │
│  - Job queue: accept spec → dispatch to Modal       │
│  - Status: poll Modal → relay events to CLI         │
│  - Storage: S3 for dataset upload + artifact return │
│                                                     │
│  Tech: lightweight API (Go/Rust/Node, whatever)     │
│  DB: Postgres (accounts, jobs, billing)             │
└─────────────────┬───────────────────────────────────┘
                  │ Modal Python SDK
                  ▼
┌─────────────────────────────────────────────────────┐
│  Modal (GPU compute backend)                        │
│                                                     │
│  Modal App: modl-cloud                              │
│    ├── train_fn(spec) → run ai-toolkit              │
│    └── generate_fn(spec) → run diffusers            │
│                                                     │
│  Modal Volume: /models                              │
│    ├── flux-schnell/ (~12GB)                         │
│    ├── flux-dev/ (~24GB)                             │
│    ├── z-image-turbo/ (~12GB)                        │
│    └── qwen-image/ (~15GB)                           │
│                                                     │
│  GPU: A100-80GB (training), A10G (inference)        │
└─────────────────────────────────────────────────────┘
```

### Model Storage: Modal Volumes

**One Modal volume (`/models`) pre-populated with all 4 base models.**

Why volumes over baked-in images:
- Adding a model = download to volume, not rebuild the Docker image
- Images stay small (~10GB: torch + ai-toolkit + diffusers)
- Volume mounts in ~5-10s, acceptable cold start
- One volume shared across all functions

Volume setup (run once, update when adding models):

```python
# deploy/modal/volume_setup.py
vol = modal.Volume.from_name("modl-models", create_if_missing=True)

MODELS = {
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "z-image-turbo": "stabilityai/z-image-turbo",  # verify actual HF ID
    "qwen-image": "Qwen/Qwen-Image",               # verify actual HF ID
}
```

### Dataset Flow (Training)

```
1. CLI: zip dataset dir → upload to S3 presigned URL
2. API: record upload, enqueue job
3. Modal fn: download dataset from S3 to /tmp, run training
4. Modal fn: upload LoRA artifact to S3
5. API: mark job complete, notify CLI
6. CLI: download .safetensors, run artifacts::collect_lora()
```

Datasets are ephemeral on cloud — uploaded per job, deleted after.
Models are persistent on the volume.

### Event Streaming

CLI needs live progress. Options:

| Approach | Latency | Complexity |
|----------|---------|-----------|
| SSE from API | Real-time | Medium — API polls Modal, streams to CLI |
| Polling | 2-5s lag | Low — CLI polls API every few seconds |
| WebSocket | Real-time | Higher — stateful connections |

**MVP: polling.** CLI hits `GET /jobs/{id}/events?after={seq}` every 2s.
Same `JobEvent` schema, just delivered over HTTP instead of stdout.

---

## CLI Changes

### New Commands

```bash
modl cloud login                    # browser OAuth or paste API key
modl cloud status                   # account, quota, usage
modl cloud logout                   # remove stored credentials

modl train --cloud                  # submit training to modl cloud
modl generate --cloud               # submit generation to modl cloud
```

### CloudExecutor (Rust)

```rust
impl Executor for CloudExecutor {
    fn submit(&mut self, spec: &TrainJobSpec) -> Result<JobHandle> {
        // 1. Upload dataset to presigned S3 URL
        // 2. POST /jobs/train with spec + dataset_url
        // 3. Return JobHandle { job_id: remote_job_id }
    }

    fn events(&mut self, job_id: &str) -> Result<Receiver<JobEvent>> {
        // Spawn thread that polls GET /jobs/{id}/events?after=N every 2s
        // Parse JSON → JobEvent → send through mpsc channel
        // Same channel type as LocalExecutor
    }
}
```

The CLI event loop doesn't change. It reads from `mpsc::Receiver<JobEvent>`
regardless of whether the executor is local or cloud.

### Config

```yaml
# ~/.modl/config.yaml
cloud:
  api_key: "mods_key_..."          # modl API key, NOT a Modal token
  api_url: "https://api.modl.run"  # overridable for dev/self-host
```

---

## Repo Structure

Three repos, clear boundaries:

| Repo | Visibility | Deploys to | Purpose |
|------|-----------|-----------|--------|
| `modl` | Public | User machines (cargo install / brew) | CLI + local runtime |
| `modl-api` | Private | Fly.io / Railway | API service: auth, billing, job orchestration, S3 |
| `modl-gpu` | Private | Modal | GPU functions: training + inference |

### Why Separate `modl-api` and `modl-gpu`

- **Different deploy targets**: API goes to a web host, GPU functions go to Modal
- **Different dependencies**: API needs Postgres/Stripe/S3 SDKs, GPU needs torch/diffusers/ai-toolkit
- **Different scaling**: API scales horizontally (cheap), GPU scales by Modal's autoscaler
- **Different change velocity**: API changes for billing/auth, GPU changes for model support
- **Replaceability**: swap Modal for RunPod by replacing `modl-gpu` only, API untouched

### `modl-api` Structure

```
modl-api/
├── src/
│   ├── main.ts             # entry point
│   ├── routes/
│   │   ├── auth.ts          # login, me, logout
│   │   ├── jobs.ts          # submit, status, events, cancel
│   │   ├── usage.ts         # billing dashboard
│   │   └── webhooks.ts      # Modal callbacks, Stripe webhooks
│   ├── services/
│   │   ├── modal.ts         # Modal SDK client (dispatch + poll)
│   │   ├── billing.ts       # Stripe metering + subscription management
│   │   ├── storage.ts       # S3 presigned URLs, cleanup
│   │   └── quota.ts         # tier limits, usage tracking
│   └── db/
│       ├── schema.sql       # accounts, subscriptions, jobs, events, artifacts
│       └── queries.ts
├── Dockerfile
├── fly.toml                 # or railway.json
└── package.json
```

Tech: Node/TypeScript (ships fast, Stripe SDK is excellent, Modal has a REST API).
Or Go if you prefer — either works for a thin API layer.

### `modl-gpu` Structure

```
modl-gpu/
├── app.py                   # Modal app definition
├── image.py                 # Container image builder
├── train_fn.py              # @app.function — training
├── generate_fn.py           # @app.function — inference
├── volume_setup.py          # Populate /models volume
├── shared/
│   ├── spec_translator.py   # TrainJobSpec → ai-toolkit config
│   ├── s3.py                # Download dataset, upload artifacts
│   └── events.py            # POST events back to modl-api
├── requirements.txt
└── README.md
```

---

## modl API (Server-Side)

### Endpoints

```
POST   /auth/login          → { api_key }
GET    /auth/me              → { account, plan, quota }
DELETE /auth/logout          → { ok }

POST   /jobs/train           → { job_id, upload_url }
POST   /jobs/generate        → { job_id }
GET    /jobs/{id}            → { status, events_url }
GET    /jobs/{id}/events     → [JobEvent, ...]
GET    /jobs/{id}/artifacts  → [{ url, sha256, size }]
POST   /jobs/{id}/cancel     → { ok }

GET    /usage                → { plan, jobs_this_month, images_this_month, limits }
GET    /models               → [{ id, name, supports_lora, available }]
```

### Job Lifecycle

```
CLI submit → API queued → Modal running → Modal completed → API completed → CLI downloads
                                ↓
                          Modal failed → API failed → CLI shows error
```

### Inference Flow (Cloud Generation)

```
1. CLI: POST /jobs/generate with spec (prompt, model, lora, params)
2. API: check quota (images remaining), enqueue
3. API: call Modal generate_fn.remote(spec)
4. Modal: load pipeline from /models volume, load LoRA if specified
5. Modal: generate image(s), upload to S3
6. API: mark complete, return artifact URLs
7. CLI: download images, save to ~/modl/outputs/
```

Inference is faster than training (~5-10s per image) so the CLI can
blocking-wait instead of polling. Or poll at 1s intervals for consistency.

LoRA files for cloud inference: stored on a separate Modal volume (`/loras`)
or downloaded from S3 per-request. Volume is better for repeat usage.

### Tech Stack (keep it minimal)

- API: single service (Node/TypeScript or Go)
- DB: Postgres (accounts, subscriptions, jobs, events, artifacts, usage)
- Storage: S3-compatible (dataset uploads, generated images, LoRA artifacts)
- Auth: API keys (modl-issued), Stripe for billing + subscriptions
- Deploy: Fly.io or Railway (not Modal — separate your compute from your API)

---

## Modal App

### Image

```python
# deploy/modal/image.py
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers",
        "diffusers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "safetensors",
        "pyyaml",
        "boto3",          # for S3 upload/download
    )
    .pip_install("ai-toolkit")  # or git install
)
```

### Training Function

```python
@app.function(
    gpu="A100",
    image=training_image,
    volumes={"/models": model_volume},
    timeout=3600,       # 1 hour max
    retries=0,          # don't retry training
)
def train(spec: dict, dataset_url: str, callback_url: str) -> dict:
    """
    1. Download dataset from S3
    2. Translate spec → ai-toolkit config
    3. Set base_model_path to /models/{model_id}/
    4. Run training (same logic as train_adapter.py)
    5. Upload LoRA artifact to S3
    6. POST completion to callback_url
    """
```

### Generation Function (Cloud Inference)

```python
lora_volume = modal.Volume.from_name("modl-loras", create_if_missing=True)

@app.function(
    gpu="A10G",           # cheaper GPU for inference (A100 for larger models)
    image=training_image,
    volumes={
        "/models": model_volume,
        "/loras": lora_volume,     # user LoRAs persist here
    },
    timeout=300,          # 5 min max
    keep_warm=1,          # keep one container warm for low latency
)
def generate(spec: dict, lora_s3_url: str | None = None) -> dict:
    """
    1. Load pipeline from /models/{model_id}/
    2. Load LoRA:
       a. Check /loras/{user_id}/{lora_name}/ (cached from previous training)
       b. If not cached, download from S3 lora_s3_url → cache to volume
    3. Generate image(s)
    4. Upload to S3
    5. Return artifact URLs + metadata
    """
```

#### LoRA Caching Strategy

When a user trains a LoRA on cloud, the artifact is:
1. Uploaded to S3 (for CLI download)
2. Saved to `/loras/{user_id}/{lora_name}/` on the Modal volume

When that user generates with `--cloud --lora myface-v1`, the generate function
finds it already on the volume — no download needed. This makes repeat inference
fast and avoids re-uploading LoRAs for every generation.

Volume cleanup: LoRAs unused for 30 days get evicted. S3 is the source of truth.

---

## Implementation Phases

### Phase 1: Foundation (no cloud yet)
- [x] Executor trait + LocalExecutor
- [x] CloudExecutor stub + --cloud flag
- [x] CaptionJobSpec + caption adapter
- [ ] E2E GPU validation (local)

### Phase 2: modl API
- [ ] API service scaffold (auth, jobs, events endpoints)
- [ ] Stripe integration (API key provisioning, usage metering)
- [ ] S3 bucket setup (dataset uploads, artifact storage)
- [ ] `modl cloud login` / `modl cloud status` CLI commands

### Phase 3: Modal Backend
- [ ] Modal app + training image
- [ ] Model volume setup (4 base models)
- [ ] train_fn: spec → ai-toolkit → artifact
- [ ] generate_fn: spec → diffusers → image
- [ ] Event callback → API → CLI polling

### Phase 4: CloudExecutor (Rust)
- [ ] Dataset zip + upload to presigned S3 URL
- [ ] POST /jobs/train with spec
- [ ] Polling thread for events
- [ ] Artifact download + collect_lora()
- [ ] Same progress bar UX as local

### Phase 5: Polish
- [ ] `modl cloud usage` — billing dashboard in terminal
- [ ] Job timeout handling + auto-cancel
- [ ] Retry on transient Modal failures
- [ ] Rate limiting per account

---

## What's Explicitly Deferred

- **BYOK (bring-your-own-Modal-key)**: maybe later as a power-user feature,
  but not at launch. Dilutes the product.
- **Multi-provider**: RunPod/Replicate as alternative backends. Good to have
  the abstraction, but ship with Modal only.
- **Self-hosted cloud**: `modl cloud` pointing at user's own infra.
  Interesting for enterprise, not for launch.
- **Fine-tuning beyond LoRA**: full fine-tune, DreamBooth, textual inversion.
  LoRA only at launch.
- **Custom base models on cloud**: users can only pick from the 4 supported
  models. Local has no restrictions.

---

## Open Questions

1. **Soft limit tuning**: the internal limits (8-10 jobs, 300-500 images for
   Pro) are guesses. Need real usage data to calibrate. Start conservative,
   loosen if margins hold.
2. **API tech**: Node/TS ships fastest (Stripe SDK, Modal REST). Go is leaner
   but more boilerplate. Pick one, don't overthink it.
3. **"Private LoRAs" as feature gate**: 3 for Pro, 10 for Max. This is the
   Ollama "private models" equivalent. Good upsell axis.
4. **Cold start messaging**: 60-90s is noticeable. UX should show "spinning
   up cloud GPU..." with a progress indicator so it feels intentional, not
   broken. Can note "first image takes longer, subsequent ones are fast."
5. **Domain**: api.modl.run? api.modl.so? Match the marketing site.
6. **Max tier necessity at launch**: maybe launch with just Free + Pro ($20).
   Add Max later when there's demand. Fewer tiers = simpler launch.
7. **A100 40GB vs 80GB**: flux-schnell and z-image-turbo might train fine
   on 40GB ($2.10/hr vs $2.50), saving ~16% on those jobs. Needs testing.
