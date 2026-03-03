# Multi-Architecture LoRA Training — Gap Analysis & Plan

## Current state (updated)

The adapter (`train_adapter.py`) uses a **data-driven `ARCH_CONFIGS` table** supporting:
- `flux`, `flux_schnell`, `zimage_turbo`, `zimage`, `chroma`, `sdxl`, `sd15`

Architecture detection uses `_detect_arch()` — registry lookup first, then substring heuristics.

The Rust `BaseModelFamily` enum now has: `Flux`, `FluxSchnell`, `ZImage`, `Chroma`, `Sdxl`, `Sd15`.

### What's working

| Model | Status | Notes |
|---|---|---|
| flux-dev | ✅ Works | Full pipeline |
| flux-schnell | ✅ Works | `assistant_lora_path` + 4-step sampling + guidance 1.0 |
| sdxl-base-1.0 | ✅ Works | Full pipeline |
| sdxl-turbo | ✅ Works | Full pipeline |
| sd-1.5 | ✅ Works | Full pipeline |
| z-image-turbo | 🟡 Config ready | Correct YAML generated, needs end-to-end testing |
| z-image | 🟡 Config ready | Correct YAML generated, needs end-to-end testing |
| chroma | 🟡 Config ready | Correct YAML generated, needs end-to-end testing |

---

## 80/20 priority — what's worth supporting

Based on the ai-toolkit author's (Ostris) tutorial videos and view counts, these
are the models real users actually train LoRAs on, ranked by demand:

### Tier 1 — High demand, image-only (ship first)

| Model | Evidence | Effort | Status |
|---|---|---|---|
| **Z-Image-Turbo** | 89k views, 3 months ago, fast growing | Done | Config ready, needs e2e test |
| **FLUX.1 Kontext** | 31k views, instruction editing LoRAs | ~1h | Needs `arch: "flux_kontext"`, paired dataset (`control_path`) |
| **Chroma** | Popular open model, simple flowmatch | Done | Config ready |
| **FLUX.2** | 26k views, 3 months old | ~30min | Needs `arch: "flux2"` entry |

### Tier 2 — High demand, image edit / multimodal

| Model | Evidence | Effort | Notes |
|---|---|---|---|
| **Qwen-Image** | 37k + 15k views, character LoRAs | ~1h | `arch: "qwen_image"`, needs `qtype` + `cache_text_embeddings` for 24GB |
| **Qwen Image Edit** | 26k + 21k + 11k views, 3 tutorials | ~1h | `arch: "qwen_image_edit"`, 32GB min without heavy quant |

### Tier 3 — Video models (different pipeline, defer)

| Model | Evidence | Effort | Notes |
|---|---|---|---|
| **Wan 2.2 14B T2V/I2V** | 43k views | ~3h | Video pipeline: num_frames, fps, MOE switching |
| **Wan 2.2 5B I2V** | 23k views | ~2h | Simpler but still video |
| **Wan 2.1 14B** | 23k + 9k views | ~2h | Older, image-from-video trick |
| **LTX-2** | 20k views, 1 month old (growing) | ~2h | Video + audio, `num_frames: 121` |

### Tier 4 — Niche (skip for now)

| Model | Notes |
|---|---|
| Concept Slider | 10k views, specialized trainer type (`concept_slider`) |
| HiDream | Low demand, 48GB example config |
| OmniGen2 | Low visibility |
| Flex.1/2 | Ostris's own models, lower adoption |

### Recommendation

**Ship Tiers 1-2 image models first.** That covers ~80% of what users actually
want to train. Video models (Tier 3) are a separate pipeline and can wait.

Priority order:
1. ~~Z-Image-Turbo~~ ✅ done
2. ~~flux-schnell~~ ✅ done
3. Flux Kontext — needs paired dataset support
4. FLUX.2 — straightforward `arch` entry
5. Qwen-Image — needs 24GB quant config
6. Qwen Image Edit — needs 32GB or heavy quant
7. Chroma ✅ done (config ready)

---

## Gap inventory (updated)

### ~~1. Architecture detection is too simplistic~~
✅ Fixed — `_detect_arch()` uses `MODEL_REGISTRY` lookup + substring heuristics.

### ~~2. No `arch` field in model config~~
✅ Fixed — `ARCH_CONFIGS` table emits `arch` for extension models (zimage, chroma).

### ~~3. `is_flux: true` is legacy~~
✅ Fixed — only flux/flux_schnell use `is_flux`; extension models use `arch`.

### ~~4. Flux sub-variants handled identically~~
✅ Fixed — `flux_schnell` has its own entry with `assistant_lora_path`, 4-step sampling, guidance 1.0.

### ~~5. HF_MODEL_MAP is tiny (5 entries)~~
✅ Fixed — `MODEL_REGISTRY` maps 8 models with both arch key and HF path.

### ~~6. Sampling config is hardcoded per 3 families~~
✅ Fixed — `ARCH_CONFIGS` table drives sampling per model.

### ~~7. Presets don't vary by architecture~~
Partially addressed — `BaseModelFamily` enum expanded. Preset tuning per arch
is a future enhancement (current defaults are reasonable for all flowmatch models).

### 8. No VRAM-aware config tuning
Still needed. Z-Image gets `quantize_te + low_vram` always; ideally this would
be conditional on VRAM. Qwen-Image *requires* `cache_text_embeddings + qtype`
on 24GB — can't train without it.

### ~~9. Rust `BaseModelFamily` enum has only 3 variants~~
✅ Fixed — now has `Flux`, `FluxSchnell`, `ZImage`, `Chroma`, `Sdxl`, `Sd15`.

### 10. No model-specific train block settings
Partially addressed — `extra_train` field in `ARCH_CONFIGS` handles `timestep_type`,
`max_denoising_steps`, etc. More models will need specific extras (e.g., wan's
`switch_boundary_every`, qwen's `cache_text_embeddings`).

### 11. Flux Kontext needs paired dataset
`flux_kontext` requires `control_path` in the dataset config for source images.
The dataset spec doesn't support this yet.

### 12. Qwen-Image needs `qtype` for quantization
Standard `quantize: true` isn't enough — needs `qtype: "uint3|<adapter_path>"`
for 24GB VRAM. The adapter doesn't emit `qtype` yet.

---

## Implementation plan (revised)

### Phase 1: ~~Architecture config table~~ ✅ DONE
- `ARCH_CONFIGS` table in `train_adapter.py`
- `_detect_arch()` + `_resolve_model_path()`
- `MODEL_REGISTRY` with 8 models
- Rust `BaseModelFamily` expanded

### Phase 2: Flux Kontext support
**Goal**: Train instruction-editing LoRAs with paired before/after images.

**Changes**:
- Add `flux_kontext` to `ARCH_CONFIGS` with `arch: "flux_kontext"`, `timestep_type: "weighted"`
- Add `control_path` support in dataset config (source images for edits)
- Add `flux-kontext` to `MODEL_REGISTRY` → `black-forest-labs/FLUX.1-Kontext-dev`
- Resolution capped at `[512, 768]` (OOM at 1024 on 24GB)

**Files**: `train_adapter.py`

### Phase 3: FLUX.2 + Qwen-Image support
**Goal**: Two more high-demand models.

**Changes**:
- Add `flux2` to `ARCH_CONFIGS` (similar to flux, `arch: "flux2"`)
- Add `qwen_image` with `qtype`, `cache_text_embeddings`, `low_vram`
- Add `qwen_image_edit` variant
- Expand `MODEL_REGISTRY`

**Files**: `train_adapter.py`

### Phase 4: VRAM-aware config
**Goal**: Auto-tune quantization/offloading based on GPU memory.

**Changes**:
- Pass `vram_mb` through spec into Python adapter
- Per-arch VRAM thresholds (e.g., qwen_image < 32GB → uint3 quant)
- `quantize_te`, `low_vram`, `cache_text_embeddings` gated on VRAM

**Files**: `train_adapter.py`, `job.rs`

### Phase 5: Video models (Wan, LTX-2)
**Goal**: LoRA training for video generation models.

**Changes**:
- Separate dataset config for video (num_frames, fps)
- `wan22_14b`, `wan22_5b`, `ltx2` arch configs
- MOE-specific settings (`switch_boundary_every`, `train_high_noise`/`train_low_noise`)

**Files**: `train_adapter.py`, potentially new video dataset handling

### Phase 6: Generation adapter for new models
**Goal**: `modl generate` works with z-image, chroma, etc.

**Files**: `gen_adapter.py`

---

## Priority order (revised)

| # | Phase | Effort | Unlocks |
|---|---|---|---|
| 1 | ~~Arch config table~~ | ~~2h~~ | ✅ DONE — z-image, chroma, flux-schnell |
| 2 | Flux Kontext | ~1h | Instruction-editing LoRAs (31k views) |
| 3 | FLUX.2 + Qwen-Image | ~2h | Two high-demand models (63k+ views combined) |
| 4 | VRAM-aware config | ~1h | 24GB users can train qwen-image, z-image |
| 5 | Video models | ~4h | Wan 2.2, LTX-2 (86k+ views combined) |
| 6 | Gen adapter | ~2h | End-to-end generate with new models |

## What NOT to do yet

- Video model support — totally different pipeline, needs design
- Full model registry integration (reading arch from registry manifests) — overengineered for now
- UI changes — preview server already handles any model's samples generically
- Concept Sliders — different trainer type (`concept_slider`), niche
- HiDream — requires 48GB, tiny user base
