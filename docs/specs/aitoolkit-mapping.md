# ai-toolkit Model Mapping

> **STATUS:** Flux-dev/schnell fully working. All other architectures have structural scaffolding but incorrect or missing config mappings. This doc tracks what's done, what's broken, and the plan to support every ai-toolkit architecture.

---

## How the Pipeline Works

```
CLI (train.rs)
  │  resolve dataset, base model, preset
  │  assemble TrainJobSpec
  ▼
LocalExecutor (executor.rs)
  │  serialize spec → YAML → /tmp/jobs/{id}.yaml
  │  spawn: python -m modl_worker.main train --config <spec.yaml>
  │  env: PYTHONPATH=modl_worker:ai-toolkit, MODL_AITOOLKIT_ROOT=~/.modl/runtime/ai-toolkit
  ▼
Python worker (main.py → train_adapter.py)
  │  spec_to_aitoolkit_config(spec) → ai-toolkit's nested YAML
  │  write translated config → tmpfile
  │  subprocess: python run.py <aitk_config.yaml>
  │  parse stdout for step/loss → emit JSON events on stdout
  ▼
ai-toolkit (run.py → ExtensionJob → SDTrainer)
  │  load model, prepare LoRA, train, save checkpoints
  ▼
Rust event loop (train.rs)
  │  read JSON events → progress bar, DB persistence
  │  on completion: collect .safetensors → ~/.modl/store/
```

### Key files

| File | Role |
|------|------|
| `src/cli/train.rs` | Interactive prompts, spec assembly, progress display |
| `src/core/presets.rs` | Pure functions: dataset stats + GPU info → training params |
| `src/core/job.rs` | `TrainJobSpec`, `TrainingParams`, `ModelRef`, serialization |
| `src/core/executor.rs` | `LocalExecutor`: spawn Python worker, parse events |
| `python/modl_worker/adapters/train_adapter.py` | `spec_to_aitoolkit_config()` — the mapping function |
| `python/modl_worker/protocol.py` | `EventEmitter` — JSON-line protocol over stdout |

---

## Current Mapping: Modl Spec → ai-toolkit Config

| Modl spec field | ai-toolkit config path | Status |
|---|---|---|
| `model.base_model_id` | `process[0].model.name_or_path` | ✅ Translated via `HF_MODEL_MAP` for Flux |
| `model.base_model_path` | `process[0].model.name_or_path` (fallback) | ✅ Used for non-Flux |
| `output.lora_name` | `config.name` | ✅ |
| `output.destination_dir` | `process[0].training_folder` | ✅ |
| `params.trigger_word` | `process[0].trigger_word` + `datasets[0].default_caption` | ✅ |
| `params.rank` | `network.linear` + `network.linear_alpha` | ✅ (alpha always = rank) |
| `params.steps` | `train.steps` + `save.save_every` + `sample.sample_every` | ✅ |
| `params.learning_rate` | `train.lr` | ✅ |
| `params.optimizer` | `train.optimizer` | ✅ |
| `params.resolution` | `datasets[0].resolution` + `sample.width/height` | ✅ (Flux uses multi-res) |
| `params.seed` | `sample.seed` + `train.seed` | ✅ |
| `params.quantize` | `model.quantize` | ❌ **Ignored** — hardcoded `True` for Flux |
| `dataset.path` | `datasets[0].folder_path` | ✅ |
| `dataset.image_count` | — | ❌ Not mapped (could tune batch_size) |
| `dataset.caption_coverage` | — | ❌ Not mapped (could tune caption_dropout_rate) |

### Hardcoded values (not exposed in spec)

| ai-toolkit field | Current hardcoded value | Should be |
|---|---|---|
| `train.noise_scheduler` | `"flowmatch"` for all models | Per-arch: `"ddpm"` for SD1.5/SDXL |
| `train.dtype` | `"bf16"` (Flux) / `"fp16"` (other) | Per-arch |
| `train.batch_size` | `1` | Expose in params or auto-tune from VRAM |
| `train.gradient_accumulation_steps` | `1` | Expose in Advanced preset |
| `train.train_text_encoder` | `False` | Per-arch: often `True` for SD1.5 |
| `train.train_unet` | `True` | Always true (ai-toolkit handles naming) |
| `model.quantize` | `True` for Flux, absent for others | Use `params.quantize` |
| `model.low_vram` | `True` for Flux | Use `params.quantize` as signal |
| `sample.sampler` | `"flowmatch"` | Per-arch: `"ddpm"` for SD1.5/SDXL |
| `sample.guidance_scale` | `4` | Per-arch: 7.5 for SDXL, 7 for SD1.5 |
| `sample.sample_steps` | `20` | Per-arch: 20-30 for ddpm, 4 for schnell |
| `datasets[0].caption_ext` | `"txt"` | OK as default |
| `datasets[0].caption_dropout_rate` | `0.05` | OK as default |
| `datasets[0].cache_latents_to_disk` | `True` | OK as default |
| `train.ema_config.use_ema` | `True` | Probably fine, but not all archs benefit |
| `save.dtype` | `"float16"` | OK as default |
| `network.type` | `"lora"` | OK (only training type we support) |

---

## Model Architecture Support

### What ai-toolkit supports (full list)

ai-toolkit has two model loading systems:

**Legacy system** (flags in `ModelConfig`, loaded by `StableDiffusion` class):

| Arch | Flag | Noise scheduler | Dtype | Quantize | Notes |
|------|------|----------------|-------|----------|-------|
| `sd1` | default | `ddpm` | fp16 | No | Stable Diffusion 1.5 |
| `sd2` | `is_v2` | `ddpm` | fp16 | No | Stable Diffusion 2.x |
| `sdxl` | `is_xl` | `ddpm` | fp16 | No | SDXL |
| `sd3` | `is_v3` | `flowmatch` | bf16 | Yes (transformer+T5) | SD 3.5 |
| `flux` | `is_flux` | `flowmatch` | bf16 | Yes (qfloat8) | FLUX.1-dev/schnell |
| `flex1` | → `flux` | `flowmatch` | bf16 | Yes | Flex.1-alpha (+ `bypass_guidance_embedding`) |
| `pixart` | `is_pixart` | `ddpm` | fp16 | TE only | PixArt-Alpha |
| `pixart_sigma` | `is_pixart_sigma` | `ddpm` | fp16 | TE only | PixArt-Sigma |
| `auraflow` | `is_auraflow` | `flowmatch` | bf16 | TE only | AuraFlow |
| `lumina2` | `is_lumina2` | `flowmatch` | bf16 | TE only | Lumina-Image-2.0 |

**New extension system** (`BaseModel` subclasses, registered via `AI_TOOLKIT_MODELS`):

| Arch | Class | Notes |
|------|-------|-------|
| `wan21` | `Wan21` | Video generation (1.3B / 14B), `flowmatch` |
| `wan22_5b/14b` | `Wan225bModel`, `Wan2214bModel` | Video, MOE, `flowmatch`, uint4+ARA |
| `cogview4` | `CogView4` | Image generation |
| `chroma` | `ChromaModel` | Image, `flowmatch` |
| `hidream` | `HidreamModel` | Image, MoE, 48GB+ |
| `flux_kontext` | `FluxKontextModel` | Kontext-dev, paired dataset |
| `flux2` | `Flux2Model` | FLUX.2 variants |
| `omnigen2` | `OmniGen2Model` | Multi-task |
| `qwen_image` | `QwenImageModel` | Image, uint3+ARA |
| `ltx2` | `LTX2Model` | Video |
| `z_image` | `ZImageModel` | Z-Image |
| `f_light` | `FLiteModel` | Lightweight |

### What modl currently supports

| Model | Status | Notes |
|-------|--------|-------|
| `flux-dev` | ✅ Working | E2E tested, quantize + low_vram, flowmatch |
| `flux-schnell` | 🟡 Mapped | Config generated but needs `assistant_lora_path` for proper training |
| `sdxl` | ❌ Broken | Would crash: `flowmatch` scheduler is wrong (needs `ddpm`) |
| `sd1.5` | ❌ Broken | Would crash: `flowmatch` scheduler, wrong resolution, no TE training |
| `sd3` | ❌ Not mapped | Needs `is_v3` flag, flowmatch scheduler (would be easy) |
| Everything else | ❌ Not mapped | No arch detection, no config generation |

### Interactive model list (train.rs)

Currently hardcoded:
```rust
let models = &["flux-dev", "flux-schnell"];
```

Should expand to include at minimum the models in the registry.

---

## The Plan

### Phase 1: Fix the Flux path (done, validating)

The Flux path works end-to-end. Remaining items:
- [x] `spec_to_aitoolkit_config()` generates valid Flux config
- [x] Progress parsing works (step/loss extraction)
- [x] Artifact collection works (.safetensors → store)
- [ ] Wire `params.quantize` through instead of hardcoding `True`
- [ ] Add sample prompts support (currently empty list)
- [ ] Handle `flux-schnell` properly (needs `assistant_lora_path`, `guidance_scale=1`, `sample_steps=4`)

### Phase 2: Architecture-aware config generation

The key insight: `spec_to_aitoolkit_config()` currently has one `is_flux` branch. It needs a model profile system that provides the correct defaults per architecture.

**Add a `MODEL_PROFILES` dict** in `train_adapter.py`:

```python
MODEL_PROFILES = {
    "flux-dev": {
        "arch": "flux",
        "hf_id": "black-forest-labs/FLUX.1-dev",
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "quantize": True,
        "low_vram": True,
        "multi_resolution": [512, 768, 1024],
        "guidance_scale": 4,
        "sample_steps": 20,
        "sampler": "flowmatch",
    },
    "flux-schnell": {
        "arch": "flux",
        "hf_id": "black-forest-labs/FLUX.1-schnell",
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "quantize": True,
        "low_vram": True,
        "multi_resolution": [512, 768, 1024],
        "guidance_scale": 1,
        "sample_steps": 4,
        "sampler": "flowmatch",
        "assistant_lora_path": "ostris/FLUX.1-schnell-training-adapter",
    },
    "sdxl": {
        "arch": "sdxl",
        "hf_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "noise_scheduler": "ddpm",
        "dtype": "fp16",
        "quantize": False,
        "low_vram": False,
        "multi_resolution": None,  # single resolution
        "guidance_scale": 7.5,
        "sample_steps": 25,
        "sampler": "ddpm",
        "train_text_encoder": False,
    },
    "sd-1.5": {
        "arch": "sd1",
        "hf_id": "runwayml/stable-diffusion-v1-5",
        "noise_scheduler": "ddpm",
        "dtype": "fp16",
        "quantize": False,
        "low_vram": False,
        "multi_resolution": None,
        "guidance_scale": 7.0,
        "sample_steps": 25,
        "sampler": "ddpm",
        "default_resolution": 512,
        "train_text_encoder": True,
    },
    "sd3.5-large": {
        "arch": "sd3",
        "hf_id": "stabilityai/stable-diffusion-3.5-large",
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "quantize": True,
        "low_vram": True,
        "multi_resolution": [512, 768, 1024],
        "guidance_scale": 4,
        "sample_steps": 20,
        "sampler": "flowmatch",
    },
}
```

Then refactor `spec_to_aitoolkit_config()` to look up the profile:

```python
profile = MODEL_PROFILES.get(base_model_id, {})
arch = profile.get("arch", "flux")  # default to flux for unknown
```

And use `profile` values instead of `is_flux` branches throughout.

### Phase 3: Expand the Rust model list

1. **`presets.rs`**: Expand `BaseModelFamily` to include `Sd3`, `Chroma`, `Wan` etc. with appropriate default resolutions
2. **`train.rs`**: Replace hardcoded model list with models from the registry (or a static list that matches `MODEL_PROFILES`)
3. **`job.rs`**: No changes needed — `TrainJobSpec` is already model-agnostic

### Phase 4: Video and new-arch models

These require structural changes beyond config mapping:
- **Video models** (Wan2.1, Wan2.2, LTX2): Need `num_frames`, `fps` in the spec, different dataset format
- **MoE models** (HiDream, Wan2.2-14B): Need `switch_boundary_every`, layer exclusions, high VRAM (48GB+)
- **Kontext/Flex2**: Need paired datasets (control images), special `model_kwargs`
- **Ultra-low-bit quantization** (uint3, uint4+ARA): Need `accuracy_recovery_adapter` paths

These are lower priority — focus on getting image-only models right first.

---

## Architecture Cheat Sheet

Quick reference for what each model family needs in ai-toolkit config:

| Model | `noise_scheduler` | `dtype` | `quantize` | `sample.sampler` | `guidance_scale` | `sample_steps` | Resolution |
|-------|-------------------|---------|-----------|-------------------|-----------------|----------------|-----------|
| flux-dev | `flowmatch` | `bf16` | Yes | `flowmatch` | 4 | 20 | [512, 768, 1024] |
| flux-schnell | `flowmatch` | `bf16` | Yes | `flowmatch` | 1 | 4 | [512, 768, 1024] |
| sdxl | `ddpm` | `fp16` | No | `ddpm` | 7.5 | 25 | 1024 |
| sd-1.5 | `ddpm` | `fp16` | No | `ddpm` | 7.0 | 25 | 512 |
| sd3.5 | `flowmatch` | `bf16` | Yes | `flowmatch` | 4 | 20 | [512, 768, 1024] |
| chroma | `flowmatch` | `bf16` | Yes | `flowmatch` | 4 | 20 | [512, 768, 1024] |
| lumina2 | `flowmatch` | `bf16` | TE only | `flowmatch` | 4 | 20 | [512, 768, 1024] |
| wan21 (video) | `flowmatch` | `bf16` | Yes | `flowmatch` | — | 30 | ~480p |

---

## What Needs to Change (file-by-file)

### `python/modl_worker/adapters/train_adapter.py`

1. **Add `MODEL_PROFILES` dict** with per-model defaults (scheduler, dtype, quantize, sampler, guidance, resolution)
2. **Replace `is_flux` branches** with profile lookups
3. **Wire `params.quantize`** through to `model.quantize` instead of hardcoding
4. **Add `arch` field** to model config for new-system models (chroma, wan, etc.)
5. **Add `assistant_lora_path`** for flux-schnell
6. **Fix non-flux models**: use `ddpm` scheduler, don't set `is_flux`, etc.

### `src/core/presets.rs`

1. **Expand `BaseModelFamily`** — add `Sd3`, `Chroma`
2. **Model-specific defaults** — `train_text_encoder` suggestion for SD1.5
3. **Keep it simple** — presets are about step/rank/LR scaling, not per-model config

### `src/cli/train.rs`

1. **Expand model list** — pull from registry or static list
2. **Handle model-specific prompts** — e.g., warn if schnell selected ("requires training adapter download")

### `src/core/job.rs`

No changes. `TrainJobSpec` is already model-agnostic. The `TrainingParams` struct has all needed fields. Model-specific translation happens in the Python adapter.

---

## Testing Strategy

Each new model profile should be testable without a GPU:

```bash
# Generate the spec (no GPU needed)
modl train --dataset test --base sdxl --preset quick --dry-run

# Translate to ai-toolkit config (no GPU needed)
python -c "
from modl_worker.adapters.train_adapter import spec_to_aitoolkit_config
import yaml
spec = yaml.safe_load(open('spec.yaml'))
print(yaml.dump(spec_to_aitoolkit_config(spec)))
"

# Validate against ai-toolkit's config parser (no GPU needed)
python -c "
from toolkit.config_modules import ModelConfig
ModelConfig(**config['model'])  # should not throw
"
```

GPU-required E2E tests should run per model family in CI (when available).
