# Vendored edit pipelines

Self-contained third-party pipeline code for Krea 2 reference-editing (K6).

## Contents

| File | Source | License |
|------|--------|---------|
| `krea2_ostris_edit.py` | `yijunwang2/krea2-outpaint` `pipeline.py` (pins `ostris/Krea2OstrisEdit`) | Apache-2.0 |
| `krea2_outpaint_placement.py` | `yijunwang2/krea2-outpaint` `outpaint.py` | Apache-2.0 |
| `PIPELINE_LICENSE.apache-2.0` | upstream `PIPELINE_LICENSE` | — |
| `NOTICE.krea2-ostris-edit` | upstream `NOTICE` | — |

The **outpaint** `pipeline.py` is a strict superset of the **reid** one (identical
class/method structure; the outpaint copy adds an optional `reference_placements`
path — 43 lines). Vendoring the superset gives a single `Krea2OstrisEditPipeline`
that serves both ReID (full-frame identity reference) and outpaint (placed
reference) — and the same code path K7 / edit-LoRA-training inference will use.

The LoRA **weights** (`krea2_reid_rank32`, `krea2_outpaint_rank32`) are NOT
vendored — they are Krea 2 Community License, curated in modl-registry with
hash-pins (reid `a80349fa…944`, outpaint `1de7c106…e76`) and pulled on demand.

## What this pipeline bundles (and why edit arches use it, not the gen one)

`krea2_ostris_edit.py` is self-contained: its own `Krea2Transformer2DModel`
carries reference-conditioning our generation `Krea2Transformer2DModel` lacks —
`precompute_ref_kv()` runs the clean reference tokens through the blocks once at
flow t=0 and caches each block's rotary-embedded K/V (reference tokens attend
only to each other, so their K/V are timestep-independent), and the block
`forward` accepts `ref_kv_cache=`/`kv_capture=`. Same layer names/shapes as the
base krea2 transformer, so base weights load through the existing checkpoint
converter — only the compute paths differ. It also bundles the same
`_convert_non_diffusers_krea2_lora_to_diffusers` we reimplemented in
`adapters/lora_utils.py`, so reid/outpaint LoRAs load through the krea2 path.

## Public API (from upstream example.py)

```python
pipe(prompt=..., image=<reference>, width, height,
     num_inference_steps=8, guidance_scale=0.0,
     reference_max_pixels=384*384,
     encode_reference_in_prompt=True,        # ReID: identity ref in the prompt
     # OR, for outpaint:
     reference_placements=[{"bbox_normalized": [x0,y0,x1,y1]}],
     encode_reference_in_prompt=False,
     kv_cache=True)
```

## Integration TODO (not yet wired — see docs/plans krea2 K6)

1. Component-assembly loader: instantiate `Krea2OstrisEditPipeline.__init__` with
   modl store components (this vendored transformer + our TE/VAE/tokenizer/
   scheduler), instead of upstream's `from_pretrained(custom_pipeline=…)`.
2. `arch_config.py`: `krea2_reid` / `krea2_outpaint` entries →
   `pipeline_class: Krea2OstrisEditPipeline` (this module), base krea2 weights +
   the reid/outpaint LoRA as a dependency.
3. `edit_adapter.py`: route `krea-2-reid` / `krea-2-outpaint`; pass the edit
   source as `image=` plus the conditioning kwargs above; outpaint two-pass
   interior placement compiles to two chained step-jobs.
4. `models.toml` + registry manifests (gated on the feature working +
   GPU-verified — do NOT surface the models before the path runs end-to-end).
