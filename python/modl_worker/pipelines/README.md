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

## Local modifications (Apache-2.0 §4b)

`krea2_ostris_edit.py` differs from upstream in exactly one place:

- **Removed the diffusers namespace registration.** Upstream ends the
  transformer section with `diffusers.Krea2Transformer2DModel =
  Krea2Transformer2DModel`, so that `DiffusionPipeline.from_pretrained` can
  resolve the Krea 2 repos' `model_index.json` on diffusers releases that predate
  Krea 2. modl assembles pipelines from local store components with explicitly
  named classes and never loads those repos via `from_pretrained`, so the
  registration is unnecessary here — and actively harmful: it is process-global
  and permanent, so importing this module for one edit job would silently
  replace the transformer class used by every later *generation* job in the same
  process (the persistent worker keeps one process alive across jobs). The two
  classes have identical `state_dict` keys (430/430), so the swap loads cleanly
  and only changes the compute path — it never raises. The call site is replaced
  by a comment explaining this; nothing else is touched.

Vendored classes are addressed through the `modl.` namespace
(`modl.Krea2OstrisEditPipeline`, `modl.Krea2Transformer2DModel`) and resolved in
`modl_worker/pipelines/__init__.py`, which makes the diffusers-vs-vendored choice
explicit at every call site.

## Integration status

Wired 2026-07-18. Arch key **`krea2_edit`** in `arch_config.py`:

1. ✅ Component-assembly loader — `assemble_pipeline` builds
   `Krea2OstrisEditPipeline` from modl store components; the transformer is this
   module's class, loaded through the existing krea2 checkpoint converter.
2. ✅ `arch_config.py` — `krea2_edit` entry; `krea-2-edit` / `krea2-edit` model
   ids, plus `detect_arch` routing for `*edit*` / `*reid*` / `*outpaint*` krea
   ids. Weights reuse the **Raw** checkpoint: these edit models are LoRAs, not
   separate checkpoints, so the user supplies the LoRA.
3. ✅ `edit_adapter.py` — `editing_mode: "krea2_reference"` passes the source as
   `image=` with the conditioning kwargs above.
4. ⬜ `models.toml` + registry manifests — deliberately not done. Nothing
   surfaces `krea-2-edit` to users yet.
5. ⬜ Outpaint placement (`reference_placements` + `krea2_outpaint_placement.py`
   two-pass) — `krea2_edit` currently covers the single-reference ReID/identity
   path only.

**Operational constraints (measured on a 4090, 2026-07-18):**

- `enable_model_cpu_offload()` **does not work** with this pipeline.
  `precompute_ref_kv()` is a custom method, not `forward()`, so accelerate's
  hooks never fire and the transformer stays on CPU while its inputs are on GPU.
  The arch sets `model_flags.requires_resident`, which makes `assemble_pipeline`
  place the pipeline resident instead.
- fp8 layerwise casting is **mandatory**: 1024 + CFG peaked at 24.35 GB of a
  24.56 GB card. 24 GB is the floor for 1024; smaller cards need 512–768.
