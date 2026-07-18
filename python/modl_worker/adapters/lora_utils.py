"""Shared LoRA key conversion and fp8-compatible loading utilities.

Handles the diffusion_model.* -> transformer.* key prefix conversion
required when loading ai-toolkit or some CivitAI LoRAs with diffusers.

Also handles deferred fp8 casting: fp8 models are loaded as bf16 so that
LoRA fuse works (bf16 + bf16), then layerwise casting is applied after
fuse to get fp8 storage / bf16 compute on GPU.
"""

from __future__ import annotations

import os
import re
import tempfile
import logging

logger = logging.getLogger(__name__)


# Original krea-ai/krea-2 (ComfyUI / ai-toolkit) module names → the
# Krea2Transformer2DModel names diffusers uses. Mirrors diffusers'
# `_convert_non_diffusers_krea2_lora_to_diffusers`, but our loader DROPS
# residual keys that have no diffusers target instead of raising — the
# diffusers refactor folded the original per-block/final `modulation.lin`
# projections into a shared `time_mod_proj` + per-block `scale_shift_table`
# parameters, so those LoRA tensors (2/530 in the community distill extract)
# are unmappable and simply skipped.
_KREA2_ATTN_MAP = {"wq": "to_q", "wk": "to_k", "wv": "to_v", "wo": "to_out.0", "gate": "to_gate"}
_KREA2_FF_MAP = {"gate": "ff.gate", "up": "ff.up", "down": "ff.down"}
_KREA2_STANDALONE_MAP = {
    "first": "img_in",
    "last.linear": "final_layer.linear",
    "tmlp.0": "time_embed.linear_1",
    "tmlp.2": "time_embed.linear_2",
    "tproj.1": "time_mod_proj",
    "txtmlp.1": "txt_in.linear_1",
    "txtmlp.3": "txt_in.linear_2",
    "txtfusion.projector": "text_fusion.projector",
}


def _krea2_map_module(module: str) -> str | None:
    m = re.match(r"blocks\.(\d+)\.(attn|mlp)\.(\w+)$", module)
    if m:
        idx, kind, sub = m.groups()
        if kind == "attn" and sub in _KREA2_ATTN_MAP:
            return f"transformer_blocks.{idx}.attn.{_KREA2_ATTN_MAP[sub]}"
        if kind == "mlp" and sub in _KREA2_FF_MAP:
            return f"transformer_blocks.{idx}.{_KREA2_FF_MAP[sub]}"
        return None
    m = re.match(r"txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(attn|mlp)\.(\w+)$", module)
    if m:
        block, idx, kind, sub = m.groups()
        if kind == "attn" and sub in _KREA2_ATTN_MAP:
            return f"text_fusion.{block}.{idx}.attn.{_KREA2_ATTN_MAP[sub]}"
        if kind == "mlp" and sub in _KREA2_FF_MAP:
            return f"text_fusion.{block}.{idx}.{_KREA2_FF_MAP[sub]}"
        return None
    return _KREA2_STANDALONE_MAP.get(module)


def _is_krea2_lora(state_dict: dict) -> bool:
    """True if the LoRA uses original krea module names (needs remapping)."""
    for k in state_dict:
        base = k.replace("base_model.model.", "").replace("diffusion_model.", "")
        if re.match(r"blocks\.\d+\.attn\.(wq|wk|wv|wo)", base) or base.startswith("txtfusion."):
            return True
    return False


def convert_krea2_lora_to_diffusers(state_dict: dict) -> tuple[dict, int]:
    """Map an original-format krea2 LoRA onto Krea2Transformer2DModel names.

    Returns ``(converted, dropped)`` where converted keys are prefixed with
    ``transformer.`` (diffusers PEFT format) and *dropped* counts the tensors
    with no diffusers target (unmappable modulation residuals — logged, not
    fatal). Non-lora keys (e.g. ``.alpha``) are carried through when their
    module maps.
    """
    stripped = {
        k.replace("base_model.model.", "").replace("diffusion_model.", ""): v
        for k, v in state_dict.items()
    }
    converted: dict = {}
    dropped = 0
    for key, val in stripped.items():
        m = re.search(r"\.(lora_[AB]\.weight|alpha)$", key)
        if m is None:
            dropped += 1
            continue
        module = _krea2_map_module(key[: m.start()])
        if module is None:
            dropped += 1
            continue
        converted[f"transformer.{module}{key[m.start():]}"] = val
    return converted, dropped


def _maybe_convert_krea2_lora(lora_path: str, emitter=None) -> str | None:
    """If *lora_path* is an original-format krea2 LoRA, write a diffusers-key
    copy to a temp file and return its path; otherwise return None.

    The caller owns the temp file and must delete it after loading.

    Detection is header-only (``safe_open`` key list) so the common
    non-krea2 path doesn't read the whole file just to sniff names —
    ``load_file`` only runs once a krea2 LoRA is confirmed.
    """
    try:
        from safetensors import safe_open

        with safe_open(lora_path, framework="pt") as f:
            keys = list(f.keys())
        if not _is_krea2_lora(keys):
            return None

        from safetensors.torch import load_file, save_file

        sd = load_file(lora_path)
        converted, dropped = convert_krea2_lora_to_diffusers(sd)
        if not converted:
            return None
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp_path = tmp.name
            save_file(converted, tmp_path)
        if emitter:
            note = f" ({dropped} unmappable tensor(s) skipped)" if dropped else ""
            emitter.info(f"  → Converted krea2 LoRA keys to diffusers format{note}")
        return tmp_path
    except Exception as exc:
        if emitter:
            emitter.info(f"  krea2 LoRA conversion skipped: {exc}")
        return None


def convert_lora_keys_if_needed(lora_path: str) -> tuple[str | None, str | None]:
    """Convert LoRA state dict keys from diffusion_model.* to transformer.*.

    If the LoRA has keys with the ``diffusion_model.`` prefix (ai-toolkit
    format), converts them to ``transformer.`` (diffusers format), saves to
    a temp file, and returns (tmp_path, None).

    If no conversion is needed, returns (None, None).

    On failure, returns (None, error_message) so callers can warn and
    continue without LoRA.

    The caller is responsible for deleting the temp file after use.
    """
    try:
        from safetensors.torch import load_file, save_file

        raw_sd = load_file(lora_path)

        old_prefix = "diffusion_model."
        new_prefix = "transformer."
        needs_conversion = any(k.startswith(old_prefix) for k in raw_sd)

        if not needs_conversion:
            return None, None

        converted = {
            new_prefix + k[len(old_prefix):] if k.startswith(old_prefix) else k: v
            for k, v in raw_sd.items()
        }

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp_path = tmp.name
            save_file(converted, tmp_path)

        return tmp_path, None

    except Exception as exc:
        return None, str(exc)


def _apply_deferred_fp8_casting(pipeline, emitter=None) -> None:
    """Apply deferred fp8 layerwise casting on the transformer.

    fp8 models are loaded as bf16 (so LoRA fuse works). After fuse,
    this applies enable_layerwise_casting to get fp8 storage / bf16
    compute, saving ~50% VRAM. Only acts if the transformer was
    marked with ``_modl_needs_fp8_casting``.
    """
    import torch

    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        return
    if not getattr(transformer, "_modl_needs_fp8_casting", False):
        return

    transformer.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
    )
    del transformer._modl_needs_fp8_casting
    if emitter:
        emitter.info("  → Applied deferred fp8 layerwise casting")


def _is_diff_patch(lora_path: str) -> bool:
    """True if the file is a ComfyUI-style additive weight patch.

    Diff patches (e.g. krea2-filter-bypass) ship raw weight deltas instead
    of lora_A/lora_B factors: every tensor key ends in ``.diff`` and the
    delta is added directly onto the matching model weight.
    """
    try:
        from safetensors import safe_open

        with safe_open(lora_path, framework="pt") as f:
            keys = list(f.keys())
        return bool(keys) and all(k.endswith(".diff") for k in keys)
    except Exception:
        return False


def apply_diff_patch(pipeline, lora_path: str, weight: float = 1.0, emitter=None) -> bool:
    """Apply a ComfyUI-style additive weight patch: W += weight * diff.

    Keys use ComfyUI naming (``diffusion_model.<module>.diff``, targeting the
    module's ``.weight``); they are converted to diffusers names with the same
    per-class converter used for checkpoint loading.

    The original values of every patched tensor are stashed on the
    *transformer* (``_modl_diff_patch_originals`` — the transformer is shared
    across ``from_pipe()`` mode clones) so the persistent worker can restore
    them exactly when the LoRA changes: a diff patch cannot be unfused
    through PEFT.
    """
    import torch
    from safetensors.torch import load_file

    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        _warn_lora_failed(emitter, "diff patch: pipeline has no transformer")
        return False

    sd = load_file(lora_path)

    # Strip prefix + `.diff`; ComfyUI diff patches target the module weight.
    raw = {}
    for key, delta in sd.items():
        name = key[: -len(".diff")]
        for prefix in ("diffusion_model.", "transformer."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        if not name.endswith((".weight", ".bias", ".scale")):
            name += ".weight"
        raw[name] = delta

    try:
        from modl_worker.adapters.pipeline_loader import _get_checkpoint_converter

        converter = _get_checkpoint_converter(type(transformer).__name__)
        converted = converter(raw) if converter is not None else raw
    except Exception:
        converted = raw

    params = dict(transformer.named_parameters())
    matched = []
    for name, delta in converted.items():
        p = params.get(name)
        if p is None or p.shape != delta.shape:
            _warn_lora_failed(
                emitter,
                f"diff patch key `{name}` does not match any transformer weight "
                f"(shape {'mismatch' if p is not None else 'n/a'})",
            )
            return False
        matched.append((name, p, delta))

    originals = getattr(transformer, "_modl_diff_patch_originals", {})
    with torch.no_grad():
        for name, p, delta in matched:
            if name not in originals:
                originals[name] = p.detach().clone()
            # Upcast so the addition also works on fp8-stored weights.
            new = p.detach().to(torch.float32) + float(weight) * delta.to(
                device=p.device, dtype=torch.float32
            )
            p.copy_(new.to(p.dtype))
    transformer._modl_diff_patch_originals = originals

    _apply_deferred_fp8_casting(pipeline, emitter)
    if emitter:
        emitter.info(f"  → Applied diff patch: {len(matched)} tensor(s) at weight {weight}")
    return True


def revert_diff_patch(pipeline, emitter=None) -> bool:
    """Restore weights patched by `apply_diff_patch`.

    Returns True if a patch was present and reverted, False if there was
    nothing to revert (so callers can fall back to PEFT unfuse).
    """
    import torch

    transformer = getattr(pipeline, "transformer", None)
    originals = getattr(transformer, "_modl_diff_patch_originals", None) if transformer else None
    if not originals:
        return False

    params = dict(transformer.named_parameters())
    with torch.no_grad():
        for name, orig in originals.items():
            p = params.get(name)
            if p is not None:
                p.copy_(orig.to(device=p.device, dtype=p.dtype))
    del transformer._modl_diff_patch_originals
    if emitter:
        emitter.info(f"  → Reverted diff patch ({len(originals)} tensor(s))")
    return True


def load_lora_with_conversion(
    pipeline,
    lora_path: str,
    lora_weight: float = 1.0,
    emitter=None,
) -> bool:
    """Load a LoRA onto a pipeline, with automatic key conversion fallback.

    Tries direct loading first.  On failure, attempts key prefix conversion
    (diffusion_model.* -> transformer.*).  On second failure, logs a warning
    and returns False so the caller can continue without LoRA.

    After successful fuse, applies deferred fp8 casting if the model was
    loaded as bf16 for LoRA compatibility.

    Returns True if the LoRA was successfully loaded and fused, False otherwise.
    """
    if _is_diff_patch(lora_path):
        return apply_diff_patch(pipeline, lora_path, lora_weight, emitter)

    # Krea 2 LoRAs use original krea module names (wq/wk/wv/wo, txtfusion.*,
    # last.modulation.lin). Diffusers' native converter RAISES on the
    # unmappable modulation residual, so pre-convert to diffusers keys here
    # (dropping the orphans) and load that instead of the raw file.
    krea2_tmp = _maybe_convert_krea2_lora(lora_path, emitter)
    if krea2_tmp is not None:
        try:
            pipeline.load_lora_weights(
                os.path.dirname(krea2_tmp),
                weight_name=os.path.basename(krea2_tmp),
                adapter_name="default",
            )
            # Fuse at strength into the bf16 base, then unload the adapter and
            # apply fp8 casting to the fused weights. Krea 2's full stack
            # (12.9B transformer + Qwen3-VL TE + a 1.9GB r256 LoRA) doesn't fit
            # resident on 24GB if the LoRA stays a separate bf16 adapter, and
            # the cpu-offload + layerwise-cast + PEFT path OOMs. Fusing in bf16
            # first (base is loaded as bf16 for exactly this) keeps the fp8
            # storage / bf16 compute footprint identical to a bare generate —
            # the path that already fits and runs fully resident.
            pipeline.fuse_lora(adapter_names=["default"], lora_scale=lora_weight)
            pipeline.unload_lora_weights()
            _apply_deferred_fp8_casting(pipeline, emitter)
            # Mark the fuse as irreversible: we unloaded the adapter (to fit
            # 24GB), so a later unfuse_lora() is a no-op and would leave the
            # base contaminated. The persistent worker checks this to force a
            # full reload instead of an in-place LoRA swap.
            pipeline._modl_lora_fused_irreversible = True
            if emitter:
                emitter.info(f"  → Fused krea2 LoRA at strength {lora_weight}")
            return True
        except Exception as exc:
            # If load_lora_weights succeeded but fuse/casting failed, the
            # adapter is still attached — reporting False ("continue without
            # LoRA") while it silently affects generation would be worse.
            try:
                pipeline.unload_lora_weights()
            except Exception:
                pass
            _warn_lora_failed(emitter, f"krea2 LoRA load failed after conversion: {exc}")
            return False
        finally:
            try:
                os.unlink(krea2_tmp)
            except OSError:
                pass

    lora_dir = os.path.dirname(lora_path)
    lora_file = os.path.basename(lora_path)

    try:
        pipeline.load_lora_weights(lora_dir, weight_name=lora_file, adapter_name="default")
        # Apply deferred fp8 casting (base weights → fp8 storage).
        # Keep LoRA UNfused: PEFT computes bf16 base + bf16 LoRA delta at
        # runtime.  Fusing then quantizing to fp8 loses the LoRA signal.
        _apply_deferred_fp8_casting(pipeline, emitter)
        # Set adapter scale (PEFT handles this at forward time)
        pipeline.set_adapters(["default"], adapter_weights=[lora_weight])
        return True
    except Exception as first_err:
        if emitter:
            emitter.info(f"  Retrying with key conversion (first error: {first_err})")

        tmp_path, convert_err = convert_lora_keys_if_needed(lora_path)

        if convert_err:
            _warn_lora_failed(emitter, convert_err)
            return False

        if tmp_path is None:
            # No conversion needed but direct load failed — incompatible LoRA
            _warn_lora_failed(emitter, str(first_err))
            return False

        try:
            pipeline.load_lora_weights(
                os.path.dirname(tmp_path),
                weight_name=os.path.basename(tmp_path),
                adapter_name="default",
            )
            _apply_deferred_fp8_casting(pipeline, emitter)
            pipeline.set_adapters(["default"], adapter_weights=[lora_weight])
            return True
        except Exception as second_err:
            _warn_lora_failed(emitter, str(second_err))
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def apply_lora_from_spec(pipeline, spec: dict, emitter) -> bool:
    """Load and fuse a LoRA based on the job spec's ``lora`` block.

    Handles:
      - Skipping when no LoRA is specified (applies deferred fp8 if needed)
      - GGUF incompatibility guard (raises RuntimeError)
      - Missing file warning
      - Key conversion fallback via ``load_lora_with_conversion``

    Afterwards — once LoRA / deferred-fp8 state is final and the pipeline's
    true footprint is known — promotes the pipeline to full GPU residency
    when it fits VRAM (skipped for ControlNet jobs, which need the offload
    headroom for the ControlNet weights).

    Returns True if LoRA was applied, False otherwise.
    """
    applied = _apply_lora_from_spec_inner(pipeline, spec, emitter)
    # A pipeline flagged _modl_needs_resident was deliberately left on CPU by
    # assemble_pipeline and cannot run until it is placed — that is not an
    # optimisation, so it happens even for ControlNet jobs.
    needs_resident = getattr(pipeline, "_modl_needs_resident", False)
    if needs_resident or not spec.get("params", {}).get("controlnet"):
        from modl_worker.device import rebalance_pipe_placement
        rebalance_pipe_placement(pipeline, emitter)
    return applied


def _apply_lora_from_spec_inner(pipeline, spec: dict, emitter) -> bool:
    lora_info = spec.get("lora")
    if not lora_info:
        # No LoRA — apply deferred fp8 casting now (was deferred for LoRA compat)
        _apply_deferred_fp8_casting(pipeline, emitter)
        return False

    lora_path = lora_info.get("path")
    lora_weight = lora_info.get("weight", 1.0)
    lora_name = lora_info.get("name", "unnamed")

    # GGUF models support PEFT LoRA via diffusers' GGUFQuantizationConfig.
    # The LoRA adapters compute in bf16 on top of the quantized base weights.

    if not lora_path:
        _apply_deferred_fp8_casting(pipeline, emitter)
        return False

    if not os.path.exists(lora_path):
        if emitter:
            emitter.warning("LORA_NOT_FOUND", f"LoRA file not found: {lora_path}")
        _apply_deferred_fp8_casting(pipeline, emitter)
        return False

    if emitter:
        emitter.info(f"Loading LoRA: {lora_name} (weight={lora_weight})")
    return load_lora_with_conversion(pipeline, lora_path, lora_weight, emitter)


def _warn_lora_failed(emitter, message: str) -> None:
    """Emit a warning about LoRA loading failure."""
    if emitter:
        emitter.warning(
            "LORA_INCOMPATIBLE",
            f"Could not load LoRA (incompatible with model?): {message}. "
            f"Generating without LoRA.",
        )
    else:
        logger.warning("LoRA load failed: %s — generating without LoRA", message)
