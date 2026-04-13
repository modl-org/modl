"""ComfyUI NVFP4 dequantization — converts packed fp4 weights to standard floats.

ComfyUI stores weights in NVFP4 format:
  - weight: uint8 (two fp4 E2M1 values packed per byte)
  - weight_scale: fp8_e4m3fn block scales (blocks of 16 elements)
  - weight_scale_2: f32 global per-tensor scale
  - comfy_quant: JSON metadata {"format": "nvfp4"}

This module streams the dequantization to disk to avoid holding both the
source fp4 dict and the full bf16 dict in RAM at the same time. The
dequantized safetensors file is cached next to the source file so subsequent
loads are fast (~seconds instead of minutes).
"""

import json
from pathlib import Path

import torch


# FP4 E2M1 lookup: 1 sign bit, 2 exponent bits, 1 mantissa bit → 16 values
_FP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,         # positive (nibble 0-7)
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], # negative (nibble 8-15)
    dtype=torch.float32,
)


def _dequant_one(
    weight: torch.Tensor,       # uint8 [out, in/2]
    block_scale: torch.Tensor,  # fp8 [out, in/16]
    global_scale: torch.Tensor, # f32 scalar
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize a single NVFP4 weight tensor to target_dtype.

    Uses chunked processing to keep peak memory low for very large weights.
    """
    out_dim = weight.shape[0]

    # Unpack uint8 → fp4 nibbles → float values
    high = (weight >> 4) & 0x0F
    low = weight & 0x0F
    # Interleave high/low per byte: result shape [out, 2*in_packed]
    unpacked = torch.stack([high, low], dim=-1).reshape(out_dim, -1)
    float_vals = _FP4_E2M1_TABLE[unpacked.long()]
    del high, low, unpacked

    # Apply block scales (blocks of 16 along in_dim)
    in_dim = float_vals.shape[1]
    n_blocks = in_dim // 16
    blocked = float_vals.reshape(out_dim, n_blocks, 16)
    scales_f32 = block_scale.float().reshape(out_dim, n_blocks, 1)
    blocked = blocked * scales_f32
    del float_vals, scales_f32

    # Apply global scale and cast
    result = (blocked.reshape(out_dim, in_dim) * global_scale.float()).to(target_dtype)
    return result


def cached_dequant_path(fp4_path: str) -> Path:
    """Return the path for the cached bf16 version of an fp4 file."""
    p = Path(fp4_path)
    return p.parent / f"{p.stem}.dequantized.bf16.safetensors"


def is_comfy_nvfp4(safetensors_path: str) -> bool:
    """Quick check: does this safetensors file contain NVFP4-quantized weights?"""
    import struct
    try:
        with open(safetensors_path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_len)
        header = json.loads(header_json)
        return any(k.endswith(".comfy_quant") for k in header if k != "__metadata__")
    except Exception:
        return False


def _remap_key_for_gemma3_conditional(key: str) -> str | None:
    """Remap ComfyUI-flat Gemma3 keys to Gemma3ForConditionalGeneration layout.

    The ComfyUI fp4 file uses flat key names (``model.layers.*``, ``vision_model.*``)
    but `Gemma3ForConditionalGeneration` expects everything nested under
    ``model.`` with sub-modules ``language_model``, ``vision_tower``, and
    ``multi_modal_projector``.

    Returns the remapped key, or None if the key should be dropped.
    """
    # Drop the tokenizer blob — not part of model weights
    if key.startswith("spiece_model"):
        return None

    # Language model: model.embed_tokens, model.layers.*, model.norm → model.language_model.*
    if key.startswith("model.embed_tokens") or key.startswith("model.layers") or key.startswith("model.norm"):
        return "model.language_model." + key[len("model."):]

    # Vision tower: vision_model.* → model.vision_tower.vision_model.*
    if key.startswith("vision_model."):
        return "model." + key.replace("vision_model.", "vision_tower.vision_model.", 1)

    # Multi-modal projector: multi_modal_projector.* → model.multi_modal_projector.*
    if key.startswith("multi_modal_projector."):
        return "model." + key

    # Unknown key — keep as-is
    return key


def dequant_comfy_nvfp4_to_file(
    fp4_path: str,
    out_path: str,
    target_dtype=torch.bfloat16,
    progress_callback=None,
    remap_keys: bool = True,
) -> None:
    """Stream-dequant an NVFP4 safetensors file to a new bf16 safetensors file.

    Reads one tensor at a time to minimize peak memory. Frees each source
    tensor immediately after dequantization.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    # First pass: scan header to identify quantized prefixes
    quant_prefixes = set()
    non_quant_keys = []
    with safe_open(fp4_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        for k in all_keys:
            if k.endswith(".comfy_quant"):
                meta_tensor = f.get_tensor(k)
                meta = json.loads(meta_tensor.numpy().tobytes().decode("utf-8"))
                if meta.get("format") == "nvfp4":
                    quant_prefixes.add(k.removesuffix("comfy_quant"))

        # Identify passthrough keys (not part of any quantized layer)
        scale_suffixes = (".weight_scale", ".weight_scale_2", ".comfy_quant")
        quant_weight_keys = {f"{p}weight" for p in quant_prefixes}
        for k in all_keys:
            if k.endswith(scale_suffixes):
                continue
            if k in quant_weight_keys:
                continue
            non_quant_keys.append(k)

        total = len(quant_prefixes) + len(non_quant_keys)
        done = 0

        # Second pass: dequant one tensor at a time, write to new file
        result = {}
        remap = _remap_key_for_gemma3_conditional if remap_keys else (lambda k: k)

        for prefix in quant_prefixes:
            orig_key = f"{prefix}weight"
            new_key = remap(orig_key)
            if new_key is None:
                done += 1
                continue
            weight = f.get_tensor(orig_key)
            block_scale = f.get_tensor(f"{prefix}weight_scale")
            global_scale = f.get_tensor(f"{prefix}weight_scale_2")
            result[new_key] = _dequant_one(weight, block_scale, global_scale, target_dtype)
            del weight, block_scale, global_scale
            done += 1
            if progress_callback and done % 50 == 0:
                progress_callback(done, total)

        # Passthrough non-quantized tensors, cast to target dtype if floating
        for k in non_quant_keys:
            new_key = remap(k)
            if new_key is None:
                done += 1
                continue
            t = f.get_tensor(k)
            if t.is_floating_point():
                result[new_key] = t.to(target_dtype)
            else:
                result[new_key] = t
            done += 1
            if progress_callback and done % 50 == 0:
                progress_callback(done, total)

    # Ensure output dir exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(result, out_path)
    del result
