"""Pipeline loading — format detection, component assembly, and strategy dispatch."""

import json
import os
from pathlib import Path

from modl_worker.protocol import EventEmitter
from modl_worker.device import get_inference_dtype, is_mps
from modl_worker.adapters.arch_config import (
    resolve_model_path,
    resolve_pipeline_class,
    resolve_gen_assembly,
    resolve_gen_components,
    ARCH_CONFIGS,
    detect_arch,
)

# Bundled config files directory
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


# ---------------------------------------------------------------------------
# Pipeline resolution (delegates to arch_config)
# ---------------------------------------------------------------------------


def _resolve_pipeline_class(base_model_id: str) -> str:
    """Determine diffusers pipeline class from base model id."""
    return resolve_pipeline_class(base_model_id)


def _get_pipeline(cls_name: str):
    """Import and return the pipeline class from diffusers."""
    import diffusers

    return getattr(diffusers, cls_name)


# ---------------------------------------------------------------------------
# Model format detection
# ---------------------------------------------------------------------------


def detect_model_format(model_source: str) -> str:
    """Detect the format of a model source path.

    Returns one of:
        "hf_directory"      — directory with model_index.json
        "full_checkpoint"   — single safetensors with UNet+VAE+TE keys
        "gguf"              — GGUF quantized model
        "transformer_only"  — safetensors with only transformer keys
        "hf_repo"           — HuggingFace repo identifier
    """
    import struct

    # Directory with model_index.json = HF pretrained layout
    if os.path.isdir(model_source):
        if os.path.exists(os.path.join(model_source, "model_index.json")):
            return "hf_directory"
        return "hf_directory"  # assume any dir is HF layout

    # GGUF file
    if model_source.endswith(".gguf"):
        return "gguf"

    # Safetensors — peek at header to detect full checkpoint vs transformer-only
    if model_source.endswith(".safetensors") and os.path.exists(model_source):
        try:
            with open(model_source, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                # Read header JSON (cap at 4MB to avoid reading weight data)
                header_bytes = f.read(min(header_size, 4 * 1024 * 1024))
                import json as _json
                header = _json.loads(header_bytes)

            keys = set(header.keys()) - {"__metadata__"}
            key_prefixes = {k.split(".")[0] for k in keys}

            # Full checkpoint = has VAE/TE alongside the diffusion model.
            # "model.diffusion_model.*" alone is just ComfyUI-format
            # transformer weights (e.g. Qwen-Image fp8), NOT a full ckpt.
            has_vae_or_te = key_prefixes & {
                "conditioner", "first_stage_model", "cond_stage_model",
            }
            if has_vae_or_te:
                return "full_checkpoint"

            # If we got here with safetensors keys, it's transformer-only
            return "transformer_only"
        except Exception:
            return "transformer_only"  # assume transformer-only if header read fails

    # Not a local file — treat as HF repo identifier
    return "hf_repo"


# ---------------------------------------------------------------------------
# Pipeline assembly from local components
# ---------------------------------------------------------------------------


def _import_class(class_name: str):
    """Import a class from diffusers or transformers."""
    import importlib
    for mod in ["diffusers", "transformers"]:
        try:
            module = importlib.import_module(mod)
            cls = getattr(module, class_name, None)
            if cls is not None:
                return cls
        except ImportError:
            continue
    raise ImportError(f"Cannot find class {class_name} in diffusers or transformers")


def _materialize_meta_tensors(model):
    """Replace any remaining meta tensors with zeros on CPU.

    After init_empty_weights() + load_state_dict(assign=True), some
    buffers (e.g. position_ids) may remain on the meta device because
    they weren't in the state dict.  This materializes them so the
    model can be moved to a real device.
    """
    import torch

    for module in model.modules():
        for name in list(module._parameters.keys()):
            p = module._parameters[name]
            if p is not None and p.device == torch.device("meta"):
                module._parameters[name] = torch.nn.Parameter(
                    torch.zeros(p.shape, dtype=p.dtype, device="cpu"),
                    requires_grad=p.requires_grad,
                )
        for name in list(module._buffers.keys()):
            b = module._buffers[name]
            if b is not None and b.device == torch.device("meta"):
                module._buffers[name] = torch.zeros(
                    b.shape, dtype=b.dtype, device="cpu"
                )


def _prepare_fp8_model(model):
    """Prepare an fp8 model for inference with enable_model_cpu_offload.

    1. Cast all non-Linear-weight parameters (norms, biases, embeddings)
       to bf16.  These are small and PyTorch can't auto-promote fp8.
    2. Add PyTorch hooks to each nn.Linear to cast the fp8 weight to
       bf16 before forward and restore after.  This keeps the model at
       ~12GB (fp8 weights) on GPU while computing in bf16.
    """
    import torch

    # ── Step 1: cast small params to bf16 ──────────────────────────────
    # Everything except nn.Linear .weight stays as bf16.
    cast_count = 0
    for module in model.modules():
        is_linear = isinstance(module, torch.nn.Linear)
        for name in list(module._parameters.keys()):
            p = module._parameters[name]
            if p is None or p.dtype != torch.float8_e4m3fn:
                continue
            # Keep Linear weights as fp8 (handled by hooks below)
            if is_linear and name == "weight":
                continue
            module._parameters[name] = torch.nn.Parameter(
                p.to(torch.bfloat16), requires_grad=False,
            )
            cast_count += 1
        for name in list(module._buffers.keys()):
            b = module._buffers[name]
            if b is not None and b.dtype == torch.float8_e4m3fn:
                module._buffers[name] = b.to(torch.bfloat16)
                cast_count += 1

    # ── Step 2: per-layer fp8→bf16 hooks on Linear weights ─────────────
    def _pre_hook(module, args):
        if module.weight.dtype == torch.float8_e4m3fn:
            module._fp8_orig_weight = module.weight.data
            module.weight.data = module.weight.data.to(torch.bfloat16)

    def _post_hook(module, args, output):
        if hasattr(module, "_fp8_orig_weight"):
            module.weight.data = module._fp8_orig_weight
            del module._fp8_orig_weight
        return output

    hook_count = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.register_forward_pre_hook(_pre_hook)
            m.register_forward_hook(_post_hook)
            hook_count += 1

    return hook_count, cast_count




def _strip_comfy_prefix(checkpoint, **kwargs):
    """Strip 'model.diffusion_model.' prefix from ComfyUI checkpoint keys.

    Generic fallback for models whose ComfyUI key names match diffusers
    exactly after removing the prefix (e.g. QwenImage).
    """
    PREFIX = "model.diffusion_model."
    return {
        (k[len(PREFIX):] if k.startswith(PREFIX) else k): v
        for k, v in checkpoint.items()
    }


def _get_checkpoint_converter(model_class_name: str):
    """Get the right ComfyUI→diffusers key converter for a transformer class.

    Models with complex key remapping (Flux, Chroma, Z-Image) have dedicated
    converters in diffusers.  Models whose ComfyUI keys match diffusers after
    prefix stripping (QwenImage) use the generic ``_strip_comfy_prefix``.
    """
    _CONVERTER_MAP = {
        "FluxTransformer2DModel": "convert_flux_transformer_checkpoint_to_diffusers",
        "Flux2Transformer2DModel": "convert_flux2_transformer_checkpoint_to_diffusers",
        "ChromaTransformer2DModel": "convert_chroma_transformer_checkpoint_to_diffusers",
        "ZImageTransformer2DModel": "convert_z_image_transformer_checkpoint_to_diffusers",
    }
    fn_name = _CONVERTER_MAP.get(model_class_name)
    if fn_name is not None:
        import importlib
        sfu = importlib.import_module("diffusers.loaders.single_file_utils")
        return getattr(sfu, fn_name)
    # Fallback: generic prefix strip
    return _strip_comfy_prefix


def _detect_weight_dtype(filepath: str) -> str:
    """Detect the dominant weight dtype from a safetensors file header.

    Returns a human-readable string like "fp8_e4m3fn", "bf16", "fp16", "fp32".
    """
    import struct

    try:
        with open(filepath, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(min(header_size, 4 * 1024 * 1024))
            header = json.loads(header_bytes)

        # Weight by total bytes, not tensor count. fp8 files have many
        # small F32 scale/bias tensors but the large weight tensors are F8.
        import math
        dtype_bytes: dict[str, int] = {}
        dtype_sizes = {"F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "I16": 2, "I32": 4, "I64": 8}
        for key, info in header.items():
            if key == "__metadata__":
                continue
            dt = info.get("dtype", "")
            shape = info.get("shape", [])
            numel = math.prod(shape) if shape else 0
            nbytes = numel * dtype_sizes.get(dt, 4)
            dtype_bytes[dt] = dtype_bytes.get(dt, 0) + nbytes

        if not dtype_bytes:
            return "unknown"

        dominant = max(dtype_bytes, key=dtype_bytes.get)
        # Map safetensors dtype names to readable names
        dtype_map = {
            "F8_E4M3": "fp8_e4m3fn",
            "BF16": "bf16",
            "F16": "fp16",
            "F32": "fp32",
            "F64": "fp64",
            "I8": "int8",
            "I16": "int16",
            "I32": "int32",
        }
        return dtype_map.get(dominant, dominant.lower())
    except Exception:
        return "unknown"


def assemble_pipeline(
    base_model_id: str,
    base_model_path: str,
    cls_name: str,
    emitter: EventEmitter,
    force_fp8: bool = False,
    no_offload: bool = False,
):
    """Assemble a pipeline from locally installed components.

    Uses bundled config files + model weights from the modl store.
    This is the core of the strategy pattern — each component is loaded
    individually and then composed into a pipeline.

    Uses accelerate's init_empty_weights() to avoid allocating full fp32
    models in RAM before loading the actual weights (which may be fp8/bf16).
    """
    import torch
    import safetensors.torch
    from accelerate import init_empty_weights

    assembly = resolve_gen_assembly(base_model_id)
    if not assembly:
        raise RuntimeError(
            f"No assembly spec for {base_model_id}. "
            f"Cannot load transformer-only file without component configs."
        )

    PipelineClass = _get_pipeline(cls_name)
    components = {}
    # Track loaded model files for metadata/visibility
    loaded_files: dict[str, dict] = {}

    for param_name, spec in assembly.items():
        model_class_name = spec["model_class"]
        config_dir = CONFIGS_DIR / spec["config_dir"]
        resolved_path = spec.get("resolved_path")
        ModelClass = _import_class(model_class_name)

        if param_name == "transformer":
            # Transformer weights come from base_model_path.
            # from_single_file handles ComfyUI→diffusers key conversion.
            is_gguf = base_model_path.endswith(".gguf")
            weight_dtype = "gguf" if is_gguf else _detect_weight_dtype(base_model_path)
            is_fp8 = weight_dtype.startswith("fp8")
            filename = Path(base_model_path).name
            emitter.info(
                f"Loading transformer: {filename} (weights={weight_dtype})"
            )
            loaded_files["transformer"] = {
                "file": filename,
                "path": base_model_path,
                "weight_dtype": weight_dtype,
                "class": model_class_name,
            }

            if is_gguf:
                # GGUF files: load directly via from_single_file with
                # GGUFQuantizationConfig. Weights stay quantized on GPU.
                from diffusers import GGUFQuantizationConfig
                dtype = get_inference_dtype()
                quantization_config = GGUFQuantizationConfig(
                    compute_dtype=dtype,
                )
                model = ModelClass.from_single_file(
                    base_model_path,
                    config=str(config_dir),
                    quantization_config=quantization_config,
                    torch_dtype=dtype,
                )
                emitter.info(f"  → GGUF quantized model loaded")
            else:
                # Safetensors: load transformer weights.
                checkpoint = safetensors.torch.load_file(base_model_path)

                # Detect ComfyUI fp8 format (has weight_scale tensors).
                weight_scale_keys = [
                    k for k in checkpoint if k.endswith(".weight_scale")
                ]
                has_comfy_fp8_scales = len(weight_scale_keys) > 0

                # For small models (≤5B params / ≤8GB fp8), dequantize to inference dtype
                # for maximum quality. For larger models, keep fp8 and use
                # layerwise casting to avoid OOM during loading.
                # MPS does not support float8 — always dequantize.
                file_size_gb = Path(base_model_path).stat().st_size / (1024**3)
                use_fp8_inference = is_fp8 and file_size_gb > 8.0 and not is_mps()

                dequant_count = 0
                if has_comfy_fp8_scales:
                    for sk in weight_scale_keys:
                        wk = sk.removesuffix("_scale")
                        if wk in checkpoint and checkpoint[wk].dtype == torch.float8_e4m3fn:
                            actual = checkpoint[wk].float() * checkpoint[sk].float()
                            if use_fp8_inference:
                                # Re-quantize: apply scale → cast back to fp8.
                                # Peak memory: one tensor at a time (not whole model).
                                checkpoint[wk] = actual.to(torch.float8_e4m3fn)
                            else:
                                # Small model / MPS: dequant to inference dtype.
                                checkpoint[wk] = actual.to(get_inference_dtype())
                            dequant_count += 1

                # Strip all scale/quant tensors (not needed after dequant,
                # and from_single_file doesn't expect them).
                scale_keys = [
                    k for k in checkpoint
                    if k.endswith("_scale") or k.endswith("_scale_2")
                    or k.endswith(".input_scale")
                ]
                for k in scale_keys:
                    del checkpoint[k]

                if dequant_count:
                    mode = "fp8 (re-quantized)" if use_fp8_inference else "bf16"
                    emitter.info(
                        f"  → Dequantized {dequant_count} fp8 weight tensors → {mode}"
                    )

                if use_fp8_inference:
                    # fp8 inference: create model from config with empty weights,
                    # convert checkpoint keys, load fp8 state dict directly.
                    # Layerwise casting (fp8 storage, bf16 compute) is applied
                    # AFTER enable_model_cpu_offload so hooks fire in the
                    # right order: move-to-GPU → cast-fp8→bf16 → forward.
                    convert_fn = _get_checkpoint_converter(model_class_name)
                    config_dict = ModelClass.load_config(str(config_dir))
                    with init_empty_weights():
                        model = ModelClass.from_config(config_dict)
                    converted = convert_fn(checkpoint)
                    del checkpoint  # free memory
                    model.load_state_dict(converted, strict=False, assign=True)
                    del converted
                    _materialize_meta_tensors(model)
                    # Cast small params (norms, biases) to bf16 and add
                    # per-layer hooks for Linear weights (fp8→bf16 on
                    # forward, restore after).
                    n_hooked, n_cast = _prepare_fp8_model(model)
                    emitter.info(
                        f"  → fp8 model prepared: {n_hooked} linear layers "
                        f"hooked, {n_cast} small params cast to bf16"
                    )
                elif has_comfy_fp8_scales:
                    # Small fp8 model dequantized to bf16: the in-memory
                    # checkpoint is already clean bf16. We can't call
                    # from_single_file (it would re-read the raw fp8 file
                    # and choke on scale tensors). Use from_config +
                    # load_state_dict with the dequantized checkpoint.
                    convert_fn = _get_checkpoint_converter(model_class_name)
                    config_dict = ModelClass.load_config(str(config_dir))
                    with init_empty_weights():
                        model = ModelClass.from_config(config_dict)
                    converted = convert_fn(checkpoint)
                    del checkpoint
                    model.load_state_dict(converted, strict=False, assign=True)
                    del converted
                    _materialize_meta_tensors(model)
                    dtype = get_inference_dtype()
                    model = model.to(dtype)
                    emitter.info(
                        f"  → Loaded dequantized fp8 → {dtype} via from_config"
                    )
                else:
                    # Clean bf16/fp16 safetensors — convert keys and load
                    # via from_config (avoids diffusers from_single_file
                    # bugs where the checkpoint_mapping_fn is identity).
                    convert_fn = _get_checkpoint_converter(model_class_name)
                    config_dict = ModelClass.load_config(str(config_dir))
                    with init_empty_weights():
                        model = ModelClass.from_config(config_dict)
                    converted = convert_fn(checkpoint)
                    del checkpoint
                    model.load_state_dict(converted, strict=False, assign=True)
                    del converted
                    _materialize_meta_tensors(model)
                    dtype = get_inference_dtype()
                    model = model.to(dtype)
                    emitter.info(
                        f"  → Loaded via from_config + convert ({dtype})"
                    )

            # Apply fp8 layerwise casting to reduce VRAM: weights stored in
            # fp8, compute in bf16. Auto for large models (>15GB), or forced
            # when ControlNet is active (need room for CN weights on GPU).
            # MPS does not support float8 — skip layerwise casting.
            if not is_fp8 and not is_gguf and not is_mps():
                file_size_gb = Path(base_model_path).stat().st_size / (1024**3)
                if file_size_gb > 15.0 or (force_fp8 and file_size_gb > 5.0):
                    model.enable_layerwise_casting(
                        storage_dtype=torch.float8_e4m3fn,
                        compute_dtype=torch.bfloat16,
                    )
                    emitter.info(
                        f"  → Enabled fp8 layerwise casting ({file_size_gb:.1f}GB bf16 → ~{file_size_gb/2:.1f}GB fp8)"
                    )

            components["transformer"] = model

        elif param_name in ("scheduler",):
            components[param_name] = ModelClass.from_pretrained(str(config_dir))

        elif param_name.startswith("tokenizer") or param_name == "processor":
            components[param_name] = ModelClass.from_pretrained(str(config_dir))

        elif resolved_path:
            weight_dtype = _detect_weight_dtype(resolved_path)
            filename = Path(resolved_path).name
            emitter.info(
                f"Loading {param_name}: {filename} (weights={weight_dtype})"
            )
            loaded_files[param_name] = {
                "file": filename,
                "path": resolved_path,
                "weight_dtype": weight_dtype,
                "class": model_class_name,
            }

            # Check if this component needs NF4 quantization and/or
            # HF directory-style loading (e.g. Flux2's Mistral3 text encoder)
            quantize_nf4 = spec.get("quantize_nf4", False)
            use_hf_dir = spec.get("hf_dir", False) or os.path.isdir(resolved_path)

            if use_hf_dir and not os.path.isdir(resolved_path):
                # Single safetensors file but component needs from_pretrained.
                # Create a synthetic HF directory with config + weights symlink.
                hf_dir = Path(resolved_path).parent / "hf_layout"
                hf_dir.mkdir(exist_ok=True)
                link = hf_dir / "model.safetensors"
                if not link.exists():
                    link.symlink_to(resolved_path)
                # Copy config files into the HF directory
                import shutil
                for cfg_file in config_dir.iterdir():
                    dst = hf_dir / cfg_file.name
                    if not dst.exists():
                        shutil.copy2(str(cfg_file), str(dst))
                resolved_path = str(hf_dir)
                use_hf_dir = True

            if hasattr(ModelClass, "from_single_file") and not use_hf_dir:
                # Diffusers models (VAE, etc.) — from_single_file handles
                # ComfyUI→diffusers key conversion.  Some newer model
                # classes (e.g. AutoencoderKLFlux2) have from_single_file
                # but aren't in the allowlist yet — fall back to manual
                # loading if the call fails.
                try:
                    components[param_name] = ModelClass.from_single_file(
                        resolved_path,
                        config=str(config_dir),
                        torch_dtype=get_inference_dtype(),
                    )
                except (ValueError, NotImplementedError):
                    # Fall back: load config → create model → load weights
                    config_dict = ModelClass.load_config(str(config_dir))
                    model = ModelClass.from_config(config_dict)
                    state_dict = safetensors.torch.load_file(resolved_path)
                    # Check key overlap before loading — zero overlap means
                    # the file has non-diffusers keys (e.g. ComfyUI/original format)
                    model_keys = set(model.state_dict().keys())
                    file_keys = set(state_dict.keys())
                    overlap = model_keys & file_keys
                    if not overlap:
                        emitter.info(
                            f"  ⚠ {param_name}: 0/{len(file_keys)} keys match "
                            f"diffusers format — weights NOT loaded. "
                            f"Re-install with `modl pull` to get compatible weights."
                        )
                    model.load_state_dict(state_dict, strict=False)
                    model = model.to(get_inference_dtype())
                    components[param_name] = model
            elif use_hf_dir:
                # HF directory layout — use from_pretrained with optional
                # NF4 quantization (e.g. Flux2's 24B Mistral3 text encoder)
                load_kwargs = {"torch_dtype": get_inference_dtype()}
                if quantize_nf4:
                    try:
                        from transformers import BitsAndBytesConfig
                        load_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        )
                        emitter.info(f"  → NF4 quantization enabled for {param_name}")
                    except ImportError:
                        emitter.info(f"  → bitsandbytes not available, loading in bf16")
                model = ModelClass.from_pretrained(
                    resolved_path, **load_kwargs,
                )
                components[param_name] = model
            else:
                # Transformers models (CLIP, T5) — use empty weights to
                # avoid ~44GB fp32 allocation for large models like T5-XXL
                config_obj = ModelClass.config_class.from_pretrained(str(config_dir))
                with init_empty_weights():
                    model = ModelClass(config_obj)
                state_dict = safetensors.torch.load_file(resolved_path)
                model.load_state_dict(state_dict, strict=False, assign=True)
                _materialize_meta_tensors(model)
                # Always cast text encoders to inference dtype for numerical stability.
                # fp8 text encoders produce unreliable embeddings.
                model = model.to(get_inference_dtype())
                components[param_name] = model
        else:
            emitter.info(f"Skipping {param_name} (no weights found)")

    emitter.info(f"Assembling {cls_name} from {len(components)} components")
    # Pass any extra pipeline constructor kwargs (e.g. is_distilled for Klein)
    arch_name = detect_arch(base_model_id)
    pipeline_kwargs = ARCH_CONFIGS.get(arch_name, {}).get("pipeline_kwargs", {})
    pipe = PipelineClass(**components, **pipeline_kwargs)
    if not no_offload:
        from modl_worker.device import move_pipe_to_device
        move_pipe_to_device(pipe)
    # Attach loaded file info for downstream metadata embedding
    pipe._modl_loaded_files = loaded_files
    return pipe


# ---------------------------------------------------------------------------
# Main pipeline loader (strategy dispatch)
# ---------------------------------------------------------------------------


def load_pipeline(
    base_model_id: str,
    base_model_path: str | None,
    cls_name: str,
    emitter: EventEmitter,
    force_fp8: bool = False,
):
    """Load a diffusers pipeline from disk or HuggingFace.

    This is the single loading path used by both one-shot mode
    (``run_generate()``) and the persistent worker (``ModelCache``).

    Strategy:
        1. HF directory layout  → from_pretrained(dir)
        2. Full checkpoint       → from_single_file(path)
        3. Transformer-only      → assemble from local components
        4. GGUF                  → assemble with GGUFQuantizationConfig
        5. HF repo identifier    → from_pretrained(repo_id)
    """
    import torch

    dtype = get_inference_dtype()
    PipelineClass = _get_pipeline(cls_name)
    model_source = base_model_path or resolve_model_path(base_model_id)
    fmt = detect_model_format(model_source)

    emitter.info(f"Model source: {model_source} (format={fmt})")

    if fmt == "hf_directory":
        pipe = PipelineClass.from_pretrained(
            model_source,
            torch_dtype=dtype,
        )
        emitter.info(f"Loaded from directory: {model_source}")
    elif fmt == "full_checkpoint":
        weight_dtype = _detect_weight_dtype(model_source)
        filename = Path(model_source).name
        emitter.info(f"Loading checkpoint: {filename} (weights={weight_dtype})")
        # full_checkpoint (e.g. SDXL) needs HF Hub access for component config
        # resolution during from_single_file(). Temporarily allow it.
        from huggingface_hub import constants as hf_constants
        was_offline = hf_constants.HF_HUB_OFFLINE
        hf_constants.HF_HUB_OFFLINE = False
        try:
            pipe = PipelineClass.from_single_file(
                model_source,
                torch_dtype=dtype,
            )
        finally:
            hf_constants.HF_HUB_OFFLINE = was_offline
        pipe._modl_loaded_files = {
            "checkpoint": {"file": filename, "path": model_source, "weight_dtype": weight_dtype},
        }
    elif fmt == "transformer_only":
        # Assemble pipeline from locally installed components
        assembly = resolve_gen_assembly(base_model_id)
        if assembly:
            return assemble_pipeline(base_model_id, model_source, cls_name, emitter, force_fp8=force_fp8)
        else:
            # Fallback: try from_pretrained with HF repo
            hf_repo = resolve_model_path(base_model_id)
            emitter.info(f"No assembly spec, falling back to HF: {hf_repo}")
            pipe = PipelineClass.from_pretrained(
                hf_repo,
                torch_dtype=dtype,
            )
    elif fmt == "gguf":
        # GGUF models need assembly with quantization config
        assembly = resolve_gen_assembly(base_model_id)
        if assembly:
            return assemble_pipeline(base_model_id, model_source, cls_name, emitter, force_fp8=force_fp8)
        else:
            raise RuntimeError(
                f"GGUF model {model_source} requires assembly spec in arch_config"
            )
    else:
        # HF repo identifier
        pipe = PipelineClass.from_pretrained(
            model_source,
            torch_dtype=dtype,
        )

    if force_fp8:
        # When fp8 is forced (ControlNet mode), use cpu_offload
        # so that text encoder and transformer don't both occupy GPU
        # simultaneously during inference.
        from modl_worker.device import move_pipe_to_device
        move_pipe_to_device(pipe)
        return pipe
    from modl_worker.device import get_device
    return pipe.to(get_device())
