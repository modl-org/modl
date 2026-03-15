"""Generate adapter — runs diffusers inference and emits events.

Translates a GenerateJobSpec (parsed from YAML) into a diffusers pipeline call.
Pipeline class and default params are resolved via arch_config — the single
source of truth for all model-specific settings.

Outputs are saved as PNG and emitted as artifact events.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from modl_worker.protocol import EventEmitter
from modl_worker.adapters.arch_config import (
    resolve_model_path,
    resolve_pipeline_class,
    resolve_pipeline_class_for_mode,
    resolve_gen_defaults,
    resolve_gen_components,
    resolve_gen_assembly,
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
            # Sample a few keys to determine format
            sample_keys = list(keys)[:50]
            key_str = " ".join(sample_keys)

            # Full checkpoint indicators (SD/SDXL format)
            full_ckpt_prefixes = [
                "conditioner.", "first_stage_model.", "model.diffusion_model.",
                "cond_stage_model.",
            ]
            if any(p in key_str for p in full_ckpt_prefixes):
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
                quantization_config = GGUFQuantizationConfig(
                    compute_dtype=torch.bfloat16,
                )
                model = ModelClass.from_single_file(
                    base_model_path,
                    config=str(config_dir),
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
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

                # For small models (≤5B params / ≤8GB fp8), dequantize to bf16
                # for maximum quality. For larger models, keep fp8 and use
                # layerwise casting to avoid OOM during loading.
                file_size_gb = Path(base_model_path).stat().st_size / (1024**3)
                use_fp8_inference = is_fp8 and file_size_gb > 8.0

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
                                # Small model: keep as bf16 for best quality.
                                checkpoint[wk] = actual.to(torch.bfloat16)
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
                    # Pick the right key converter for the model architecture.
                    if model_class_name == "FluxTransformer2DModel":
                        from diffusers.loaders.single_file_utils import (
                            convert_flux_transformer_checkpoint_to_diffusers,
                        )
                        convert_fn = convert_flux_transformer_checkpoint_to_diffusers
                    else:
                        from diffusers.loaders.single_file_utils import (
                            convert_flux2_transformer_checkpoint_to_diffusers,
                        )
                        convert_fn = convert_flux2_transformer_checkpoint_to_diffusers
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
                    if model_class_name == "FluxTransformer2DModel":
                        from diffusers.loaders.single_file_utils import (
                            convert_flux_transformer_checkpoint_to_diffusers,
                        )
                        convert_fn = convert_flux_transformer_checkpoint_to_diffusers
                    else:
                        from diffusers.loaders.single_file_utils import (
                            convert_flux2_transformer_checkpoint_to_diffusers,
                        )
                        convert_fn = convert_flux2_transformer_checkpoint_to_diffusers
                    config_dict = ModelClass.load_config(str(config_dir))
                    with init_empty_weights():
                        model = ModelClass.from_config(config_dict)
                    converted = convert_fn(checkpoint)
                    del checkpoint
                    model.load_state_dict(converted, strict=False, assign=True)
                    del converted
                    _materialize_meta_tensors(model)
                    model = model.to(torch.bfloat16)
                    emitter.info(
                        f"  → Loaded dequantized fp8 → bf16 via from_config"
                    )
                else:
                    model = ModelClass.from_single_file(
                        base_model_path, config=str(config_dir),
                        torch_dtype=torch.bfloat16,
                    )
                    emitter.info(
                        f"  → Loaded via from_single_file (bf16)"
                    )

            # Apply fp8 layerwise casting to reduce VRAM: weights stored in
            # fp8, compute in bf16. Auto for large models (>15GB), or forced
            # when ControlNet is active (need room for CN weights on GPU).
            if not is_fp8 and not is_gguf:
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
                        torch_dtype=torch.bfloat16,
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
                    model = model.to(torch.bfloat16)
                    components[param_name] = model
            elif use_hf_dir:
                # HF directory layout — use from_pretrained with optional
                # NF4 quantization (e.g. Flux2's 24B Mistral3 text encoder)
                load_kwargs = {"torch_dtype": torch.bfloat16}
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
                # Always cast text encoders to bf16 for numerical stability.
                # fp8 text encoders produce unreliable embeddings.
                model = model.to(torch.bfloat16)
                components[param_name] = model
        else:
            emitter.info(f"Skipping {param_name} (no weights found)")

    emitter.info(f"Assembling {cls_name} from {len(components)} components")
    # Pass any extra pipeline constructor kwargs (e.g. is_distilled for Klein)
    from .arch_config import ARCH_CONFIGS, detect_arch
    arch_name = detect_arch(base_model_id)
    pipeline_kwargs = ARCH_CONFIGS.get(arch_name, {}).get("pipeline_kwargs", {})
    pipe = PipelineClass(**components, **pipeline_kwargs)
    pipe.enable_model_cpu_offload()
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

    PipelineClass = _get_pipeline(cls_name)
    model_source = base_model_path or resolve_model_path(base_model_id)
    fmt = detect_model_format(model_source)

    emitter.info(f"Model source: {model_source} (format={fmt})")

    if fmt == "hf_directory":
        pipe = PipelineClass.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
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
                torch_dtype=torch.bfloat16,
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
                torch_dtype=torch.bfloat16,
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
            torch_dtype=torch.bfloat16,
        )

    if force_fp8:
        # When fp8 is forced (ControlNet mode), use model_cpu_offload
        # instead of .to("cuda") so that text encoder and transformer
        # don't both occupy GPU simultaneously during inference.
        pipe.enable_model_cpu_offload()
        return pipe
    return pipe.to("cuda")


# ---------------------------------------------------------------------------
# Size presets
# ---------------------------------------------------------------------------

SIZE_PRESETS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_generate(config_path: Path, emitter: EventEmitter) -> int:
    """Run image generation from a GenerateJobSpec YAML file (one-shot mode).

    Loads the pipeline from scratch, runs inference, then exits. For
    persistent-worker mode, see ``run_generate_with_pipeline()``.
    """
    import yaml

    if not config_path.exists():
        emitter.error(
            "SPEC_NOT_FOUND",
            f"Generate spec not found: {config_path}",
            recoverable=False,
        )
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    model_info = spec.get("model", {})
    base_model_id = model_info.get("base_model_id", "flux-schnell")
    base_model_path = model_info.get("base_model_path")
    lora_info = spec.get("lora")

    # Detect generation mode for cold-start pipeline selection
    params = spec.get("params", {})
    init_image_path = params.get("init_image")
    mask_path = params.get("mask")
    if mask_path and init_image_path:
        cold_mode = "inpaint"
    elif init_image_path:
        cold_mode = "img2img"
    else:
        cold_mode = "txt2img"

    # -------------------------------------------------------------------
    # 1. Load pipeline (cold start)
    # -------------------------------------------------------------------
    emitter.info(f"Loading pipeline for {base_model_id} (mode={cold_mode})...")
    count = params.get("count", 1)
    emitter.progress(stage="load", step=0, total_steps=count)

    try:
        # For cold start, load the mode-specific pipeline directly.
        # Force fp8 when ControlNet is active — bf16 transformer + controlnet
        # + text encoder exceeds 24GB VRAM.
        cls_name = resolve_pipeline_class_for_mode(base_model_id, cold_mode)
        has_controlnet = bool(params.get("controlnet"))
        pipe = load_pipeline(
            base_model_id, base_model_path, cls_name, emitter,
            force_fp8=has_controlnet,
        )

        # Load LoRA if specified
        if lora_info:
            lora_path = lora_info.get("path")
            lora_weight = lora_info.get("weight", 1.0)
            if lora_path and os.path.exists(lora_path):
                emitter.info(f"Loading LoRA: {lora_info.get('name', 'unnamed')} (weight={lora_weight})")
                from modl_worker.adapters.lora_utils import load_lora_with_conversion
                load_lora_with_conversion(pipe, lora_path, lora_weight, emitter)

        emitter.job_started(config=str(config_path))

    except Exception as exc:
        emitter.error(
            "PIPELINE_LOAD_FAILED",
            f"Failed to load diffusers pipeline: {exc}",
            recoverable=False,
        )
        return 1

    # -------------------------------------------------------------------
    # 2. Delegate to shared inference loop
    # -------------------------------------------------------------------
    return run_generate_with_pipeline(spec, emitter, pipe, cls_name)


# ---------------------------------------------------------------------------
# ControlNet support
# ---------------------------------------------------------------------------

# Control type → mode index for union controlnets
# These indices map to the mode embedding in union ControlNet weights
FLUX_CONTROLNET_MODES = {
    "canny": 0, "tile": 1, "depth": 2, "blur": 3,
    "pose": 4, "gray": 5, "softedge": 5,
}

SDXL_CONTROLNET_MODES = {
    "canny": 0, "tile": 1, "depth": 2, "blur": 3,
    "pose": 4, "gray": 5, "softedge": 5, "normal": 6,
    "scribble": 6, "hed": 6, "mlsd": 7,
}

ZIMAGE_CONTROLNET_MODES = {
    "canny": 0, "hed": 1, "depth": 2, "pose": 3,
    "mlsd": 4, "scribble": 5, "gray": 6,
}

QWEN_IMAGE_CONTROLNET_MODES = {
    "canny": 0, "depth": 1, "pose": 2, "softedge": 3,
}


# ControlNet model paths indexed by (arch_key, manifest_id)
# Maps to HuggingFace repos — resolved from installed store paths at runtime
CONTROLNET_CONFIGS = {
    "flux": {
        "repo": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        "model_class": "FluxControlNetModel",
        "pipeline_class": "FluxControlNetPipeline",
        "modes": FLUX_CONTROLNET_MODES,
    },
    "sdxl": {
        "repo": "xinsir/controlnet-union-sdxl-1.0",
        "model_class": "ControlNetModel",
        "pipeline_class": "StableDiffusionXLControlNetPipeline",
        "modes": SDXL_CONTROLNET_MODES,
    },
    "qwen_image": {
        "repo": "InstantX/Qwen-Image-ControlNet-Union",
        "model_class": "QwenImageControlNetModel",
        "pipeline_class": "QwenImageControlNetPipeline",
        "modes": QWEN_IMAGE_CONTROLNET_MODES,
    },
    "zimage_turbo": {
        "model_class": "ZImageControlNetModel",
        "pipeline_class": "ZImageControlNetPipeline",
        "modes": ZIMAGE_CONTROLNET_MODES,
    },
    "zimage": {
        "model_class": "ZImageControlNetModel",
        "pipeline_class": "ZImageControlNetPipeline",
        "modes": ZIMAGE_CONTROLNET_MODES,
    },
    "flux_schnell": {
        "repo": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        "model_class": "FluxControlNetModel",
        "pipeline_class": "FluxControlNetPipeline",
        "modes": FLUX_CONTROLNET_MODES,
    },
}


def _resolve_controlnet_path(arch: str) -> str | None:
    """Resolve installed ControlNet path from modl store, fallback to HF repo."""
    config = CONTROLNET_CONFIGS.get(arch)
    if not config:
        return None

    # Try to find installed controlnet from modl's SQLite DB
    from .arch_config import _get_installed_path
    # Try multiple manifest IDs per arch (prefer newer versions first)
    manifest_id_candidates = {
        "flux": ["flux-dev-controlnet-union"],
        "flux_schnell": ["flux-dev-controlnet-union"],
        "sdxl": ["sdxl-controlnet-union"],
        "qwen_image": ["qwen-image-controlnet-union"],
        # Prefer v2.1 (better quality, same VRAM). Diffusers 0.38+ has full
        # support — auto-pads control_in_dim from 16→33 in the pipeline.
        # Falls back to v1 (6 control layers, control_in_dim=16) if v2.1 not installed.
        "zimage_turbo": ["z-image-turbo-controlnet-union-2.1", "z-image-turbo-controlnet-union"],
        "zimage": ["z-image-turbo-controlnet-union-2.1", "z-image-turbo-controlnet-union"],
    }
    candidates = manifest_id_candidates.get(arch, [])
    for manifest_id in candidates:
        path = _get_installed_path(manifest_id)
        if path:
            return path

    # Fallback to HuggingFace repo if configured (will download)
    return config.get("repo")


def _resolve_controlnet_config(arch: str, model_path: str) -> str | None:
    """Return bundled config path for controlnets that need it.

    Z-Image: diffusers misdetects the lite v2.1 variant as v1.0.
    Qwen-Image: from_single_file needs a config to avoid HF downloads.
    """
    configs_dir = Path(__file__).parent.parent / "configs"

    if arch == "qwen_image":
        return str(configs_dir / "qwen-image-controlnet")

    if arch in ("flux", "flux_schnell"):
        return str(configs_dir / "flux-controlnet-union")

    if arch not in ("zimage_turbo", "zimage"):
        return None

    import safetensors.torch

    configs_dir = Path(__file__).parent.parent / "configs"
    # Peek at the embedder shape to determine control_in_dim
    try:
        metadata = safetensors.torch.load_file(model_path, device="cpu")
        emb_key = "control_all_x_embedder.2-1.weight"
        if emb_key not in metadata:
            return None
        emb_shape = metadata[emb_key].shape
        del metadata  # free memory
        # patch_size=2: embed_dim = 2*2*control_in_dim
        control_in_dim = emb_shape[1] // 4
    except Exception:
        return None

    if control_in_dim == 16:
        return str(configs_dir / "zimage-controlnet-v1")
    elif control_in_dim == 33:
        # Distinguish full (15 layers) from lite (3 layers) by file size.
        # Full v2.1 ~6.7GB, lite ~2GB.
        file_size_gb = Path(model_path).stat().st_size / (1024**3)
        if file_size_gb < 4.0:
            return str(configs_dir / "zimage-controlnet-v21-lite")
        else:
            return str(configs_dir / "zimage-controlnet-v21")
    return None


def _load_controlnet(
    cn_inputs: list[dict],
    base_model_id: str,
    arch: str,
    pipeline: object,
    emitter: EventEmitter,
) -> tuple:
    """Load ControlNet model and prepare control images.

    Returns (cn_pipeline, control_images, scales, end_values).
    cn_pipeline is a new pipeline with controlnet attached, or None on failure.
    """
    import torch

    config = CONTROLNET_CONFIGS.get(arch)
    if not config:
        emitter.warning(
            "CONTROLNET_NOT_SUPPORTED",
            f"ControlNet not supported for architecture '{arch}'. Generating without ControlNet.",
        )
        return None, [], [], []

    # Load ControlNet model
    model_path = _resolve_controlnet_path(arch)
    if not model_path:
        emitter.warning(
            "CONTROLNET_NOT_FOUND",
            f"ControlNet weights not found for {arch}. Generating without ControlNet.",
        )
        return None, [], [], []

    emitter.info(f"Loading ControlNet: {Path(model_path).name}")

    import diffusers
    cn_cls = getattr(diffusers, config["model_class"])

    # Load ControlNet in bf16 — these are lightweight (control blocks only,
    # ~2-3GB). The heavy shared modules (embedders, refiners) come from the
    # base transformer via from_transformer() inside the pipeline constructor.
    dtype = torch.bfloat16

    try:
        if model_path.endswith(".safetensors"):
            cn_config = _resolve_controlnet_config(arch, model_path)
            try:
                if cn_config:
                    cn_model = cn_cls.from_single_file(
                        model_path, config=cn_config, torch_dtype=dtype,
                    )
                else:
                    cn_model = cn_cls.from_single_file(model_path, torch_dtype=dtype)
            except (ValueError, NotImplementedError, AttributeError):
                # from_single_file not supported or missing (e.g. FluxControlNetModel,
                # QwenImageControlNetModel). Fall back to from_config + load_state_dict.
                if not cn_config:
                    raise
                import safetensors.torch as sf
                config_dict = cn_cls.load_config(cn_config)
                cn_model = cn_cls.from_config(config_dict)
                state_dict = sf.load_file(model_path)
                cn_model.load_state_dict(state_dict, strict=False)
                cn_model = cn_model.to(dtype)
        else:
            cn_model = cn_cls.from_pretrained(model_path, torch_dtype=dtype)
    except Exception as exc:
        emitter.error(
            "CONTROLNET_LOAD_FAILED",
            f"Failed to load ControlNet weights: {exc}",
            recoverable=True,
        )
        return None, [], [], []

    # Remove accelerate cpu_offload hooks from the base pipeline so the
    # new CN pipeline can set up its own offload sequence cleanly.
    from accelerate.hooks import remove_hook_from_module
    for name in ["text_encoder", "transformer", "vae"]:
        mod = getattr(pipeline, name, None)
        if mod is not None:
            remove_hook_from_module(mod, recurse=True)

    # Check if everything fits on GPU. With force_fp8, the transformer is
    # ~5.7GB. Lite controlnet adds ~2GB → fits. Full controlnet (6GB) needs
    # the text encoder offloaded during denoising → use wrapper approach.
    cn_model_size_gb = sum(
        p.numel() * p.element_size() for p in cn_model.parameters()
    ) / (1024**3)
    use_wrapper = cn_model_size_gb > 4.0  # Full v2.1 is ~6GB

    if use_wrapper and arch in ("zimage_turbo", "zimage"):
        # Single-model wrapper: combines transformer + controlnet into one
        # nn.Module so model_cpu_offload can offload the text encoder during
        # denoising. The wrapper's forward runs controlnet → transformer.
        from .z_image_control import ZImageControlWrapper

        wrapper = ZImageControlWrapper(pipeline.transformer, cn_model)
        # Use the base ZImagePipeline (not ControlNet variant) — the wrapper
        # handles controlnet internally, the pipeline just calls transformer.
        from diffusers import ZImagePipeline
        cn_pipe = ZImagePipeline(
            transformer=wrapper,
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            scheduler=pipeline.scheduler,
        )
        cn_pipe.enable_model_cpu_offload()
        emitter.info(f"ControlNet loaded ({cn_model_size_gb:.1f}GB, CPU offload via wrapper)")
    else:
        # Standard approach: construct the ControlNet pipeline directly.
        # Forward all components from the base pipeline (some pipelines like
        # FluxControlNetPipeline need text_encoder_2, tokenizer_2, etc.)
        cn_pipe_cls = getattr(diffusers, config["pipeline_class"])
        import inspect
        cn_init_params = set(inspect.signature(cn_pipe_cls.__init__).parameters.keys())
        cn_init_params.discard("self")

        cn_kwargs = {"controlnet": cn_model}
        for param_name in cn_init_params:
            if param_name == "controlnet":
                continue
            component = getattr(pipeline, param_name, None)
            if component is not None:
                cn_kwargs[param_name] = component

        cn_pipe = cn_pipe_cls(**cn_kwargs)

        # Check if the base pipeline was using model_cpu_offload (large
        # models like qwen-image). If so, use cpu_offload on the CN
        # pipeline too and pin the controlnet on CUDA (it's independent,
        # no shared modules — safe to keep resident).
        base_was_offloaded = bool(getattr(pipeline, "_all_hooks", None))
        if base_was_offloaded:
            cn_pipe.enable_model_cpu_offload()
            cn_pipe.controlnet.to("cuda")
            emitter.info(f"ControlNet loaded ({cn_model_size_gb:.1f}GB, CPU offload)")
        else:
            cn_pipe.to("cuda")
            emitter.info(f"ControlNet loaded ({cn_model_size_gb:.1f}GB)")

    # Load control images
    from modl_worker.image_util import load_image as _load_img

    control_images = []
    scales = []
    end_values = []
    for inp in cn_inputs:
        img = _load_img(inp["image"])
        control_images.append(img)
        scales.append(inp.get("strength", 0.75))
        end_values.append(inp.get("control_end", 0.8))

    return cn_pipe, control_images, scales, end_values


def _resolve_control_modes(cn_types: list[str], arch: str) -> list[int] | None:
    """Map control type names to mode indices for union controlnets."""
    config = CONTROLNET_CONFIGS.get(arch)
    if not config or "modes" not in config:
        return None

    modes = config["modes"]
    result = []
    for ct in cn_types:
        mode_idx = modes.get(ct)
        if mode_idx is not None:
            result.append(mode_idx)
        else:
            # Unknown type — return None to skip mode setting
            return None
    return result


# ---------------------------------------------------------------------------
# Style reference / IP-Adapter support
# ---------------------------------------------------------------------------

STYLE_REF_CONFIGS = {
    "sdxl": {
        "mechanism": "ip-adapter",
        "repo": "h94/IP-Adapter",
        "subfolder": "sdxl_models",
        "weight_name": "ip-adapter-plus_sdxl_vit-h.safetensors",
        "image_encoder_repo": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    },
    "flux": {
        "mechanism": "ip-adapter",
        "repo": "InstantX/FLUX.1-dev-IP-Adapter",
        "weight_name": "ip-adapter.bin",
        "image_encoder_repo": "google/siglip-so400m-patch14-384",
    },
    "flux2_klein": {
        "mechanism": "native",
    },
    "flux2_klein_9b": {
        "mechanism": "native",
    },
}


def _load_style_ref(
    style_inputs: list[dict],
    base_model_id: str,
    arch: str,
    pipe: object,
    emitter: EventEmitter,
) -> tuple:
    """Load style reference mechanism and prepare images.

    Returns (style_images, strength, mechanism).
    """
    from PIL import Image
    from modl_worker.image_util import load_image as _load_img

    config = STYLE_REF_CONFIGS.get(arch)
    if not config:
        emitter.warning(
            "STYLE_REF_NOT_SUPPORTED",
            f"Style reference not supported for architecture '{arch}'. Generating without style-ref.",
        )
        return [], 0.0, None

    mechanism = config["mechanism"]
    images = [_load_img(inp["image"]) for inp in style_inputs]
    strength = style_inputs[0].get("strength", 0.6) if style_inputs else 0.6

    if mechanism == "ip-adapter":
        # Load IP-Adapter into the pipeline
        emitter.info(f"Loading IP-Adapter for {arch}...")
        try:
            pipe.load_ip_adapter(
                config["repo"],
                subfolder=config.get("subfolder"),
                weight_name=config["weight_name"],
            )
            pipe.set_ip_adapter_scale(strength)
            emitter.info("IP-Adapter loaded")
        except Exception as exc:
            emitter.warning(
                "IP_ADAPTER_LOAD_FAILED",
                f"Failed to load IP-Adapter: {exc}. Generating without style-ref.",
            )
            return [], 0.0, None

    return images, strength, mechanism


def run_generate_with_pipeline(
    spec: dict,
    emitter: EventEmitter,
    pipeline: object,
    cls_name: str,
) -> int:
    """Run image generation using an already-loaded pipeline.

    This is the shared inference loop used by both one-shot mode
    (``run_generate()``) and persistent-worker mode (``serve.py``).
    The caller is responsible for loading / caching the pipeline and
    handling LoRA reconciliation.

    Args:
        spec: Parsed GenerateJobSpec dict (prompt, model, params, output, etc.)
        emitter: EventEmitter to write JSONL events (stdout or socket)
        pipeline: A loaded diffusers pipeline object (already on CUDA)
        cls_name: Pipeline class name (e.g. "FluxPipeline")

    Returns:
        Exit code (0 = success, 1 = all images failed)
    """
    import torch
    from PIL import Image

    from modl_worker.image_util import load_image

    prompt = spec.get("prompt", "")
    model_info = spec.get("model", {})
    lora_info = spec.get("lora")
    output_info = spec.get("output", {})
    params = spec.get("params", {})

    base_model_id = model_info.get("base_model_id", "flux-schnell")

    # Use arch-aware defaults from ARCH_CONFIGS when user didn't specify
    gen_defaults = resolve_gen_defaults(base_model_id)
    width = params.get("width", 1024)
    height = params.get("height", 1024)
    steps = params.get("steps", gen_defaults["steps"])
    guidance = params.get("guidance", gen_defaults["guidance"])
    seed = params.get("seed")
    count = params.get("count", 1)

    # Img2img / inpainting params
    init_image_path = params.get("init_image")
    mask_path = params.get("mask")
    strength = params.get("strength", 0.75)

    # Determine generation mode
    if mask_path and init_image_path:
        mode = "inpaint"
    elif init_image_path:
        mode = "img2img"
    else:
        mode = "txt2img"

    # Load init image and mask if needed
    init_img = None
    mask_img = None
    if init_image_path:
        init_img = load_image(init_image_path)
    if mask_path:
        mask_img = load_image(mask_path)

    # Detect architecture early (needed for ControlNet/style-ref loading)
    from .arch_config import detect_arch
    arch = detect_arch(base_model_id)

    # ControlNet params
    cn_inputs = params.get("controlnet", [])

    # Switch pipeline if needed for img2img/inpaint via from_pipe()
    pipe = pipeline
    if mode != "txt2img":
        target_cls_name = resolve_pipeline_class_for_mode(base_model_id, mode)
        if target_cls_name != cls_name:
            emitter.info(f"Switching pipeline: {cls_name} -> {target_cls_name} (mode={mode})")
            TargetClass = _get_pipeline(target_cls_name)
            pipe = TargetClass.from_pipe(pipeline)
            cls_name = target_cls_name

    # Load ControlNet if requested
    cn_pipe = None
    cn_images = []
    cn_scales = []
    cn_end_values = []
    if cn_inputs:
        cn_pipe, cn_images, cn_scales, cn_end_values = _load_controlnet(
            cn_inputs, base_model_id, arch, pipe, emitter
        )
        if cn_pipe is not None:
            pipe = cn_pipe  # Use the ControlNet-wrapped pipeline

    # Load style reference if requested
    style_inputs = params.get("style_ref", [])
    style_images = []
    style_strength = 0.6
    style_mechanism = None
    if style_inputs:
        style_images, style_strength, style_mechanism = _load_style_ref(
            style_inputs, base_model_id, arch, pipe, emitter
        )

    output_dir = output_info.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    generator = torch.Generator(device="cuda")
    if seed is not None:
        generator.manual_seed(seed)

    # Build inference kwargs — different pipelines accept different params
    gen_kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "generator": generator,
    }

    # QwenImagePipeline/QwenImageEditPlusPipeline use true_cfg_scale (not guidance_scale).
    # negative_prompt=" " (space) is required to enable true CFG — without it quality degrades.
    if arch in ("qwen_image", "qwen_image_edit"):
        gen_kwargs["true_cfg_scale"] = guidance
        gen_kwargs["negative_prompt"] = " "
    else:
        gen_kwargs["guidance_scale"] = guidance

    is_flux_fill = arch in ("flux_fill", "flux_fill_onereward")

    if mode == "txt2img":
        gen_kwargs["width"] = width
        gen_kwargs["height"] = height
    elif mode == "img2img":
        gen_kwargs["image"] = init_img
        gen_kwargs["strength"] = strength
    elif mode == "inpaint":
        gen_kwargs["image"] = init_img
        gen_kwargs["mask_image"] = mask_img
        gen_kwargs["width"] = width
        gen_kwargs["height"] = height
        if not is_flux_fill:
            gen_kwargs["strength"] = strength  # Fill pipelines don't use strength

    # Add ControlNet conditioning
    _cn_wrapper = None
    if cn_pipe is not None and cn_images:
        from .z_image_control import ZImageControlWrapper

        # Check if the pipeline uses the wrapper approach (base ZImagePipeline
        # with ZImageControlWrapper as transformer) vs the standard approach
        # (ZImageControlNetPipeline with separate controlnet).
        if isinstance(getattr(pipe, "transformer", None), ZImageControlWrapper):
            # Wrapper approach: store the control image + params on the wrapper.
            # VAE encoding happens inside the generation loop (below) when
            # model_cpu_offload has moved the VAE to CUDA via its hook.
            _cn_wrapper = pipe.transformer
            _cn_wrapper._pending_control = {
                "image": cn_images[0],
                "scale": cn_scales[0] if cn_scales else 0.75,
                "height": height,
                "width": width,
            }
        else:
            # Standard approach: pass control kwargs to the ControlNet pipeline.
            control_image = cn_images[0] if len(cn_images) == 1 else cn_images
            control_scale = cn_scales[0] if len(cn_scales) == 1 else cn_scales

            # Different pipelines use different kwarg names for the control image:
            # Flux/Z-Image: control_image, SDXL/SD1.5: image
            import inspect
            pipe_params = set(inspect.signature(pipe.__call__).parameters.keys())
            if "control_image" in pipe_params:
                gen_kwargs["control_image"] = control_image
            else:
                gen_kwargs["image"] = control_image
            gen_kwargs["controlnet_conditioning_scale"] = control_scale

            if "control_guidance_end" in pipe_params:
                control_end = cn_end_values[0] if len(cn_end_values) == 1 else cn_end_values
                gen_kwargs["control_guidance_end"] = control_end
            if "control_mode" in pipe_params:
                cn_types = [inp.get("control_type", "canny") for inp in cn_inputs]
                control_modes = _resolve_control_modes(cn_types, arch)
                if control_modes is not None:
                    cm = control_modes[0] if len(control_modes) == 1 else control_modes
                    gen_kwargs["control_mode"] = cm

    # Add style reference images to generation kwargs
    if style_images and style_mechanism == "ip-adapter":
        gen_kwargs["ip_adapter_image"] = style_images if len(style_images) > 1 else style_images[0]
    elif style_images and style_mechanism == "native":
        # Flux 2 Klein native multi-ref: pass as reference_images
        gen_kwargs["reference_images"] = style_images

    # VAE-encode control image for wrapper approach (deferred until now so
    # the VAE's cpu_offload hook can move it to CUDA).
    if _cn_wrapper is not None and hasattr(_cn_wrapper, "_pending_control"):
        pending = _cn_wrapper._pending_control
        del _cn_wrapper._pending_control
        from diffusers.image_processor import VaeImageProcessor
        img_proc = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor * 2)
        ctrl_tensor = img_proc.preprocess(
            pending["image"], height=pending["height"], width=pending["width"],
        )
        # Send to the execution device — model_cpu_offload hook will move
        # the VAE to CUDA on forward(), so the input must already be there.
        ctrl_tensor = ctrl_tensor.to(dtype=pipe.vae.dtype, device=pipe._execution_device)
        with torch.no_grad():
            ctrl_latent = pipe.vae.encode(ctrl_tensor).latent_dist.mode()
        ctrl_latent = (ctrl_latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        ctrl_latent = ctrl_latent.unsqueeze(2)
        # Pad to control_in_dim if needed (v2.0+: 16ch → 33ch)
        cn_cfg = _cn_wrapper.controlnet.config
        if cn_cfg.control_in_dim and cn_cfg.control_in_dim != ctrl_latent.shape[1]:
            pad_ch = cn_cfg.control_in_dim - ctrl_latent.shape[1]
            ctrl_latent = torch.cat([
                ctrl_latent,
                torch.zeros(*ctrl_latent.shape[:1], pad_ch, *ctrl_latent.shape[2:],
                            device=ctrl_latent.device, dtype=ctrl_latent.dtype),
            ], dim=1)
        _cn_wrapper.set_control(list(ctrl_latent.unbind(dim=0)), scale=pending["scale"])

    artifact_paths = []

    for i in range(count):
        t0 = time.time()

        try:
            result = pipe(**gen_kwargs)
            image = result.images[0]
        except Exception as exc:
            emitter.error(
                "GENERATION_FAILED",
                f"Generation failed on image {i + 1}/{count}: {exc}",
                recoverable=(i + 1 < count),
            )
            continue

        elapsed = time.time() - t0

        # Advance seed for next image in batch
        if seed is not None:
            generator.manual_seed(seed + i + 1)

        image_seed = seed + i if seed is not None else None

        # Save image
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}_{i:03d}.png" if count > 1 else f"{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        # Persist provenance in PNG text chunks for portability across tools.
        save_kwargs = {}
        if filepath.lower().endswith(".png"):
            try:
                from PIL.PngImagePlugin import PngInfo

                # Collect model file info if available (from assemble_pipeline)
                model_files = {}
                if hasattr(pipe, "_modl_loaded_files"):
                    for comp, info in pipe._modl_loaded_files.items():
                        model_files[comp] = {
                            "file": info["file"],
                            "dtype": info["weight_dtype"],
                        }

                # ControlNet metadata
                cn_meta = None
                if cn_inputs:
                    cn_meta = [
                        {
                            "image": Path(inp["image"]).name,
                            "type": inp.get("control_type", "canny"),
                            "strength": inp.get("strength", 0.75),
                            "end": inp.get("control_end", 0.8),
                        }
                        for inp in cn_inputs
                    ]

                embedded_meta = {
                    "generated_with": "modl.run",
                    "prompt": prompt,
                    "base_model_id": base_model_id,
                    "lora_name": lora_info.get("name") if lora_info else None,
                    "lora_strength": lora_info.get("weight") if lora_info else None,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "guidance": guidance,
                    "seed": image_seed,
                    "image_index": i,
                    "count": count,
                    "timestamp": timestamp,
                    "model_files": model_files or None,
                    "controlnet": cn_meta,
                }
                pnginfo = PngInfo()
                pnginfo.add_text("Software", "modl.run")
                pnginfo.add_text("Comment", "generated with modl.run")
                pnginfo.add_text("modl_metadata", json.dumps(embedded_meta, separators=(",", ":")))
                save_kwargs["pnginfo"] = pnginfo
            except Exception:
                # Non-fatal: save image even if metadata embedding fails.
                pass

        image.save(filepath, **save_kwargs)

        # Hash the output
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        size_bytes = os.path.getsize(filepath)

        emitter.artifact(
            path=filepath,
            sha256=sha256.hexdigest(),
            size_bytes=size_bytes,
        )

        emitter.progress(
            stage="generate",
            step=i + 1,
            total_steps=count,
            eta_seconds=elapsed * (count - i - 1) if count > 1 else None,
        )

        artifact_paths.append(filepath)
        emitter.info(f"Image {i + 1}/{count}: {filepath} ({elapsed:.1f}s)")

    # Clean up control wrapper state
    if _cn_wrapper is not None:
        _cn_wrapper.clear_control()

    # Clean up tmp files (uploaded init images / masks) after generation
    _cleanup_tmp_files(init_image_path, mask_path)

    if artifact_paths:
        emitter.completed(f"Generated {len(artifact_paths)} image(s)")
    else:
        emitter.error(
            "NO_IMAGES_GENERATED",
            "All generation attempts failed",
            recoverable=False,
        )
        return 1

    return 0


def _cleanup_tmp_files(*paths: str | None) -> None:
    """Delete tmp files under ~/.modl/tmp/ after they've been consumed."""
    tmp_dir = str(Path.home() / ".modl" / "tmp")
    for p in paths:
        if p and p.startswith(tmp_dir):
            try:
                os.remove(p)
            except OSError:
                pass

