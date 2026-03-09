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

        dtype_counts: dict[str, int] = {}
        for key, info in header.items():
            if key == "__metadata__":
                continue
            dt = info.get("dtype", "")
            dtype_counts[dt] = dtype_counts.get(dt, 0) + 1

        if not dtype_counts:
            return "unknown"

        dominant = max(dtype_counts, key=dtype_counts.get)
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
            weight_dtype = _detect_weight_dtype(base_model_path)
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

            # Always load in bf16 first. For fp8 files, from_single_file
            # upcasts to bf16 during key conversion.
            model = ModelClass.from_single_file(
                base_model_path, config=str(config_dir), torch_dtype=torch.bfloat16,
            )
            if is_fp8:
                # Use diffusers' built-in layerwise casting: casts weights
                # down to fp8 for storage (~12GB vs 24GB) but upcasts to
                # bf16 during forward passes. Automatically keeps norms
                # and embeddings in bf16 for numerical stability.
                model.enable_layerwise_casting(
                    storage_dtype=torch.float8_e4m3fn,
                    compute_dtype=torch.bfloat16,
                )
                emitter.info(
                    f"  → fp8 layerwise casting enabled (storage=fp8, compute=bf16)"
                )
            components["transformer"] = model

        elif param_name in ("scheduler",):
            components[param_name] = ModelClass.from_pretrained(str(config_dir))

        elif param_name.startswith("tokenizer"):
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

            # Check if this is an NF4-quantized model (HF directory with
            # quantization config, e.g. Flux2's Mistral3 text encoder)
            quantize_nf4 = spec.get("quantize_nf4", False)

            if hasattr(ModelClass, "from_single_file"):
                # Diffusers models (VAE) — from_single_file handles memory
                components[param_name] = ModelClass.from_single_file(
                    resolved_path,
                    config=str(config_dir),
                    torch_dtype=torch.bfloat16,
                )
            elif os.path.isdir(resolved_path):
                # HF directory layout (e.g. quantized Mistral3 from BFL)
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
                    resolved_path, config=str(config_dir), **load_kwargs,
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
    pipe = PipelineClass(**components)
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
        pipe = PipelineClass.from_single_file(
            model_source,
            torch_dtype=torch.bfloat16,
        )
        pipe._modl_loaded_files = {
            "checkpoint": {"file": filename, "path": model_source, "weight_dtype": weight_dtype},
        }
    elif fmt == "transformer_only":
        # Assemble pipeline from locally installed components
        assembly = resolve_gen_assembly(base_model_id)
        if assembly:
            return assemble_pipeline(base_model_id, model_source, cls_name, emitter)
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
            return assemble_pipeline(base_model_id, model_source, cls_name, emitter)
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
        # For cold start, load the mode-specific pipeline directly
        cls_name = resolve_pipeline_class_for_mode(base_model_id, cold_mode)
        pipe = load_pipeline(base_model_id, base_model_path, cls_name, emitter)

        # Load LoRA if specified
        if lora_info:
            lora_path = lora_info.get("path")
            lora_weight = lora_info.get("weight", 1.0)
            if lora_path and os.path.exists(lora_path):
                emitter.info(f"Loading LoRA: {lora_info.get('name', 'unnamed')} (weight={lora_weight})")
                pipe.load_lora_weights(lora_path)
                pipe.fuse_lora(lora_scale=lora_weight)

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
        init_img = Image.open(init_image_path).convert("RGB")
    if mask_path:
        mask_img = Image.open(mask_path).convert("RGB")

    # Switch pipeline if needed for img2img/inpaint via from_pipe()
    pipe = pipeline
    if mode != "txt2img":
        target_cls_name = resolve_pipeline_class_for_mode(base_model_id, mode)
        if target_cls_name != cls_name:
            emitter.info(f"Switching pipeline: {cls_name} -> {target_cls_name} (mode={mode})")
            TargetClass = _get_pipeline(target_cls_name)
            pipe = TargetClass.from_pipe(pipeline)
            cls_name = target_cls_name

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
        "guidance_scale": guidance,
    }

    if mode == "txt2img":
        gen_kwargs["width"] = width
        gen_kwargs["height"] = height
    elif mode == "img2img":
        gen_kwargs["image"] = init_img
        gen_kwargs["strength"] = strength
    elif mode == "inpaint":
        gen_kwargs["image"] = init_img
        gen_kwargs["mask_image"] = mask_img
        gen_kwargs["strength"] = strength
        gen_kwargs["width"] = width
        gen_kwargs["height"] = height
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

