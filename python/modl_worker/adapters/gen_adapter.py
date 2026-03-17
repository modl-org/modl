"""Generate adapter — runs diffusers inference and emits events.

Translates a GenerateJobSpec (parsed from YAML) into a diffusers pipeline call.
Pipeline class and default params are resolved via arch_config — the single
source of truth for all model-specific settings.

Pipeline loading logic lives in pipeline_loader.py; ControlNet/style-ref
loading lives in controlnet.py.  This module contains the inference loop
and the one-shot entry point.

Outputs are saved as PNG and emitted as artifact events.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from modl_worker.protocol import EventEmitter
from modl_worker.adapters.arch_config import (
    resolve_pipeline_class_for_mode,
    resolve_gen_defaults,
)

# Re-exports for backward compatibility (moved to pipeline_loader)
from modl_worker.adapters.pipeline_loader import (  # noqa: F401
    load_pipeline,
    detect_model_format,
    assemble_pipeline,
    _resolve_pipeline_class,
    _get_pipeline,
)

from modl_worker.adapters.controlnet import (
    _load_controlnet,
    _resolve_control_modes,
    _load_style_ref,
)


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
    if seed is None:
        seed = generator.seed()  # capture the random seed for reproducibility
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

    # Chroma supports and benefits from negative prompts (unlike Flux).
    if arch == "chroma":
        neg = params.get("negative_prompt", "")
        if not neg:
            neg = "low quality, ugly, unfinished, out of focus, deformed, disfigured, blurry"
        gen_kwargs["negative_prompt"] = neg

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
