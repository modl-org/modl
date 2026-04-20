"""Image editing adapter — QwenImageEditPlusPipeline (Qwen-Image-Edit-2511).

Handles instruction-based image editing: takes one or more source images
plus a text prompt describing the edit, produces an output image.
"""

import os
import time
from pathlib import Path

from modl_worker.protocol import EventEmitter
from modl_worker.image_util import save_and_emit_artifact


def run_edit(config_path: Path, emitter: EventEmitter) -> int:
    """Run image editing from an EditJobSpec YAML file (one-shot mode)."""
    import yaml

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Edit spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    model_info = spec.get("model", {})
    base_model_id = model_info.get("base_model_id", "qwen-image-edit")
    base_model_path = model_info.get("base_model_path")

    params = spec.get("params", {})
    count = params.get("count", 1)

    emitter.info(f"Loading edit pipeline for {base_model_id}...")
    emitter.progress(stage="load", step=0, total_steps=count)

    try:
        pipe = _load_edit_pipeline(base_model_id, base_model_path, emitter)
        _apply_lora(pipe, spec, emitter)
        emitter.job_started(config=str(config_path))
    except Exception as exc:
        emitter.error(
            "PIPELINE_LOAD_FAILED",
            f"Failed to load edit pipeline: {exc}",
            recoverable=False,
        )
        return 1

    return run_edit_with_pipeline(spec, emitter, pipe)


def run_edit_with_pipeline(spec: dict, emitter: EventEmitter, pipeline: object) -> int:
    """Run image editing using an already-loaded pipeline.

    Shared inference loop used by both one-shot and persistent-worker modes.
    """
    import torch

    from modl_worker.image_util import load_image

    prompt = spec.get("prompt", "")
    model_info = spec.get("model", {})
    output_info = spec.get("output", {})
    params = spec.get("params", {})

    base_model_id = model_info.get("base_model_id", "qwen-image-edit")

    steps = params.get("steps", 40)
    guidance = params.get("guidance", 4.0)
    seed = params.get("seed")
    count = params.get("count", 1)

    # Detect architecture early — needed for scheduler and kwarg decisions
    from .arch_config import detect_arch
    arch = detect_arch(base_model_id, arch_key=model_info.get("arch_key"))

    # Apply scheduler overrides.
    #
    # Lightning mode and config-driven shift are mutually exclusive:
    # Lightning LoRAs are distilled with shift=3.0 and a simple linear
    # schedule — those settings must not be overwritten.  The config-driven
    # shift (e.g. 3.1 for Qwen-2511) only applies to vanilla (non-Lightning) inference.
    from .arch_config import ARCH_CONFIGS
    inf_cfg = ARCH_CONFIGS.get(arch, {}).get("inference", {})
    sched_overrides = params.get("scheduler_overrides")
    lightning_sigmas = None
    if sched_overrides and hasattr(pipeline, "scheduler"):
        # Lightning mode — shift=3.0, static schedule, custom sigmas.
        from .gen_adapter import _apply_scheduler_overrides, _compute_lightning_sigmas
        _apply_scheduler_overrides(pipeline, sched_overrides, emitter)
        lightning_sigmas = _compute_lightning_sigmas(steps)
    else:
        shift_val = inf_cfg.get("scheduler_shift")
        if shift_val is not None and hasattr(pipeline, "scheduler"):
            import math
            log_shift = math.log(shift_val)
            sched = pipeline.scheduler
            config = dict(sched.config)
            config["use_dynamic_shifting"] = True
            config["base_shift"] = log_shift
            config["max_shift"] = log_shift
            config["time_shift_type"] = "exponential"
            sched_class = type(sched)
            pipeline.scheduler = sched_class.from_config(config)
            emitter.info(f"Scheduler shift={shift_val} (base_shift=max_shift=log({shift_val})={log_shift:.4f})")

    # Debug: dump sigma schedule for comparison with ComfyUI.
    if os.environ.get("MODL_DEBUG_SIGMAS") and hasattr(pipeline, "scheduler"):
        import numpy as np
        sched = pipeline.scheduler
        # Simulate what the pipeline will do: set_timesteps with the same
        # sigmas/mu it would compute at inference time.
        debug_steps = steps
        debug_sigmas = np.linspace(1.0, 1 / debug_steps, debug_steps)
        if lightning_sigmas is not None:
            debug_sigmas = np.array(lightning_sigmas, dtype=np.float32)
        # Compute mu the same way the pipeline does (calculate_shift)
        import math as _math
        base_shift = sched.config.get("base_shift", 0.5)
        max_shift = sched.config.get("max_shift", 1.15)
        # Use 1024x1024 as reference (seq_len = (1024/16)^2 = 4096 for typical VAE)
        ref_seq_len = 4096
        m = (max_shift - base_shift) / (sched.config.get("max_image_seq_len", 4096) - sched.config.get("base_image_seq_len", 256))
        b = base_shift - m * sched.config.get("base_image_seq_len", 256)
        mu = ref_seq_len * m + b
        emitter.info(f"[DEBUG SIGMAS] steps={debug_steps}, mu={mu:.4f}, "
                     f"use_dynamic_shifting={sched.config.get('use_dynamic_shifting')}, "
                     f"shift={sched.config.get('shift', 'N/A')}, "
                     f"base_shift={base_shift:.4f}, max_shift={max_shift:.4f}")
        emitter.info(f"[DEBUG SIGMAS] raw input sigmas: {debug_sigmas.tolist()}")
        # Apply the shift manually to show final sigmas
        if sched.config.get("use_dynamic_shifting"):
            shifted = _math.exp(mu) / (_math.exp(mu) + (1.0 / debug_sigmas - 1.0))
            emitter.info(f"[DEBUG SIGMAS] shifted sigmas (exp shift={_math.exp(mu):.4f}): {shifted.tolist()}")
        else:
            s = sched.config.get("shift", 1.0)
            shifted = s * debug_sigmas / (1 + (s - 1) * debug_sigmas)
            emitter.info(f"[DEBUG SIGMAS] shifted sigmas (static shift={s}): {shifted.tolist()}")

    image_paths = params.get("image_paths", [])
    mask_path = params.get("mask_path")
    blend_mode = params.get("blend_mode", "pixel")

    if not image_paths:
        emitter.error("NO_IMAGES", "No input images provided", recoverable=False)
        return 1

    # Load source images (EXIF orientation is applied automatically)
    source_images = []
    for p in image_paths:
        try:
            img = load_image(p)
            source_images.append(img)
            emitter.info(f"Loaded input image: {p} ({img.size[0]}x{img.size[1]})")
        except Exception as exc:
            emitter.error("IMAGE_LOAD_FAILED", f"Failed to load {p}: {exc}", recoverable=False)
            return 1

    # Load or derive mask image (white=edit, black=preserve)
    mask_image = None
    if mask_path:
        mask_image = _resolve_mask(mask_path, source_images, emitter)
        if mask_image is None and mask_path not in ("auto",):
            return 1  # _resolve_mask already emitted the error

    output_dir = output_info.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    from modl_worker.device import get_generator_device
    generator = torch.Generator(device=get_generator_device())
    if seed is not None:
        generator.manual_seed(seed)

    # Build inference kwargs — different pipelines need different params
    inf_cfg = ARCH_CONFIGS.get(arch, {}).get("inference", {})
    use_native_inpaint = False
    if inf_cfg.get("editing_mode") == "native":
        # Native editing via the `image` parameter (e.g. Klein).
        # Supports multiple input images (e.g. source + reference).
        # No guidance (distilled), no negative prompt.
        # Explicit --size overrides the first image's dimensions.
        out_h = params.get("height") or source_images[0].size[1]
        out_w = params.get("width") or source_images[0].size[0]

        # Latent-space blend: swap to Flux2KleinInpaintPipeline if available
        if mask_image is not None and blend_mode == "latent":
            try:
                from diffusers import Flux2KleinInpaintPipeline
                pipeline = Flux2KleinInpaintPipeline.from_pipe(pipeline)
                use_native_inpaint = True
                emitter.info("Loaded Flux2KleinInpaintPipeline for latent-space mask blending")
            except ImportError:
                emitter.info("Flux2KleinInpaintPipeline not available (diffusers too old), will fall back to pixel blend")

        gen_kwargs = {
            "image": source_images if len(source_images) > 1 else source_images[0],
            "prompt": prompt,
            "num_inference_steps": steps,
            "height": out_h,
            "width": out_w,
            "generator": generator,
        }

        if use_native_inpaint and mask_image is not None:
            gen_kwargs["mask_image"] = mask_image
            gen_kwargs["strength"] = 0.8
    else:
        # Qwen-Image-Edit: instruction-based editing with true CFG.
        # true_cfg_scale controls actual classifier-free guidance (default 4.0).
        # guidance_scale controls diffusers noise scheduling (always 1.0 for Qwen).
        # In Lightning mode, the Rust CLI overrides guidance to 1.0 — but that's
        # meant for guidance_scale, NOT true_cfg_scale. We must preserve true_cfg_scale
        # at a reasonable level or the output is soft/blurry (no CFG).
        is_lightning = sched_overrides is not None
        true_cfg = guidance if not is_lightning else 4.0
        gen_kwargs = {
            "image": source_images if len(source_images) > 1 else source_images[0],
            "prompt": prompt,
            "true_cfg_scale": true_cfg,
            "negative_prompt": " ",
            "num_inference_steps": steps,
            "guidance_scale": 1.0,
            "generator": generator,
        }
        # Optional output dimensions (for outpainting — larger than source)
        if params.get("width"):
            gen_kwargs["width"] = params["width"]
        if params.get("height"):
            gen_kwargs["height"] = params["height"]

    # Lightning mode: use ComfyUI-style simple linear sigmas.
    if lightning_sigmas is not None:
        gen_kwargs["sigmas"] = lightning_sigmas


    artifact_paths = []

    for i in range(count):
        t0 = time.time()

        try:
            result = pipeline(**gen_kwargs)
            image = result.images[0]
        except Exception as exc:
            emitter.error(
                "EDIT_FAILED",
                f"Edit failed on image {i + 1}/{count}: {exc}",
                recoverable=(i + 1 < count),
            )
            continue

        # Masked blend: preserve source pixels outside the mask region.
        # Skip if the inpaint pipeline already handled blending natively.
        if mask_image is not None and not use_native_inpaint:
            if blend_mode == "latent":
                emitter.info("Latent blend not available for this pipeline, falling back to pixel blend")
            image = _pixel_blend(source_images[0], image, mask_image)

        elapsed = time.time() - t0

        if seed is not None:
            generator.manual_seed(seed + i + 1)

        image_seed = seed + i if seed is not None else None

        lora_info = spec.get("lora")
        embedded_meta = {
            "generated_with": "modl.run",
            "mode": "edit",
            "prompt": prompt,
            "base_model_id": base_model_id,
            "input_images": image_paths,
            "steps": steps,
            "guidance": guidance,
            "seed": image_seed,
            "lora_name": lora_info.get("name") if lora_info else None,
            "lora_strength": lora_info.get("weight") if lora_info else None,
            "image_index": i,
            "count": count,
            "mask": mask_path,
            "blend_mode": blend_mode if mask_path else None,
        }

        filepath = save_and_emit_artifact(
            image, output_dir, emitter,
            index=i, count=count, metadata=embedded_meta,
            stage="edit", elapsed=elapsed,
        )
        if filepath:
            artifact_paths.append(filepath)

    if artifact_paths:
        emitter.completed(f"Edited {len(artifact_paths)} image(s)")
    else:
        emitter.error("NO_IMAGES_GENERATED", "All edit attempts failed.", recoverable=False)

    return 0 if artifact_paths else 1


def _resolve_mask(mask_path: str, source_images: list, emitter: EventEmitter):
    """Load or derive a mask image. Returns a PIL Image in 'L' mode, or None on error.

    White pixels = edit region, black pixels = preserve.
    """
    from PIL import Image

    from modl_worker.image_util import load_image

    if mask_path == "from-alpha":
        # Derive mask from the first source image's alpha channel
        src = source_images[0]
        if src.mode not in ("RGBA", "LA", "PA"):
            emitter.error(
                "NO_ALPHA",
                "--mask from-alpha requires an image with an alpha channel (PNG with transparency)",
                recoverable=False,
            )
            return None
        # Alpha > 0 means subject is present → invert: we want to edit *around* the subject
        # Actually: alpha = where the image exists. For "from-alpha", white = where alpha exists = edit there.
        # User intent: "edit the region defined by the alpha". White = edit.
        alpha = source_images[0].getchannel("A")
        emitter.info(f"Derived mask from alpha channel ({alpha.size[0]}x{alpha.size[1]})")
        return alpha

    if mask_path == "auto":
        # "auto" is deferred to step 3 (image_reference).
        # For now, skip mask — will be wired when --reference lands.
        emitter.info("--mask auto: no reference image provided, skipping mask")
        return None

    # Load mask from file path
    try:
        mask = load_image(mask_path, mode="L")
        emitter.info(f"Loaded mask: {mask_path} ({mask.size[0]}x{mask.size[1]})")
        return mask
    except Exception as exc:
        emitter.error("MASK_LOAD_FAILED", f"Failed to load mask {mask_path}: {exc}", recoverable=False)
        return None


def _pixel_blend(source: "Image.Image", edited: "Image.Image", mask: "Image.Image") -> "Image.Image":
    """Blend edited result with source using a mask (white=edit, black=preserve).

    Resizes mask and source to match the edited image dimensions if needed,
    then pastes source pixels where mask is black (preserve region).
    """
    from PIL import ImageOps

    w, h = edited.size

    # Resize source and mask to match output if pipeline changed dimensions
    if source.size != (w, h):
        source = source.resize((w, h), resample=3)  # LANCZOS
    if mask.size != (w, h):
        mask = mask.resize((w, h), resample=0)  # NEAREST for binary mask

    # Invert mask: we need white=preserve for paste
    preserve_mask = ImageOps.invert(mask.convert("L"))
    result = edited.copy()
    result.paste(source.convert("RGB"), mask=preserve_mask)
    return result


def _apply_lora(pipeline, spec: dict, emitter: EventEmitter) -> None:
    """Load and fuse a LoRA onto the edit pipeline if specified in the spec."""
    from .lora_utils import apply_lora_from_spec
    apply_lora_from_spec(pipeline, spec, emitter)


def _load_edit_pipeline(base_model_id: str, base_model_path: str | None, emitter: EventEmitter):
    """Load the edit pipeline for the given model."""
    from .arch_config import resolve_pipeline_class
    from .pipeline_loader import load_pipeline

    cls_name = resolve_pipeline_class(base_model_id)
    return load_pipeline(base_model_id, base_model_path, cls_name, emitter)
