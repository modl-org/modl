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
    image_paths = params.get("image_paths", [])

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

    output_dir = output_info.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    from modl_worker.device import get_generator_device
    generator = torch.Generator(device=get_generator_device())
    if seed is not None:
        generator.manual_seed(seed)

    # Build inference kwargs — different pipelines need different params
    from .arch_config import detect_arch
    arch = detect_arch(base_model_id)

    if arch in ("flux2_klein", "flux2_klein_9b"):
        # Klein: native image editing via the `image` parameter.
        # Supports multiple input images (e.g. source + reference).
        # No guidance (distilled), no negative prompt.
        gen_kwargs = {
            "image": source_images if len(source_images) > 1 else source_images[0],
            "prompt": prompt,
            "num_inference_steps": steps,
            "height": source_images[0].size[1],
            "width": source_images[0].size[0],
            "generator": generator,
        }
    else:
        # Qwen-Image-Edit: instruction-based editing with true CFG.
        gen_kwargs = {
            "image": source_images if len(source_images) > 1 else source_images[0],
            "prompt": prompt,
            "true_cfg_scale": guidance,
            "negative_prompt": " ",
            "num_inference_steps": steps,
            "generator": generator,
        }
        # Optional output dimensions (for outpainting — larger than source)
        if params.get("width"):
            gen_kwargs["width"] = params["width"]
        if params.get("height"):
            gen_kwargs["height"] = params["height"]

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
