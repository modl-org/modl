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
    resolve_gen_defaults,
    resolve_gen_components,
)


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

    # -------------------------------------------------------------------
    # 1. Load pipeline (cold start)
    # -------------------------------------------------------------------
    emitter.info(f"Loading pipeline for {base_model_id}...")
    count = spec.get("params", {}).get("count", 1)
    emitter.progress(stage="load", step=0, total_steps=count)

    try:
        import torch

        cls_name = _resolve_pipeline_class(base_model_id)
        PipelineClass = _get_pipeline(cls_name)

        # Determine model source: store path or HuggingFace ID
        model_source = base_model_path or resolve_model_path(base_model_id)

        # Models with separate components (e.g. z-image-turbo with Qwen3 text
        # encoder) need from_pretrained with the full HF repo — from_single_file
        # can't discover configs for non-standard text encoders.
        components = resolve_gen_components(base_model_id)

        if components and model_source.endswith(".safetensors"):
            # Fall back to HF repo which has model_index.json + configs
            hf_repo = resolve_model_path(base_model_id)
            emitter.info(f"Loading from HF repo {hf_repo} (model has separate components)")
            pipe = PipelineClass.from_pretrained(
                hf_repo,
                torch_dtype=torch.bfloat16,
            )
        elif model_source.endswith(".safetensors"):
            pipe = PipelineClass.from_single_file(
                model_source,
                torch_dtype=torch.bfloat16,
            )
        else:
            pipe = PipelineClass.from_pretrained(
                model_source,
                torch_dtype=torch.bfloat16,
            )

        pipe = pipe.to("cuda")

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

    output_dir = output_info.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    generator = torch.Generator(device="cuda")
    if seed is not None:
        generator.manual_seed(seed)

    # Build inference kwargs — different pipelines accept different params
    gen_kwargs = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "generator": generator,
    }

    # Guidance scale
    if cls_name == "FluxPipeline":
        gen_kwargs["guidance_scale"] = guidance
    else:
        gen_kwargs["guidance_scale"] = guidance

    pipe = pipeline
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


# ---------------------------------------------------------------------------
# Backwards-compat alias (used by serve.py imports)
# ---------------------------------------------------------------------------

def _hf_id_for_model(base_model_id: str) -> str:
    """Map a modl model ID to a HuggingFace repo ID."""
    return resolve_model_path(base_model_id)
