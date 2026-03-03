"""Generate adapter — runs diffusers inference and emits events.

Translates a GenerateJobSpec (parsed from YAML) into a diffusers pipeline call.
Pipeline class is selected based on base model ID:
  - flux-*   → FluxPipeline
  - sdxl-*   → StableDiffusionXLPipeline
  - sd-*     → StableDiffusionPipeline  (fallback)

Outputs are saved as PNG and emitted as artifact events.
"""

import hashlib
import os
import time
from pathlib import Path

from modl_worker.protocol import EventEmitter


# ---------------------------------------------------------------------------
# Model → Pipeline mapping
# ---------------------------------------------------------------------------

_MODEL_PIPELINE_MAP = {
    "flux": "FluxPipeline",
    "sdxl": "StableDiffusionXLPipeline",
    "sd": "StableDiffusionPipeline",
}


def _resolve_pipeline_class(base_model_id: str) -> str:
    """Determine diffusers pipeline class from base model id."""
    model_lower = base_model_id.lower()
    for prefix, cls_name in _MODEL_PIPELINE_MAP.items():
        if model_lower.startswith(prefix):
            return cls_name
    return "FluxPipeline"  # default for modern models


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
    """Run image generation from a GenerateJobSpec YAML file."""
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

    prompt = spec.get("prompt", "")
    model_info = spec.get("model", {})
    lora_info = spec.get("lora")
    output_info = spec.get("output", {})
    params = spec.get("params", {})

    base_model_id = model_info.get("base_model_id", "flux-schnell")
    base_model_path = model_info.get("base_model_path")

    width = params.get("width", 1024)
    height = params.get("height", 1024)
    steps = params.get("steps", 28)
    guidance = params.get("guidance", 3.5)
    seed = params.get("seed")
    count = params.get("count", 1)

    output_dir = output_info.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # 1. Load pipeline
    # -------------------------------------------------------------------
    emitter.info(f"Loading pipeline for {base_model_id}...")
    emitter.progress(stage="load", step=0, total_steps=count)

    try:
        import torch

        cls_name = _resolve_pipeline_class(base_model_id)
        PipelineClass = _get_pipeline(cls_name)

        # Determine model source: store path or HuggingFace ID
        model_source = base_model_path or _hf_id_for_model(base_model_id)

        pipe = PipelineClass.from_single_file(
            model_source,
            torch_dtype=torch.bfloat16,
        ) if model_source.endswith(".safetensors") else PipelineClass.from_pretrained(
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
    # 2. Generate images
    # -------------------------------------------------------------------
    import torch

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

    # Guidance scale: Flux uses a different parameter name for some versions
    if cls_name == "FluxPipeline":
        gen_kwargs["guidance_scale"] = guidance
    else:
        gen_kwargs["guidance_scale"] = guidance

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

        # Save image
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}_{i:03d}.png" if count > 1 else f"{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)

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
# HuggingFace model ID mapping (for models not installed locally)
# ---------------------------------------------------------------------------

_HF_MODEL_IDS = {
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd-1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}


def _hf_id_for_model(base_model_id: str) -> str:
    """Map a modl model ID to a HuggingFace repo ID."""
    return _HF_MODEL_IDS.get(base_model_id, base_model_id)
