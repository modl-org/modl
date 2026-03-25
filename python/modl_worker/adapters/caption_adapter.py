"""Caption adapter — auto-generate captions for dataset images using Florence-2 or BLIP-2.

Reads a caption job spec YAML containing:
  dataset_path: str   — path to the dataset directory
  model: str           — "florence-2" (default) or "blip"
  overwrite: bool      — re-caption images that already have .txt files

Scans for images without paired .txt captions (unless overwrite=True),
runs the selected vision-language model, writes .txt files alongside images,
and emits progress + artifact events.
"""

import hashlib
import os
import time
from pathlib import Path
from typing import List, Tuple

from modl_worker.protocol import EventEmitter

# Valid image extensions (must match Rust side)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _find_images(dataset_path: Path, overwrite: bool) -> List[Path]:
    """Find images that need captioning."""
    images = []
    for f in sorted(dataset_path.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption_path = f.with_suffix(".txt")
        if not overwrite and caption_path.exists():
            continue
        images.append(f)
    return images


def _load_florence2(emitter: EventEmitter, model_path: str | None = None) -> Tuple:
    """Load Florence-2 model and processor.
    
    If model_path is provided, loads from that local directory (pre-downloaded
    via modl registry). Otherwise falls back to HuggingFace Hub download.
    """
    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch

    model_id = model_path or "microsoft/Florence-2-large"
    source = "local" if model_path else "HuggingFace Hub"
    emitter.info(f"Loading Florence-2 from {source}: {model_id}")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    from modl_worker.device import get_device
    model = model.to(get_device())

    return model, processor


def _caption_florence2(
    model, processor, image_path: Path, style_mode: bool = False
) -> str:
    """Generate a caption for a single image using Florence-2.
    
    When style_mode is True, uses <CAPTION> (shorter, factual) instead of
    <DETAILED_CAPTION> to avoid describing art style/medium — better for
    style LoRA datasets where the style should be learned implicitly.
    """
    import torch
    from PIL import Image

    from modl_worker.image_util import load_image

    image = load_image(image_path)

    # <CAPTION> gives shorter factual descriptions (good for style LoRAs)
    # <DETAILED_CAPTION> gives richer descriptions including style/medium
    task = "<CAPTION>" if style_mode else "<DETAILED_CAPTION>"
    inputs = processor(text=task, images=image, return_tensors="pt").to(
        model.device, torch.float16
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
            do_sample=False,
            # Disable KV cache — Florence-2's custom modeling code is
            # incompatible with transformers >=4.46 DynamicCache format.
            # Slightly slower but avoids shape errors in attention layers.
            use_cache=False,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=task, image_size=(image.width, image.height)
    )
    caption = parsed.get(task, text).strip()
    return caption


def _load_blip(emitter: EventEmitter, model_path: str | None = None) -> Tuple:
    """Load BLIP-2 model and processor.
    
    If model_path is provided, loads from that local directory.
    """
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    import torch

    model_id = model_path or "Salesforce/blip2-opt-2.7b"
    source = "local" if model_path else "HuggingFace Hub"
    emitter.info(f"Loading BLIP-2 from {source}: {model_id}")

    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    from modl_worker.device import get_device
    model = model.to(get_device())

    return model, processor


def _caption_blip(
    model, processor, image_path: Path, style_mode: bool = False
) -> str:
    """Generate a caption for a single image using BLIP-2.
    
    When style_mode is True, uses a text prompt instructing the model to
    describe content only, omitting art style/medium/technique — better for
    style LoRA datasets.
    """
    import torch
    from PIL import Image

    from modl_worker.image_util import load_image

    image = load_image(image_path)

    if style_mode:
        # Prompt BLIP-2 to focus on content, not visual style
        prompt = "Describe what is depicted in this image without mentioning the art style, medium, or drawing technique:"
        inputs = processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(model.device, torch.float16)
    else:
        inputs = processor(images=image, return_tensors="pt").to(
            model.device, torch.float16
        )

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_caption(config_path: Path, emitter: EventEmitter) -> int:
    """Run auto-captioning on a dataset from a CaptionJobSpec YAML file."""
    import yaml

    if not config_path.exists():
        emitter.error(
            "SPEC_NOT_FOUND",
            f"Caption spec not found: {config_path}",
            recoverable=False,
        )
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    dataset_path = Path(spec.get("dataset_path", ""))
    model_name = spec.get("model", "florence-2")
    overwrite = spec.get("overwrite", False)
    model_path = spec.get("model_path")  # Local path from registry, if available
    style_mode = spec.get("style", False)

    if not dataset_path.exists() or not dataset_path.is_dir():
        emitter.error(
            "DATASET_NOT_FOUND",
            f"Dataset directory not found: {dataset_path}",
            recoverable=False,
        )
        return 2

    # Find images that need captioning
    images = _find_images(dataset_path, overwrite)

    if not images:
        emitter.info("All images already have captions. Nothing to do.")
        emitter.completed("No images needed captioning")
        return 0

    total = len(images)
    mode_label = f"{model_name} (style mode: content only)" if style_mode else model_name
    emitter.info(f"Found {total} image(s) to caption using {mode_label}")
    emitter.job_started(config=str(config_path))

    # Load model
    try:
        if model_name.lower() in ("florence-2", "florence2", "florence"):
            model, processor = _load_florence2(emitter, model_path)
            caption_fn = lambda m, p, img: _caption_florence2(m, p, img, style_mode=style_mode)
        elif model_name.lower() in ("blip", "blip-2", "blip2"):
            model, processor = _load_blip(emitter, model_path)
            caption_fn = lambda m, p, img: _caption_blip(m, p, img, style_mode=style_mode)
        elif model_name.lower() in ("qwen", "qwen-vl", "qwen3-vl", "qwen3-vl-2b", "qwen25-vl", "qwen25-vl-3b"):
            from modl_worker.adapters.vl_common import load_qwen_vl, run_vl_inference
            model, processor = load_qwen_vl(emitter)
            if style_mode:
                _qwen_prompt = "You are captioning training data. Describe ONLY the content: subjects, objects, actions, and setting. Never use words like: drawing, painting, sketch, illustration, hand-drawn, childlike, child's, artwork, artistic, cartoon, doodle, depicted, rendered. Treat every image as a photograph of a real scene. Write one sentence."
                _max_tokens = 80
            else:
                _qwen_prompt = "Write a concise caption for this image in one or two sentences. Describe the subject, their pose or action, and the setting. Do not start with 'This is' or 'The image shows'."
                _max_tokens = 100
            caption_fn = lambda m, p, img: run_vl_inference(m, p, str(img), _qwen_prompt, max_tokens=_max_tokens)
        else:
            emitter.error(
                "UNKNOWN_MODEL",
                f"Unknown captioning model: {model_name}. Use 'florence-2', 'blip', or 'qwen'.",
                recoverable=False,
            )
            return 2
    except Exception as exc:
        emitter.error(
            "MODEL_LOAD_FAILED",
            f"Failed to load captioning model: {exc}",
            recoverable=False,
        )
        return 1

    emitter.info("Model loaded, starting captioning...")

    # Caption each image
    captioned = 0
    errors = 0

    for i, image_path in enumerate(images):
        emitter.progress(stage="caption", step=i, total_steps=total)

        try:
            t0 = time.time()
            caption = caption_fn(model, processor, image_path)
            elapsed = time.time() - t0

            # Write caption file
            caption_path = image_path.with_suffix(".txt")
            caption_path.write_text(caption, encoding="utf-8")

            # Emit artifact for the caption file
            sha256 = hashlib.sha256(caption.encode("utf-8")).hexdigest()
            size_bytes = caption_path.stat().st_size
            emitter.artifact(
                path=str(caption_path),
                sha256=sha256,
                size_bytes=size_bytes,
            )

            emitter.info(
                f"[{i + 1}/{total}] {image_path.name} ({elapsed:.1f}s): {caption}"
            )
            captioned += 1

        except Exception as exc:
            emitter.warning(
                "CAPTION_FAILED",
                f"Failed to caption {image_path.name}: {exc}",
            )
            errors += 1

    # Final progress tick
    emitter.progress(stage="caption", step=total, total_steps=total)

    summary = f"Captioned {captioned}/{total} images"
    if errors > 0:
        summary += f" ({errors} error(s))"
    emitter.completed(summary)

    return 0 if errors == 0 else 1
