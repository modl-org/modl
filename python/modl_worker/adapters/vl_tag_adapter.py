"""VL Tag adapter — automatic image labeling using Qwen2.5-VL-3B.

Generates comma-separated tags/labels for images using a vision-language model.
Unlike the dataset-focused tag_adapter.py, this operates on individual images
and returns structured JSON results.

Reads a vl-tag job spec YAML containing:
  image_paths: list[str]    — paths to images
  model: str                — "qwen25-vl-3b" (default)
  max_tags: int             — maximum number of tags (optional)
"""

import time
from pathlib import Path

from modl_worker.protocol import EventEmitter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _resolve_images(image_paths: list[str]) -> list[Path]:
    """Expand directories and filter to valid image files."""
    result = []
    for p_str in image_paths:
        p = Path(p_str)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    result.append(f)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            result.append(p)
    return result


def _load_qwen_vl(emitter: EventEmitter):
    """Load Qwen2.5-VL-3B model and processor."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    emitter.info("Loading Qwen2.5-VL-3B model...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    return model, processor


def run_vl_tag(config_path: Path, emitter: EventEmitter) -> int:
    """Run VL-based image tagging from a VlTagJobSpec YAML file."""
    import yaml
    import torch
    from qwen_vl_utils import process_vision_info

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"VL tag spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    image_paths = spec.get("image_paths", [])
    max_tags = spec.get("max_tags")

    images = _resolve_images(image_paths)
    if not images:
        emitter.error("NO_IMAGES", "No valid images found", recoverable=False)
        return 2

    total = len(images)
    emitter.info(f"Found {total} image(s) to tag")
    emitter.job_started(config=str(config_path))

    try:
        model, processor = _load_qwen_vl(emitter)
    except Exception as exc:
        emitter.error("MODEL_LOAD_FAILED", f"Failed to load Qwen2.5-VL: {exc}", recoverable=False)
        return 1

    emitter.info("Model loaded, starting tagging...")

    prompt = "List the main objects and concepts in this image as comma-separated tags. Just the tags, nothing else."

    results = []
    errors = 0

    for i, image_path in enumerate(images):
        emitter.progress(stage="vl-tag", step=i, total_steps=total)

        try:
            t0 = time.time()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=256)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            elapsed = time.time() - t0

            # Parse comma-separated tags
            tags = [t.strip() for t in response.split(",") if t.strip()]
            if max_tags and len(tags) > max_tags:
                tags = tags[:max_tags]

            results.append({
                "image": str(image_path),
                "tags": tags,
            })

            emitter.info(f"[{i + 1}/{total}] {image_path.name} ({elapsed:.1f}s): {', '.join(tags[:5])}...")

        except Exception as exc:
            emitter.warning("VL_TAG_FAILED", f"Failed to tag {image_path.name}: {exc}")
            results.append({"image": str(image_path), "tags": [], "error": str(exc)})
            errors += 1

    emitter.progress(stage="vl-tag", step=total, total_steps=total)

    emitter.result("vl_tagging", {
        "results": results,
        "images_processed": total,
        "errors": errors,
    })

    summary = f"Tagged {total - errors}/{total} images"
    if errors > 0:
        summary += f" ({errors} error(s))"
    emitter.completed(summary)

    return 0 if errors == 0 else 1
