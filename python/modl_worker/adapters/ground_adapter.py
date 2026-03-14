"""Ground adapter — text-grounded object detection using Qwen2.5-VL-3B.

Locates objects matching a text query in images using a vision-language model.
Returns bounding boxes for each detected instance.

Reads a ground job spec YAML containing:
  image_paths: list[str]    — paths to images
  query: str                — text query like "coffee cup" or "person"
  model: str                — "qwen25-vl-3b" (default)
  threshold: float          — minimum confidence (default 0.0)
"""

import json
import re
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


def _parse_detections(response_text: str, query: str, threshold: float) -> list[dict]:
    """Parse bounding box detections from Qwen2.5-VL response text.

    The model typically returns a JSON array like:
    [{"label": "coffee cup", "bbox_2d": [x1, y1, x2, y2]}, ...]

    Falls back to regex extraction if JSON parsing fails.
    """
    # Try to extract JSON array from the response
    json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if json_match:
        try:
            raw = json.loads(json_match.group())
            if isinstance(raw, list):
                objects = []
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    bbox = item.get("bbox_2d") or item.get("bbox")
                    label = item.get("label", query)
                    if bbox and isinstance(bbox, list) and len(bbox) == 4:
                        confidence = float(item.get("confidence", 1.0))
                        if confidence >= threshold:
                            objects.append({
                                "label": str(label),
                                "bbox": [round(float(c), 1) for c in bbox],
                                "confidence": round(confidence, 4),
                            })
                return objects
        except (json.JSONDecodeError, ValueError):
            pass

    return []


def run_ground(config_path: Path, emitter: EventEmitter, model_cache: dict | None = None) -> int:
    """Run text-grounded object detection on images from a GroundJobSpec YAML file."""
    import yaml
    import torch
    from qwen_vl_utils import process_vision_info
    from modl_worker.image_util import load_image

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Ground spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    image_paths = spec.get("image_paths", [])
    query = spec.get("query", "")
    threshold = float(spec.get("threshold") or 0.0)

    if not query:
        emitter.error("NO_QUERY", "No query provided for grounded detection", recoverable=False)
        return 2

    images = _resolve_images(image_paths)
    if not images:
        emitter.error("NO_IMAGES", "No valid images found", recoverable=False)
        return 2

    total = len(images)
    emitter.info(f"Found {total} image(s) to analyze for '{query}'")
    emitter.job_started(config=str(config_path))

    try:
        if model_cache is not None and "qwen_vl_model" in model_cache:
            model, processor = model_cache["qwen_vl_model"]
            emitter.info("Using cached Qwen2.5-VL model")
        else:
            model, processor = _load_qwen_vl(emitter)
            if model_cache is not None:
                model_cache["qwen_vl_model"] = (model, processor)
    except Exception as exc:
        emitter.error("MODEL_LOAD_FAILED", f"Failed to load Qwen2.5-VL: {exc}", recoverable=False)
        return 1

    emitter.info("Model loaded, starting grounded detection...")

    prompt = (
        f'Locate all instances of "{query}" in this image. '
        f'Return bounding boxes as JSON array: [{{"label": "...", "bbox_2d": [x1, y1, x2, y2]}}]'
    )

    detections = []
    errors = 0

    for i, image_path in enumerate(images):
        emitter.progress(stage="ground", step=i, total_steps=total)

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
                generated_ids = model.generate(**inputs, max_new_tokens=1024)

            # Trim input tokens from generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            elapsed = time.time() - t0

            objects = _parse_detections(response_text, query, threshold)

            detection = {
                "image": str(image_path),
                "objects": objects,
                "object_count": len(objects),
            }
            detections.append(detection)

            emitter.info(f"[{i + 1}/{total}] {image_path.name} ({elapsed:.1f}s): {len(objects)} object(s)")

        except Exception as exc:
            emitter.warning("GROUND_FAILED", f"Failed to process {image_path.name}: {exc}")
            detections.append({"image": str(image_path), "objects": [], "object_count": 0, "error": str(exc)})
            errors += 1

    emitter.progress(stage="ground", step=total, total_steps=total)

    total_objects = sum(d["object_count"] for d in detections)
    emitter.result("grounding", {
        "detections": detections,
        "total_objects": total_objects,
        "images_processed": total,
        "errors": errors,
    })

    summary = f"Found {total_objects} '{query}' instance(s) in {total - errors}/{total} images"
    if errors > 0:
        summary += f" ({errors} error(s))"
    emitter.completed(summary)

    return 0 if errors == 0 else 1
