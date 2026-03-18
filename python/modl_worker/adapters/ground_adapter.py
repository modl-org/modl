"""Ground adapter — text-grounded object detection using Qwen2.5-VL.

Locates objects matching a text query in images using a vision-language model.
Returns bounding boxes for each detected instance.

Reads a ground job spec YAML containing:
  image_paths: list[str]    — paths to images
  query: str                — text query like "coffee cup" or "person"
  model: str                — "qwen25-vl-3b" (default) or "qwen25-vl-7b"
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


def _parse_detections(response_text: str, query: str, threshold: float) -> list[dict]:
    """Parse bounding box detections from VL model response text.

    Handles multiple response formats:
    - Clean JSON array
    - Markdown-fenced ```json blocks
    - Truncated responses (parse individual objects via regex)
    - Duplicate keys in objects (common with Qwen3-VL)
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", response_text)
    text = text.strip()

    # Try parsing the full JSON array first
    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        try:
            raw = json.loads(json_match.group())
            if isinstance(raw, list):
                objects = _extract_objects(raw, query, threshold)
                if objects:
                    return objects
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: extract individual {bbox_2d: [...]} objects via regex
    # This handles truncated responses where the array isn't closed
    objects = []
    for m in re.finditer(r'\{[^{}]*"bbox_2d"\s*:\s*\[([^\]]+)\][^{}]*\}', text):
        try:
            coords = [float(x.strip()) for x in m.group(1).split(",")]
            if len(coords) == 4:
                # Extract label if present
                label_m = re.search(r'"label"\s*:\s*"([^"]*)"', m.group(0))
                label = label_m.group(1) if label_m else query
                objects.append({
                    "label": str(label),
                    "bbox": [round(c, 1) for c in coords],
                    "confidence": 1.0,
                })
        except (ValueError, IndexError):
            continue

    return objects


def _extract_objects(raw: list, query: str, threshold: float) -> list[dict]:
    """Extract bbox objects from a parsed JSON list."""
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


def run_ground(config_path: Path, emitter: EventEmitter) -> int:
    """Run text-grounded object detection on images from a GroundJobSpec YAML file."""
    import yaml
    from modl_worker.adapters.vl_common import load_qwen_vl, run_vl_inference

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
    model_id = spec.get("model") or "qwen25-vl-3b"
    threshold = float(spec.get("threshold") or 0.0)

    if not query:
        emitter.error("NO_QUERY", "No query provided for grounded detection", recoverable=False)
        return 2

    images = _resolve_images(image_paths)
    if not images:
        emitter.error("NO_IMAGES", "No valid images found", recoverable=False)
        return 2

    total = len(images)
    emitter.info(f"Found {total} image(s) to analyze for '{query}' using {model_id}")
    emitter.job_started(config=str(config_path))

    try:
        model, processor = load_qwen_vl(emitter, model_id)
    except Exception as exc:
        emitter.error("MODEL_LOAD_FAILED", f"Failed to load VL model: {exc}", recoverable=False)
        return 1

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
            response = run_vl_inference(model, processor, str(image_path), prompt)
            elapsed = time.time() - t0

            objects = _parse_detections(response, query, threshold)

            detections.append({
                "image": str(image_path),
                "objects": objects,
                "object_count": len(objects),
            })

            emitter.info(f"[{i + 1}/{total}] {image_path.name} ({elapsed:.1f}s): {len(objects)} object(s)")

        except Exception as exc:
            emitter.warning("GROUND_FAILED", f"Failed to process {image_path.name}: {exc}")
            detections.append({"image": str(image_path), "objects": [], "object_count": 0})
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
