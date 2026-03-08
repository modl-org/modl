"""Detect adapter — detect faces in images using InsightFace.

Returns bounding boxes, confidence scores, and optionally face embeddings
for identity matching across images.

Reads a detect job spec YAML containing:
  image_paths: list[str]      — paths to images
  detect_type: str            — "face" (default)
  model: str                  — "insightface-buffalo-l" (default)
  model_path: str             — optional local model path
  return_embeddings: bool     — include face embeddings for identity matching
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


def _load_insightface(emitter: EventEmitter, model_path: str | None = None):
    """Load InsightFace face analysis model."""
    from insightface.app import FaceAnalysis

    emitter.info("Loading InsightFace buffalo_l model...")

    kwargs = {}
    if model_path:
        kwargs["root"] = model_path

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"], **kwargs)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def run_detect(config_path: Path, emitter: EventEmitter, model_cache: dict | None = None) -> int:
    """Run face detection on images from a DetectJobSpec YAML file."""
    import yaml
    import numpy as np

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Detect spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    image_paths = spec.get("image_paths", [])
    detect_type = spec.get("detect_type", "face")
    return_embeddings = spec.get("return_embeddings", False)
    model_path = spec.get("model_path")

    if detect_type != "face":
        emitter.error("UNSUPPORTED_TYPE", f"Unsupported detect type: {detect_type}. Only 'face' is supported.", recoverable=False)
        return 2

    images = _resolve_images(image_paths)
    if not images:
        emitter.error("NO_IMAGES", "No valid images found", recoverable=False)
        return 2

    total = len(images)
    emitter.info(f"Found {total} image(s) to analyze")
    emitter.job_started(config=str(config_path))

    try:
        if model_cache is not None and "insightface_app" in model_cache:
            app = model_cache["insightface_app"]
            emitter.info("Using cached InsightFace model")
        else:
            app = _load_insightface(emitter, model_path)
            if model_cache is not None:
                model_cache["insightface_app"] = app
    except Exception as exc:
        emitter.error("MODEL_LOAD_FAILED", f"Failed to load InsightFace: {exc}", recoverable=False)
        return 1

    emitter.info("Model loaded, starting detection...")

    detections = []
    errors = 0

    for i, image_path in enumerate(images):
        emitter.progress(stage="detect", step=i, total_steps=total)

        try:
            import cv2
            t0 = time.time()
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")

            faces = app.get(img)
            elapsed = time.time() - t0

            face_data = []
            for face in faces:
                fd = {
                    "bbox": [round(float(x), 1) for x in face.bbox],
                    "confidence": round(float(face.det_score), 4),
                }
                if return_embeddings and face.embedding is not None:
                    fd["embedding"] = face.embedding.tolist()
                face_data.append(fd)

            detection = {
                "image": str(image_path),
                "face_count": len(faces),
                "faces": face_data,
            }
            detections.append(detection)

            emitter.info(f"[{i + 1}/{total}] {image_path.name} ({elapsed:.1f}s): {len(faces)} face(s)")

        except Exception as exc:
            emitter.warning("DETECT_FAILED", f"Failed to process {image_path.name}: {exc}")
            detections.append({"image": str(image_path), "face_count": 0, "faces": [], "error": str(exc)})
            errors += 1

    emitter.progress(stage="detect", step=total, total_steps=total)

    total_faces = sum(d["face_count"] for d in detections)
    emitter.result("detection", {
        "detections": detections,
        "total_faces": total_faces,
        "images_processed": total,
        "errors": errors,
    })

    summary = f"Detected {total_faces} face(s) in {total - errors}/{total} images"
    if errors > 0:
        summary += f" ({errors} error(s))"
    emitter.completed(summary)

    return 0 if errors == 0 else 1
