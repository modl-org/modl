"""Resize adapter — batch-resize dataset images to training resolution.

Reads a resize job spec YAML containing:
  dataset_path: str   — path to the dataset directory
  resolution: int     — max dimension in pixels (default 1024)
  method: str         — "contain" (fit inside, pad white), "cover" (crop center),
                         or "squish" (stretch to square)

Resizes images in-place, preserving paired .txt captions.
Emits progress + log events.
"""

import os
import time
from pathlib import Path
from typing import List

from modl_worker.protocol import EventEmitter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _find_images(dataset_path: Path) -> List[Path]:
    """Find all images in the dataset."""
    images = []
    for f in sorted(dataset_path.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        images.append(f)
    return images


def _resize_image(image_path: Path, resolution: int, method: str) -> dict:
    """Resize a single image. Returns info dict with old/new dimensions."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Skip if already within resolution
    if max(orig_w, orig_h) <= resolution:
        return {"skipped": True, "orig": (orig_w, orig_h), "new": (orig_w, orig_h)}

    if method == "cover":
        # Crop to fill: resize so shortest side = resolution, then center crop
        scale = resolution / min(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # Center crop to square
        left = (new_w - resolution) // 2
        top = (new_h - resolution) // 2
        img = img.crop((left, top, left + resolution, top + resolution))

    elif method == "squish":
        # Stretch to exact resolution x resolution
        img = img.resize((resolution, resolution), Image.LANCZOS)

    else:  # "contain" (default)
        # Fit inside: resize so longest side = resolution, keep aspect ratio
        scale = resolution / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Save back (preserve format)
    ext = image_path.suffix.lower()
    save_kwargs = {}
    if ext in (".jpg", ".jpeg"):
        save_kwargs["quality"] = 95
        save_kwargs["format"] = "JPEG"
    elif ext == ".png":
        save_kwargs["format"] = "PNG"
    elif ext == ".webp":
        save_kwargs["quality"] = 95
        save_kwargs["format"] = "WEBP"

    img.save(image_path, **save_kwargs)
    final_w, final_h = img.size

    return {"skipped": False, "orig": (orig_w, orig_h), "new": (final_w, final_h)}


def run_resize(config_path: Path, emitter: EventEmitter) -> int:
    """Run batch resize on a dataset from a ResizeJobSpec YAML file."""
    import yaml

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Resize spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    dataset_path = Path(spec.get("dataset_path", ""))
    resolution = int(spec.get("resolution", 1024))
    method = spec.get("method", "contain")

    if not dataset_path.exists() or not dataset_path.is_dir():
        emitter.error("DATASET_NOT_FOUND", f"Dataset directory not found: {dataset_path}", recoverable=False)
        return 2

    images = _find_images(dataset_path)
    if not images:
        emitter.info("No images found in dataset.")
        emitter.completed("No images to resize")
        return 0

    total = len(images)
    emitter.info(f"Resizing {total} image(s) to {resolution}px ({method})")
    emitter.job_started(config=str(config_path))

    resized = 0
    skipped = 0
    errors = 0

    for i, image_path in enumerate(images):
        emitter.progress(stage="resize", step=i, total_steps=total)

        try:
            t0 = time.time()
            result = _resize_image(image_path, resolution, method)
            elapsed = time.time() - t0

            if result["skipped"]:
                skipped += 1
                emitter.info(
                    f"[{i + 1}/{total}] {image_path.name} ({elapsed:.1f}s): "
                    f"already {result['orig'][0]}x{result['orig'][1]}, skipped"
                )
            else:
                resized += 1
                ow, oh = result["orig"]
                nw, nh = result["new"]
                emitter.info(
                    f"[{i + 1}/{total}] {image_path.name} ({elapsed:.1f}s): "
                    f"{ow}x{oh} → {nw}x{nh}"
                )

        except Exception as exc:
            emitter.warning("RESIZE_FAILED", f"Failed to resize {image_path.name}: {exc}")
            errors += 1

    emitter.progress(stage="resize", step=total, total_steps=total)

    summary = f"Resized {resized}/{total} images to {resolution}px"
    if skipped > 0:
        summary += f" ({skipped} already within size)"
    if errors > 0:
        summary += f" ({errors} error(s))"
    emitter.completed(summary)

    return 0 if errors == 0 else 1
