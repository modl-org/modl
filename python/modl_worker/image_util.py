"""Shared image loading and output utilities."""

import hashlib
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageOps


# Valid image extensions — canonical set used across all adapters
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_image(path: str | Path, mode: str = "RGB") -> Image.Image:
    """Open an image, apply EXIF orientation, and convert to the given mode.

    Phone cameras typically store images in landscape with an EXIF orientation
    tag. ``ImageOps.exif_transpose`` physically rotates the pixels to match the
    intended orientation and strips the tag so downstream code sees the correct
    geometry.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if mode and img.mode != mode:
        img = img.convert(mode)
    return img


def resolve_images(image_paths: list[str], extensions: set[str] | None = None) -> list[Path]:
    """Expand directories and filter to valid image files.

    Args:
        image_paths: List of file or directory paths.
        extensions: Set of lowercase extensions to accept. Defaults to IMAGE_EXTENSIONS.
    """
    exts = extensions or IMAGE_EXTENSIONS
    result = []
    for p_str in image_paths:
        p = Path(p_str)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.is_file() and f.suffix.lower() in exts:
                    result.append(f)
        elif p.is_file() and p.suffix.lower() in exts:
            result.append(p)
    return result


def save_and_emit_artifact(
    image: Image.Image,
    output_dir: str,
    emitter,
    *,
    index: int = 0,
    count: int = 1,
    metadata: dict | None = None,
    stage: str = "generate",
    elapsed: float | None = None,
) -> str | None:
    """Save a PIL image as PNG with metadata, hash it, and emit artifact + progress events.

    Args:
        image: PIL Image to save.
        output_dir: Directory to write the file into.
        emitter: EventEmitter for artifact/progress events.
        index: Image index within a batch (0-based).
        count: Total images in the batch.
        metadata: Dict to embed as ``modl_metadata`` PNG text chunk.
        stage: Stage name for progress events.
        elapsed: Time taken for this image (used for ETA calculation).

    Returns:
        The output filepath, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}_{index:03d}.png" if count > 1 else f"{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Build PNG metadata
    save_kwargs = {}
    if metadata:
        try:
            from PIL.PngImagePlugin import PngInfo
            pnginfo = PngInfo()
            pnginfo.add_text("Software", "modl.run")
            pnginfo.add_text("Comment", f"{stage}d with modl.run")
            pnginfo.add_text("modl_metadata", json.dumps(metadata, separators=(",", ":")))
            save_kwargs["pnginfo"] = pnginfo
        except Exception:
            pass  # Non-fatal: save image even if metadata embedding fails

    image.save(filepath, **save_kwargs)

    # Hash the output
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    size_bytes = os.path.getsize(filepath)

    emitter.artifact(path=filepath, sha256=sha256.hexdigest(), size_bytes=size_bytes)
    emitter.progress(
        stage=stage,
        step=index + 1,
        total_steps=count,
        eta_seconds=elapsed * (count - index - 1) if elapsed and count > 1 else None,
    )
    emitter.info(f"Image {index + 1}/{count}: {filepath}" + (f" ({elapsed:.1f}s)" if elapsed else ""))

    return filepath
