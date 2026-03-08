"""Upscale adapter — magnify images using Real-ESRGAN.

Reads an upscale job spec YAML containing:
  image_paths: list[str]  — paths to images to upscale
  output_dir: str         — where to save upscaled images
  scale: int              — upscaling factor (2 or 4, default: 4)
  model_path: str         — path to .pth weights in modl store
"""

import hashlib
import time
from pathlib import Path

import numpy as np
from PIL import Image

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


def _load_realesrgan(model_path: str, scale: int, emitter: EventEmitter):
    """Load Real-ESRGAN model from a local .pth file."""
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as exc:
        raise ImportError(
            f"Missing dependency: {exc}. Install with: pip install realesrgan"
        ) from exc

    # RealESRGAN x4plus uses RRDBNet with these params
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
    )
    return upsampler


def run_upscale(config_path: Path, emitter: EventEmitter, model_cache: dict | None = None) -> int:
    """Run upscaling from an UpscaleJobSpec YAML file."""
    import yaml

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Upscale spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    image_paths = spec.get("image_paths", [])
    output_dir = Path(spec.get("output_dir", "."))
    scale = spec.get("scale", 4)
    model_path = spec.get("model_path")

    if not model_path or not Path(model_path).exists():
        emitter.error(
            "MODEL_NOT_FOUND",
            f"Upscaler weights not found: {model_path}. Run `modl pull realesrgan-x4plus` first.",
            recoverable=False,
        )
        return 2

    images = _resolve_images(image_paths)
    if not images:
        emitter.error("NO_IMAGES", "No valid images found to upscale", recoverable=False)
        return 2

    total = len(images)
    emitter.info(f"Found {total} image(s) to upscale ({scale}x)")
    emitter.job_started(config=str(config_path))

    # Load model (use cache if available)
    cache_key = f"realesrgan_{scale}x"
    try:
        if model_cache is not None and cache_key in model_cache:
            upsampler = model_cache[cache_key]
            emitter.info("Using cached Real-ESRGAN model")
        else:
            emitter.info(f"Loading Real-ESRGAN ({scale}x)...")
            upsampler = _load_realesrgan(model_path, scale, emitter)
            if model_cache is not None:
                model_cache[cache_key] = upsampler
    except Exception as exc:
        emitter.error("MODEL_LOAD_FAILED", f"Failed to load upscaler: {exc}", recoverable=False)
        return 1

    emitter.info("Model loaded, starting upscaling...")

    output_dir.mkdir(parents=True, exist_ok=True)
    upscaled = 0
    errors = 0

    for i, image_path in enumerate(images):
        emitter.progress(stage="upscale", step=i, total_steps=total)

        try:
            t0 = time.time()

            # Load with OpenCV (Real-ESRGAN expects BGR numpy array)
            import cv2
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Run upscaling
            output_img, _ = upsampler.enhance(img, outscale=scale)

            # Save output
            output_path = output_dir / f"{image_path.stem}_{scale}x.png"
            cv2.imwrite(str(output_path), output_img)

            elapsed = time.time() - t0

            # Hash output for artifact tracking
            with open(output_path, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
            size_bytes = output_path.stat().st_size

            h, w = img.shape[:2]
            oh, ow = output_img.shape[:2]

            emitter.artifact(path=str(output_path), sha256=sha256, size_bytes=size_bytes)
            emitter.info(f"[{i + 1}/{total}] {image_path.name} ({w}x{h} → {ow}x{oh}, {elapsed:.1f}s)")
            upscaled += 1

        except Exception as exc:
            emitter.warning("UPSCALE_FAILED", f"Failed to upscale {image_path.name}: {exc}")
            errors += 1

    emitter.progress(stage="upscale", step=total, total_steps=total)

    emitter.result("upscale", {
        "upscaled": upscaled,
        "errors": errors,
        "output_dir": str(output_dir),
        "scale": scale,
    })

    summary = f"Upscaled {upscaled}/{total} images ({scale}x)"
    if errors > 0:
        summary += f" ({errors} error(s))"
    emitter.completed(summary)

    return 0 if errors == 0 else 1
