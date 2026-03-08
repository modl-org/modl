"""Remove-bg adapter — remove background and output transparent RGBA PNG.

Uses BiRefNet for foreground segmentation, then applies the mask as an
alpha channel to produce a transparent PNG.

Reads a RemoveBgJobSpec YAML containing:
  image_paths: list[str]  — paths to images
  output_dir: str         — where to save transparent PNGs
  model_path: str         — path to BiRefNet weights in modl store
"""

import hashlib
import time
from pathlib import Path

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


def _load_birefnet(model_path: str, emitter: EventEmitter, model_cache: dict | None = None):
    """Load BiRefNet model, with optional caching."""
    import torch

    if model_cache is not None and "birefnet_model" in model_cache:
        emitter.info("Using cached BiRefNet model")
        return model_cache["birefnet_model"]

    emitter.info("Loading BiRefNet model...")
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    # Load local weights from modl store
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    if model_cache is not None:
        model_cache["birefnet_model"] = model

    return model


def _remove_background(image_path: Path, model, emitter: EventEmitter) -> Image.Image:
    """Run BiRefNet on an image and return RGBA with transparent background."""
    import torch
    from torchvision import transforms

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    mask_np = (pred * 255).byte().numpy()
    mask = Image.fromarray(mask_np).resize((w, h), Image.BILINEAR)

    # Compose RGBA: original RGB + mask as alpha
    rgba = img.copy().convert("RGBA")
    rgba.putalpha(mask)

    return rgba


def run_remove_bg(config_path: Path, emitter: EventEmitter, model_cache: dict | None = None) -> int:
    """Run background removal from a RemoveBgJobSpec YAML file."""
    import yaml

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Remove-bg spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    image_paths = spec.get("image_paths", [])
    output_dir = Path(spec.get("output_dir", "."))
    model_path = spec.get("model_path")

    if not model_path or not Path(model_path).exists():
        emitter.error(
            "MODEL_NOT_FOUND",
            "BiRefNet weights not found. Run `modl pull birefnet-dis` first.",
            recoverable=False,
        )
        return 2

    images = _resolve_images(image_paths)
    if not images:
        emitter.error("NO_IMAGES", "No valid images found", recoverable=False)
        return 2

    total = len(images)
    emitter.info(f"Removing background from {total} image(s)")
    emitter.job_started(config=str(config_path))

    # Load model
    try:
        model = _load_birefnet(model_path, emitter, model_cache)
    except Exception as exc:
        emitter.error("MODEL_LOAD_FAILED", f"Failed to load BiRefNet: {exc}", recoverable=False)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    errors = 0

    for i, image_path in enumerate(images):
        emitter.progress(stage="remove-bg", step=i, total_steps=total)

        try:
            t0 = time.time()
            rgba = _remove_background(image_path, model, emitter)

            output_path = output_dir / f"{image_path.stem}_nobg.png"
            rgba.save(str(output_path))
            elapsed = time.time() - t0

            with open(output_path, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
            size_bytes = output_path.stat().st_size

            emitter.artifact(path=str(output_path), sha256=sha256, size_bytes=size_bytes)
            emitter.info(f"[{i + 1}/{total}] {image_path.name} ({elapsed:.1f}s)")
            processed += 1

        except Exception as exc:
            emitter.warning("REMOVE_BG_FAILED", f"Failed on {image_path.name}: {exc}")
            errors += 1

    emitter.progress(stage="remove-bg", step=total, total_steps=total)

    emitter.result("remove-bg", {
        "processed": processed,
        "errors": errors,
        "output_dir": str(output_dir),
    })

    summary = f"Removed background from {processed}/{total} image(s)"
    if errors > 0:
        summary += f" ({errors} error(s))"
    emitter.completed(summary)

    return 0 if errors == 0 else 1
