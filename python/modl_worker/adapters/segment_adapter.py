"""Segment adapter — generate masks for targeted inpainting.

Supports multiple methods:
  - bbox: Rectangle mask from bounding box coordinates (with feathering)
  - background: BiRefNet background removal (foreground mask)
  - sam: Segment Anything Model with point/box prompt

Reads a segment job spec YAML containing:
  image_path: str           — path to input image
  output_mask_path: str     — where to write the mask
  method: str               — "bbox", "background", or "sam"
  bbox: [x1, y1, x2, y2]   — bounding box (for bbox/sam methods)
  point: [x, y]             — point prompt (for sam method)
  model: str                — model name
  model_path: str           — optional local model path
  expand_px: int            — expand mask by N pixels
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from modl_worker.protocol import EventEmitter


def _mask_from_bbox(image_path: Path, bbox: list[float], expand_px: int) -> Image.Image:
    """Create a mask from a bounding box with optional expansion and feathering."""
    img = Image.open(image_path)
    w, h = img.size
    mask = Image.new("L", (w, h), 0)

    x1, y1, x2, y2 = bbox
    # Expand
    x1 = max(0, x1 - expand_px)
    y1 = max(0, y1 - expand_px)
    x2 = min(w, x2 + expand_px)
    y2 = min(h, y2 + expand_px)

    # Draw white rectangle
    draw = ImageDraw.Draw(mask)
    draw.rectangle([int(x1), int(y1), int(x2), int(y2)], fill=255)

    # Feather edges with Gaussian blur
    if expand_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=expand_px // 2))

    return mask


def _mask_from_birefnet(
    image_path: Path, emitter: EventEmitter, model_path: str | None = None, model_cache: dict | None = None
) -> Image.Image:
    """Generate foreground mask using BiRefNet loaded from modl store."""
    import torch

    if not model_path or not Path(model_path).exists():
        raise ValueError("BiRefNet weights not found. Run `modl pull birefnet-dis` first.")

    if model_cache is not None and "birefnet_model" in model_cache:
        model = model_cache["birefnet_model"]
        emitter.info("Using cached BiRefNet model")
    else:
        emitter.info("Loading BiRefNet model...")
        try:
            from transformers import AutoModelForImageSegmentation
            model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet", trust_remote_code=True
            )
            # Load local weights
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict, strict=False)
            model = model.cuda().eval()
        except Exception:
            # Fallback: try loading via transformers pipeline with local model
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline(
                "image-segmentation", model="ZhengPeng7/BiRefNet",
                trust_remote_code=True, device="cuda"
            )
            if model_cache is not None:
                model_cache["birefnet_pipe"] = pipe
            img = Image.open(image_path).convert("RGB")
            result = pipe(img)
            if result and isinstance(result, list):
                return result[0]["mask"].convert("L")
            raise ValueError("BiRefNet failed to produce a mask")

        if model_cache is not None:
            model_cache["birefnet_model"] = model

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
    mask = (pred * 255).byte().numpy()
    mask_img = Image.fromarray(mask).resize((w, h), Image.BILINEAR)

    return mask_img


def _mask_from_sam(
    image_path: Path,
    emitter: EventEmitter,
    bbox: list[float] | None = None,
    point: list[float] | None = None,
    model_path: str | None = None,
    model_cache: dict | None = None,
) -> Image.Image:
    """Generate mask using Segment Anything Model."""
    import torch

    if model_cache is not None and "sam_predictor" in model_cache:
        predictor = model_cache["sam_predictor"]
        emitter.info("Using cached SAM predictor")
    else:
        emitter.info("Loading SAM model...")

        sam_checkpoint = model_path
        if not sam_checkpoint or not Path(sam_checkpoint).exists():
            raise ValueError("SAM weights not found. Run `modl pull sam-vit-base` first.")

        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        sam.to("cuda")
        predictor = SamPredictor(sam)
        if model_cache is not None:
            model_cache["sam_predictor"] = predictor

    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    predictor.set_image(img)

    if bbox:
        input_box = np.array(bbox)
        masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
    elif point:
        input_point = np.array([point])
        input_label = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=input_point, point_labels=input_label, multimask_output=True
        )
    else:
        raise ValueError("SAM requires either bbox or point prompt")

    # Use highest-scoring mask
    best_idx = scores.argmax()
    mask_np = (masks[best_idx] * 255).astype(np.uint8)

    return Image.fromarray(mask_np)


def run_segment(config_path: Path, emitter: EventEmitter, model_cache: dict | None = None) -> int:
    """Run segmentation from a SegmentJobSpec YAML file."""
    import yaml

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Segment spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    image_path = Path(spec.get("image_path", ""))
    output_mask_path = Path(spec.get("output_mask_path", ""))
    method = spec.get("method", "bbox")
    bbox = spec.get("bbox")
    point = spec.get("point")
    model_path = spec.get("model_path")
    expand_px = spec.get("expand_px", 10)

    if not image_path.is_file():
        emitter.error("IMAGE_NOT_FOUND", f"Image not found: {image_path}", recoverable=False)
        return 2

    emitter.info(f"Segmenting {image_path.name} using method={method}")
    emitter.job_started(config=str(config_path))

    try:
        t0 = time.time()

        if method == "bbox":
            if not bbox:
                emitter.error("NO_BBOX", "bbox method requires --bbox parameter", recoverable=False)
                return 2
            mask = _mask_from_bbox(image_path, bbox, expand_px)

        elif method == "background":
            mask = _mask_from_birefnet(image_path, emitter, model_path, model_cache=model_cache)

        elif method == "sam":
            mask = _mask_from_sam(image_path, emitter, bbox=bbox, point=point, model_path=model_path, model_cache=model_cache)

        else:
            emitter.error("UNKNOWN_METHOD", f"Unknown segmentation method: {method}", recoverable=False)
            return 2

        elapsed = time.time() - t0

        # Save mask
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(str(output_mask_path))

        emitter.artifact(path=str(output_mask_path))
        emitter.result("segment", {
            "mask_path": str(output_mask_path),
            "method": method,
            "image": str(image_path),
            "elapsed_seconds": round(elapsed, 2),
        })
        emitter.completed(f"Mask saved to {output_mask_path} ({elapsed:.1f}s)")

    except Exception as exc:
        emitter.error("SEGMENT_FAILED", f"Segmentation failed: {exc}", recoverable=False)
        return 1

    return 0
