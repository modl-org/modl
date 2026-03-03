"""Tag adapter — auto-generate structured tags for dataset images using VL models.

Reads a tag job spec YAML containing:
  dataset_path: str   — path to the dataset directory
  model: str          — "florence-2" (default) or "wd-tagger"
  overwrite: bool     — re-tag images that already have .txt files

Unlike captioning (which produces free-form sentences), tagging produces
comma-separated labels that are more useful for LoRA training captions.

Florence-2 uses the <MORE_DETAILED_CAPTION> + <OD> tasks to produce rich tags.
WD-Tagger uses SwinV2-based tagger for anime/booru-style tags.

When a .txt file already exists and overwrite=True, the existing content is
**prepended** as a tag prefix (preserving user-supplied trigger words and
folder tags from dataset creation).
"""

import hashlib
import os
import time
from pathlib import Path
from typing import List, Tuple

from modl_worker.protocol import EventEmitter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _find_images(dataset_path: Path, overwrite: bool) -> List[Tuple[Path, str]]:
    """Find images that need tagging. Returns (image_path, existing_caption_or_empty)."""
    images = []
    for f in sorted(dataset_path.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption_path = f.with_suffix(".txt")
        existing = ""
        if caption_path.exists():
            if not overwrite:
                continue
            existing = caption_path.read_text(encoding="utf-8").strip()
        images.append((f, existing))
    return images


# ---------------------------------------------------------------------------
# Florence-2 tagger
# ---------------------------------------------------------------------------


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
    ).to("cuda")

    return model, processor


def _tag_florence2(model, processor, image_path: Path) -> str:
    """Generate structured tags for an image using Florence-2.

    Uses multiple tasks and merges the results:
    - <MORE_DETAILED_CAPTION> for a rich description
    - <OD> (object detection) for subject identification
    """
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    tags = []

    # 1) Detailed caption → extract key phrases
    for task in ("<MORE_DETAILED_CAPTION>", "<OD>"):
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
                use_cache=False,
            )
        text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            text, task=task, image_size=(image.width, image.height)
        )

        if task == "<MORE_DETAILED_CAPTION>":
            caption = parsed.get(task, "").strip()
            # Keep the full detailed caption as the primary tag line
            if caption:
                tags.append(caption)
        elif task == "<OD>":
            # Extract detected object labels
            od_result = parsed.get(task, {})
            labels = od_result.get("labels", [])
            # Deduplicate and lowercase
            seen = set()
            for label in labels:
                l = label.strip().lower()
                if l and l not in seen:
                    seen.add(l)
                    tags.append(l)

    return ", ".join(tags)


# ---------------------------------------------------------------------------
# WD Tagger (SwinV2-based, good for anime/illustration styles)
# ---------------------------------------------------------------------------


_WD_MODEL = None
_WD_LABELS = None


def _load_wd_tagger(emitter: EventEmitter):
    """Load WD SwinV2 tagger."""
    global _WD_MODEL, _WD_LABELS

    import numpy as np

    model_id = "SmilingWolf/wd-swinv2-tagger-v3"
    emitter.info(f"Loading {model_id}...")

    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        import pandas as pd

        # Download model and labels
        model_path = hf_hub_download(model_id, "model.onnx")
        labels_path = hf_hub_download(model_id, "selected_tags.csv")

        _WD_MODEL = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        _WD_LABELS = pd.read_csv(labels_path)

        emitter.info(f"Loaded {model_id} ({len(_WD_LABELS)} tags)")
    except ImportError as e:
        emitter.error(
            "MISSING_DEPS",
            f"WD tagger requires onnxruntime and pandas: {e}. "
            f"Install with: pip install onnxruntime-gpu pandas",
            recoverable=False,
        )
        raise


def _tag_wd(image_path: Path, threshold: float = 0.35) -> str:
    """Generate booru-style tags using WD tagger."""
    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    # WD tagger expects 448x448
    image = image.resize((448, 448), Image.LANCZOS)
    img_array = np.array(image).astype(np.float32)
    # Normalize and add batch dimension
    img_array = img_array[:, :, ::-1]  # RGB -> BGR
    img_array = np.expand_dims(img_array, 0)

    input_name = _WD_MODEL.get_inputs()[0].name
    output_name = _WD_MODEL.get_outputs()[0].name
    probs = _WD_MODEL.run([output_name], {input_name: img_array})[0][0]

    # Filter tags above threshold
    tags = []
    for i, prob in enumerate(probs):
        if prob >= threshold and i < len(_WD_LABELS):
            tag = _WD_LABELS.iloc[i]["name"]
            tag = tag.replace("_", " ").strip()
            if tag:
                tags.append(tag)

    return ", ".join(tags)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_tag(config_path: Path, emitter: EventEmitter) -> int:
    """Run auto-tagging on a dataset from a TagJobSpec YAML file."""
    import yaml

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Tag spec not found: {config_path}", recoverable=False)
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

    if not dataset_path.exists() or not dataset_path.is_dir():
        emitter.error(
            "DATASET_NOT_FOUND",
            f"Dataset directory not found: {dataset_path}",
            recoverable=False,
        )
        return 2

    images = _find_images(dataset_path, overwrite)
    if not images:
        emitter.info("All images already have tags. Nothing to do.")
        emitter.completed("No images needed tagging")
        return 0

    total = len(images)
    emitter.info(f"Found {total} image(s) to tag using {model_name}")
    emitter.job_started(config=str(config_path))

    # Load model
    try:
        if model_name.lower() in ("florence-2", "florence2", "florence"):
            model, processor = _load_florence2(emitter, model_path)
            tag_fn = lambda img_path: _tag_florence2(model, processor, img_path)
        elif model_name.lower() in ("wd-tagger", "wd", "wdtagger"):
            _load_wd_tagger(emitter)
            tag_fn = lambda img_path: _tag_wd(img_path)
        else:
            emitter.error(
                "UNKNOWN_MODEL",
                f"Unknown tagging model: {model_name}. Use 'florence-2' or 'wd-tagger'.",
                recoverable=False,
            )
            return 2
    except Exception as exc:
        emitter.error(
            "MODEL_LOAD_FAILED",
            f"Failed to load tagging model: {exc}",
            recoverable=False,
        )
        return 1

    emitter.info("Model loaded, starting tagging...")

    tagged = 0
    errors = 0

    for i, (image_path, existing_caption) in enumerate(images):
        emitter.progress(stage="tag", step=i, total_steps=total)

        try:
            t0 = time.time()
            new_tags = tag_fn(image_path)
            elapsed = time.time() - t0

            # Merge: if there's an existing caption (e.g. folder tag or trigger
            # word), prepend it. Avoid duplicating if existing is already a
            # substring.
            if existing_caption:
                # Check if existing content looks like a simple folder tag
                # (single word/phrase, no comma-separated list) vs a full caption
                if "," not in existing_caption and len(existing_caption.split()) <= 3:
                    # Simple tag prefix — prepend
                    final = f"{existing_caption}, {new_tags}"
                else:
                    # Overwrite with new tags
                    final = new_tags
            else:
                final = new_tags

            # Write tag file
            caption_path = image_path.with_suffix(".txt")
            caption_path.write_text(final, encoding="utf-8")

            sha256 = hashlib.sha256(final.encode("utf-8")).hexdigest()
            size_bytes = caption_path.stat().st_size
            emitter.artifact(path=str(caption_path), sha256=sha256, size_bytes=size_bytes)

            emitter.info(f"[{i + 1}/{total}] {image_path.name} ({elapsed:.1f}s): {final}")
            tagged += 1

        except Exception as exc:
            emitter.warning("TAG_FAILED", f"Failed to tag {image_path.name}: {exc}")
            errors += 1

    emitter.progress(stage="tag", step=total, total_steps=total)

    summary = f"Tagged {tagged}/{total} images"
    if errors > 0:
        summary += f" ({errors} error(s))"
    emitter.completed(summary)

    return 0 if errors == 0 else 1
