"""Compare adapter — compute CLIP cosine similarity between images.

Supports three modes:
  - Pairwise: compare two images
  - Reference: compare all images against a reference
  - Matrix: compute NxN similarity matrix for a batch

Reads a compare job spec YAML containing:
  image_paths: list[str]      — paths to images
  reference_path: str         — optional reference image (one-vs-many mode)
  model: str                  — "clip-vit-large-patch14" (default)
  clip_model_path: str        — optional local path to CLIP model
"""

import time
from pathlib import Path

import torch
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


def _load_clip(emitter: EventEmitter, model_path: str | None = None):
    """Load CLIP ViT-L/14 model and processor."""
    from transformers import CLIPModel, CLIPProcessor

    model_id = model_path or "openai/clip-vit-large-patch14"
    source = "local" if model_path else "HuggingFace Hub"
    emitter.info(f"Loading CLIP ViT-L/14 from {source}: {model_id}")

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to("cuda").eval()

    return model, processor


def _embed_image(model, processor, image_path: Path) -> torch.Tensor:
    """Get normalized CLIP embedding for a single image."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0)


def run_compare(config_path: Path, emitter: EventEmitter, model_cache: dict | None = None) -> int:
    """Run image comparison from a CompareJobSpec YAML file."""
    import yaml

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Compare spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    image_paths = spec.get("image_paths", [])
    reference_path = spec.get("reference_path")
    clip_model_path = spec.get("clip_model_path")

    images = _resolve_images(image_paths)
    if not images:
        emitter.error("NO_IMAGES", "No valid images found", recoverable=False)
        return 2

    if reference_path:
        ref = Path(reference_path)
        if not ref.is_file():
            emitter.error("REF_NOT_FOUND", f"Reference image not found: {reference_path}", recoverable=False)
            return 2

    total = len(images) + (1 if reference_path else 0)
    emitter.info(f"Found {len(images)} image(s) to compare")
    emitter.job_started(config=str(config_path))

    try:
        if model_cache is not None and "clip_model" in model_cache:
            clip_model = model_cache["clip_model"]
            clip_processor = model_cache["clip_preprocess"]
            emitter.info("Using cached CLIP model")
        else:
            clip_model, clip_processor = _load_clip(emitter, clip_model_path)
            if model_cache is not None:
                model_cache["clip_model"] = clip_model
                model_cache["clip_preprocess"] = clip_processor
    except Exception as exc:
        emitter.error("MODEL_LOAD_FAILED", f"Failed to load CLIP: {exc}", recoverable=False)
        return 1

    emitter.info("Model loaded, computing embeddings...")

    # Compute embeddings
    embeddings = []
    image_names = []
    step = 0

    if reference_path:
        ref_path = Path(reference_path)
        emitter.progress(stage="compare", step=step, total_steps=total)
        ref_embedding = _embed_image(clip_model, clip_processor, ref_path)
        step += 1

    for i, img_path in enumerate(images):
        emitter.progress(stage="compare", step=step, total_steps=total)
        try:
            emb = _embed_image(clip_model, clip_processor, img_path)
            embeddings.append(emb)
            image_names.append(img_path.name)
        except Exception as exc:
            emitter.warning("EMBED_FAILED", f"Failed to embed {img_path.name}: {exc}")
            image_names.append(img_path.name)
            embeddings.append(None)
        step += 1

    emitter.progress(stage="compare", step=total, total_steps=total)

    # Compute similarities
    if reference_path:
        # Reference mode: compare each image to the reference
        similarities = []
        for i, emb in enumerate(embeddings):
            if emb is not None:
                sim = torch.cosine_similarity(ref_embedding.unsqueeze(0), emb.unsqueeze(0)).item()
                similarities.append(round(sim, 4))
            else:
                similarities.append(None)

        valid_sims = [s for s in similarities if s is not None]
        mean_sim = sum(valid_sims) / len(valid_sims) if valid_sims else 0.0

        emitter.result("comparison", {
            "mode": "reference",
            "reference": ref_path.name,
            "images": image_names,
            "similarities": similarities,
            "mean_similarity": round(mean_sim, 4),
        })

        emitter.completed(f"Compared {len(images)} images to reference (mean similarity: {mean_sim:.3f})")

    elif len(images) == 2 and embeddings[0] is not None and embeddings[1] is not None:
        # Pairwise mode
        sim = torch.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()

        emitter.result("comparison", {
            "mode": "pairwise",
            "images": image_names,
            "similarity": round(sim, 4),
        })

        emitter.completed(f"Similarity: {sim:.4f}")

    else:
        # Matrix mode
        n = len(embeddings)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if embeddings[i] is not None and embeddings[j] is not None:
                    sim = torch.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
                    matrix[i][j] = round(sim, 4)

        # Mean off-diagonal similarity
        off_diag = []
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i][j] > 0:
                    off_diag.append(matrix[i][j])
        mean_sim = sum(off_diag) / len(off_diag) if off_diag else 0.0

        emitter.result("comparison", {
            "mode": "matrix",
            "images": image_names,
            "similarities": matrix,
            "mean_similarity": round(mean_sim, 4),
        })

        emitter.completed(f"Computed {n}x{n} similarity matrix (mean: {mean_sim:.3f})")

    return 0
