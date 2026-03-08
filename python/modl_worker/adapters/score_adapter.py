"""Score adapter — predict aesthetic quality scores for images.

Uses LAION aesthetic predictor v2 (tiny MLP) on top of CLIP ViT-L/14
embeddings. Returns scores on a 1-10 scale per image.

Reads a score job spec YAML containing:
  image_paths: list[str]  — paths to images to score
  model: str              — "laion-aesthetic-v2" (default)
  clip_model_path: str    — optional local path to CLIP model
  predictor_path: str     — optional local path to aesthetic predictor weights
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from modl_worker.protocol import EventEmitter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class AestheticPredictor(nn.Module):
    """LAION aesthetic predictor — a single linear layer on CLIP embeddings."""

    def __init__(self, input_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


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


def _load_aesthetic_predictor(
    emitter: EventEmitter, predictor_path: str | None = None
) -> AestheticPredictor:
    """Load the LAION aesthetic predictor MLP."""
    predictor = AestheticPredictor(768)  # CLIP ViT-L/14 has 768-dim embeddings

    if predictor_path:
        emitter.info(f"Loading aesthetic predictor from: {predictor_path}")
        state_dict = torch.load(predictor_path, map_location="cpu", weights_only=True)
    else:
        emitter.info("Downloading LAION aesthetic predictor v2...")
        url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=True)

    predictor.load_state_dict(state_dict)
    predictor.to("cuda").eval()
    return predictor


def run_score(config_path: Path, emitter: EventEmitter, model_cache: dict | None = None) -> int:
    """Run aesthetic scoring on images from a ScoreJobSpec YAML file."""
    import yaml

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Score spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    image_paths = spec.get("image_paths", [])
    clip_model_path = spec.get("clip_model_path")
    predictor_path = spec.get("predictor_path")

    images = _resolve_images(image_paths)
    if not images:
        emitter.error("NO_IMAGES", "No valid images found to score", recoverable=False)
        return 2

    total = len(images)
    emitter.info(f"Found {total} image(s) to score")
    emitter.job_started(config=str(config_path))

    # Load models (use cache if available)
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

        if model_cache is not None and "aesthetic_predictor" in model_cache:
            predictor = model_cache["aesthetic_predictor"]
            emitter.info("Using cached aesthetic predictor")
        else:
            predictor = _load_aesthetic_predictor(emitter, predictor_path)
            if model_cache is not None:
                model_cache["aesthetic_predictor"] = predictor
    except Exception as exc:
        emitter.error("MODEL_LOAD_FAILED", f"Failed to load scoring models: {exc}", recoverable=False)
        return 1

    emitter.info("Models loaded, starting scoring...")

    scores = []
    errors = 0

    for i, image_path in enumerate(images):
        emitter.progress(stage="score", step=i, total_steps=total)

        try:
            t0 = time.time()
            image = Image.open(image_path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt").to("cuda")

            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                score = predictor(image_features).item()

            elapsed = time.time() - t0
            scores.append({"image": str(image_path), "score": round(score, 4)})

            emitter.info(f"[{i + 1}/{total}] {image_path.name} ({elapsed:.1f}s): score={score:.2f}")

        except Exception as exc:
            emitter.warning("SCORE_FAILED", f"Failed to score {image_path.name}: {exc}")
            scores.append({"image": str(image_path), "score": None, "error": str(exc)})
            errors += 1

    emitter.progress(stage="score", step=total, total_steps=total)

    # Emit structured result
    valid_scores = [s["score"] for s in scores if s["score"] is not None]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    emitter.result("score", {
        "scores": scores,
        "mean_score": round(mean_score, 4),
        "count": total,
        "errors": errors,
    })

    summary = f"Scored {total - errors}/{total} images (mean: {mean_score:.2f})"
    if errors > 0:
        summary += f" ({errors} error(s))"
    emitter.completed(summary)

    return 0 if errors == 0 else 1
