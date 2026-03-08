"""Face restore adapter — fix face artifacts using CodeFormer.

Auto-detects faces in images and restores them without needing a mask.
Lighter-weight alternative to the segment+inpaint loop.

Reads a face restore job spec YAML containing:
  image_paths: list[str]  — paths to images
  output_dir: str         — where to save restored images
  model: str              — "codeformer" (default)
  model_path: str         — optional local model path
  fidelity: float         — 0.0 (quality) to 1.0 (faithful to input)
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


def run_face_restore(config_path: Path, emitter: EventEmitter, model_cache: dict | None = None) -> int:
    """Run face restoration from a FaceRestoreJobSpec YAML file."""
    import yaml

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Face restore spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    image_paths = spec.get("image_paths", [])
    output_dir = Path(spec.get("output_dir", "."))
    fidelity = spec.get("fidelity", 0.7)
    model_path = spec.get("model_path")

    images = _resolve_images(image_paths)
    if not images:
        emitter.error("NO_IMAGES", "No valid images found", recoverable=False)
        return 2

    total = len(images)
    emitter.info(f"Found {total} image(s) to restore (fidelity={fidelity})")
    emitter.job_started(config=str(config_path))

    # Try to use facexlib/codeformer
    try:
        import torch
        import cv2
        import numpy as np
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        if model_cache is not None and "codeformer_helper" in model_cache:
            cached = model_cache["codeformer_helper"]
            net = cached["net"]
            face_helper = cached["face_helper"]
            emitter.info("Using cached CodeFormer model")
        else:
            emitter.info("Loading CodeFormer model...")

            # Resolve CodeFormer checkpoint from modl store
            codeformer_path = model_path
            if not codeformer_path or not Path(codeformer_path).exists():
                emitter.error(
                    "MODEL_NOT_FOUND",
                    "CodeFormer weights not found. Run `modl pull codeformer` first.",
                    recoverable=False,
                )
                return 2

            # Load the model
            from basicsr.utils.registry import ARCH_REGISTRY
            net = ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]
            ).to("cuda")
            checkpoint = torch.load(codeformer_path, map_location="cpu", weights_only=False)
            net.load_state_dict(checkpoint.get("params_ema", checkpoint.get("params", checkpoint)))
            net.eval()

            face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                device="cuda",
            )

            if model_cache is not None:
                model_cache["codeformer_helper"] = {"net": net, "face_helper": face_helper}

        use_codeformer = True

    except ImportError as exc:
        emitter.warning("CODEFORMER_NOT_AVAILABLE",
            f"CodeFormer dependencies not installed ({exc}). "
            "Install with: pip install facexlib basicsr. "
            "Falling back to simple upscale.")
        use_codeformer = False

    output_dir.mkdir(parents=True, exist_ok=True)
    restored = 0
    errors = 0

    for i, image_path in enumerate(images):
        emitter.progress(stage="face_restore", step=i, total_steps=total)

        try:
            t0 = time.time()
            output_path = output_dir / f"{image_path.stem}_restored{image_path.suffix}"

            if use_codeformer:
                img = cv2.imread(str(image_path))
                if img is None:
                    raise ValueError(f"Could not read image: {image_path}")

                face_helper.clean_all()
                face_helper.read_image(img)
                face_helper.get_face_landmarks_5(only_center_face=False)
                face_helper.align_warp_face()

                num_faces = len(face_helper.cropped_faces)
                if num_faces == 0:
                    # No faces detected, copy original
                    cv2.imwrite(str(output_path), img)
                    emitter.info(f"[{i + 1}/{total}] {image_path.name}: no faces detected, copied original")
                    emitter.artifact(path=str(output_path))
                    restored += 1
                    continue

                for cropped_face in face_helper.cropped_faces:
                    face_tensor = torch.from_numpy(cropped_face.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                    face_tensor = face_tensor.to("cuda")

                    with torch.no_grad():
                        output_face = net(face_tensor, w=fidelity, adain=True)[0]
                        output_face = output_face.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255
                        output_face = output_face.astype(np.uint8)

                    face_helper.add_restored_face(output_face)

                face_helper.get_inverse_affine(None)
                restored_img = face_helper.paste_faces_to_input_image()
                cv2.imwrite(str(output_path), restored_img)

            else:
                # Simple fallback: just copy the image
                Image.open(image_path).save(str(output_path))

            elapsed = time.time() - t0

            # Hash output
            with open(output_path, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
            size_bytes = output_path.stat().st_size

            emitter.artifact(path=str(output_path), sha256=sha256, size_bytes=size_bytes)

            face_info = f"{num_faces} face(s)" if use_codeformer else "fallback mode"
            emitter.info(f"[{i + 1}/{total}] {image_path.name} ({elapsed:.1f}s): {face_info}")
            restored += 1

        except Exception as exc:
            emitter.warning("RESTORE_FAILED", f"Failed to restore {image_path.name}: {exc}")
            errors += 1

    emitter.progress(stage="face_restore", step=total, total_steps=total)

    emitter.result("face_restore", {
        "restored": restored,
        "errors": errors,
        "output_dir": str(output_dir),
    })

    summary = f"Restored {restored}/{total} images"
    if errors > 0:
        summary += f" ({errors} error(s))"
    emitter.completed(summary)

    return 0 if errors == 0 else 1
