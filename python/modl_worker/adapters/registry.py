"""Single adapter registry — the source of truth for all adapters.

Both ``main.py`` (one-shot CLI) and ``serve.py`` (persistent daemon)
import from here instead of maintaining separate dispatch maps.

Each adapter is registered with:
  - ``name``:           CLI command name (e.g. "generate", "score")
  - ``run_fn``:         One-shot entry point  ``(Path, EventEmitter) -> int``
  - ``daemon``:         Whether the daemon should handle this command
  - ``daemon_handler``: "pipeline" (gen/edit with model cache) or "analysis" (utility cache)
  - ``cache_keys``:     Utility cache keys this adapter uses (for analysis adapters)
  - ``description``:    Help text for argparse
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class AdapterEntry:
    name: str
    run_fn: Callable
    description: str = ""
    daemon: bool = False
    daemon_handler: str = "analysis"  # "pipeline" | "analysis"
    cache_keys: list[str] = field(default_factory=list)


# Populated by _register_all() on first access.
_ADAPTERS: dict[str, AdapterEntry] | None = None


def get_adapters() -> dict[str, AdapterEntry]:
    """Return the adapter registry, lazily initialised."""
    global _ADAPTERS
    if _ADAPTERS is None:
        _ADAPTERS = _register_all()
    return _ADAPTERS


def _register_all() -> dict[str, AdapterEntry]:
    """Import adapters and build the registry.

    Kept in one place so that (a) the import is deferred until needed and
    (b) every adapter is visible in a single table.
    """
    from modl_worker.adapters import (
        run_train, run_generate, run_edit, run_caption, run_resize, run_tag,
        run_score, run_detect, run_compare,
        run_segment, run_face_restore, run_upscale, run_remove_bg,
        run_face_crop, run_ground, run_describe, run_vl_tag,
        run_preprocess, run_lanpaint,
    )

    entries = [
        # ── Primary actions ──────────────────────────────────────────────
        AdapterEntry("train",        run_train,        "Run training adapter"),
        AdapterEntry("generate",     run_generate,     "Run inference/generation adapter",
                     daemon=True, daemon_handler="pipeline"),
        AdapterEntry("edit",         run_edit,         "Run image editing adapter",
                     daemon=True, daemon_handler="pipeline"),

        # ── Dataset / captioning (one-shot only) ─────────────────────────
        AdapterEntry("caption",      run_caption,      "Run auto-captioning adapter"),
        AdapterEntry("resize",       run_resize,       "Run batch image resize"),
        AdapterEntry("tag",          run_tag,          "Run auto-tagging adapter"),

        # ── Analysis (daemon-capable with utility cache) ─────────────────
        AdapterEntry("score",        run_score,        "Run aesthetic scoring adapter",
                     daemon=True, cache_keys=["clip_model", "clip_preprocess", "aesthetic_predictor"]),
        AdapterEntry("detect",       run_detect,       "Run face detection adapter",
                     daemon=True, cache_keys=["insightface_app"]),
        AdapterEntry("compare",      run_compare,      "Run image comparison adapter",
                     daemon=True, cache_keys=["clip_model", "clip_preprocess"]),
        AdapterEntry("segment",      run_segment,      "Run image segmentation adapter",
                     daemon=True, cache_keys=["birefnet_model", "sam_model"]),
        AdapterEntry("face-restore", run_face_restore,  "Run face restoration adapter",
                     daemon=True, cache_keys=["codeformer_model"]),
        AdapterEntry("upscale",      run_upscale,      "Run image upscaling adapter",
                     daemon=True, cache_keys=["upscaler_model"]),
        AdapterEntry("remove-bg",    run_remove_bg,    "Run background removal adapter",
                     daemon=True, cache_keys=["birefnet_model"]),
        AdapterEntry("preprocess",   run_preprocess,   "Run control image preprocessing",
                     daemon=True, cache_keys=[]),

        # ── Vision-language (one-shot only) ──────────────────────────────
        AdapterEntry("face-crop",    run_face_crop,    "Detect faces and create close-up crops"),
        AdapterEntry("ground",       run_ground,       "Run text-grounded object detection"),
        AdapterEntry("describe",     run_describe,     "Run image captioning/description"),
        AdapterEntry("vl-tag",       run_vl_tag,       "Run VL-based image tagging"),

        # ── Inpainting ───────────────────────────────────────────────────
        AdapterEntry("lanpaint",     run_lanpaint,     "Run LanPaint training-free inpainting"),
    ]

    return {e.name: e for e in entries}
