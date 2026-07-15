"""Regression tests for spec_to_aitoolkit_config.

These build full ai-toolkit configs from minimal TrainJobSpecs — the exact
path the worker runs on a pod (no local store, HF fallback). A NameError
here shipped to production once; every trainable arch gets a smoke build.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modl_worker.adapters.arch_config import ARCH_CONFIGS  # noqa: E402
from modl_worker.adapters.config_builder import spec_to_aitoolkit_config  # noqa: E402


def _spec(base_model_id: str, lora_type: str = "object") -> dict:
    return {
        "dataset": {"name": "ds", "path": "/tmp/ds", "image_count": 6, "caption_coverage": 1.0},
        "model": {"base_model_id": base_model_id, "base_model_path": None},
        "output": {"lora_name": "test-lora", "destination_dir": "/tmp/out"},
        "params": {
            "preset": "quick",
            "lora_type": lora_type,
            "trigger_word": "ohwx",
            "steps": 100,
            "rank": 16,
            "learning_rate": 1e-4,
            "optimizer": "adamw8bit",
            "resolution": 1024,
            "quantize": True,
            "batch_size": 0,
            "num_repeats": 0,
            "caption_dropout_rate": -1.0,
        },
    }


TRAINABLE_MODELS = [
    "flux-dev",
    "flux-schnell",
    "chroma",
    "flux2-dev",
    "flux2-klein-4b",
    "flux2-klein-9b",
    "z-image",
    "z-image-turbo",
    "qwen-image",
    "sdxl",
    "sd-1.5",
]


def test_config_builds_for_every_trainable_model():
    for model_id in TRAINABLE_MODELS:
        for lora_type in ("object", "style", "character"):
            config = spec_to_aitoolkit_config(_spec(model_id, lora_type))
            process = config["config"]["process"][0]
            assert process["type"] == "sd_trainer", model_id
            assert process["datasets"][0]["folder_path"] == "/tmp/ds"
            assert "cache_text_embeddings" in process["datasets"][0]
            # With no store path, every trainable id must resolve to an
            # HF-style repo ref (owner/name) — a bare id would make
            # ai-toolkit fail at model load on pods.
            name_or_path = process["model"]["name_or_path"]
            assert "/" in name_or_path and not name_or_path.startswith("/"), (
                f"{model_id}: unresolved model path {name_or_path!r}"
            )


def test_cache_text_embeddings_follows_arch_config():
    config = spec_to_aitoolkit_config(_spec("sdxl"))
    dataset = config["config"]["process"][0]["datasets"][0]
    expected = ARCH_CONFIGS["sdxl"].get("training", {}).get("cache_text_embeddings", False)
    assert dataset["cache_text_embeddings"] == expected


def test_hf_fallback_when_no_store_path():
    # Pod path: base_model_path is None → model name must be an HF-style ref,
    # never a local store path.
    config = spec_to_aitoolkit_config(_spec("sdxl"))
    name_or_path = config["config"]["process"][0]["model"]["name_or_path"]
    assert "/" in name_or_path
    assert not name_or_path.startswith("/")
