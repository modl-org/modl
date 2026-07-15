"""Regression tests for spec_to_aitoolkit_config.

These build full ai-toolkit configs from minimal TrainJobSpecs — the exact
path the worker runs on a pod (no local store, HF fallback). A NameError
here shipped to production once; every trainable arch gets a smoke build.
"""

import importlib.util
import sys
import types
from pathlib import Path

# Load config_builder + arch_config directly, bypassing modl_worker.adapters'
# __init__.py (which eagerly imports every adapter and pulls PIL/torch —
# unavailable in the pytest-only CI env). Same approach as test_arch_config,
# extended with a synthetic package so config_builder's relative import of
# arch_config resolves.
_ADAPTERS = Path(__file__).resolve().parents[1] / "modl_worker" / "adapters"
_pkg = types.ModuleType("_cb_isolated")
_pkg.__path__ = [str(_ADAPTERS)]
sys.modules["_cb_isolated"] = _pkg


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_cb_isolated.{name}", _ADAPTERS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_cb_isolated.{name}"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_arch_config = _load("arch_config")
_config_builder = _load("config_builder")

ARCH_CONFIGS = _arch_config.ARCH_CONFIGS
spec_to_aitoolkit_config = _config_builder.spec_to_aitoolkit_config


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
