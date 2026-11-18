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
    "moss-image",
    "moss-image-base",
    "qwen-image-2-1",
    "qwen-image-2-1-turbo",
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


def _process(model_id: str, lora_type: str = "character") -> dict:
    return spec_to_aitoolkit_config(_spec(model_id, lora_type))["config"]["process"][0]


def test_moss_uses_the_trainer_arch_not_the_inference_flags():
    base = _process("moss-image-base")
    assert base["model"]["arch"] == "moss_image:base"
    assert base["model"]["name_or_path"] == "OpenMOSS-Team/MOSS-Image-Base"
    assert base["model"]["qtype"] == "convrot8" and base["model"]["low_vram"] is True
    # modl-only inference flags must not leak into ai-toolkit's model block
    assert "park_text_encoder" not in base["model"]
    assert "vae_fp32" not in base["model"]
    # block 22 is skipped in the uncond branch, so the LoRA leaves it alone
    assert base["network"]["network_kwargs"] == {"ignore_if_contains": ["layers.22."]}
    assert base["train"]["timestep_type"] == "shift"
    assert base["datasets"][0]["cache_text_embeddings"] is True
    assert base["sample"]["sample_steps"] == 30 and base["sample"]["guidance_scale"] == 3.0

    sft = _process("moss-image")
    assert sft["model"]["arch"] == "moss_image"
    assert sft["model"]["name_or_path"] == "OpenMOSS-Team/MOSS-Image-SFT"


def test_qwen21_trains_on_base_through_qwen_image_2():
    for model_id in ("qwen-image-2-1", "qwen-image-2-1-turbo"):
        process = _process(model_id)
        assert process["model"]["arch"] == "qwen_image_2", model_id
        assert process["model"]["name_or_path"] == "Comfy-Org/Qwen-Image-2.1", model_id
        assert process["model"]["model_kwargs"] == {"rgba": False}
        assert "rgba_output" not in process["model"]
        assert "network_kwargs" not in process["network"]
        # previews use CFG, unlike the CFG-1 inference default
        assert process["sample"]["guidance_scale"] == 3.0
        assert process["sample"]["sample_steps"] == 30


def test_training_model_block_is_not_shared_between_builds():
    first = _process("moss-image-base")
    first["model"]["model_kwargs"]["uncond_skip_layers"].append(99)
    first["network"]["network_kwargs"]["ignore_if_contains"].append("x")
    second = _process("moss-image-base")
    assert second["model"]["model_kwargs"]["uncond_skip_layers"] == [22]
    assert second["network"]["network_kwargs"]["ignore_if_contains"] == ["layers.22."]


def test_store_path_does_not_override_the_training_repo():
    spec = _spec("moss-image-base")
    spec["model"]["base_model_path"] = "/tmp"  # a directory, as a store dir would be
    process = spec_to_aitoolkit_config(spec)["config"]["process"][0]
    assert process["model"]["name_or_path"] == "OpenMOSS-Team/MOSS-Image-Base"
