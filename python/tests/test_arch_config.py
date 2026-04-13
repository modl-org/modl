"""Schema and contract tests for ARCH_CONFIGS.

These tests are the cheap safety net for the Python worker: no GPU, no model
downloads, no torch import required. They run in well under a second and catch
the most common breakage class — drift between ARCH_CONFIGS, models.toml,
bundled scheduler configs, and the diffusers pipeline surface.

Intentionally kept free of heavy dependencies so the suite runs in CI on a
plain Python image. The diffusers-class resolution check is gated behind
`importorskip` so it activates automatically in dev/GPU environments.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "python" / "modl_worker" / "configs"
MODELS_TOML = REPO_ROOT / "models.toml"

# Load arch_config.py directly instead of importing modl_worker.adapters.
# The package __init__.py eagerly imports every adapter (gen, edit, caption, …)
# which pulls PIL/torch/diffusers and defeats the "no heavy deps" promise of
# this test module. arch_config.py itself only uses os/sqlite3/pathlib, so a
# direct file load keeps the schema tests installable with just pytest.
_ARCH_CONFIG_PATH = REPO_ROOT / "python" / "modl_worker" / "adapters" / "arch_config.py"
_spec = importlib.util.spec_from_file_location("_arch_config_isolated", _ARCH_CONFIG_PATH)
_arch_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_arch_config)  # type: ignore[union-attr]

ARCH_CONFIGS = _arch_config.ARCH_CONFIGS
MODEL_REGISTRY = _arch_config.MODEL_REGISTRY
detect_arch = _arch_config.detect_arch
resolve_pipeline_class = _arch_config.resolve_pipeline_class
resolve_pipeline_class_for_mode = _arch_config.resolve_pipeline_class_for_mode
resolve_gen_defaults = _arch_config.resolve_gen_defaults

ARCH_KEYS = sorted(ARCH_CONFIGS.keys())

REQUIRED_TOP_LEVEL_KEYS = {
    "pipeline_class",
    "model_flags",
    "noise_scheduler",
    "dtype",
    "train_text_encoder",
    "resolutions",
    "default_resolution",
    "sample",
}

VALID_DTYPES = {"fp16", "bf16", "fp32"}
VALID_NOISE_SCHEDULERS = {"flowmatch", "ddpm"}

VALID_GUIDANCE_PARAMS = {
    "guidance_scale",
    "true_cfg_scale",
    "guidance",
}

VALID_EDITING_MODES = {"standard", "native"}


# ---------------------------------------------------------------------------
# Per-arch schema tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCH_KEYS)
def test_required_keys_present(arch: str) -> None:
    cfg = ARCH_CONFIGS[arch]
    missing = REQUIRED_TOP_LEVEL_KEYS - set(cfg.keys())
    assert not missing, f"{arch} missing required keys: {missing}"


@pytest.mark.parametrize("arch", ARCH_KEYS)
def test_pipeline_class_is_nonempty_string(arch: str) -> None:
    cfg = ARCH_CONFIGS[arch]
    pc = cfg["pipeline_class"]
    assert isinstance(pc, str) and pc, f"{arch}.pipeline_class must be a non-empty string"
    # Pipeline class names are CamelCase and end in "Pipeline".
    assert pc[0].isupper(), f"{arch}.pipeline_class should be CamelCase, got {pc!r}"
    assert pc.endswith("Pipeline"), f"{arch}.pipeline_class should end in 'Pipeline', got {pc!r}"


@pytest.mark.parametrize("arch", ARCH_KEYS)
def test_dtype_valid(arch: str) -> None:
    dtype = ARCH_CONFIGS[arch]["dtype"]
    assert dtype in VALID_DTYPES, f"{arch}.dtype={dtype!r} not in {VALID_DTYPES}"


@pytest.mark.parametrize("arch", ARCH_KEYS)
def test_noise_scheduler_valid(arch: str) -> None:
    ns = ARCH_CONFIGS[arch]["noise_scheduler"]
    assert ns in VALID_NOISE_SCHEDULERS, f"{arch}.noise_scheduler={ns!r} not in {VALID_NOISE_SCHEDULERS}"


@pytest.mark.parametrize("arch", ARCH_KEYS)
def test_resolutions_are_positive_ints(arch: str) -> None:
    cfg = ARCH_CONFIGS[arch]
    resolutions = cfg["resolutions"]
    assert isinstance(resolutions, list) and resolutions, f"{arch}.resolutions must be a non-empty list"
    for r in resolutions:
        assert isinstance(r, int) and r > 0, f"{arch}.resolutions contains invalid value {r!r}"
    default = cfg["default_resolution"]
    assert isinstance(default, int) and default > 0, f"{arch}.default_resolution must be a positive int"


@pytest.mark.parametrize("arch", ARCH_KEYS)
def test_sample_block_shape(arch: str) -> None:
    sample = ARCH_CONFIGS[arch]["sample"]
    assert isinstance(sample, dict)
    for key in ("sampler", "steps", "guidance", "neg"):
        assert key in sample, f"{arch}.sample missing {key!r}"
    assert isinstance(sample["steps"], int) and sample["steps"] > 0, f"{arch}.sample.steps must be a positive int"
    assert isinstance(sample["guidance"], (int, float)), f"{arch}.sample.guidance must be numeric"
    assert isinstance(sample["neg"], str), f"{arch}.sample.neg must be a string"


@pytest.mark.parametrize("arch", ARCH_KEYS)
def test_inference_subdict_shape(arch: str) -> None:
    inference = ARCH_CONFIGS[arch].get("inference")
    if inference is None:
        return
    assert isinstance(inference, dict), f"{arch}.inference must be a dict"

    gp = inference.get("guidance_param")
    if gp is not None:
        assert gp in VALID_GUIDANCE_PARAMS, f"{arch}.inference.guidance_param={gp!r} not in {VALID_GUIDANCE_PARAMS}"

    em = inference.get("editing_mode")
    if em is not None:
        assert em in VALID_EDITING_MODES, f"{arch}.inference.editing_mode={em!r} not in {VALID_EDITING_MODES}"

    shift = inference.get("scheduler_shift")
    if shift is not None:
        assert isinstance(shift, (int, float)), f"{arch}.inference.scheduler_shift must be numeric"

    for bool_key in ("supports_negative_prompt", "skip_strength_in_inpaint"):
        if bool_key in inference:
            assert isinstance(inference[bool_key], bool), f"{arch}.inference.{bool_key} must be a bool"


@pytest.mark.parametrize("arch", ARCH_KEYS)
def test_training_subdict_shape(arch: str) -> None:
    training = ARCH_CONFIGS[arch].get("training")
    if training is None:
        return
    assert isinstance(training, dict), f"{arch}.training must be a dict"

    max_lr = training.get("max_learning_rate")
    if max_lr is not None:
        assert isinstance(max_lr, (int, float)) and max_lr > 0, (
            f"{arch}.training.max_learning_rate must be a positive number"
        )

    noise_offset = training.get("noise_offset")
    if noise_offset is not None:
        assert isinstance(noise_offset, dict), f"{arch}.training.noise_offset must be a dict"
        for k, v in noise_offset.items():
            assert isinstance(v, (int, float)), f"{arch}.training.noise_offset.{k} must be numeric"

    if "use_ema" in training:
        assert isinstance(training["use_ema"], bool), f"{arch}.training.use_ema must be a bool"


# ---------------------------------------------------------------------------
# Cross-reference tests (bundled configs, gen_components)
# ---------------------------------------------------------------------------


def _iter_gen_components(arch: str):
    gen_components = ARCH_CONFIGS[arch].get("gen_components") or {}
    for component_type, spec in gen_components.items():
        if isinstance(spec, dict):
            yield component_type, spec


@pytest.mark.parametrize("arch", ARCH_KEYS)
def test_gen_components_have_required_keys(arch: str) -> None:
    for component_type, spec in _iter_gen_components(arch):
        assert "model_class" in spec, f"{arch}.gen_components.{component_type} missing model_class"
        assert isinstance(spec["model_class"], str) and spec["model_class"], (
            f"{arch}.gen_components.{component_type}.model_class must be a non-empty string"
        )


@pytest.mark.parametrize("arch", ARCH_KEYS)
def test_gen_component_config_dirs_exist(arch: str) -> None:
    """Every gen_component that references a bundled config_dir must point at
    an existing directory under python/modl_worker/configs/."""
    for component_type, spec in _iter_gen_components(arch):
        config_dir = spec.get("config_dir")
        if config_dir is None:
            continue
        path = CONFIGS_DIR / config_dir
        assert path.is_dir(), (
            f"{arch}.gen_components.{component_type}.config_dir={config_dir!r} "
            f"does not exist at {path}"
        )


@pytest.mark.parametrize("arch", ARCH_KEYS)
def test_scheduler_config_file_present(arch: str) -> None:
    """If the arch declares a scheduler with a config_dir, the bundled
    scheduler_config.json must be present."""
    gen_components = ARCH_CONFIGS[arch].get("gen_components") or {}
    scheduler_spec = gen_components.get("scheduler")
    if not isinstance(scheduler_spec, dict):
        return
    config_dir = scheduler_spec.get("config_dir")
    if not config_dir:
        return
    scheduler_json = CONFIGS_DIR / config_dir / "scheduler_config.json"
    assert scheduler_json.is_file(), (
        f"{arch}: bundled scheduler config missing at {scheduler_json}"
    )


# ---------------------------------------------------------------------------
# Sync with models.toml (replaces scripts/validate-arch-sync.py)
# ---------------------------------------------------------------------------


def _load_toml_arch_keys() -> set[str]:
    with open(MODELS_TOML, "rb") as f:
        data = tomllib.load(f)
    keys = set()
    for family in data["families"].values():
        for model in family["models"].values():
            keys.add(model["arch_key"])
    return keys


def test_models_toml_arch_keys_have_python_configs() -> None:
    """Every arch_key referenced from models.toml must be implemented in
    ARCH_CONFIGS. The reverse is allowed — Python may define internal
    base variants without user-facing models."""
    toml_keys = _load_toml_arch_keys()
    missing = toml_keys - set(ARCH_CONFIGS.keys())
    assert not missing, (
        f"models.toml references arch_keys with no ARCH_CONFIGS entry: {sorted(missing)}"
    )


def test_model_registry_arch_keys_exist() -> None:
    """Every entry in MODEL_REGISTRY must point at a known ARCH_CONFIGS key."""
    bad = {
        model_id: arch for model_id, (arch, _) in MODEL_REGISTRY.items()
        if arch not in ARCH_CONFIGS
    }
    assert not bad, f"MODEL_REGISTRY entries point at unknown arch_keys: {bad}"


# ---------------------------------------------------------------------------
# Resolver helpers — behavior smoke
# ---------------------------------------------------------------------------


def test_detect_arch_respects_explicit_key() -> None:
    assert detect_arch("anything", arch_key="flux") == "flux"
    assert detect_arch("anything", arch_key="sdxl") == "sdxl"


def test_detect_arch_uses_registry() -> None:
    for model_id, (expected_arch, _) in MODEL_REGISTRY.items():
        assert detect_arch(model_id) == expected_arch, (
            f"detect_arch({model_id!r}) should resolve to {expected_arch!r}"
        )


def test_resolve_pipeline_class_matches_config() -> None:
    for model_id, (arch, _) in MODEL_REGISTRY.items():
        assert resolve_pipeline_class(model_id) == ARCH_CONFIGS[arch]["pipeline_class"]


def test_resolve_pipeline_class_for_mode_falls_back_to_txt2img() -> None:
    # flux2 does not currently expose img2img/inpaint classes — resolver must
    # fall back to the base pipeline rather than raising.
    base = resolve_pipeline_class_for_mode("flux2-dev", "txt2img")
    img2img = resolve_pipeline_class_for_mode("flux2-dev", "img2img")
    inpaint = resolve_pipeline_class_for_mode("flux2-dev", "inpaint")
    assert base == "Flux2Pipeline"
    assert img2img == base
    assert inpaint == base


def test_resolve_gen_defaults_returns_positive_values() -> None:
    for model_id in MODEL_REGISTRY:
        defaults = resolve_gen_defaults(model_id)
        assert defaults["steps"] > 0
        assert defaults["guidance"] >= 0


# ---------------------------------------------------------------------------
# Optional: verify pipeline class names resolve against the installed diffusers.
# Gated behind importorskip so CI without diffusers still passes the cheap
# schema layer; local dev + GPU CI catches diffusers renames automatically.
# ---------------------------------------------------------------------------


def test_pipeline_classes_exist_in_diffusers() -> None:
    diffusers = pytest.importorskip("diffusers")

    missing: list[tuple[str, str, str]] = []
    for arch, cfg in ARCH_CONFIGS.items():
        for key in ("pipeline_class", "img2img_class", "inpaint_class"):
            name = cfg.get(key)
            if not name:
                continue
            if not hasattr(diffusers, name):
                missing.append((arch, key, name))

    assert not missing, (
        "Pipeline classes referenced in ARCH_CONFIGS not found in installed diffusers: "
        + ", ".join(f"{a}.{k}={n}" for a, k, n in missing)
    )
