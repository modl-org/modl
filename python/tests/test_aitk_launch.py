"""aitk_launch shims the imports ai-toolkit expects from an older diffusers."""

import importlib.util
import sys
import types
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "modl_worker" / "aitk_launch.py"
_spec = importlib.util.spec_from_file_location("_aitk_launch_isolated", _PATH)
aitk_launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aitk_launch)


def _fake_diffusers(monkeypatch, ltx2_has_encode_video):
    def encode_video():
        return "utils"

    names = [
        "diffusers",
        "diffusers.pipelines",
        "diffusers.pipelines.ltx2",
        "diffusers.pipelines.ltx2.export_utils",
        "diffusers.utils",
        "diffusers.utils.export_utils",
    ]
    mods = {n: types.ModuleType(n) for n in names}
    for n in names[:5]:
        mods[n].__path__ = []
    mods["diffusers.utils.export_utils"].encode_video = encode_video
    if ltx2_has_encode_video:
        mods["diffusers.pipelines.ltx2.export_utils"].encode_video = lambda: "ltx2"
    for n, m in mods.items():
        monkeypatch.setitem(sys.modules, n, m)
    # keep the transformers shim out of the diffusers tests
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", None)
    return mods["diffusers.pipelines.ltx2.export_utils"]


def test_adds_encode_video_where_ai_toolkit_looks_for_it(monkeypatch):
    ltx2_export = _fake_diffusers(monkeypatch, ltx2_has_encode_video=False)
    assert aitk_launch.apply_shims() == ["ltx2.encode_video"]
    assert ltx2_export.encode_video() == "utils"


def test_leaves_an_existing_encode_video_alone(monkeypatch):
    ltx2_export = _fake_diffusers(monkeypatch, ltx2_has_encode_video=True)
    assert aitk_launch.apply_shims() == []
    assert ltx2_export.encode_video() == "ltx2"


def test_no_diffusers_is_not_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "diffusers", None)
    assert aitk_launch.apply_shims() == []


def _fake_transformers(monkeypatch):
    registry = {}

    class PreTrainedModel:
        _can_record_outputs = None
        seen = []

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            PreTrainedModel.seen.append(cls.__name__)

        def __init__(self):
            registry[str(self.__class__)] = self._can_record_outputs

    names = [
        "transformers",
        "transformers.modeling_utils",
        "transformers.utils",
        "transformers.utils.output_capturing",
    ]
    mods = {n: types.ModuleType(n) for n in names}
    mods["transformers"].__path__ = []
    mods["transformers.utils"].__path__ = []
    mods["transformers.utils"].output_capturing = mods[
        "transformers.utils.output_capturing"
    ]
    mods["transformers.modeling_utils"].PreTrainedModel = PreTrainedModel
    mods["transformers.utils.output_capturing"]._CAN_RECORD_REGISTRY = registry
    for n, m in mods.items():
        monkeypatch.setitem(sys.modules, n, m)
    return PreTrainedModel, registry


def test_class_swapped_model_keeps_its_capture_specs(monkeypatch):
    """ai-toolkit rebinds a loaded model onto a subclass whose __init__ never
    ran; transformers must still find what that class can record."""
    base, registry = _fake_transformers(monkeypatch)

    class Clip(base):
        _can_record_outputs = {"hidden_states": "layer"}

    model = Clip()
    assert "transformers.capture_registry" in aitk_launch.apply_shims()

    class Wrapper(Clip):
        pass

    model.__class__ = Wrapper
    assert registry[str(model.__class__)] == {"hidden_states": "layer"}
    # the original hook still runs, and a second apply does not wrap twice
    assert base.seen == ["Clip", "Wrapper"]
    assert "transformers.capture_registry" not in aitk_launch.apply_shims()


def test_models_without_capture_specs_are_not_registered(monkeypatch):
    base, registry = _fake_transformers(monkeypatch)
    aitk_launch.apply_shims()

    class Plain(base):
        pass

    assert str(Plain) not in registry
