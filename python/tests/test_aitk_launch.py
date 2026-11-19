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
