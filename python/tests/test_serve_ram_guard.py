"""Host-RAM guard in the worker's model cache.

The cache is capped by model count; these check that it also evicts when the
incoming model would not fit in host RAM next to what is already cached.
"""

import importlib.util
import sys
import time
import types

import pytest

from modl_worker import serve
from modl_worker.serve import CachedPipeline, CacheKey, ModelCache

GIB = 1024**3


@pytest.fixture(autouse=True)
def _torch_stub(monkeypatch):
    """Eviction calls ``torch.cuda.empty_cache()``; CI runs without torch."""
    if importlib.util.find_spec("torch") is None:
        stub = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
        monkeypatch.setitem(sys.modules, "torch", stub)


class _Emitter:
    def __init__(self):
        self.lines = []

    def info(self, message):
        self.lines.append(message)


def _cache_with(*model_ids):
    cache = ModelCache(max_models=4)
    for i, model_id in enumerate(model_ids):
        key = CacheKey(model_id=model_id, dtype="bfloat16", mode="txt2img")
        cache._cache[key] = CachedPipeline(
            pipeline=object(),
            cls_name="X",
            loaded_at=time.time(),
            last_used=float(i),  # first is least recently used
            vram_estimate_mb=0,
        )
    return cache


def _ids(cache):
    return [k.model_id for k in cache._cache]


def test_keeps_cache_when_ram_is_plentiful(monkeypatch):
    monkeypatch.setattr(serve, "_estimate_load_ram_bytes", lambda *_: 20 * GIB)
    monkeypatch.setattr(serve, "_available_ram_bytes", lambda: 40 * GIB)
    cache = _cache_with("a", "b")
    cache._make_room_in_ram("c", None, _Emitter())
    assert _ids(cache) == ["a", "b"]


def test_evicts_lru_until_the_incoming_model_fits(monkeypatch):
    monkeypatch.setattr(serve, "_estimate_load_ram_bytes", lambda *_: 30 * GIB)
    available = iter([10 * GIB, 25 * GIB, 45 * GIB])
    monkeypatch.setattr(serve, "_available_ram_bytes", lambda: next(available))
    cache = _cache_with("a", "b", "c")
    emitter = _Emitter()
    cache._make_room_in_ram("d", None, emitter)
    assert _ids(cache) == ["c"]
    assert any("Host RAM is tight" in line for line in emitter.lines)


def test_stops_when_the_cache_is_empty(monkeypatch):
    monkeypatch.setattr(serve, "_estimate_load_ram_bytes", lambda *_: 100 * GIB)
    monkeypatch.setattr(serve, "_available_ram_bytes", lambda: 1 * GIB)
    cache = _cache_with("a")
    cache._make_room_in_ram("b", None, _Emitter())
    assert _ids(cache) == []


def test_skipped_where_meminfo_is_unavailable(monkeypatch):
    monkeypatch.setattr(serve, "_estimate_load_ram_bytes", lambda *_: 100 * GIB)
    monkeypatch.setattr(serve, "_available_ram_bytes", lambda: None)
    cache = _cache_with("a", "b")
    cache._make_room_in_ram("c", None, _Emitter())
    assert _ids(cache) == ["a", "b"]


def test_path_bytes_handles_files_dirs_and_missing(tmp_path):
    (tmp_path / "w.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.json").write_bytes(b"y" * 5)
    assert serve._path_bytes(str(tmp_path / "w.safetensors")) == 10
    assert serve._path_bytes(str(tmp_path)) == 15
    assert serve._path_bytes(str(tmp_path / "nope")) == 0
    assert serve._path_bytes(None) == 0
