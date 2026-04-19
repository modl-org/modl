"""Tests for ModelCache VRAM-aware eviction and OOM recovery."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

# If torch isn't importable (CPU-only CI, minimal env), install a lightweight
# stub so ModelCache's deferred `import torch` inside its methods doesn't
# crash. When torch IS installed we leave it alone — polluting sys.modules
# would break every other test that exercises real diffusers code.
try:
    import torch  # noqa: F401
except ImportError:
    torch_stub = types.ModuleType("torch")
    cuda_stub = types.ModuleType("torch.cuda")
    cuda_stub.is_available = lambda: True
    cuda_stub.mem_get_info = lambda: (0, 0)
    cuda_stub.empty_cache = lambda: None
    cuda_stub.memory_allocated = lambda: 0
    cuda_stub.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    torch_stub.cuda = cuda_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.cuda"] = cuda_stub


def _install_loader_stub(load_pipeline_impl):
    """Replace modl_worker.adapters.pipeline_loader with a stub module.

    The real loader depends on diffusers and is unsafe to import in a CPU-only
    test environment. Serve.py does a deferred `from … import load_pipeline`
    inside _load_with_oom_recovery, so replacing the sys.modules entry is
    enough."""
    pkg = sys.modules.get("modl_worker.adapters")
    if pkg is None:
        pkg = types.ModuleType("modl_worker.adapters")
        sys.modules["modl_worker.adapters"] = pkg
    loader = types.ModuleType("modl_worker.adapters.pipeline_loader")
    loader.load_pipeline = load_pipeline_impl
    sys.modules["modl_worker.adapters.pipeline_loader"] = loader
    pkg.pipeline_loader = loader


from modl_worker.protocol import EventEmitter  # noqa: E402
from modl_worker.serve import CacheKey, CachedPipeline, ModelCache  # noqa: E402


class _RecordingEmitter(EventEmitter):
    """Captures events in-memory for assertions."""

    def __init__(self) -> None:
        super().__init__(source="test", job_id="")
        self.events: list[dict] = []

    def emit(self, event: dict) -> None:
        self.events.append(event)


def _stub_pipeline(tag: str = "fake") -> object:
    """Cheap sentinel for a diffusers pipeline."""
    return types.SimpleNamespace(kind=tag)


def _seed_cache(cache: ModelCache, ids: list[str]) -> None:
    """Pre-populate the cache with fake pipelines so eviction has victims."""
    now = 0.0
    for i, model_id in enumerate(ids):
        key = CacheKey(model_id=model_id, dtype="bfloat16", mode="txt2img")
        now += 1.0
        cache._cache[key] = CachedPipeline(
            pipeline=_stub_pipeline(model_id),
            cls_name="FakePipeline",
            loaded_at=now,
            last_used=now,
            vram_estimate_mb=8000,
        )


def test_ensure_vram_headroom_no_eviction_when_plenty_free():
    cache = ModelCache(max_models=4)
    _seed_cache(cache, ["a", "b"])
    emitter = _RecordingEmitter()

    # Simulate 20 GB free — well above the 4 GB headroom.
    with patch.object(ModelCache, "_free_vram_mb", return_value=20_000):
        cache._ensure_vram_headroom(emitter)

    assert len(cache._cache) == 2, "no models should be evicted when VRAM is plentiful"


def test_ensure_vram_headroom_evicts_lru_until_enough_free():
    cache = ModelCache(max_models=4)
    _seed_cache(cache, ["oldest", "newer", "newest"])
    emitter = _RecordingEmitter()

    # 500 MB free to start — below 4 GB headroom; rises after each eviction.
    free_values = iter([500, 2_000, 5_000])

    def fake_free_mb(self):
        return next(free_values)

    with patch.object(ModelCache, "_free_vram_mb", fake_free_mb):
        cache._ensure_vram_headroom(emitter)

    # 500 → evict → 2000 (still < 4096) → evict → 5000 → stop. One model left.
    assert len(cache._cache) == 1
    remaining_ids = [k.model_id for k in cache._cache.keys()]
    assert remaining_ids == ["newest"], "oldest entries should evict first"


def test_ensure_vram_headroom_stops_when_cache_empty():
    cache = ModelCache(max_models=4)
    _seed_cache(cache, ["only"])
    emitter = _RecordingEmitter()

    # Free VRAM never crosses the headroom — would loop forever without the
    # cache-empty guard.
    with patch.object(ModelCache, "_free_vram_mb", return_value=100):
        cache._ensure_vram_headroom(emitter)

    assert cache._cache == {}, "cache should be empty after exhausting evictions"


def test_ensure_vram_headroom_skips_when_cuda_unavailable():
    cache = ModelCache(max_models=4)
    _seed_cache(cache, ["a", "b"])
    emitter = _RecordingEmitter()

    # _free_vram_mb returns None when CUDA is unavailable — no eviction.
    with patch.object(ModelCache, "_free_vram_mb", return_value=None):
        cache._ensure_vram_headroom(emitter)

    assert len(cache._cache) == 2, "CPU-only hosts must not trigger eviction"


def test_load_with_oom_recovery_retries_after_evicting_all():
    cache = ModelCache(max_models=4)
    _seed_cache(cache, ["stale"])
    emitter = _RecordingEmitter()

    import torch
    calls = {"n": 0}
    succeeded_pipeline = _stub_pipeline("new")

    def flaky_loader(model_id, path, cls_name, em):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return succeeded_pipeline

    _install_loader_stub(flaky_loader)
    result = cache._load_with_oom_recovery(
        "new-model", None, "FakePipeline", emitter
    )

    assert result is succeeded_pipeline
    assert calls["n"] == 2, "should have retried exactly once"
    assert cache._cache == {}, "retry path must evict everything before retrying"


def test_load_with_oom_recovery_raises_on_second_oom():
    cache = ModelCache(max_models=4)
    _seed_cache(cache, ["stale"])
    emitter = _RecordingEmitter()

    import torch

    def always_oom(model_id, path, cls_name, em):
        raise torch.cuda.OutOfMemoryError("won't fit at any price")

    _install_loader_stub(always_oom)
    with pytest.raises(RuntimeError, match="even after evicting"):
        cache._load_with_oom_recovery(
            "huge-model", None, "FakePipeline", emitter
        )


def test_load_with_oom_recovery_raises_when_cache_already_empty():
    cache = ModelCache(max_models=4)
    emitter = _RecordingEmitter()

    import torch

    def always_oom(model_id, path, cls_name, em):
        raise torch.cuda.OutOfMemoryError("nothing to evict")

    _install_loader_stub(always_oom)
    with pytest.raises(RuntimeError, match="CUDA out of memory loading huge-model"):
        cache._load_with_oom_recovery(
            "huge-model", None, "FakePipeline", emitter
        )
