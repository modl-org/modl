"""Apple Silicon (MPS) device decisions, tested without a Mac by faking the device."""

import pytest

torch = pytest.importorskip("torch")

from modl_worker import device


@pytest.fixture
def fake_mps(monkeypatch):
    """Make the worker believe it runs on MPS; reset the cached decisions."""
    monkeypatch.delenv("MODL_DTYPE", raising=False)
    monkeypatch.delenv("MODL_FORCE_OFFLOAD", raising=False)
    monkeypatch.setattr(device, "get_device", lambda: "mps")
    device.get_inference_dtype.cache_clear()
    yield monkeypatch
    device.get_inference_dtype.cache_clear()


def test_mps_uses_bf16_when_supported(fake_mps):
    fake_mps.setattr(device, "_mps_supports_bf16", lambda: True)
    assert device.get_inference_dtype() == torch.bfloat16


def test_mps_falls_back_to_fp16_without_bf16(fake_mps):
    fake_mps.setattr(device, "_mps_supports_bf16", lambda: False)
    assert device.get_inference_dtype() == torch.float16


def test_dtype_env_override(fake_mps):
    fake_mps.setattr(device, "_mps_supports_bf16", lambda: True)
    fake_mps.setenv("MODL_DTYPE", "fp32")
    assert device.get_inference_dtype() == torch.float32


class _FakePipe:
    def __init__(self, n_params):
        self.unet = torch.nn.Linear(n_params, 1, bias=False)
        self.components = {"unet": self.unet}
        self.placed = None

    def to(self, dev):
        self.placed = ("resident", dev)
        return self

    def enable_model_cpu_offload(self, device=None):
        self.placed = ("offload", device)


def _set_cap(monkeypatch, cap_bytes):
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: cap_bytes, raising=False)


def test_place_on_mps_resident_when_it_fits(fake_mps):
    pipe = _FakePipe(1000)  # 4000 bytes of fp32 weights
    _set_cap(fake_mps, 100_000)
    device.place_on_mps(pipe)
    assert pipe.placed == ("resident", "mps")


def test_place_on_mps_offloads_over_working_set(fake_mps):
    pipe = _FakePipe(1000)
    _set_cap(fake_mps, 5000)  # 4000 > 0.7 * 5000
    device.place_on_mps(pipe)
    assert pipe.placed == ("offload", "mps")


def test_moss_time_embedding_matches_float64_reference():
    """The MPS-safe sinusoid must equal upstream's on-device float64 formula."""
    pytest.importorskip("diffusers")
    from modl_worker.pipelines.moss_image import _timestep_sinusoid

    half = 128
    times = torch.tensor([0.25, 731.0, 999.5])
    freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    reference = torch.cat((phase.cos(), phase.sin()), -1).float()
    assert torch.equal(_timestep_sinusoid(times, half), reference)
