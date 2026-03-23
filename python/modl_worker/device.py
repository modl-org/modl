"""Device resolution for modl worker.

Reads MODL_DEVICE env var (set by the Rust CLI based on detected hardware),
with fallback to PyTorch runtime detection.

Usage:
    from modl_worker.device import get_device, get_generator_device
    pipe.to(get_device())
    generator = torch.Generator(device=get_generator_device())
"""

import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def get_device() -> str:
    """Return the torch device string: 'cuda', 'mps', or 'cpu'."""
    env = os.environ.get("MODL_DEVICE", "").lower()
    if env in ("cuda", "mps", "cpu"):
        return env
    # Auto-detect
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_generator_device() -> str:
    """Return the device for torch.Generator.

    MPS has quirks with Generator — some ops require CPU generator even when
    running on MPS. Use 'cpu' for MPS, actual device otherwise.
    """
    dev = get_device()
    if dev == "mps":
        return "cpu"
    return dev


def get_inference_dtype():
    """Return the best inference dtype for the active device.

    MPS has limited bfloat16 support — use float16 for reliability.
    CUDA uses bfloat16 for best quality.
    """
    if get_device() == "mps":
        return torch.float16
    return torch.bfloat16


def is_mps() -> bool:
    """Return True if running on MPS (Apple Silicon)."""
    return get_device() == "mps"


def move_pipe_to_device(pipe):
    """Move a pipeline to the active device, using the right strategy.

    On CUDA: use enable_model_cpu_offload() for memory efficiency.
    On MPS: use pipe.to("mps") since cpu_offload is CUDA-only.
    On CPU: use pipe.to("cpu").
    """
    dev = get_device()
    if dev == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(dev)


def empty_cache():
    """Free GPU cache for the active device."""
    dev = get_device()
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "mps":
        torch.mps.empty_cache()
