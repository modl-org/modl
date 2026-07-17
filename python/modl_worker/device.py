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


def _pipe_weight_bytes(pipe) -> int:
    """Sum of parameter + buffer bytes across all pipeline components.

    fp8-stored weights count 1 byte/element, so a layerwise-cast
    transformer is sized at its actual GPU footprint.
    """
    total = 0
    for name in getattr(pipe, "components", {}) or {}:
        comp = getattr(pipe, name, None)
        if comp is None or not hasattr(comp, "parameters"):
            continue
        for p in comp.parameters():
            total += p.numel() * p.element_size()
        for b in comp.buffers():
            total += b.numel() * b.element_size()
    return total


def _has_accelerate_hooks(pipe) -> bool:
    """True if any component carries an accelerate offload hook."""
    for name in getattr(pipe, "components", {}) or {}:
        comp = getattr(pipe, name, None)
        if comp is not None and getattr(comp, "_hf_hook", None) is not None:
            return True
    return False


# Headroom for activations, per-layer dtype-cast transients, VAE decode and
# CUDA context when deciding whether a pipeline can live fully resident.
# Calibrated on Krea 2 Turbo (fp8, 18.4GB weights): denoise transients
# measured ~7GB over weights (~38%) — layerwise-cast bf16 compute buffers
# dominate. 35% of weights with a 6GB floor keeps the decision honest.
_RESIDENT_RESERVE_FLOOR_BYTES = 6 * 1024**3
_RESIDENT_RESERVE_WEIGHT_FRACTION = 0.35


def rebalance_pipe_placement(pipe, emitter=None) -> None:
    """Promote an offloaded pipeline to full GPU residency when it fits.

    ``enable_model_cpu_offload`` re-transfers weights CPU→GPU on every job —
    for a stack that genuinely fits VRAM (SDXL, SD15, Klein 4B on a 24GB
    card) that transfer is pure per-job overhead. Called after LoRA /
    deferred-fp8 state is final, because the footprint before layerwise
    casting (bf16) can be ~2x the end state.

    No-op on non-CUDA devices, when the pipeline is already resident, when
    the stack doesn't fit with reserve, or when MODL_FORCE_OFFLOAD=1.
    """
    if get_device() != "cuda":
        return
    if os.environ.get("MODL_FORCE_OFFLOAD") == "1":
        return
    if not _has_accelerate_hooks(pipe):
        return  # already resident (or never offloaded)

    weight_bytes = _pipe_weight_bytes(pipe)
    free_bytes, _total = torch.cuda.mem_get_info()
    reserve = max(
        _RESIDENT_RESERVE_FLOOR_BYTES,
        int(weight_bytes * _RESIDENT_RESERVE_WEIGHT_FRACTION),
    )
    needed = weight_bytes + reserve
    if needed > free_bytes:
        if emitter:
            emitter.info(
                f"  → Keeping cpu offload: {weight_bytes / 1024**3:.1f}GB weights "
                f"+ {reserve / 1024**3:.1f}GB reserve > {free_bytes / 1024**3:.1f}GB free"
            )
        return

    try:
        pipe.remove_all_hooks()
        pipe.to("cuda")
        if emitter:
            emitter.info(
                f"  → Pipeline fits VRAM ({weight_bytes / 1024**3:.1f}GB weights) — "
                f"running fully resident (no per-job offload transfers)"
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pipe.enable_model_cpu_offload()
        if emitter:
            emitter.info("  → Resident placement OOMed, reverting to cpu offload")


def empty_cache():
    """Free GPU cache for the active device."""
    dev = get_device()
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "mps":
        torch.mps.empty_cache()
