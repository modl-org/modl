"""Device resolution for modl worker.

Reads MODL_DEVICE env var (set by the Rust CLI based on detected hardware),
with fallback to PyTorch runtime detection.

Usage:
    from modl_worker.device import get_device, get_generator_device
    pipe.to(get_device())
    generator = torch.Generator(device=get_generator_device())
"""

import os
import weakref
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
# Calibrated on Krea 2 Turbo (fp8, 17.4GiB weights): denoise transients
# measured ≥8GB over weights (layerwise-cast bf16 compute buffers dominate)
# — and a borderline pass promoted it straight into an OOM. 50% of weights
# with an 8GB floor plus a 5% free-VRAM margin keeps the decision solidly
# conservative; a wrong promote is additionally self-healed at runtime by
# ``demote_pipe_to_offload``.
_RESIDENT_RESERVE_FLOOR_BYTES = 8 * 1024**3
_RESIDENT_RESERVE_WEIGHT_FRACTION = 0.50


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

    if getattr(pipe, "_modl_needs_resident", False):
        # Pipeline cannot be offloaded at all (custom-method transformer, e.g.
        # Krea2 edit's precompute_ref_kv — accelerate hooks only fire on
        # forward). assemble_pipeline left it on CPU so the deferred fp8 cast
        # could shrink it first; place it now that the footprint is final.
        # MODL_FORCE_OFFLOAD cannot apply here — offload does not work.
        pipe.to("cuda")
        pipe._modl_needs_resident = False
        if emitter:
            emitter.info(
                f"  → Resident: {_pipe_weight_bytes(pipe) / 1024**3:.1f}GB weights on GPU"
            )
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
    if needed > int(free_bytes * 0.95) and getattr(pipe, "_modl_park_te", False):
        place_with_parked_text_encoder(pipe, emitter)
        return
    if needed > int(free_bytes * 0.95):
        if emitter:
            emitter.info(
                f"  → Keeping cpu offload: {weight_bytes / 1024**3:.1f}GB weights "
                f"+ {reserve / 1024**3:.1f}GB reserve > {free_bytes / 1024**3:.1f}GB free"
            )
        return

    try:
        pipe.remove_all_hooks()
        pipe.to("cuda")
        _RESIDENT.add(pipe)
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


def _module_bytes(module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters()) + sum(
        b.numel() * b.element_size() for b in module.buffers()
    )


def _remove_park_hooks(pipe) -> None:
    _PARKED.discard(pipe)
    for handle in getattr(pipe, "_modl_park_handles", ()):
        handle.remove()
    pipe._modl_park_handles = []
    original = getattr(pipe, "_modl_unparked_class", None)
    if original is not None:
        pipe.__class__ = original
        del pipe._modl_unparked_class


def _pin_execution_device(pipe) -> None:
    """Make ``pipe._execution_device`` report cuda while the TE is parked.

    Without accelerate hooks diffusers answers ``_execution_device`` with
    ``pipe.device`` — the device of the first module in *alphabetical*
    component order, which is the parked (CPU) ``text_encoder``. Prompt
    tokens and latents would then be created on the CPU. Swap in a
    same-named subclass that pins the property; undone by _remove_park_hooks.
    """
    if getattr(pipe, "_modl_unparked_class", None) is not None:
        return
    original = type(pipe)
    pipe.__class__ = type(
        original.__name__,
        (original,),
        {"_execution_device": property(lambda self: torch.device("cuda"))},
    )
    pipe._modl_unparked_class = original


# Pipelines placed by place_with_parked_text_encoder. Their GPU residency is
# arbitrated here: two cached MOSS pipelines (Base + SFT) cannot both keep a
# 10GB DiT on a 24GB card *and* swap a 16GB text encoder in, so whoever needs
# VRAM next evicts the others' modules to CPU first.
_PARKED: "weakref.WeakSet" = weakref.WeakSet()
# Ordinary pipelines promoted to full residency by rebalance_pipe_placement.
# A parked pipeline that still lacks room demotes these to cpu offload.
_RESIDENT: "weakref.WeakSet" = weakref.WeakSet()

# Activation headroom kept free when bringing a module up. The DiT peak at
# 1024² is ~2.5GB over weights (MOSS, measured 12.7GiB total).
_PARK_DENOISE_HEADROOM = 4 * 1024**3
_PARK_ENCODE_HEADROOM = 2 * 1024**3
# VAE encode of edit references (Qwen-Image-2.1: up to 10 images at ~1MP).
_PARK_VAE_HEADROOM = 3 * 1024**3
# Decode working memory per output megapixel (Qwen-Image-2.1 VAE at 1024²
# measured ~5.6GB over weights).
_PARK_DECODE_BYTES_PER_MP = 6 * 1024**3


def _vae_spatial_factor(vae) -> int:
    """Pixels per latent cell along one side: 16 for Qwen-Image-2.1, 8 for
    the classic KL VAEs (2 ** (levels - 1))."""
    cfg = getattr(vae, "config", None)
    factor = getattr(cfg, "scale_factor_spatial", None)
    if factor:
        return int(factor)
    blocks = getattr(cfg, "block_out_channels", None)
    return 2 ** (len(blocks) - 1) if blocks else 8


def _park_move(module, device: str) -> None:
    """Move a parked-pipeline module, keeping a persistent CPU copy of params.

    ``module.to("cpu")`` allocates fresh pageable memory and copies — 8s for
    the 16GB MOSS text encoder, 3.5s for the DiT, on every prompt change.
    These weights never change after load (MOSS LoRAs are unmerged wrappers
    with their own params), so the first upload remembers each parameter's
    CPU tensor and moving back is just re-pointing ``param.data``. Buffers are
    small and may be reassigned by the model (RoPE tables), so they move
    normally. Costs no extra RAM while the module is on the CPU; while it is
    on the GPU the CPU copy is kept (that's the point).
    """
    with torch.no_grad():
        for param in module.parameters():
            if device == "cuda":
                if param.device.type != "cuda":
                    param._modl_cpu = param.data
                    param.data = param.data.to("cuda", non_blocking=True)
            else:
                cpu = getattr(param, "_modl_cpu", None)
                if param.device.type == "cuda":
                    param.data = cpu if cpu is not None else param.data.to("cpu")
        for mod in module.modules():
            for name, buf in mod._buffers.items():
                if buf is not None and buf.device.type != device:
                    mod._buffers[name] = buf.to(device)
    if device == "cuda":
        torch.cuda.synchronize()


def _on_cuda(module) -> bool:
    p = next(module.parameters(), None)
    return p is not None and p.device.type == "cuda"


def _make_room(needed: int, keep: tuple = ()) -> None:
    """Move parked pipelines' modules to CPU until ``needed`` bytes are free.

    Text encoders go first (they are re-uploaded per encode anyway), then
    other pipelines' denoisers, then this pipeline's own (``keep`` is never
    moved).
    """
    free, _ = torch.cuda.mem_get_info()
    if free >= needed:
        return
    candidates = []
    for p in list(_PARKED):
        for rank, name in ((0, "text_encoder"), (1, "transformer"), (2, "vae")):
            comp = getattr(p, name, None)
            if comp is not None and comp not in keep and _on_cuda(comp):
                candidates.append((rank, id(comp), comp))
    for _rank, _id, comp in sorted(candidates, key=lambda c: c[0]):
        _park_move(comp, "cpu")
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        if free >= needed:
            return
    # Last resort: another cached model sits fully resident. Demote it to
    # cpu offload — it keeps working, just with per-job transfers.
    for p in list(_RESIDENT):
        _RESIDENT.discard(p)
        p.enable_model_cpu_offload()
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        if free >= needed:
            return


def yield_parked_gpu(pipe) -> None:
    """Before a job on ``pipe``, move *other* parked pipelines off the GPU.

    A parked pipeline's DiT stays resident between jobs; an ordinary
    (offloaded or resident) pipeline knows nothing about it and would OOM —
    e.g. a MOSS → Klein edit workflow in one worker. Thanks to the persistent
    CPU copies in ``_park_move`` this is a pointer swap, and the parked
    pipeline's own pre-hooks bring everything back on its next job.
    """
    if get_device() != "cuda":
        return
    moved = False
    for p in list(_PARKED):
        if p is pipe:
            continue
        for name in ("text_encoder", "transformer", "vae"):
            comp = getattr(p, name, None)
            if comp is not None and comp is not getattr(pipe, name, None) and _on_cuda(comp):
                _park_move(comp, "cpu")
                moved = True
    if moved:
        torch.cuda.empty_cache()


def _guard_parked_vae(vae, te) -> None:
    """Make ``vae.encode()`` / ``vae.decode()`` safe under parked placement.

    ``encode`` and ``decode`` are not ``forward``, so the park pre-hooks
    don't fire for them:
    - encode: the denoiser (transformer + VAE) steps out while the text
      encoder encodes and only returns on the first transformer forward, but
      edit pipelines encode their reference images in between (Qwen-Image-2.1).
      ``encode_prompt`` has finished by then, so the text encoder may be
      parked to make room (a 1536px reference needs several GB).
    - decode: an edit's reference KV cache is still allocated when the pipeline
      decodes, so with the DiT resident a 1024px decode can OOM. Make room
      first; that parks the DiT only when memory is short (a pointer swap
      thanks to the persistent CPU copies), and its pre-hook brings it back
      on the next job.
    Installed once per (possibly shared) VAE; a no-op under accelerate
    offload, whose own hooks place it.
    """
    if getattr(vae, "_modl_park_guard", False) or not hasattr(vae, "encode"):
        return
    original_encode, original_decode = vae.encode, vae.decode

    def _parked() -> bool:
        return getattr(vae, "_hf_hook", None) is None

    def encode(x, *args, **kwargs):
        if _parked():
            missing = 0 if _on_cuda(vae) else _module_bytes(vae)
            megapixels = x.shape[-1] * x.shape[-2] / 1e6
            working = max(_PARK_VAE_HEADROOM, int(megapixels * _PARK_DECODE_BYTES_PER_MP))
            _make_room(missing + working, keep=(vae,))
            if missing:
                _park_move(vae, "cuda")
        return original_encode(x, *args, **kwargs)

    def decode(z, *args, **kwargs):
        if _parked():
            if not _on_cuda(vae):
                _make_room(_module_bytes(vae), keep=(vae,))
                _park_move(vae, "cuda")
            megapixels = z.shape[-1] * z.shape[-2] * _vae_spatial_factor(vae) ** 2 / 1e6
            needed = int(max(megapixels, 0.25) * _PARK_DECODE_BYTES_PER_MP)
            _make_room(needed, keep=(vae,))
            # Large outputs (2048² ≈ 25GB of decode activations) can't fit even
            # with everything else parked: decode in tiles for this call only.
            can_tile = hasattr(vae, "enable_tiling") and not getattr(vae, "use_tiling", False)
            if can_tile and torch.cuda.mem_get_info()[0] < needed:
                return _tiled(z, *args, **kwargs)
            try:
                return original_decode(z, *args, **kwargs)
            except torch.cuda.OutOfMemoryError:
                # The estimate missed (fragmentation, a large edit KV cache).
                if not can_tile:
                    raise
                torch.cuda.empty_cache()
                return _tiled(z, *args, **kwargs)
        return original_decode(z, *args, **kwargs)

    def _tiled(z, *args, **kwargs):
        vae.enable_tiling()
        try:
            return original_decode(z, *args, **kwargs)
        finally:
            vae.disable_tiling()

    vae.encode, vae.decode = encode, decode
    vae._modl_park_guard = True


def place_with_parked_text_encoder(pipe, emitter=None) -> None:
    """Keep the denoiser resident and swap only the text encoder in to encode.

    For stacks whose text encoder is the big half (MOSS-Image: 16GB Qwen3.5-9B
    vs a 10GB DiT), ``enable_model_cpu_offload`` re-uploads *both* every job.
    Here the transformer + VAE live on the GPU and the text encoder on the
    CPU; a forward pre-hook moves it up to encode. If the two don't fit
    together, the transformer steps out for the encode and its own pre-hook
    brings it back (parking the text encoder again) on the first denoise
    step. Plain torch hooks, not accelerate ones; ``_execution_device`` is
    pinned to cuda because diffusers would otherwise report the parked text
    encoder's device.
    """
    te = getattr(pipe, "text_encoder", None)
    transformer = getattr(pipe, "transformer", None)
    if te is None or transformer is None:
        return

    pipe.remove_all_hooks()
    _remove_park_hooks(pipe)
    if _on_cuda(te):
        _park_move(te, "cpu")
    torch.cuda.empty_cache()
    vae = getattr(pipe, "vae", None)
    denoiser = [m for m in (transformer, vae) if m is not None]
    denoiser_bytes = sum(_module_bytes(m) for m in denoiser)
    _PARKED.add(pipe)

    def bring_up_denoiser():
        missing = [m for m in denoiser if not _on_cuda(m)]
        if not missing:
            return
        _make_room(
            sum(_module_bytes(m) for m in missing) + _PARK_DENOISE_HEADROOM,
            keep=tuple(denoiser),
        )
        for m in missing:
            _park_move(m, "cuda")

    te_bytes = _module_bytes(te)

    def te_pre_hook(module, args, kwargs):
        # The text encoder may be shared by several cached pipelines (Base and
        # SFT load the same file); every pipeline's hook is idempotent.
        if not _on_cuda(module):
            _make_room(te_bytes + _PARK_ENCODE_HEADROOM, keep=(module,))
            _park_move(module, "cuda")

    def transformer_pre_hook(module, args, kwargs):
        if _on_cuda(te):
            _park_move(te, "cpu")
            torch.cuda.empty_cache()
        bring_up_denoiser()

    if vae is not None:
        _guard_parked_vae(vae, te)
    bring_up_denoiser()
    _pin_execution_device(pipe)
    pipe._modl_park_handles = [
        te.register_forward_pre_hook(te_pre_hook, with_kwargs=True),
        transformer.register_forward_pre_hook(transformer_pre_hook, with_kwargs=True),
    ]
    if emitter:
        emitter.info(
            f"  → Denoiser resident ({denoiser_bytes / 1024**3:.1f}GB), text encoder "
            f"parked on CPU ({te_bytes / 1024**3:.1f}GB, swapped in to encode)"
        )


def settle_parked_after_error(pipe) -> None:
    """After a failed job, put a parked pipeline back in its idle state.

    An exception mid-pipeline (typically an OOM) can leave the text encoder
    on the GPU next to whatever else was up, so every following job OOMs too.
    Park the text encoder and free the cache; the pre-hooks bring the rest
    back on the next job. No-op for other pipelines.
    """
    if pipe not in _PARKED:
        return
    te = getattr(pipe, "text_encoder", None)
    if te is not None and _on_cuda(te):
        _park_move(te, "cpu")
    torch.cuda.empty_cache()


def demote_pipe_to_offload(pipe, exc, emitter=None) -> bool:
    """Demote a resident pipeline back to cpu offload after a CUDA OOM.

    Runtime safety net for ``rebalance_pipe_placement``: if a promoted
    pipeline OOMs on denoise transients the placement estimate missed,
    re-enable offload so the job can retry instead of failing.

    Returns True if a retry makes sense (the pipeline was resident and has
    been demoted), False otherwise (not an OOM, or already offloaded).
    """
    if not isinstance(exc, torch.cuda.OutOfMemoryError):
        return False
    if get_device() != "cuda":
        return False
    if _has_accelerate_hooks(pipe):
        return False  # already offloaded — a retry would just OOM again
    if pipe in _PARKED:
        # Parking already is the low-memory placement; layering accelerate
        # offload on top leaves half-parked modules and breaks the *next* job
        # with a cpu/cuda mismatch. Fail just this image.
        settle_parked_after_error(pipe)
        return False

    _remove_park_hooks(pipe)
    torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload()
    torch.cuda.empty_cache()
    if emitter:
        emitter.info(
            "  → OOM while fully resident — demoted to cpu offload, retrying"
        )
    return True


def empty_cache():
    """Free GPU cache for the active device."""
    dev = get_device()
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "mps":
        torch.mps.empty_cache()
