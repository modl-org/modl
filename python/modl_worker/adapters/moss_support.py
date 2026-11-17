"""MOSS-Image glue: prompt-embedding cache and unmerged LoRA application.

The vendored pipeline (``modl_worker.pipelines.moss_image``) is a plain
``DiffusionPipeline`` — no ``LoraLoaderMixin``, no PEFT integration — so LoRAs
are applied here by wrapping the target ``nn.Linear`` modules.
"""

from __future__ import annotations

import re
from collections import OrderedDict

_EMBED_CACHE_SIZE = 8


def prepare_moss_pipeline(pipe) -> None:
    """Wrap ``encode_prompt`` with a small LRU cache.

    The text encoder is a 9B model that has to be swapped onto the GPU for
    every encode. A job with ``count > 1``, a seed sweep, or the empty
    negative prompt re-encodes the same strings — cache the embeddings
    (≈1MB each) so those cost nothing.
    """
    if getattr(pipe, "_modl_embed_cache", None) is not None:
        return
    cache: OrderedDict = OrderedDict()
    original = pipe.encode_prompt

    def encode_prompt(prompt, max_sequence_length=4096):
        key = (tuple([prompt] if isinstance(prompt, str) else prompt), max_sequence_length)
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            return hit
        result = original(prompt, max_sequence_length)
        cache[key] = result
        while len(cache) > _EMBED_CACHE_SIZE:
            cache.popitem(last=False)
        return result

    pipe.encode_prompt = encode_prompt
    pipe._modl_embed_cache = cache


def is_moss_pipeline(pipeline) -> bool:
    transformer = getattr(pipeline, "transformer", None)
    return type(transformer).__name__ == "MOSSImageTransformer"


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

_PREFIXES = ("diffusion_model.", "transformer.", "model.diffusion_model.")
_SUFFIX = re.compile(r"\.(lora_down|lora_up|lora_A|lora_B)\.weight$|\.alpha$")


def _make_wrapper(base, down, up, scale):
    import torch

    class LoraLinear(torch.nn.Module):
        """``base(x) + scale * up(down(x))`` with the LoRA branch in fp32.

        Merging a DMD-style LoRA into bf16 weights loses most of the delta
        (bf16 has ~3 significant digits; the per-weight update is far below
        that), so the branch stays separate and accumulates in fp32.
        """

        def __init__(self):
            super().__init__()
            self.base = base
            self.scale = scale
            self.down = torch.nn.Parameter(down.float(), requires_grad=False)
            self.up = torch.nn.Parameter(up.float(), requires_grad=False)

        @property
        def weight(self):  # upstream reads proj[...].weight.dtype
            return self.base.weight

        def forward(self, x):
            delta = (x.float() @ self.down.T) @ self.up.T
            return self.base(x) + (self.scale * delta).to(x.dtype)

    return LoraLinear()


def _group_lora(state_dict: dict) -> dict[str, dict]:
    """Group LoRA tensors by target module name (prefix-stripped)."""
    groups: dict[str, dict] = {}
    for key, tensor in state_dict.items():
        m = _SUFFIX.search(key)
        if not m:
            continue
        module = key[: m.start()]
        for prefix in _PREFIXES:
            if module.startswith(prefix):
                module = module[len(prefix):]
                break
        kind = m.group(1) or "alpha"
        kind = {"lora_A": "down", "lora_B": "up", "lora_down": "down", "lora_up": "up"}.get(kind, kind)
        groups.setdefault(module, {})[kind] = tensor
    return groups


def apply_moss_lora(pipeline, lora_path: str, weight: float = 1.0, emitter=None) -> bool:
    """Apply a kohya/PEFT-format MOSS LoRA as unmerged fp32 branches.

    Targets are upstream module names (``layers.N.proj.q`` …). A LoRA trained
    against the Comfy fused layout (``layers.N.proj.qkv``) is split: the shared
    ``down`` feeds q/k/v and ``up`` is cut into thirds. Originals are stashed on
    the transformer so ``revert_moss_lora`` restores them exactly.
    """
    import torch
    from safetensors.torch import load_file

    transformer = pipeline.transformer
    revert_moss_lora(pipeline)
    groups = _group_lora(load_file(lora_path))

    targets: list[tuple[str, torch.Tensor, torch.Tensor, float]] = []
    skipped = []
    for module, parts in groups.items():
        down, up = parts.get("down"), parts.get("up")
        if down is None or up is None:
            skipped.append(module)
            continue
        rank = down.shape[0]
        alpha = float(parts["alpha"]) if "alpha" in parts else float(rank)
        scale = alpha / rank * float(weight)
        if module.endswith(".proj.qkv"):
            stem = module[: -len("qkv")]
            for name, up_part in zip("qkv", up.chunk(3, dim=0)):
                targets.append((stem + name, down, up_part, scale))
        else:
            targets.append((module, down, up, scale))

    wrapped: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []
    for module, down, up, scale in targets:
        parent_name, _, child = module.rpartition(".")
        try:
            parent = transformer.get_submodule(parent_name)
            base = parent[child] if isinstance(parent, torch.nn.ModuleDict) else getattr(parent, child)
        except (AttributeError, KeyError):
            skipped.append(module)
            continue
        if not isinstance(base, torch.nn.Linear) or base.weight.shape != (up.shape[0], down.shape[1]):
            skipped.append(module)
            continue
        wrapper = _make_wrapper(base, down, up, scale).to(base.weight.device)
        if isinstance(parent, torch.nn.ModuleDict):
            parent[child] = wrapper
        else:
            setattr(parent, child, wrapper)
        wrapped.append((parent, child, base))

    if not wrapped:
        if emitter:
            emitter.warning(
                "LORA_INCOMPATIBLE",
                f"No MOSS-Image modules matched the LoRA ({len(skipped)} keys). "
                f"Generating without LoRA.",
            )
        return False

    transformer._modl_moss_lora = wrapped
    if emitter:
        note = f", {len(skipped)} unmatched" if skipped else ""
        emitter.info(
            f"  → Applied MOSS LoRA unmerged (fp32 branch): {len(wrapped)} modules "
            f"at strength {weight}{note}"
        )
    return True


def revert_moss_lora(pipeline, emitter=None) -> bool:
    """Restore the Linear modules replaced by ``apply_moss_lora``."""
    import torch

    transformer = getattr(pipeline, "transformer", None)
    wrapped = getattr(transformer, "_modl_moss_lora", None) if transformer is not None else None
    if not wrapped:
        return False
    for parent, child, base in wrapped:
        if isinstance(parent, torch.nn.ModuleDict):
            parent[child] = base
        else:
            setattr(parent, child, base)
    del transformer._modl_moss_lora
    if emitter:
        emitter.info(f"  → Reverted MOSS LoRA ({len(wrapped)} modules)")
    return True
