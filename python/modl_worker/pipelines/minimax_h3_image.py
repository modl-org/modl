"""MiniMax-H3-Image on a single 24 GB GPU.

MiniMax-H3-Image is a 6-step, guidance-distilled text-to-image and editing
model: a 33B DiT (pruned to ~20B in the INT8 release) conditioned on hidden
state 50 of a Qwen3-VL-32B text encoder. diffusers (>= 0.43) ships the
transformer, VAE and a modular pipeline; this module adds what a 24 GB card
needs on top:

- **DiT**: the INT8 ConvRot checkpoint (``unsloth/MiniMax-H3-Image-FP8``),
  loaded without torchao. Every quantized linear stores int8 weights with a
  per-row scale; activations are quantized per token on the fly and multiplied
  with ``torch._int_mm`` (W8A8, as the checkpoint declares). The checkpoint
  uses the pruned "curve" AdaLN (a rank-8 time table, no SiLU) and ConvRot:
  the weights were stored rotated by a block Hadamard, so the activations of
  those linears are rotated by the same matrix before the matmul.
- **Text encoder**: ComfyUI's INT8 ConvRot Qwen3-VL-32B, already truncated to
  the 50 layers H3 reads and unnormalized (27 GB). It never sits on the GPU:
  the weights live in pinned host memory and stream through the GPU one
  decoder layer at a time, so the 20 GB DiT stays resident between prompts.
- **Placement**: DiT and VAE resident; the DiT steps out (and is reloaded from
  disk) only when another model needs the room.

Licence: the model weights are under the MiniMax-H3-Image Community License,
which excludes some territories. This file is modl's own code (Apache-2.0).
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import types
from pathlib import Path

import torch
from torch import nn

# 6-step distilled schedule; the checkpoint has one PDD head per step.
NATIVE_STEPS = 6
# Hidden state H3 conditions on. The bundled encoder file holds exactly this
# many layers, with no final norm.
TEXT_ENCODER_LAYER = 50
# The pipeline renders below this many pixels larger (upstream defaults to
# 4 MP). modl passes the requested size through unchanged instead.
MIN_GENERATION_AREA = 1024


# ---------------------------------------------------------------------------
# ConvRot: block Hadamard rotation of activations
# ---------------------------------------------------------------------------

_HADAMARD: dict = {}


def hadamard(size: int, device, dtype) -> torch.Tensor:
    """Normalized Kronecker power of the 4x4 Hadamard (symmetric, orthogonal).

    Entry (i, j) is negative when an odd number of the base-4 digits of
    ``i ^ j`` equal 3. ``size`` must be a power of 4.
    """
    key = (size, str(device), dtype)
    if key not in _HADAMARD:
        levels = (size.bit_length() - 1) // 2
        if 4**levels != size:
            raise ValueError(f"ConvRot group size must be a power of 4, got {size}")
        idx = torch.arange(size)
        x = idx[:, None] ^ idx[None, :]
        threes = x & (x >> 1) & 0x5555555555555555
        parity = torch.zeros_like(threes)
        for _ in range(levels):
            parity ^= threes & 1
            threes >>= 2
        scale = 2.0**-levels
        h = torch.full((size, size), scale).masked_fill(parity.bool(), -scale)
        _HADAMARD[key] = h.to(device=device, dtype=dtype)
    return _HADAMARD[key]


def rotate(x: torch.Tensor, group: int) -> torch.Tensor:
    features = x.shape[-1]
    h = hadamard(group, x.device, x.dtype)
    return (x.reshape(-1, features // group, group) @ h).reshape(x.shape)


# ---------------------------------------------------------------------------
# Quantized linears
# ---------------------------------------------------------------------------


_F32_EPS = torch.finfo(torch.float32).eps
_W8A8_ROWS = 4096


class Int8Linear(nn.Module):
    """Linear with int8 weights and a per-output-row scale.

    ``w8a8`` quantizes the activation per token and runs ``torch._int_mm``
    (the DiT checkpoint is W8A8). Otherwise the weight is dequantized per call
    (the text encoder, whose matmuls are tiny next to its weight transfers).
    ``group`` > 0 rotates the activation first (ConvRot).
    """

    def __init__(self, qweight, scale, bias=None, group: int = 0, w8a8: bool = False):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale.reshape(1, -1).float())
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.group = group
        self.w8a8 = w8a8
        self.out_features, self.in_features = qweight.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.group:
            x = rotate(x, self.group)
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if self.w8a8 and x2.is_cuda and x2.shape[0] > 16:
            # Dynamic per-token symmetric int8, numerically matching torchao's
            # Int8Tensor (the format this checkpoint was made for): the scale is
            # amax / 127.5 in the activation dtype, and the int32 product is
            # rounded to that dtype before the weight scale is applied.
            # Row chunks bound the fp32/int32 temporaries, which at 4 MP with
            # edit references would otherwise cost several GB per FF layer.
            y = torch.empty(x2.shape[0], self.out_features, dtype=x.dtype, device=x.device)
            wt = self.qweight.t()
            for i in range(0, x2.shape[0], _W8A8_ROWS):
                xc = x2[i : i + _W8A8_ROWS]
                if xc.shape[0] <= 16:  # _int_mm needs > 16 rows (a short tail chunk)
                    yc = torch.nn.functional.linear(xc, self.qweight.to(x.dtype)) * self.scale
                else:
                    xs = (xc.abs().amax(dim=-1, keepdim=True) / 127.5).clamp(min=_F32_EPS).float()
                    xq = torch.round(xc * (1.0 / xs)).clamp_(-128, 127).to(torch.int8)
                    yc = (torch._int_mm(xq, wt) * xs).to(x.dtype) * self.scale
                if self.bias is not None:
                    yc = yc + self.bias
                y[i : i + _W8A8_ROWS] = yc
            return y.reshape(*shape[:-1], self.out_features)
        else:
            w = self.qweight.to(x.dtype) * self.scale.t().to(x.dtype)
            y = torch.nn.functional.linear(x2, w)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.reshape(*shape[:-1], self.out_features)


def _set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, child, module)


# ---------------------------------------------------------------------------
# DiT (unsloth INT8 ConvRot, curve-form AdaLN)
# ---------------------------------------------------------------------------


class CurveTimeEmbedder(nn.Module):
    """Pruned AdaLN time embedding: a rank-``dim`` curve sampled on ``grid``
    points over t in [0, 1], linearly interpolated."""

    def __init__(self, grid: int, dim: int):
        super().__init__()
        self.register_buffer("table", torch.empty(grid, dim, dtype=torch.float32))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        grid = self.table.shape[0]
        pos = t.float().clamp(0, 1) * (grid - 1)
        i0 = pos.floor().long().clamp(max=grid - 2)
        return torch.lerp(self.table[i0], self.table[i0 + 1], (pos - i0).unsqueeze(1))


class _RawTimestep(nn.Module):
    def forward(self, t):
        return t


def _to_curve_form(dit, dim: int, grid: int, out_dtype=torch.bfloat16) -> None:
    """Swap the dense AdaLN for the checkpoint's pruned curve form (on meta)."""

    def modulation_forward(self, temb):
        # No SiLU: the curve already is the post-activation embedding.
        temb = self.linear(temb.to(self.linear.weight.dtype)).to(out_dtype)
        return temb.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)

    def norm_out_forward(self, hidden_states, temb, idx):
        shift, scale = self.linear(temb.to(self.linear.weight.dtype)).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)

    for block in dit.transformer_blocks:
        proj = block.adaln_proj
        proj.linear = nn.Linear(dim, proj.linear.out_features, device="meta", dtype=torch.float32)
        proj.forward = types.MethodType(modulation_forward, proj)
    norm_out = dit.norm_out
    norm_out.linear = nn.Linear(dim, norm_out.linear.out_features, device="meta", dtype=torch.float32)
    norm_out.forward = types.MethodType(norm_out_forward, norm_out)
    dit.time_embedder = CurveTimeEmbedder(grid, dim).to("meta")
    dit.time_proj = _RawTimestep()


class _ChunkedFF(nn.Module):
    """Run a per-token feed-forward over sequence chunks.

    The SwiGLU intermediate is tokens x 2*14336; at 4 MP plus edit references
    (~25k tokens) that is ~1.4 GB per call on top of a 19 GB resident DiT.
    """

    def __init__(self, ff: nn.Module, tokens: int = 4096):
        super().__init__()
        self.ff = ff
        self.tokens = tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] <= self.tokens:
            return self.ff(x)
        return torch.cat([self.ff(c) for c in x.split(self.tokens, dim=-2)], dim=-2)


def build_transformer(dit_path: str, config_dir: Path, device: str = "cuda"):
    """Build the INT8 DiT straight onto ``device``."""
    from diffusers import MiniMaxH3ImageTransformer2DModel
    from diffusers.models.transformers.transformer_minimax_h3_image import (
        MiniMaxH3ImageRotaryPosEmbed,
    )
    from safetensors import safe_open

    cfg = MiniMaxH3ImageTransformer2DModel.load_config(str(config_dir / "transformer"))
    with torch.device("meta"):
        dit = MiniMaxH3ImageTransformer2DModel.from_config(cfg)

    with safe_open(dit_path, "pt", device=device) as f:
        meta = f.metadata() or {}
        info = json.loads(meta.get("unsloth_metadata", "{}"))
        if info.get("adaln_form", "curve") != "curve" or "curve_dim" not in info:
            raise RuntimeError(
                f"{Path(dit_path).name}: expected the unsloth INT8 ConvRot H3-Image DiT "
                "(curve-form AdaLN metadata missing)"
            )
        _to_curve_form(dit, int(info["curve_dim"]), int(info["curve_grid"]))
        rotated = set(info.get("activation_rotation_fqns", []))
        group = int(info.get("activation_rotation_group", 256))

        quantized = sorted(
            k[: -len("._weight_qdata")] for k in f.keys() if k.endswith("._weight_qdata")
        )
        plain = {}
        for k in f.keys():
            if "._weight_" in k:
                continue
            plain[k] = f.get_tensor(k)
        for name in quantized:
            zero = f.get_tensor(f"{name}._weight_zero_point")
            if zero.any():
                raise RuntimeError(f"{name}: asymmetric int8 weights are not supported")
            del zero
            bias = plain.pop(f"{name}.bias", None)
            _set_module(
                dit,
                name,
                Int8Linear(
                    f.get_tensor(f"{name}._weight_qdata"),
                    f.get_tensor(f"{name}._weight_scale"),
                    bias=bias,
                    group=group if name in rotated else 0,
                    w8a8=os.environ.get("MODL_H3_W8A8", "1") != "0",
                ),
            )

    missing, unexpected = dit.load_state_dict(plain, strict=False, assign=True)
    qset = set(quantized)
    missing = [m for m in missing if m.rsplit(".", 1)[0] not in qset]
    if missing or unexpected:
        raise RuntimeError(
            f"H3 DiT load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    # Non-persistent RoPE buffers were left on meta by the meta init.
    dit.rope = MiniMaxH3ImageRotaryPosEmbed(dit.rope.rope_freq_dim, cfg.get("rope_theta", 10000.0))
    dit.to(device)
    left = [n for n, t in list(dit.named_parameters()) + list(dit.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f"H3 DiT tensors left uninitialized: {left[:5]}")
    for block in dit.modules():
        ff = getattr(block, "ff", None)
        if ff is not None and not isinstance(ff, _ChunkedFF) and not isinstance(block, _ChunkedFF):
            block.ff = _ChunkedFF(ff)
    return dit.eval().requires_grad_(False)


# ---------------------------------------------------------------------------
# Text encoder (ComfyUI INT8 ConvRot Qwen3-VL-32B), streamed from pinned RAM
# ---------------------------------------------------------------------------


def _te_name(key: str) -> str:
    if key.startswith("model."):
        return "model.language_model." + key[len("model.") :]
    if key.startswith("visual."):
        return "model." + key
    return key


class _CpuEmbedding(nn.Module):
    """Token embedding looked up on the CPU (1.5 GB we never move)."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.inner = embedding

    @property
    def weight(self):
        return self.inner.weight

    def forward(self, ids):
        return self.inner(ids.cpu()).to(ids.device)


def _stream_through_gpu(module: nn.Module, device: str = "cuda") -> None:
    """Keep ``module``'s tensors in pinned host memory and upload them only for
    the duration of its forward."""
    host: list[tuple[nn.Module, str, bool, torch.Tensor]] = []
    for sub in module.modules():
        for name, p in list(sub._parameters.items()):
            if p is not None:
                p.data = p.data.pin_memory() if torch.cuda.is_available() else p.data
                host.append((sub, name, True, p.data))
        for name, b in list(sub._buffers.items()):
            if b is not None:
                pinned = b.pin_memory() if torch.cuda.is_available() else b
                sub._buffers[name] = pinned
                host.append((sub, name, False, pinned))

    def up(_module, _args):
        for sub, name, is_param, t in host:
            gpu = t.to(device, non_blocking=True)
            if is_param:
                sub._parameters[name].data = gpu
            else:
                sub._buffers[name] = gpu

    def down(_module, _args, _out):
        for sub, name, is_param, t in host:
            if is_param:
                sub._parameters[name].data = t
            else:
                sub._buffers[name] = t

    module.register_forward_pre_hook(up)
    module.register_forward_hook(down)


def build_text_encoder(te_path: str, config_dir: Path, device: str = "cuda"):
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    quant: dict[str, tuple] = {}
    plain: dict[str, torch.Tensor] = {}
    with safe_open(te_path, "pt") as f:
        meta = f.metadata() or {}
        n_layers = json.loads(meta.get("minimax_h3_te", "{}")).get("num_hidden_layers", TEXT_ENCODER_LAYER)
        keys = list(f.keys())
        for k in keys:
            if not k.endswith(".comfy_quant"):
                continue
            base = k[: -len(".comfy_quant")]
            conf = json.loads(bytes(f.get_tensor(k).tolist()).decode() or "{}")
            conf = {**conf.get("params", {}), **conf}
            group = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 0
            quant[base] = (f.get_tensor(base + ".weight"), f.get_tensor(base + ".weight_scale"), group)
        for k in keys:
            if k.endswith((".comfy_quant", ".weight_scale")) or k[: -len(".weight")] in quant:
                continue
            plain[_te_name(k)] = f.get_tensor(k)
    if n_layers != TEXT_ENCODER_LAYER:
        raise RuntimeError(
            f"{Path(te_path).name}: expected the {TEXT_ENCODER_LAYER}-layer H3 text encoder, found {n_layers}"
        )

    cfg = AutoConfig.from_pretrained(str(config_dir / "text_encoder"))
    cfg.text_config.num_hidden_layers = n_layers
    with torch.device("meta"):
        te = Qwen3VLForConditionalGeneration(cfg)
    lm = te.model.language_model
    # The file holds the stream after layer 50, unnormalized: drop the final
    # norm and report one more layer, so the pipeline's guard passes and
    # hidden_states[50] is that unnormalized stream.
    lm.norm = nn.Identity()
    te.config.text_config.num_hidden_layers = n_layers + 1
    for base, (q, s, group) in quant.items():
        bias = plain.pop(_te_name(base) + ".bias", None)
        _set_module(te, _te_name(base), Int8Linear(q, s, bias=bias, group=group))
    missing, unexpected = te.load_state_dict(plain, strict=False, assign=True)
    qnames = {_te_name(b) for b in quant}
    missing = [m for m in missing if "lm_head" not in m and m.rsplit(".", 1)[0] not in qnames]
    if missing or unexpected:
        raise RuntimeError(f"H3 text encoder mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    te.lm_head = nn.Identity()
    # Non-persistent rotary buffers were left on meta: rebuild those modules.
    visual = te.model.visual
    old = visual.rotary_pos_emb
    # transformers 5.5 builds it from (dim, theta); later 5.x from the config.
    visual.rotary_pos_emb = (
        type(old)(old.dim, old.theta) if hasattr(old, "dim") else type(old)(cfg.vision_config)
    )
    lm.rotary_emb = type(lm.rotary_emb)(config=te.config.text_config)
    left = [n for n, t in list(te.named_parameters()) + list(te.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f"H3 text encoder tensors left uninitialized: {left[:5]}")
    te.eval().requires_grad_(False)

    lm.embed_tokens = _CpuEmbedding(lm.embed_tokens)
    lm.rotary_emb.to(device)
    for layer in lm.layers:
        _stream_through_gpu(layer, device)
    _stream_through_gpu(visual, device)
    return te


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class _Output:
    def __init__(self, images):
        self.images = images


def _module_bytes(module) -> int:
    return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers()))


class MiniMaxH3ImagePipeline:
    """Callable like a diffusers pipeline: ``pipe(prompt=..., image=...,
    width=..., height=..., generator=...).images``.

    ``image`` (one image or a list) makes it an edit: the first picture is the
    one to edit, and the prompt can refer to each as ``<Picture n>``.
    """

    # Loaded through load_for_modl() instead of the generic diffusers paths.
    _modl_custom_loader = True

    def __init__(self, dit_path: str, te_path: str, vae_path: str, config_dir: Path, emitter=None):
        self.dit_path, self.te_path, self.vae_path = dit_path, te_path, vae_path
        self.config_dir = Path(config_dir)
        self.emitter = emitter
        self.components = {}  # no accelerate hooks: device.py leaves us alone
        self._dit = None
        self._dit_bytes = 0

        from diffusers.modular_pipelines import MiniMaxH3ImageBlocks, SequentialPipelineBlocks
        from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

        t0 = time.time()
        self.text_encoder_streamed = build_text_encoder(te_path, self.config_dir)
        self._info(f"H3 text encoder ready in {time.time() - t0:.0f}s (pinned RAM, streamed per layer)")

        self.vae = self._load_vae(vae_path).to("cuda").eval().requires_grad_(False)

        processor = Qwen3VLProcessor.from_pretrained(str(self.config_dir / "processor"))
        tokenizer = Qwen2TokenizerFast.from_pretrained(str(self.config_dir / "processor"))
        from diffusers import MiniMaxH3Scheduler

        scheduler = MiniMaxH3Scheduler.from_pretrained(str(self.config_dir / "scheduler"))

        blocks = MiniMaxH3ImageBlocks()
        cuda_device = property(lambda _self: torch.device("cuda"))

        def sub(names):
            seq = SequentialPipelineBlocks.from_blocks_dict({n: blocks.sub_blocks[n] for n in names})
            pipe = seq.init_pipeline()
            # Components live on mixed devices (streamed encoder, CPU embedding).
            pipe.__class__ = type("H3" + type(pipe).__name__, (type(pipe),), {"_execution_device": cuda_device})
            return pipe

        self._encode = sub(["before_encode", "text_encoder", "vae_encoder"])
        self._denoise = sub(["denoise", "decode"])
        self._encode.update_components(
            text_encoder=self.text_encoder_streamed, tokenizer=tokenizer, processor=processor, vae=self.vae
        )
        self._denoise.update_components(scheduler=scheduler, vae=self.vae)
        self._ensure_dit()

    # -- helpers ----------------------------------------------------------------

    def _info(self, msg: str) -> None:
        if self.emitter is not None:
            self.emitter.info(msg)
        print(msg, file=sys.stderr, flush=True)  # also in ~/.modl/worker.log

    def _load_vae(self, vae_path: str):
        from diffusers import AutoencoderKLMiniMaxH3Image
        from safetensors.torch import load_file

        cfg = AutoencoderKLMiniMaxH3Image.load_config(str(self.config_dir / "vae"))
        vae = AutoencoderKLMiniMaxH3Image.from_config(cfg)
        vae.load_state_dict(load_file(vae_path), strict=True)
        return vae.to(torch.bfloat16)

    @property
    def transformer(self):
        return self._dit

    def _lend_dit_blocks(self, nbytes: int) -> list:
        """Move trailing DiT blocks to CPU until ``nbytes`` are freed."""
        moved, freed = [], 0
        for blk in reversed(list(self._dit.transformer_blocks)):
            if freed >= nbytes:
                break
            freed += _module_bytes(blk)
            blk.to("cpu")
            moved.append(blk)
        gc.collect()
        torch.cuda.empty_cache()
        return moved

    def _park_dit(self, device: str) -> None:
        """device.py's park hook: drop the DiT to make room (it reloads from
        disk on the next job) — no 20 GB host copy on a 64 GB machine."""
        if device == "cpu" and self._dit is not None:
            self._denoise.update_components(transformer=None)
            self._dit = None
            gc.collect()
            torch.cuda.empty_cache()
            self._info("H3 DiT released to make room for another model")

    def _ensure_dit(self) -> None:
        if self._dit is not None:
            return
        from modl_worker.device import _make_room

        if self._dit_bytes:
            _make_room(self._dit_bytes + 3 * 1024**3, keep=(self.vae,))
        t0 = time.time()
        dit = build_transformer(self.dit_path, self.config_dir)
        dit._modl_park_move = self._park_dit
        self._dit_bytes = _module_bytes(dit)
        self._dit = dit
        self._denoise.update_components(transformer=dit)
        self._info(f"H3 DiT resident ({self._dit_bytes / 1024**3:.1f} GB) in {time.time() - t0:.0f}s")

    def _ensure_vae(self) -> None:
        from modl_worker.device import _make_room, _on_cuda, _park_move

        if not _on_cuda(self.vae):
            _make_room(_module_bytes(self.vae), keep=(self.vae,))
            _park_move(self.vae, "cuda")

    # -- diffusers-like surface used by modl's adapters ------------------------

    def to(self, *args, **kwargs):
        return self

    def remove_all_hooks(self):
        pass

    def enable_model_cpu_offload(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image=None,
        width: int | None = None,
        height: int | None = None,
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
        **_ignored,
    ) -> _Output:
        if num_inference_steps not in (None, NATIVE_STEPS):
            self._info(f"H3-Image always samples its {NATIVE_STEPS} distilled steps; ignoring steps={num_inference_steps}")
        images = None
        if image is not None:
            images = [im.convert("RGB") for im in (image if isinstance(image, (list, tuple)) else [image])]
        if not width or not height:
            width, height = self._default_size(images)

        self._ensure_dit()
        self._ensure_vae()
        request = dict(prompt=prompt, width=int(width), height=int(height),
                       min_generation_area=MIN_GENERATION_AREA, generator=generator)
        kwargs = dict(request)
        if images:
            kwargs["image"] = images

        t0 = time.time()
        try:
            if images and sum(im.width * im.height for im in images) > 1.5e6:
                raise torch.cuda.OutOfMemoryError  # known not to fit next to the DiT
            state = self._encode(**kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Large references (VAE encode at up to 2048 short edge) next to the
            # resident DiT: free the DiT for the encode and reload it after.
            # Large references next to the resident DiT: lend the encode ~4 GB by
            # moving the last DiT blocks to RAM for the duration (~1 s each way,
            # instead of a 28 s reload from disk).
            moved = self._lend_dit_blocks(4 << 30)
            self._info(f"H3 encode needs the room: moved {len(moved)} DiT blocks to RAM for it")
            try:
                state = self._encode(**kwargs)
            finally:
                for blk in moved:
                    blk.to("cuda")
        t_enc = time.time() - t0
        for k, v in request.items():  # denoise reads the request from the state
            state.set(k, v)

        t1 = time.time()
        _, state = self._denoise.blocks.sub_blocks["denoise"](self._denoise, state)
        torch.cuda.synchronize()
        t_den = time.time() - t1
        t2 = time.time()
        _, state = self._denoise.blocks.sub_blocks["decode"](self._denoise, state)
        out = state.get("images")
        self._info(
            f"H3 {width}x{height}: encode {t_enc:.1f}s, denoise {t_den:.1f}s, decode {time.time() - t2:.1f}s"
        )
        return _Output(out)

    @staticmethod
    def _default_size(images) -> tuple[int, int]:
        """2048² for text-to-image; an edit keeps the first picture's aspect
        ratio at ~4 MP (H3 is weak at low resolution)."""
        if not images:
            return 2048, 2048
        w, h = images[0].size
        scale = (2048 * 2048 / (w * h)) ** 0.5
        return max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)

    # -- modl loader ------------------------------------------------------------

    @classmethod
    def load_for_modl(cls, base_model_id: str, base_model_path: str | None, emitter):
        from modl_worker.adapters.arch_config import ARCH_CONFIGS, detect_arch, resolve_gen_assembly
        from modl_worker.adapters.pipeline_loader import resolve_model_path

        assembly = resolve_gen_assembly(base_model_id, download_missing=base_model_path is None) or {}
        dit_path = base_model_path or assembly.get("transformer", {}).get("resolved_path") or resolve_model_path(base_model_id)
        te_path = assembly.get("text_encoder", {}).get("resolved_path")
        vae_path = assembly.get("vae", {}).get("resolved_path")
        missing = [n for n, p in (("text encoder", te_path), ("VAE", vae_path)) if not p]
        if missing or not dit_path or not Path(dit_path).is_file():
            raise RuntimeError(
                f"MiniMax-H3-Image is missing its {', '.join(missing) or 'transformer'}; run `modl pull {base_model_id}`"
            )
        config_dir = Path(__file__).resolve().parent.parent / "configs" / ARCH_CONFIGS[detect_arch(base_model_id)]["config_dir"]
        emitter.info(f"Loading MiniMax-H3-Image: DiT {Path(dit_path).name}, text encoder {Path(te_path).name}")
        pipe = cls(dit_path, te_path, vae_path, config_dir, emitter)
        pipe._modl_loaded_files = {
            "transformer": {"file": Path(dit_path).name, "path": dit_path, "weight_dtype": "int8"},
            "text_encoder": {"file": Path(te_path).name, "path": te_path, "weight_dtype": "int8"},
            "vae": {"file": Path(vae_path).name, "path": vae_path, "weight_dtype": "bf16"},
        }
        from modl_worker.device import _PARKED

        _PARKED.add(pipe)  # takes part in VRAM arbitration with other cached models
        return pipe
