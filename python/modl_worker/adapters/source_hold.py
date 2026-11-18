"""Source hold: keep the parts of an edit you did not ask to change.

Edit models redraw the whole picture from noise, so every edit shifts, darkens
or re-textures the untouched areas a little, and a chain of edits compounds it.
The source hold pulls the sampler's clean-image estimate back toward the source
latent at every step:

    x0' = x0 + w * (src - x0)          (then v' = (x_t - x0') / sigma)

Early steps decide layout, late steps decide what survives. Early, the pull is
weak (``hold`` strength) so the edit can form; late, it is strong but gated by
distance: tokens whose estimate is already far from the source are where the
model is making the edit, so they are left alone. An optional mask (white =
edit area) holds everything outside it firmly.

Ported from comfyui-krea2edit-advanced by FurrySphere (Apache-2.0,
https://github.com/FurrySphere/comfyui-krea2edit-advanced), adapted from a
ComfyUI post-CFG hook to diffusers flow-match pipelines: the hold wraps
``scheduler.step`` and finds the source latent in the reference tokens the
pipeline already feeds its transformer, so no extra VAE encode is needed.

Supported pipelines: FLUX.2 Klein, Qwen-Image-Edit(-Plus), Qwen-Image-2.1. The
source must be encoded at the output size (the default for all three, unless
--size changes it); otherwise it is resampled and a warning is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class HoldConfig:
    ref_guide: float = 0.9   # pull strength 0..1; 1 = copy the source where the weight is full
    crossover: float = 0.5   # early phase = sigma above crossover * sigma_start
    late_gain: float = 10.0  # late strength = min(1, late_gain * ref_guide)
    early_gain: float = 0.8  # early strength outside the edit area = early_gain * ref_guide
    early_hold: float = 0.7  # the user knob: early pull inside the edit area (0 = none)
    dist_tau: float = 0.02   # distance-gate tolerance in source-std units; 0 = off
    dist_pow: float = 8.0    # gate steepness: d^n / (d^n + tol^n)
    smooth: float = 0.5      # across-step memory of the distance map


def step_weights(sigmas, ref_guide, crossover, late_gain, early_gain=1.0, early_decay=0.0):
    """Per-step (early, late) weights. A step straddling the crossover is split by
    its share of the pull, so the total early pull does not depend on step count."""
    sigmas = [float(s) for s in sigmas]
    sigma_c = crossover * sigmas[0]
    late = min(1.0, late_gain * ref_guide)
    plan = []
    for s, sn in zip(sigmas[:-1], sigmas[1:]):
        f = min(1.0, max(0.0, (1 - sigma_c / s) / (1 - sn / s))) if s > sigma_c else 0.0
        if s > sigma_c and sigmas[0] > sigma_c:
            fall = ((s - sigma_c) / (sigmas[0] - sigma_c)) ** early_decay
        else:
            fall = 1.0
        plan.append((early_gain * ref_guide * f * fall, late * (1 - f)))
    return plan


class Guide:
    """State of one sampling run, on a (B, D, H, W) token grid."""

    def __init__(self, cfg: HoldConfig, sigmas):
        self.cfg = cfg
        self.plan = step_weights(sigmas, cfg.ref_guide, cfg.crossover, cfg.late_gain, cfg.early_gain)
        if cfg.early_hold > 0:
            self.hold = [e for e, _ in step_weights(
                sigmas, cfg.ref_guide, cfg.crossover, cfg.late_gain,
                min(cfg.early_hold, 1.0), 2.0 / cfg.early_hold)]
        else:
            self.hold = [0.0] * len(self.plan)
        self.ref_std = None
        self.dist = None

    def _freedom(self, x0, ref):
        cfg = self.cfg
        if cfg.dist_tau <= 0 or cfg.ref_guide >= 1:
            return 0.0
        if self.ref_std is None:
            self.ref_std = ref.float().std()
        d = (x0 - ref).float().pow(2).mean(1, keepdim=True).sqrt() / self.ref_std
        self.dist = d if self.dist is None else cfg.smooth * self.dist + (1 - cfg.smooth) * d
        tau = cfg.dist_tau * cfg.ref_guide / (1 - cfg.ref_guide)
        return self.dist.pow(cfg.dist_pow) / (self.dist.pow(cfg.dist_pow) + tau ** cfg.dist_pow)

    def apply(self, i, x0, ref, keep):
        """keep: (1, 1, H, W), 1 = keep as the source, 0 = edit area."""
        if i >= len(self.plan):
            return x0
        early, late = self.plan[i]
        hold = self.hold[i]
        if early == 0 and late == 0 and hold == 0:
            return x0
        free = self._freedom(x0, ref) * (1 - keep) if late > 0 else 0.0
        w = late * (1 - free) + keep * early + (1 - keep) * hold
        return torch.lerp(x0, ref, w)


# --- locating the source tokens -------------------------------------------------

def pipeline_family(pipeline) -> str | None:
    name = type(pipeline).__name__
    if name.startswith("Flux2Klein"):
        return "klein"
    if name.startswith("QwenImage21"):
        return "qwen21"
    if name.startswith("QwenImageEdit"):
        return "qwen_edit"
    return None


def _grid(tokens, h, w):
    """(B, N, D) row-major tokens -> (B, D, h, w)."""
    b, n, d = tokens.shape
    assert n == h * w, f"token count {n} != {h}x{w}"
    return tokens.transpose(1, 2).reshape(b, d, h, w)


def _tokens(grid):
    b, d, h, w = grid.shape
    return grid.reshape(b, d, h * w).transpose(1, 2)


def locate_source(family, kwargs):
    """From the transformer's first-call kwargs, return (target_hw, ref_grid, order).

    order is None for row-major target tokens, or a LongTensor permutation that
    maps row-major grid positions to the pipeline's token order (Klein ids)."""
    hs = kwargs["hidden_states"]
    if family in ("qwen21", "qwen_edit"):
        shapes = kwargs["img_shapes"][0]
        if len(shapes) < 2:
            return None
        sizes = [int(s[1]) * int(s[2]) for s in shapes]
        if family == "qwen21":           # refs first, target last
            th, tw = int(shapes[-1][1]), int(shapes[-1][2])
            rh, rw = int(shapes[0][1]), int(shapes[0][2])
            ref = hs[:, : sizes[0]]
        else:                            # target first, then refs
            th, tw = int(shapes[0][1]), int(shapes[0][2])
            rh, rw = int(shapes[1][1]), int(shapes[1][2])
            ref = hs[:, sizes[0]: sizes[0] + sizes[1]]
        return (th, tw), _grid(ref, rh, rw), None

    if family == "klein":
        ids = kwargs.get("img_ids")
        if ids is None:
            return None
        if ids.dim() == 3:
            ids = ids[0]
        t = ids[:, 0]
        frames = torch.unique(t, sorted=True)
        if len(frames) < 2:
            return None
        tgt, src = t == frames[0], t == frames[1]
        th = int(ids[tgt, 1].max()) + 1
        tw = int(ids[tgt, 2].max()) + 1
        rh = int(ids[src, 1].max()) + 1
        rw = int(ids[src, 2].max()) + 1
        # Scatter reference tokens onto their (h, w) grid.
        ref_tok = hs[:, src]
        pos = (ids[src, 1] * rw + ids[src, 2]).long().to(hs.device)
        ref = torch.zeros(hs.shape[0], rh * rw, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        ref[:, pos] = ref_tok
        order = (ids[tgt, 1] * tw + ids[tgt, 2]).long().to(hs.device)
        return (th, tw), _grid(ref, rh, rw), order
    return None


class SourceHold:
    """Installs the hold on a pipeline for one call; use as a context manager."""

    def __init__(self, pipeline, strength: float, edit_mask=None, emitter=None):
        self.pipeline = pipeline
        self.family = pipeline_family(pipeline)
        self.cfg = HoldConfig(early_hold=max(0.0, float(strength)))
        self.edit_mask = edit_mask  # PIL "L", white = edit area
        self.emitter = emitter
        self._hook = None
        self._orig_step = None
        self.ref = None      # (B, D, H, W) source on the target grid, float32
        self.keep = None     # (1, 1, H, W)
        self.order = None
        self.hw = None
        self.guide = None
        self.resampled = False

    @property
    def supported(self) -> bool:
        return self.family is not None

    def _info(self, msg):
        if self.emitter is not None:
            self.emitter.info(msg)

    def _capture(self, module, args, kwargs):
        if self.ref is not None:
            return None
        found = locate_source(self.family, kwargs)
        if found is None:
            return None
        (th, tw), ref, order = found
        ref = ref.float()
        if ref.shape[-2:] != (th, tw):
            self.resampled = True
            self._info(
                f"Source hold: source encoded at {tuple(ref.shape[-2:])} tokens, output is "
                f"{(th, tw)}; resampling (softer hold). Edit at the source size for best results."
            )
            ref = F.interpolate(ref, size=(th, tw), mode="bilinear", align_corners=False)
        self.ref, self.order, self.hw = ref, order, (th, tw)
        keep = torch.zeros(1, 1, th, tw, device=ref.device)
        if self.edit_mask is not None:
            import numpy as np
            m = np.asarray(self.edit_mask.convert("L"), dtype="float32") / 255.0
            m = torch.from_numpy(m)[None, None].to(ref.device)
            m = F.interpolate(m, size=(th, tw), mode="area")
            keep = 1 - m.clamp(0, 1)
        self.keep = keep
        return None

    def _to_grid(self, packed):
        th, tw = self.hw
        if self.order is None:
            return _grid(packed, th, tw)
        b, n, d = packed.shape
        grid = torch.zeros(b, th * tw, d, device=packed.device, dtype=packed.dtype)
        grid[:, self.order] = packed
        return _grid(grid, th, tw)

    def _from_grid(self, grid):
        toks = _tokens(grid)
        return toks if self.order is None else toks[:, self.order]

    def _step(self, model_output, timestep, sample, *args, **kwargs):
        sched = self.pipeline.scheduler
        if self.ref is None:
            return self._orig_step(model_output, timestep, sample, *args, **kwargs)
        if sched.step_index is None:
            sched._init_step_index(timestep)
        i = sched.step_index
        if self.guide is None:
            self.guide = Guide(self.cfg, sched.sigmas.tolist())
        sigma = float(sched.sigmas[i])
        if sigma <= 0:
            return self._orig_step(model_output, timestep, sample, *args, **kwargs)
        v = model_output.float()
        x = sample.float()
        x0 = self._to_grid(x - sigma * v)
        ref = self.ref.to(x0.device).expand(x0.shape[0], -1, -1, -1)
        x0 = self._from_grid(self.guide.apply(i, x0, ref, self.keep.to(x0.device)))
        v_new = ((x - x0) / sigma).to(model_output.dtype)
        return self._orig_step(v_new, timestep, sample, *args, **kwargs)

    def __enter__(self):
        if not self.supported:
            return self
        self._hook = self.pipeline.transformer.register_forward_pre_hook(self._capture, with_kwargs=True)
        sched = self.pipeline.scheduler
        self._orig_step = sched.step
        sched.step = self._step
        self._info(f"Source hold on (strength {self.cfg.early_hold:g}"
                   f"{', masked' if self.edit_mask is not None else ''})")
        return self

    def __exit__(self, *exc):
        if self._hook is not None:
            self._hook.remove()
        if self._orig_step is not None:
            # Drop the instance attribute so the class method is used again.
            try:
                del self.pipeline.scheduler.step
            except AttributeError:
                self.pipeline.scheduler.step = self._orig_step
        return False
