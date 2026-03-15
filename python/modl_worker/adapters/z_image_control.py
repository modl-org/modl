"""Z-Image ControlNet: single-model wrapper for model_cpu_offload compatibility.

Diffusers' two-model approach (ZImageControlNetModel + ZImageTransformer2DModel)
uses from_transformer() to share modules between them. This breaks
model_cpu_offload because the controlnet forward runs before the transformer's
forward, and the shared modules haven't been moved to CUDA yet.

This wrapper combines both models into a single nn.Module so model_cpu_offload
moves everything to CUDA together. The wrapper's forward runs the controlnet
first (to compute hints), then the transformer with those hints.

The control context (VAE-encoded control image) is set as state on the wrapper
before the pipeline runs, so no pipeline modification is needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.controlnets.controlnet_z_image import ZImageControlNetModel


class ZImageControlWrapper(nn.Module):
    """Wraps a ZImageTransformer2DModel and ZImageControlNetModel into one module.

    Usage::

        wrapper = ZImageControlWrapper(transformer, controlnet)
        wrapper.set_control(control_latents, scale=0.75)
        pipeline.transformer = wrapper
        pipeline.enable_model_cpu_offload()
        result = pipeline(...)  # wrapper.forward() handles controlnet internally
        wrapper.clear_control()
    """

    def __init__(self, transformer, controlnet):
        super().__init__()
        self.transformer = transformer
        # from_transformer shares embedders/refiners from the transformer
        # into the controlnet. Since both are submodules of this wrapper,
        # model_cpu_offload moves everything together — no device mismatch.
        self.controlnet = ZImageControlNetModel.from_transformer(controlnet, transformer)

        # Control state (set before pipeline run)
        self._control_context = None
        self._control_scale = 0.75

    def set_control(self, control_context: list[torch.Tensor], scale: float = 0.75):
        """Set the control context for the next pipeline run."""
        self._control_context = control_context
        self._control_scale = scale

    def clear_control(self):
        """Clear control state after pipeline run."""
        self._control_context = None

    # Forward proxy attributes that the pipeline/diffusers expects on the
    # transformer (config, dtype, device, etc.)
    @property
    def config(self):
        return self.transformer.config

    @property
    def dtype(self):
        return self.transformer.dtype

    @property
    def device(self):
        return self.transformer.device

    @property
    def in_channels(self):
        return self.transformer.in_channels

    def forward(self, x, t, cap_feats, **kwargs):
        # If control context is set, run the controlnet to get hints
        controlnet_block_samples = kwargs.pop("controlnet_block_samples", None)
        if self._control_context is not None and controlnet_block_samples is None:
            device = x[0].device
            dtype = x[0].dtype
            control_ctx = [c.to(device=device, dtype=dtype) for c in self._control_context]

            # When CFG is active, x has 2x batch (positive + negative).
            # Replicate control context to match.
            if len(x) > len(control_ctx):
                control_ctx = control_ctx * (len(x) // len(control_ctx))

            controlnet_block_samples = self.controlnet(
                x, t, cap_feats, control_ctx,
                conditioning_scale=self._control_scale,
                patch_size=kwargs.get("patch_size", 2),
                f_patch_size=kwargs.get("f_patch_size", 1),
            )

        return self.transformer(
            x, t, cap_feats,
            controlnet_block_samples=controlnet_block_samples,
            **kwargs,
        )
