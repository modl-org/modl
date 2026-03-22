"""Step callback infrastructure for latent-space primitives.

Provides `build_step_callback()` for composing per-step callbacks,
and `StepProgress` for per-step progress events.

NOTE: Mask blend inpainting is implemented in mask_blend_loop.py using
the monkey-patch approach (Approach D). The old LatentMaskBlend callback
and prepare_mask_blend() have been removed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modl_worker.protocol import EventEmitter


class StepProgress:
    """Emit per-step progress events."""

    def __init__(self, emitter: EventEmitter, total_steps: int, image_index: int, count: int):
        self.emitter = emitter
        self.total_steps = total_steps
        self.image_index = image_index
        self.count = count

    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        if self.total_steps > 0 and (step_index + 1) % max(1, self.total_steps // 10) == 0:
            self.emitter.progress(
                stage="denoise",
                step=step_index + 1,
                total_steps=self.total_steps,
            )
        return callback_kwargs


def build_step_callback(primitives: list) -> tuple:
    """Build a callback_on_step_end function from a list of primitives.

    Returns (callback_fn, tensor_inputs) or (None, None) if no primitives.
    """
    if not primitives:
        return None, None

    def callback_fn(pipe, step_index, timestep, callback_kwargs):
        for primitive in primitives:
            callback_kwargs = primitive(pipe, step_index, timestep, callback_kwargs)
        return callback_kwargs

    tensor_inputs = ["latents"]
    return callback_fn, tensor_inputs
