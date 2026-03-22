"""Step callback infrastructure for latent-space primitives.

Provides `build_step_callback()` which returns a callback function suitable
for diffusers' `callback_on_step_end` parameter.  Registered primitives
are dispatched in order at each denoising step.

Primitives implemented here:
  - Latent mask blend: universal inpainting for any model via per-step
    blending of denoised latents with correctly-noised originals.
  - Per-step progress events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modl_worker.protocol import EventEmitter


# -----------------------------------------------------------------------
# Scheduler type detection
# -----------------------------------------------------------------------

def _is_flow_matching(scheduler) -> bool:
    """Detect if a scheduler uses flow-matching (Flux, Z-Image, Qwen, Chroma)
    vs DDPM noise scheduling (SDXL, SD 1.5)."""
    cls_name = type(scheduler).__name__
    if "FlowMatch" in cls_name or "Flow" in cls_name:
        return True
    # Check config for prediction_type
    if hasattr(scheduler, "config"):
        pred = getattr(scheduler.config, "prediction_type", None)
        if pred == "flow_matching":
            return True
    return False


# -----------------------------------------------------------------------
# Latent mask blend primitive
# -----------------------------------------------------------------------

class LatentMaskBlend:
    """Per-step latent blending for universal inpainting.

    At each denoising step, replaces unmasked latents with correctly-noised
    originals, ensuring the original content is preserved while the masked
    region is regenerated.

    Works with both flow-matching (Flux, Z-Image, Qwen, Chroma) and DDPM
    (SDXL, SD 1.5) schedulers.
    """

    def __init__(self, clean_latents, mask_latents, noise, scheduler):
        """
        Args:
            clean_latents: VAE-encoded original image latents.
            mask_latents: Binary mask in latent space (1 = regenerate, 0 = preserve).
            noise: Random noise tensor (same shape as latents).
            scheduler: The pipeline's scheduler (for noise computation).
        """
        import torch
        self.clean_latents = clean_latents
        self.mask = mask_latents
        self.noise = noise
        self.scheduler = scheduler
        self.is_flow = _is_flow_matching(scheduler)

    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        """Blend at each denoising step."""
        import torch

        latents = callback_kwargs["latents"]

        if self.is_flow:
            # Flow-matching: noised = (1 - t) * clean + t * noise
            # Diffusers flow-match schedulers store sigmas; the current sigma
            # represents `t` in the flow equation.
            sigmas = self.scheduler.sigmas
            if step_index < len(sigmas):
                t = sigmas[step_index].to(device=latents.device, dtype=latents.dtype)
            else:
                t = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)

            # Reshape t for broadcasting
            while t.dim() < latents.dim():
                t = t.unsqueeze(-1)

            noised_original = (1.0 - t) * self.clean_latents + t * self.noise
        else:
            # DDPM: use scheduler.add_noise for correct noise level
            noised_original = self.scheduler.add_noise(
                self.clean_latents, self.noise, timestep.unsqueeze(0)
            )

        # Blend: mask=1 keeps denoised (regenerate), mask=0 keeps noised original (preserve)
        mask = self.mask.to(device=latents.device, dtype=latents.dtype)
        callback_kwargs["latents"] = mask * latents + (1.0 - mask) * noised_original
        return callback_kwargs


# -----------------------------------------------------------------------
# Step progress primitive
# -----------------------------------------------------------------------

class StepProgress:
    """Emit per-step progress events."""

    def __init__(self, emitter: EventEmitter, total_steps: int, image_index: int, count: int):
        self.emitter = emitter
        self.total_steps = total_steps
        self.image_index = image_index
        self.count = count

    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        # Emit progress at 10% intervals to avoid flooding
        if self.total_steps > 0 and (step_index + 1) % max(1, self.total_steps // 10) == 0:
            self.emitter.progress(
                stage="denoise",
                step=step_index + 1,
                total_steps=self.total_steps,
            )
        return callback_kwargs


# -----------------------------------------------------------------------
# Callback builder
# -----------------------------------------------------------------------

def build_step_callback(
    primitives: list,
) -> tuple:
    """Build a callback_on_step_end function from a list of primitives.

    Each primitive is a callable with signature:
        (pipe, step_index, timestep, callback_kwargs) -> callback_kwargs

    Returns:
        (callback_fn, tensor_inputs) tuple for passing to the pipeline.
        Returns (None, None) if no primitives are registered.
    """
    if not primitives:
        return None, None

    def callback_fn(pipe, step_index, timestep, callback_kwargs):
        for primitive in primitives:
            callback_kwargs = primitive(pipe, step_index, timestep, callback_kwargs)
        return callback_kwargs

    # We always need latents; prompt_embeds may be needed for prompt scheduling (future)
    tensor_inputs = ["latents"]

    return callback_fn, tensor_inputs


def prepare_mask_blend(
    pipe,
    init_image,
    mask_image,
    generator,
    arch: str,
    width: int,
    height: int,
    emitter: EventEmitter,
):
    """Prepare the LatentMaskBlend primitive.

    Encodes the init image to latent space, prepares the mask in latent space,
    and generates the noise tensor.

    Args:
        pipe: The loaded diffusers pipeline.
        init_image: PIL Image (original/clean image).
        mask_image: PIL Image (white=regenerate, black=preserve).
        generator: torch.Generator for reproducibility.
        arch: Architecture key from detect_arch().
        width: Target width.
        height: Target height.
        emitter: EventEmitter for logging.

    Returns:
        LatentMaskBlend instance.
    """
    import torch
    import numpy as np
    from PIL import Image
    from diffusers.image_processor import VaeImageProcessor

    emitter.info("Preparing latent mask blend...")

    # Resolve device — with model_cpu_offload the execution device is set
    # on the pipeline but individual components may be on CPU until called.
    device = getattr(pipe, '_execution_device', None)
    if device is None:
        import torch as _t
        device = "cuda" if _t.cuda.is_available() else "cpu"
    dtype = pipe.vae.dtype if hasattr(pipe, 'vae') else torch.bfloat16

    # VAE scale factor (for image preprocessing — NOT including latent packing)
    vae_scale = getattr(pipe, 'vae_scale_factor', 8)
    # DiT models (Flux, Z-Image, Qwen, Chroma) pack 2x2 spatial into channels
    packing_factor = 2 if arch not in ("sdxl", "sd15") else 1

    img_processor = VaeImageProcessor(vae_scale_factor=vae_scale)

    # Encode init image to latents.
    # Move VAE to execution device explicitly — with model_cpu_offload,
    # the VAE may still be on CPU at this point.
    init_resized = init_image.resize((width, height), Image.LANCZOS)
    init_tensor = img_processor.preprocess(init_resized, height=height, width=width)
    init_tensor = init_tensor.to(device=device, dtype=dtype)

    vae_was_on_cpu = next(pipe.vae.parameters()).device.type == "cpu"
    if vae_was_on_cpu:
        pipe.vae.to(device)

    with torch.no_grad():
        latent_dist = pipe.vae.encode(init_tensor)
        if hasattr(latent_dist, 'latent_dist'):
            clean_latents = latent_dist.latent_dist.mode()
        else:
            clean_latents = latent_dist.mode()

    if vae_was_on_cpu:
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()

    # Apply VAE scaling
    if hasattr(pipe.vae.config, 'scaling_factor'):
        clean_latents = clean_latents * pipe.vae.config.scaling_factor
    if hasattr(pipe.vae.config, 'shift_factor'):
        clean_latents = clean_latents - pipe.vae.config.shift_factor

    # Determine latent format: Flux/Qwen/Chroma pipelines use _pack_latents
    # (sequence format: B, seq, C*4) while Z-Image/SDXL keep spatial (B, C, H, W).
    uses_sequence_packing = hasattr(pipe, '_pack_latents')

    # Pack latents to match the format used inside the pipeline's denoising loop.
    latent_h = clean_latents.shape[2]
    latent_w = clean_latents.shape[3]
    if uses_sequence_packing:
        # Flux sequence format: (B, H/2*W/2, C*4)
        clean_latents = _pack_latents_flux(clean_latents)
        latent_h = latent_h // 2
        latent_w = latent_w // 2
    # Spatial format: latents stay as (B, C, H, W) — no packing needed

    # Prepare mask in latent space
    mask_resized = mask_image.convert("L").resize((width, height), Image.NEAREST)
    mask_np = np.array(mask_resized).astype(np.float32) / 255.0
    mask_np = (mask_np > 0.5).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

    # Downsample mask to latent resolution
    mask_latent = torch.nn.functional.interpolate(
        mask_tensor, size=(latent_h, latent_w), mode="nearest"
    )

    # Pack mask to match latent format
    if uses_sequence_packing:
        # Sequence format: (B, 1, H, W) → (B, H*W, 1) to broadcast against (B, seq, C*4)
        mask_latent = mask_latent.squeeze(1).reshape(1, latent_h * latent_w, 1)
    mask_latent = mask_latent.expand_as(clean_latents).to(device=device, dtype=clean_latents.dtype)

    # Generate noise — use CPU generator for compatibility (CUDA generators
    # can't be used with device="cpu" tensors).
    cpu_gen = torch.Generator(device="cpu")
    if generator.device.type == "cpu":
        cpu_gen.set_state(generator.get_state())
    else:
        # Seed CPU gen from the current CUDA generator's initial seed
        cpu_gen.manual_seed(generator.initial_seed())
    noise = torch.randn(
        clean_latents.shape, generator=cpu_gen, device="cpu", dtype=torch.float32,
    ).to(device=device, dtype=clean_latents.dtype)

    emitter.info(f"Mask blend ready: latent shape={list(clean_latents.shape)}, "
                 f"scheduler={'flow-matching' if _is_flow_matching(pipe.scheduler) else 'DDPM'}")

    return LatentMaskBlend(clean_latents, mask_latent, noise, pipe.scheduler)


def _pack_latents_flux(latents):
    """Pack latents from (B, C, H, W) to Flux sequence format (B, H/2*W/2, C*4).

    This matches FluxPipeline._pack_latents exactly — the format latents are in
    during the denoising loop (and thus inside callback_on_step_end).
    """
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
    return latents


def _pack_latents_spatial(latents):
    """Pack latents from (B, C, H, W) to (B, C*4, H//2, W//2).

    Used by non-Flux architectures that keep spatial dims (e.g. UNet-based).
    """
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    return latents
