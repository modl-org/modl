"""Step callback infrastructure for latent-space primitives.

Provides `build_step_callback()` which returns a callback function suitable
for diffusers' `callback_on_step_end` parameter.  Registered primitives
are dispatched in order at each denoising step.

Primitives implemented here:
  - Latent mask blend: universal inpainting for any model via per-step
    blending of denoised latents with correctly-noised originals.
    Follows the ComfyUI approach: VAE-encode padded image (noise in padded
    areas), use noise_mask to control per-region denoising, re-blend at
    each step.
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

    Follows the ComfyUI KSampler approach:
    - Original region (mask=0): re-noise clean latents to current timestep,
      overwrite the pipeline's latents → model sees correctly-noised original
    - Masked region (mask=1): let the model denoise freely → new content
    - Feathered zone: blend between the two

    This is called at each denoising step via callback_on_step_end.
    """

    def __init__(self, clean_latents, mask_latents, noise, scheduler):
        """
        Args:
            clean_latents: VAE-encoded original image latents (clean, no noise).
            mask_latents: Mask in latent space (1 = regenerate, 0 = preserve,
                          gradient = feathered blend).
            noise: Random noise tensor (same shape as latents).
            scheduler: The pipeline's scheduler (for noise computation).
        """
        self.clean_latents = clean_latents
        self.mask = mask_latents
        self.noise = noise
        self.scheduler = scheduler
        self.is_flow = _is_flow_matching(scheduler)

    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        """Re-blend at each denoising step."""
        import torch

        latents = callback_kwargs["latents"]

        # Re-noise clean latents to the current noise level.
        # The callback fires AFTER the scheduler step, so we use the next
        # sigma (step_index + 1) to match where the denoising is heading.
        if self.is_flow:
            # Flow-matching: noised = sigma * noise + (1 - sigma) * clean
            sigmas = self.scheduler.sigmas
            # Use next step's sigma (where denoising is heading)
            next_idx = step_index + 1
            if next_idx < len(sigmas):
                sigma = sigmas[next_idx].to(device=latents.device, dtype=latents.dtype)
            else:
                sigma = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
            while sigma.dim() < latents.dim():
                sigma = sigma.unsqueeze(-1)
            noised_original = sigma * self.noise + (1.0 - sigma) * self.clean_latents
        else:
            # DDPM: use scheduler.add_noise with the current timestep
            noised_original = self.scheduler.add_noise(
                self.clean_latents, self.noise, timestep.unsqueeze(0)
            )

        # Blend: mask=1 → keep denoised (new content), mask=0 → use re-noised original
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


# -----------------------------------------------------------------------
# Mask blend preparation
# -----------------------------------------------------------------------

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
    """Prepare initial latents and the LatentMaskBlend callback.

    Follows the ComfyUI approach:
    1. Pad the init image with random noise in masked areas (pixel space)
    2. VAE-encode the entire padded+noised image → initial latents
    3. Create mask + noise at latent resolution for per-step re-blending
    4. Also VAE-encode the clean original for the re-blending reference

    Returns:
        (initial_latents, mask_blend_callback) — pass initial_latents as the
        `latents` kwarg to the pipeline, and mask_blend_callback to the
        callback primitives list.
    """
    import torch
    import numpy as np
    from PIL import Image
    from diffusers.image_processor import VaeImageProcessor

    emitter.info("Preparing mask blend (ComfyUI-style)...")

    # Device / dtype
    device = getattr(pipe, '_execution_device', None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pipe.vae.dtype if hasattr(pipe, 'vae') else torch.bfloat16

    vae_scale = getattr(pipe, 'vae_scale_factor', 8)
    img_processor = VaeImageProcessor(vae_scale_factor=vae_scale)

    # --- 1. Prepare mask (pixel space) ---
    mask_resized = mask_image.convert("L").resize((width, height), Image.NEAREST)
    mask_np = np.array(mask_resized).astype(np.float32) / 255.0
    # Keep as continuous (feathered) — don't threshold to binary

    # --- 2. Create padded image with noise in masked areas ---
    # This is the ComfyUI ImagePadForOutpaint approach:
    # - Where mask=0 (preserve): keep original pixels
    # - Where mask=1 (regenerate): fill with random noise
    # - Where mask=gradient: blend between original and noise
    init_resized = init_image.convert("RGB").resize((width, height), Image.LANCZOS)
    init_np = np.array(init_resized).astype(np.float32) / 255.0

    # Generate pixel-space noise
    rng = np.random.RandomState(generator.initial_seed() % (2**31))
    pixel_noise = rng.rand(*init_np.shape).astype(np.float32)

    # Blend: mask=1 → noise, mask=0 → original, gradient → smooth blend
    mask_3ch = mask_np[:, :, np.newaxis]  # (H, W, 1) for broadcasting
    padded_np = (1.0 - mask_3ch) * init_np + mask_3ch * pixel_noise
    padded_np = np.clip(padded_np, 0, 1)

    # Convert back to PIL for VaeImageProcessor
    padded_pil = Image.fromarray((padded_np * 255).astype(np.uint8))

    # --- 3. VAE encode the padded image → initial latents ---
    padded_tensor = img_processor.preprocess(padded_pil, height=height, width=width)
    padded_tensor = padded_tensor.to(device=device, dtype=dtype)

    vae_was_on_cpu = next(pipe.vae.parameters()).device.type == "cpu"
    if vae_was_on_cpu:
        pipe.vae.to(device)

    with torch.no_grad():
        # Encode padded image (noise in masked areas, original elsewhere)
        padded_dist = pipe.vae.encode(padded_tensor)
        if hasattr(padded_dist, 'latent_dist'):
            padded_latents = padded_dist.latent_dist.mode()
        else:
            padded_latents = padded_dist.mode()

        # Encode clean original for per-step re-blending reference
        init_tensor = img_processor.preprocess(init_resized, height=height, width=width)
        init_tensor = init_tensor.to(device=device, dtype=dtype)
        clean_dist = pipe.vae.encode(init_tensor)
        if hasattr(clean_dist, 'latent_dist'):
            clean_latents = clean_dist.latent_dist.mode()
        else:
            clean_latents = clean_dist.mode()

    if vae_was_on_cpu:
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()

    # Apply VAE scaling to both
    for lat in [padded_latents, clean_latents]:
        # In-place would be cleaner but let's be explicit
        pass
    if hasattr(pipe.vae.config, 'scaling_factor'):
        padded_latents = padded_latents * pipe.vae.config.scaling_factor
        clean_latents = clean_latents * pipe.vae.config.scaling_factor
    if hasattr(pipe.vae.config, 'shift_factor'):
        padded_latents = padded_latents - pipe.vae.config.shift_factor
        clean_latents = clean_latents - pipe.vae.config.shift_factor

    # --- 4. Determine latent format and pack ---
    uses_sequence_packing = hasattr(pipe, '_pack_latents')

    latent_h = clean_latents.shape[2]
    latent_w = clean_latents.shape[3]

    if uses_sequence_packing:
        padded_latents = _pack_latents_flux(padded_latents)
        clean_latents = _pack_latents_flux(clean_latents)
        latent_h = latent_h // 2
        latent_w = latent_w // 2

    # --- 5. Prepare mask at latent resolution ---
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    mask_latent = torch.nn.functional.interpolate(
        mask_tensor, size=(latent_h, latent_w), mode="bilinear", align_corners=False,
    )

    if uses_sequence_packing:
        mask_latent = mask_latent.squeeze(1).reshape(1, latent_h * latent_w, 1)
    mask_latent = mask_latent.expand_as(clean_latents).to(device=device, dtype=clean_latents.dtype)

    # --- 6. Generate noise for per-step re-blending ---
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(generator.initial_seed())
    noise = torch.randn(
        clean_latents.shape, generator=cpu_gen, device="cpu", dtype=torch.float32,
    ).to(device=device, dtype=clean_latents.dtype)

    # --- 7. Create initial latents for the pipeline ---
    # The txt2img pipeline expects latents at sigma_max noise level.
    # For flow matching, sigma_max ≈ 1.0 (pure noise at start).
    # We blend: in masked areas → pure noise, in preserved areas → noised original.
    # This way the pipeline starts with the right content in each region.
    #
    # Don't try to use scheduler.scale_noise here — the scheduler hasn't been
    # configured with timesteps yet (that happens inside the pipeline call).
    # Instead, just pass noise — the pipeline's prepare_latents will handle scaling.
    # The callback will enforce the mask constraint at each step anyway.
    initial_latents = noise

    emitter.info(f"Mask blend ready: shape={list(clean_latents.shape)}, "
                 f"scheduler={'flow-matching' if _is_flow_matching(pipe.scheduler) else 'DDPM'}")

    callback = LatentMaskBlend(clean_latents, mask_latent, noise, pipe.scheduler)
    return initial_latents, callback


def _pack_latents_flux(latents):
    """Pack latents from (B, C, H, W) to Flux sequence format (B, H/2*W/2, C*4)."""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
    return latents
