"""Mask-blend inpainting via monkey-patched prepare_latents.

Approach D: don't fight the pipeline. Instead:
1. Pre-encode prompt with the ORIGINAL image (CLIP style anchor)
2. VAE-encode the padded image, create latent mask, seed with noise
3. Monkey-patch prepare_latents to return our seeded latents
4. Use callback_on_step_end for per-step mask re-blending
5. Call pipe() normally — it handles dtype, hooks, VRAM, packing

This matches ComfyUI's ImagePadForOutpaint → VAEEncode → KSampler flow.
"""

import torch
import numpy as np
from PIL import Image
from diffusers.image_processor import VaeImageProcessor


def run_mask_blend(
    pipe,
    prompt: str,
    init_image: Image.Image,
    mask_image: Image.Image,
    condition_image: Image.Image | None,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int | None,
    arch: str,
    emitter,
    negative_prompt: str = "",
    gen_kwargs: dict | None = None,
) -> Image.Image:
    """Run mask-blend inpainting using the pipeline's own denoising loop.

    Returns the output PIL Image.
    """
    import inspect

    if gen_kwargs is None:
        gen_kwargs = {}

    device = getattr(pipe, '_execution_device', None) or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pipe.vae.dtype
    vae_scale = getattr(pipe, 'vae_scale_factor', 8)

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    # --- 1. CLIP-encode prompt with ORIGINAL image (style anchor) ---
    # The vision encoder should see the clean scene, not the padded canvas.
    encoder_image = condition_image or init_image
    emitter.info("Encoding prompt with original image (split routing)...")

    prompt_embeds, neg_embeds = _encode_prompt_split(
        pipe, prompt, negative_prompt, encoder_image, arch, device,
    )

    # --- 2. VAE-encode padded image + create seeded latents ---
    emitter.info("Preparing latent canvas...")

    # Resize images to target dimensions
    init_resized = init_image.convert("RGB").resize((width, height), Image.LANCZOS)
    mask_resized = mask_image.convert("L").resize((width, height), Image.NEAREST)

    # Prepare mask as float tensor
    mask_np = np.array(mask_resized).astype(np.float32) / 255.0
    # Keep continuous values for feathered blending

    # Create padded image with noise in masked areas (pixel space)
    init_np = np.array(init_resized).astype(np.float32) / 255.0
    rng = np.random.RandomState(seed % (2**31) if seed else 42)
    pixel_noise = rng.rand(*init_np.shape).astype(np.float32)
    mask_3ch = mask_np[:, :, np.newaxis]
    padded_np = (1.0 - mask_3ch) * init_np + mask_3ch * pixel_noise
    padded_pil = Image.fromarray((np.clip(padded_np, 0, 1) * 255).astype(np.uint8))

    # VAE encode the padded image
    img_processor = VaeImageProcessor(vae_scale_factor=vae_scale)
    padded_tensor = img_processor.preprocess(padded_pil, height=height, width=width)
    padded_tensor = padded_tensor.to(device=device, dtype=dtype)

    with torch.no_grad():
        enc = pipe.vae.encode(padded_tensor)
        padded_latents = enc.latent_dist.mode() if hasattr(enc, 'latent_dist') else enc.mode()

    if hasattr(pipe.vae.config, 'scaling_factor'):
        padded_latents = padded_latents * pipe.vae.config.scaling_factor
    if hasattr(pipe.vae.config, 'shift_factor'):
        padded_latents = padded_latents - pipe.vae.config.shift_factor

    # Also encode clean original for the re-blend reference
    clean_tensor = img_processor.preprocess(init_resized, height=height, width=width)
    clean_tensor = clean_tensor.to(device=device, dtype=dtype)

    with torch.no_grad():
        enc2 = pipe.vae.encode(clean_tensor)
        clean_latents = enc2.latent_dist.mode() if hasattr(enc2, 'latent_dist') else enc2.mode()

    if hasattr(pipe.vae.config, 'scaling_factor'):
        clean_latents = clean_latents * pipe.vae.config.scaling_factor
    if hasattr(pipe.vae.config, 'shift_factor'):
        clean_latents = clean_latents - pipe.vae.config.shift_factor

    # Prepare latent-resolution mask
    latent_h, latent_w = padded_latents.shape[2], padded_latents.shape[3]
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    latent_mask = torch.nn.functional.interpolate(
        mask_tensor, size=(latent_h, latent_w), mode="bilinear", align_corners=False,
    ).to(device=device, dtype=padded_latents.dtype)

    # Generate noise and create initial seeded latents
    noise = torch.randn(
        padded_latents.shape, generator=generator, device="cpu", dtype=torch.float32,
    ).to(device=device, dtype=padded_latents.dtype)

    # Seed: original areas get noised-original, masked areas get pure noise
    # This is what ComfyUI's VAEEncode + noise_mask does
    seeded_latents = latent_mask * noise + (1.0 - latent_mask) * padded_latents

    emitter.info(f"Latent canvas: {latent_h}x{latent_w}, mask coverage: {mask_np.mean():.0%}")

    # --- 3. Monkey-patch prepare_latents ---
    # Let the pipeline generate random noise in its own format (handles packing),
    # then overwrite the values with our seeded latents.
    real_prepare = pipe.prepare_latents

    def patched_prepare_latents(*args, **kwargs):
        """Replace random noise with our seeded latents in the pipeline's format.

        The pipeline creates noise at its expected channel count (may be > VAE channels
        for models like Klein where in_channels=128 but VAE outputs 16ch). We fold and
        pack our seeded latents to match, then overwrite in the result tensor.
        """
        result = real_prepare(*args, **kwargs)
        packed_latents = result[0] if isinstance(result, tuple) else result

        # Fold our (B, C_vae, H, W) latents to match pipeline's spatial format
        sl = seeded_latents
        b, c_vae, h, w = sl.shape

        if h % 2 == 0 and w % 2 == 0:
            # Spatial fold: (B, C, H, W) → (B, C*4, H/2, W/2)
            folded = sl.reshape(b, c_vae, h // 2, 2, w // 2, 2)
            folded = folded.permute(0, 1, 3, 5, 2, 4).reshape(b, c_vae * 4, h // 2, w // 2)
        else:
            folded = sl

        # Pipeline may use more channels than VAE output (e.g. Klein: 128 vs 64).
        # Pad with the pipeline's noise in the extra channels.
        target_ch = packed_latents.shape[-1] if packed_latents.dim() == 3 else packed_latents.shape[1]
        if folded.shape[1] < target_ch:
            # Need to pad channels — unpack pipeline's noise to get the extra channels
            # Actually, we can just pack our folded and let the extra channels keep pipeline noise
            pass  # handled below via selective copy

        # Pack to sequence format if pipeline uses it
        if packed_latents.dim() == 3 and folded.dim() == 4 and hasattr(pipe, '_pack_latents'):
            try:
                folded_packed = pipe._pack_latents(folded, *folded.shape)
            except TypeError:
                folded_packed = pipe._pack_latents(folded)
        else:
            folded_packed = folded

        folded_packed = folded_packed.to(device=packed_latents.device, dtype=packed_latents.dtype)

        # If channel counts match, direct copy
        if folded_packed.shape == packed_latents.shape:
            packed_latents.copy_(folded_packed)
        elif folded_packed.dim() == 3 and packed_latents.dim() == 3:
            # Sequence format: (B, seq, C). Copy our channels, keep pipeline noise for rest
            our_ch = folded_packed.shape[-1]
            packed_latents[:, :, :our_ch] = folded_packed
        elif folded_packed.dim() == 4 and packed_latents.dim() == 4:
            # Spatial: (B, C, H, W). Copy our channels
            our_ch = folded_packed.shape[1]
            packed_latents[:, :our_ch] = folded_packed
        else:
            # Fallback: just copy what fits
            emitter.info(f"WARNING: shape mismatch in prepare_latents: ours={folded_packed.shape} vs pipeline={packed_latents.shape}")
            packed_latents.copy_(folded_packed)

        return result

    pipe.prepare_latents = patched_prepare_latents

    # --- 4. Callback for per-step mask re-blend ---
    # Blend in spatial (B, C, H, W) format regardless of pipeline's internal packing.
    # On each step: unpack → blend → repack. Simple and format-agnostic.
    is_flow = _is_flow_matching(pipe.scheduler)

    # Store spatial-format tensors for the callback.
    # clean_latents, noise, latent_mask are already (B, C_vae, H, W).
    # The callback will work in this spatial space.
    spatial_h, spatial_w = latent_h, latent_w  # VAE output spatial dims
    c_vae = clean_latents.shape[1]

    def reblend_callback(pipe_ref, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        lat_device = latents.device
        lat_dtype = latents.dtype

        # --- Unpack to spatial if needed ---
        if latents.dim() == 3:
            # Sequence format (B, seq_len, C_packed)
            b, seq_len, c_packed = latents.shape
            # Reshape to spatial: (B, C_packed, sqrt(seq), sqrt(seq)) or known dims
            spatial = latents.permute(0, 2, 1).reshape(b, c_packed, spatial_h // 2, spatial_w // 2)
            was_packed = True
        else:
            spatial = latents
            was_packed = False

        # The pipeline's latent channels may exceed VAE channels (Klein: 128 vs 64).
        # We only blend the VAE channels; leave extra channels untouched.
        c_lat = spatial.shape[1]
        c_blend = min(c_vae * 4 if was_packed else c_vae, c_lat)

        # Get spatial clean/noise at matching resolution
        cl = clean_latents.to(device=lat_device, dtype=lat_dtype)
        ns = noise.to(device=lat_device, dtype=lat_dtype)
        mk = latent_mask.to(device=lat_device, dtype=lat_dtype)

        # If pipeline folded spatially (H/2, W/2 with C*4), fold our references too
        if was_packed and cl.shape[2] != spatial.shape[2]:
            b2, c2, h2, w2 = cl.shape
            cl = cl.reshape(b2, c2, h2 // 2, 2, w2 // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b2, c2 * 4, h2 // 2, w2 // 2)
            ns = ns.reshape(b2, c2, h2 // 2, 2, w2 // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b2, c2 * 4, h2 // 2, w2 // 2)
            mk = torch.nn.functional.interpolate(mk, size=(h2 // 2, w2 // 2), mode="nearest")

        # Expand mask to blend channels
        mk_blend = mk.expand(1, c_blend, mk.shape[2], mk.shape[3])

        # Re-noise original to current noise level
        if is_flow:
            sigmas = pipe_ref.scheduler.sigmas
            next_idx = step_index + 1
            if next_idx < len(sigmas):
                sigma = sigmas[next_idx].to(device=lat_device, dtype=lat_dtype)
            else:
                sigma = torch.tensor(0.0, device=lat_device, dtype=lat_dtype)
            sigma = sigma.reshape(1, 1, 1, 1)
            noised_orig = sigma * ns + (1.0 - sigma) * cl
        else:
            noised_orig = pipe_ref.scheduler.add_noise(cl, ns, timestep.unsqueeze(0))

        # Blend VAE channels only; keep extra channels from model prediction
        blended = spatial.clone()
        blended[:, :c_blend] = mk_blend * spatial[:, :c_blend] + (1.0 - mk_blend) * noised_orig[:, :c_blend]

        # --- Repack if needed ---
        if was_packed:
            callback_kwargs["latents"] = blended.reshape(b, c_packed, -1).permute(0, 2, 1)
        else:
            callback_kwargs["latents"] = blended

        return callback_kwargs

    # --- 5. Build pipeline kwargs ---
    pipe_kwargs = dict(gen_kwargs)  # copy base kwargs
    pipe_kwargs["prompt"] = prompt
    pipe_kwargs["num_inference_steps"] = steps
    pipe_kwargs["generator"] = generator
    pipe_kwargs["width"] = width
    pipe_kwargs["height"] = height
    pipe_kwargs["callback_on_step_end"] = reblend_callback
    pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    # Inject pre-computed prompt embeddings for vision-language models (split routing).
    # For text-only encoders, skip pre-encoding — the encoder doesn't see the image anyway.
    if prompt_embeds is not None and arch in ("qwen_image", "qwen_image_edit"):
        pipe_kwargs["prompt_embeds"] = prompt_embeds
        pipe_kwargs.pop("prompt", None)
        if neg_embeds is not None:
            pipe_kwargs["negative_prompt_embeds"] = neg_embeds

    # Architecture-specific kwargs
    _add_arch_kwargs(pipe_kwargs, arch, guidance)

    # --- 6. Run pipeline ---
    emitter.info(f"Denoising {steps} steps...")
    try:
        result = pipe(**pipe_kwargs)
        image = result.images[0]
    finally:
        pipe.prepare_latents = real_prepare  # always restore

    return image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_flow_matching(scheduler) -> bool:
    cls_name = type(scheduler).__name__
    if "FlowMatch" in cls_name or "Flow" in cls_name:
        return True
    if hasattr(scheduler, "config"):
        if getattr(scheduler.config, "prediction_type", None) == "flow_matching":
            return True
    return False


def _add_arch_kwargs(kwargs, arch, guidance):
    """Add architecture-specific pipeline kwargs."""
    if arch in ("qwen_image", "qwen_image_edit"):
        kwargs["true_cfg_scale"] = guidance
        kwargs.setdefault("negative_prompt", " ")
    elif arch == "chroma":
        kwargs["guidance_scale"] = guidance
        kwargs.setdefault("negative_prompt", "low quality, ugly, deformed")
    else:
        kwargs["guidance_scale"] = guidance


def _encode_prompt_split(pipe, prompt, negative_prompt, original_image, arch, device):
    """Encode prompt with the original (unpadded) image for CLIP style anchoring.

    Returns (prompt_embeds, neg_embeds) or (None, None) if the pipeline
    doesn't support prompt_embeds injection.
    """
    import inspect

    # Check if pipeline supports prompt_embeds parameter
    call_sig = inspect.signature(pipe.__call__)
    if "prompt_embeds" not in call_sig.parameters:
        return None, None

    # Check if encode_prompt accepts an image parameter (vision-language models)
    encode_sig = inspect.signature(pipe.encode_prompt)
    encode_params = set(encode_sig.parameters.keys())

    try:
        if "image" in encode_params:
            # Vision-language encoder (Qwen, Klein) — pass original image
            result = pipe.encode_prompt(prompt=prompt, image=original_image, device=device)
        else:
            # Text-only encoder (Flux, Z-Image, SDXL)
            encode_kwargs = {"prompt": prompt, "device": device}
            if "prompt_2" in encode_params:
                encode_kwargs["prompt_2"] = prompt
            result = pipe.encode_prompt(**encode_kwargs)

        # Handle various return formats
        if isinstance(result, tuple):
            prompt_embeds = result[0]
        else:
            prompt_embeds = result

        # Encode negative prompt
        neg_embeds = None
        if negative_prompt:
            if "image" in encode_params:
                neg_result = pipe.encode_prompt(prompt=negative_prompt, image=original_image, device=device)
            else:
                neg_kwargs = {"prompt": negative_prompt, "device": device}
                if "prompt_2" in encode_params:
                    neg_kwargs["prompt_2"] = negative_prompt
                neg_result = pipe.encode_prompt(**neg_kwargs)

            neg_embeds = neg_result[0] if isinstance(neg_result, tuple) else neg_result

        return prompt_embeds, neg_embeds

    except Exception as e:
        # If encoding fails, return None and let the pipeline handle it
        import traceback
        traceback.print_exc()
        return None, None
