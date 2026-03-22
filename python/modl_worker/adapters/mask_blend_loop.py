"""Mask-blend inpainting via monkey-patched prepare_latents.

Approach D: don't fight the pipeline. Instead:
1. Pre-encode prompt with the ORIGINAL image (CLIP style anchor)
2. VAE-encode the padded image, create latent mask, seed with noise
3. Monkey-patch prepare_latents to inject our seeded latents
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
    """Run mask-blend inpainting using the pipeline's own denoising loop."""
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
    encoder_image = condition_image or init_image
    emitter.info("Encoding prompt with original image (split routing)...")
    prompt_embeds, neg_embeds = _encode_prompt_split(
        pipe, prompt, negative_prompt, encoder_image, arch, device,
    )

    # --- 2. VAE-encode + create seeded latents ---
    emitter.info("Preparing latent canvas...")

    init_resized = init_image.convert("RGB").resize((width, height), Image.LANCZOS)
    mask_resized = mask_image.convert("L").resize((width, height), Image.NEAREST)
    mask_np = np.array(mask_resized).astype(np.float32) / 255.0

    # Pixel-space noise in masked areas (ComfyUI's ImagePadForOutpaint)
    init_np = np.array(init_resized).astype(np.float32) / 255.0
    rng = np.random.RandomState(seed % (2**31) if seed else 42)
    pixel_noise = rng.rand(*init_np.shape).astype(np.float32)
    mask_3ch = mask_np[:, :, np.newaxis]
    padded_np = (1.0 - mask_3ch) * init_np + mask_3ch * pixel_noise
    padded_pil = Image.fromarray((np.clip(padded_np, 0, 1) * 255).astype(np.uint8))

    img_processor = VaeImageProcessor(vae_scale_factor=vae_scale)

    # VAE encode padded image
    padded_tensor = img_processor.preprocess(padded_pil, height=height, width=width)
    padded_tensor = padded_tensor.to(device=device, dtype=dtype)
    with torch.no_grad():
        enc = pipe.vae.encode(padded_tensor)
        padded_latents = enc.latent_dist.mode() if hasattr(enc, 'latent_dist') else enc.mode()

    # VAE encode clean original (for per-step re-blend reference)
    clean_tensor = img_processor.preprocess(init_resized, height=height, width=width)
    clean_tensor = clean_tensor.to(device=device, dtype=dtype)
    with torch.no_grad():
        enc2 = pipe.vae.encode(clean_tensor)
        clean_latents = enc2.latent_dist.mode() if hasattr(enc2, 'latent_dist') else enc2.mode()

    # Apply VAE scaling — CORRECT ORDER: (raw - shift) * scale
    # Inverse of decode which does: (latents / scale) + shift
    for lat in [padded_latents, clean_latents]:
        if hasattr(pipe.vae.config, 'shift_factor'):
            lat.sub_(pipe.vae.config.shift_factor)
        if hasattr(pipe.vae.config, 'scaling_factor'):
            lat.mul_(pipe.vae.config.scaling_factor)

    # Latent-resolution mask
    latent_h, latent_w = padded_latents.shape[2], padded_latents.shape[3]
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    latent_mask = torch.nn.functional.interpolate(
        mask_tensor, size=(latent_h, latent_w), mode="bilinear", align_corners=False,
    ).to(device=device, dtype=padded_latents.dtype)

    # Generate noise
    noise = torch.randn(
        padded_latents.shape, generator=generator, device="cpu", dtype=torch.float32,
    ).to(device=device, dtype=padded_latents.dtype)

    # Seeded latents: preserved areas = padded (encoded original), masked = noise
    seeded_latents = latent_mask * noise + (1.0 - latent_mask) * padded_latents

    emitter.info(f"Latent canvas: {latent_h}x{latent_w}, mask coverage: {mask_np.mean():.0%}")

    # --- 3. Monkey-patch prepare_latents ---
    real_prepare = pipe.prepare_latents

    def patched_prepare_latents(*args, **kwargs):
        """Inject seeded latents into the pipeline's own noise tensor."""
        result = real_prepare(*args, **kwargs)
        packed = result[0] if isinstance(result, tuple) else result

        # Transform seeded_latents from (B, C_vae, H, W) to pipeline format.
        sl = seeded_latents
        b, c_vae, h, w = sl.shape

        # Step 1: Spatial fold (B, C, H, W) → (B, C*4, H/2, W/2)
        if h % 2 == 0 and w % 2 == 0:
            folded = sl.reshape(b, c_vae, h // 2, 2, w // 2, 2)
            folded = folded.permute(0, 1, 3, 5, 2, 4).reshape(b, c_vae * 4, h // 2, w // 2)
        else:
            folded = sl

        # Step 2: Sequence pack if pipeline uses it
        if packed.dim() == 3 and folded.dim() == 4 and hasattr(pipe, '_pack_latents'):
            try:
                folded = pipe._pack_latents(folded, *folded.shape)
            except TypeError:
                folded = pipe._pack_latents(folded)

        folded = folded.to(device=packed.device, dtype=packed.dtype)

        # Step 3: Copy our channels, keep pipeline noise in extra channels
        if folded.shape == packed.shape:
            packed.copy_(folded)
        elif folded.dim() == 3 and packed.dim() == 3:
            packed[:, :, :folded.shape[-1]] = folded
        elif folded.dim() == 4 and packed.dim() == 4:
            packed[:, :folded.shape[1]] = folded

        return result

    pipe.prepare_latents = patched_prepare_latents

    # --- 4. Callback for per-step mask re-blend ---
    # Key insight from Pedro: blend in unpacked spatial format, then repack.
    # The callback unpacks sequence → spatial, blends, repacks.
    is_flow = _is_flow_matching(pipe.scheduler)
    c_vae_val = clean_latents.shape[1]

    def reblend_callback(pipe_ref, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        lat_dev = latents.device
        lat_dt = latents.dtype

        # Unpack to spatial
        if latents.dim() == 3:
            b, seq_len, c_packed = latents.shape
            # Klein packs (B,C,H,W) → (B,H*W,C) directly
            # Flux packs (B,C,H//2,W//2) with 2x2 fold → (B,H//2*W//2,C*4)
            # Either way: permute to (B, C, H, W) spatial
            spatial = latents.permute(0, 2, 1).reshape(b, c_packed, latent_h // 2, latent_w // 2)
            was_seq = True
        else:
            spatial = latents
            was_seq = False

        # Determine how many channels to blend (VAE channels only)
        c_spatial = spatial.shape[1]
        c_blend = min(c_vae_val * 4 if was_seq else c_vae_val, c_spatial)

        # Fold clean/noise to match spatial format
        cl = clean_latents.to(device=lat_dev, dtype=lat_dt)
        ns = noise.to(device=lat_dev, dtype=lat_dt)
        mk = latent_mask.to(device=lat_dev, dtype=lat_dt)

        if was_seq and cl.shape[2] != spatial.shape[2]:
            b2, c2, h2, w2 = cl.shape
            cl = cl.reshape(b2, c2, h2//2, 2, w2//2, 2).permute(0,1,3,5,2,4).reshape(b2, c2*4, h2//2, w2//2)
            ns = ns.reshape(b2, c2, h2//2, 2, w2//2, 2).permute(0,1,3,5,2,4).reshape(b2, c2*4, h2//2, w2//2)
            mk = torch.nn.functional.interpolate(mk, size=(h2//2, w2//2), mode="nearest")

        mk_exp = mk.expand(1, c_blend, mk.shape[2], mk.shape[3])

        # Re-noise clean latents to current noise level
        if is_flow:
            sigmas = pipe_ref.scheduler.sigmas
            # callback fires AFTER scheduler step: latents are at step_index+1 noise level
            next_idx = step_index + 1
            if next_idx < len(sigmas):
                sigma = sigmas[next_idx].to(device=lat_dev, dtype=lat_dt)
            else:
                sigma = torch.tensor(0.0, device=lat_dev, dtype=lat_dt)
            sigma = sigma.reshape(1, 1, 1, 1)
            noised_orig = sigma * ns + (1.0 - sigma) * cl
        else:
            noised_orig = pipe_ref.scheduler.add_noise(cl, ns, timestep.unsqueeze(0))

        # Blend: mask=1 → keep denoised, mask=0 → keep re-noised original
        blended = spatial.clone()
        blended[:, :c_blend] = mk_exp * spatial[:, :c_blend] + (1.0 - mk_exp) * noised_orig[:, :c_blend]

        # Repack
        if was_seq:
            callback_kwargs["latents"] = blended.reshape(b, c_packed, -1).permute(0, 2, 1)
        else:
            callback_kwargs["latents"] = blended

        return callback_kwargs

    # --- 5. Build pipeline kwargs ---
    pipe_kwargs = dict(gen_kwargs)
    pipe_kwargs["prompt"] = prompt
    pipe_kwargs["num_inference_steps"] = steps
    pipe_kwargs["generator"] = generator
    pipe_kwargs["width"] = width
    pipe_kwargs["height"] = height
    pipe_kwargs["callback_on_step_end"] = reblend_callback
    pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    # Inject pre-computed embeddings for vision-language models only
    if prompt_embeds is not None and arch in ("qwen_image", "qwen_image_edit"):
        pipe_kwargs["prompt_embeds"] = prompt_embeds
        pipe_kwargs.pop("prompt", None)
        if neg_embeds is not None:
            pipe_kwargs["negative_prompt_embeds"] = neg_embeds

    _add_arch_kwargs(pipe_kwargs, arch, guidance)

    # --- 6. Run ---
    emitter.info(f"Denoising {steps} steps...")
    try:
        result = pipe(**pipe_kwargs)
        image = result.images[0]
    finally:
        pipe.prepare_latents = real_prepare
    return image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_flow_matching(scheduler) -> bool:
    cls_name = type(scheduler).__name__
    return ("FlowMatch" in cls_name or "Flow" in cls_name or
            getattr(getattr(scheduler, "config", None), "prediction_type", None) == "flow_matching")


def _add_arch_kwargs(kwargs, arch, guidance):
    if arch in ("qwen_image", "qwen_image_edit"):
        kwargs["true_cfg_scale"] = guidance
        kwargs.setdefault("negative_prompt", " ")
    elif arch == "chroma":
        kwargs["guidance_scale"] = guidance
        kwargs.setdefault("negative_prompt", "low quality, ugly, deformed")
    else:
        kwargs["guidance_scale"] = guidance


def _encode_prompt_split(pipe, prompt, negative_prompt, original_image, arch, device):
    """Encode prompt with original image for CLIP style anchoring."""
    import inspect

    call_sig = inspect.signature(pipe.__call__)
    if "prompt_embeds" not in call_sig.parameters:
        return None, None

    encode_sig = inspect.signature(pipe.encode_prompt)
    encode_params = set(encode_sig.parameters.keys())

    try:
        if "image" in encode_params:
            result = pipe.encode_prompt(prompt=prompt, image=original_image, device=device)
        else:
            kw = {"prompt": prompt, "device": device}
            if "prompt_2" in encode_params:
                kw["prompt_2"] = prompt
            result = pipe.encode_prompt(**kw)

        prompt_embeds = result[0] if isinstance(result, tuple) else result

        neg_embeds = None
        if negative_prompt:
            if "image" in encode_params:
                neg_result = pipe.encode_prompt(prompt=negative_prompt, image=original_image, device=device)
            else:
                nkw = {"prompt": negative_prompt, "device": device}
                if "prompt_2" in encode_params:
                    nkw["prompt_2"] = negative_prompt
                neg_result = pipe.encode_prompt(**nkw)
            neg_embeds = neg_result[0] if isinstance(neg_result, tuple) else neg_result

        return prompt_embeds, neg_embeds
    except Exception:
        import traceback
        traceback.print_exc()
        return None, None
