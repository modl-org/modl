"""LanPaint inpainting adapter — training-free inpainting for Z-Image.

Uses the LanPaint algorithm (Zheng et al., arXiv:2502.03491v3) to enable
inpainting on models that have no dedicated inpaint pipeline.

The algorithm code (lanpaint/) is a direct port from the original ComfyUI
implementation at https://github.com/scraed/LanPaint (MIT license).

Current implementation: Z-Image only. Other architectures need a
per-model wrapper (see ZImageModelWrapper) due to differing transformer
output conventions.
"""

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from modl_worker.protocol import EventEmitter
from modl_worker.adapters.arch_config import (
    detect_arch,
    resolve_gen_defaults,
    resolve_pipeline_class_for_mode,
)

# Models known to be distilled — LanPaint quality will degrade.
_DISTILLED_ARCHS = frozenset({
    "flux_schnell", "flux2_klein", "flux2_klein_9b", "zimage_turbo",
})

# Supported architectures (add more as we validate per-model wrappers)
_SUPPORTED_ARCHS = frozenset({"zimage"})


def run_lanpaint(config_path: Path, emitter: EventEmitter) -> int:
    """Run LanPaint inpainting from a GenerateJobSpec YAML file."""
    import yaml

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    model_info = spec.get("model", {})
    base_model_id = model_info.get("base_model_id", "z-image")
    base_model_path = model_info.get("base_model_path")
    params = spec.get("params", {})
    lora_info = spec.get("lora")

    if not params.get("init_image") or not params.get("mask"):
        emitter.error("MISSING_INPUTS", "LanPaint requires both init_image and mask", recoverable=False)
        return 2

    arch = detect_arch(base_model_id)
    if arch not in _SUPPORTED_ARCHS:
        emitter.error(
            "UNSUPPORTED_ARCH",
            f"LanPaint currently only supports Z-Image (got {base_model_id}, arch={arch}).",
            recoverable=False,
        )
        return 2

    if arch in _DISTILLED_ARCHS:
        emitter.info(f"WARNING: {base_model_id} is distilled — LanPaint quality will degrade.")

    cls_name = resolve_pipeline_class_for_mode(base_model_id, "txt2img")
    emitter.info(f"Loading {base_model_id} for LanPaint inpainting (pipeline={cls_name})...")
    emitter.progress(stage="load", step=0, total_steps=1)

    try:
        from modl_worker.adapters.pipeline_loader import assemble_pipeline

        # Load WITHOUT enable_model_cpu_offload — we manage GPU placement
        # manually. Diffusers' accelerate hooks leak CUDA memory when
        # components are moved between devices outside the pipeline flow.
        pipe = assemble_pipeline(base_model_id, base_model_path, cls_name, emitter, no_offload=True)

        if lora_info:
            lora_path = lora_info.get("path")
            lora_weight = lora_info.get("weight", 1.0)
            if lora_path and os.path.exists(lora_path):
                emitter.info(f"Loading LoRA: {lora_info.get('name', 'unnamed')}")
                from modl_worker.adapters.lora_utils import load_lora_with_conversion
                load_lora_with_conversion(pipe, lora_path, lora_weight, emitter)

        emitter.job_started(config=str(config_path))
    except Exception as exc:
        emitter.error("PIPELINE_LOAD_FAILED", str(exc), recoverable=False)
        return 1

    return _run_lanpaint_zimage(spec, emitter, pipe)


# ---------------------------------------------------------------------------
# Z-Image model wrapper + forward pass
# ---------------------------------------------------------------------------

def _zimage_forward_raw(transformer, latents, timestep, embeds, device):
    """Single forward pass through Z-Image transformer.

    Returns RAW diffusers output (no negation). Note: diffusers' Z-Image
    output is the NEGATIVE of ComfyUI's raw output. The x0 conversion
    accounts for this: x0 = x + raw_diffusers * sigma.
    """
    latent_input = latents.to(transformer.dtype)
    latent_input = latent_input.unsqueeze(2)  # Z-Image expects 5D (B,C,F,H,W)
    latent_input_list = list(latent_input.unbind(dim=0))

    embeds_gpu = [e.to(device) for e in embeds]

    with torch.no_grad():
        out_list = transformer(
            latent_input_list, timestep, embeds_gpu, return_dict=False
        )[0]

    raw_output = torch.stack([t.float() for t in out_list], dim=0)
    raw_output = raw_output.squeeze(2)
    return raw_output


class ZImageModelWrapper:
    """Wraps Z-Image pipeline to match the interface LanPaint expects.

    LanPaint calls inner_model(x, sigma) and expects (x0_std, x0_big).

    Key convention difference: diffusers' Z-Image transformer output is
    NEGATED relative to ComfyUI's. ZImagePipeline line 265 explicitly
    negates before passing to the scheduler. So:
        ComfyUI:   x0 = x - raw_comfy * sigma
        Diffusers: x0 = x + raw_diffusers * sigma  (raw_diffusers = -raw_comfy)
    """

    def __init__(self, transformer, prompt_embeds, negative_prompt_embeds, guidance, cfg_big, device):
        self.transformer = transformer
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.guidance = guidance
        self.cfg_BIG = cfg_big
        self.device = device

        # Dummy inner_model with CONST noise_scaling (used by LanPaint's replace step)
        class _ModelSampling:
            def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
                return sigma * noise + (1.0 - sigma) * latent_image

        class _InnerModel:
            model_sampling = _ModelSampling()
            model_type = type("FlowType", (), {"value": 2})()

        self.inner_model = _InnerModel()

    def __call__(self, x, sigma, model_options=None, seed=None):
        sigma_val = sigma if sigma.dim() == 0 else sigma[0]
        zimage_t = (1 - sigma_val).expand(x.shape[0])

        sigma_e = sigma_val
        while sigma_e.dim() < x.dim():
            sigma_e = sigma_e.unsqueeze(-1)

        # x0 = x + raw * sigma (diffusers sign convention)
        raw_cond = _zimage_forward_raw(self.transformer, x, zimage_t, self.prompt_embeds, self.device)
        x0_cond = x.float() + raw_cond * sigma_e

        if self.guidance <= 1.0 or self.negative_prompt_embeds is None:
            return x0_cond, x0_cond

        raw_uncond = _zimage_forward_raw(self.transformer, x, zimage_t, self.negative_prompt_embeds, self.device)
        x0_uncond = x.float() + raw_uncond * sigma_e

        x0_std = x0_uncond + self.guidance * (x0_cond - x0_uncond)
        x0_big = x0_uncond + self.cfg_BIG * (x0_cond - x0_uncond)
        return x0_std, x0_big


# ---------------------------------------------------------------------------
# Main Z-Image LanPaint flow
# ---------------------------------------------------------------------------

def _run_lanpaint_zimage(spec: dict, emitter: EventEmitter, pipe) -> int:
    from diffusers.image_processor import VaeImageProcessor
    from modl_worker.image_util import load_image
    from modl_worker.lanpaint import LanPaint

    prompt = spec.get("prompt", "")
    params = spec.get("params", {})
    output_info = spec.get("output", {})
    model_info = spec.get("model", {})
    base_model_id = model_info.get("base_model_id", "z-image")

    gen_defaults = resolve_gen_defaults(base_model_id)
    steps = params.get("steps", gen_defaults["steps"])
    guidance = params.get("guidance", gen_defaults["guidance"])
    seed = params.get("seed")
    width = params.get("width", 1024)
    height = params.get("height", 1024)
    count = params.get("count", 1)

    lp_steps = params.get("lanpaint_steps", 5)
    lp_friction = params.get("lanpaint_friction", 15.0)
    lp_lambda = params.get("lanpaint_lambda", 16.0)
    lp_step_size = params.get("lanpaint_step_size", 0.2)
    lp_beta = params.get("lanpaint_beta", 1.0)
    lp_early_stop = params.get("lanpaint_early_stop_steps", 1)
    cfg_big = params.get("lanpaint_cfg_big", guidance)

    emitter.info(
        f"LanPaint config: inner_steps={lp_steps}, lambda={lp_lambda}, "
        f"friction={lp_friction}, guidance={guidance}, cfg_big={cfg_big}"
    )

    device = torch.device("cuda")

    # --- 1. Encode prompt: text encoder → GPU → CPU → delete ---
    emitter.info("Encoding prompt...")
    pipe.text_encoder.to(device)
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt="",
            do_classifier_free_guidance=(guidance > 1.0),
            device=device,
        )
    prompt_embeds = [p.cpu() for p in prompt_embeds]
    if negative_prompt_embeds:
        negative_prompt_embeds = [n.cpu() for n in negative_prompt_embeds]

    pipe.text_encoder.to("cpu")
    del pipe.text_encoder
    pipe.text_encoder = None
    torch.cuda.empty_cache()
    gc.collect()

    # --- 2. Encode init image ---
    vae_scale = pipe.vae_scale_factor * 2
    img_processor = VaeImageProcessor(vae_scale_factor=vae_scale)

    init_img = load_image(params["init_image"]).resize((width, height), Image.LANCZOS)
    mask_img = load_image(params["mask"]).convert("L").resize((width, height), Image.NEAREST)

    init_tensor = img_processor.preprocess(init_img, height=height, width=width)
    init_tensor = init_tensor.to(device=device, dtype=pipe.vae.dtype)

    pipe.vae.to(device)
    with torch.no_grad():
        latent_dist = pipe.vae.encode(init_tensor)
        latent_image = latent_dist.latent_dist.mode() if hasattr(latent_dist, "latent_dist") else latent_dist.mode()

    # Scale only (no shift) — matches ComfyUI's process_latent_in
    if hasattr(pipe.vae.config, "scaling_factor"):
        latent_image = latent_image * pipe.vae.config.scaling_factor

    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    latent_image = latent_image.to(dtype=torch.float32)

    # --- 3. Prepare mask ---
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    mask_np = 1.0 - (mask_np > 0.5).astype(np.float32)  # white=inpaint → 0 for LanPaint
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

    latent_h, latent_w = latent_image.shape[-2], latent_image.shape[-1]
    latent_mask = torch.nn.functional.interpolate(
        mask_tensor, size=(latent_h, latent_w), mode="nearest"
    ).expand_as(latent_image).to(device=device, dtype=torch.float32)

    # --- 4. Set up scheduler ---
    from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift, retrieve_timesteps

    # Use shift=3.0 matching ComfyUI's Z-Image config (diffusers ships 6.0)
    pipe.scheduler.config["shift"] = 3.0

    image_seq_len = (latent_h // 2) * (latent_w // 2)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    pipe.scheduler.sigma_min = 0.0
    timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, device, mu=mu)
    sigmas = pipe.scheduler.sigmas.to(device=device, dtype=torch.float32)

    # --- 5. Build model wrapper ---
    model_wrapper = ZImageModelWrapper(
        pipe.transformer, prompt_embeds, negative_prompt_embeds, guidance, cfg_big, device,
    )

    # --- 6. Generate ---
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    output_dir = output_info.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)
    artifact_paths = []

    for img_idx in range(count):
        t0 = time.time()
        image_seed = (seed + img_idx) if seed is not None else None
        if image_seed is not None:
            generator.manual_seed(image_seed)

        noise = torch.randn(
            latent_image.shape, generator=generator, device="cpu", dtype=torch.float32,
        ).to(device)

        total_steps = len(timesteps)
        emitter.info(f"Running {total_steps} denoising steps...")

        pipe.transformer.to(device)

        lanpaint = LanPaint(
            Model=model_wrapper, NSteps=lp_steps, Friction=lp_friction,
            Lambda=lp_lambda, Beta=lp_beta, StepSize=lp_step_size,
            IS_FLUX=False, IS_FLOW=True,
        )

        # CONST noise scaling: x = sigma * noise + (1-sigma) * latent
        latents = sigmas[0] * noise + (1 - sigmas[0]) * latent_image

        try:
            for step_idx in range(total_steps):
                sigma = sigmas[step_idx]

                sigma_clamped = sigma.clamp(max=0.9999)
                flow_t = sigma_clamped.unsqueeze(0) if sigma_clamped.dim() == 0 else sigma_clamped
                abt = (1 - flow_t) ** 2 / ((1 - flow_t) ** 2 + flow_t ** 2)
                ve_sigma = flow_t / (1 - flow_t + 1e-8)
                current_times = (ve_sigma, abt, flow_t)

                remaining = total_steps - step_idx
                n_inner = 0 if remaining <= lp_early_stop else None

                x0_pred = lanpaint(
                    latents, latent_image, noise, flow_t,
                    latent_mask, current_times, {}, image_seed,
                    n_steps=n_inner,
                )

                # Euler step: d = (x - x0) / sigma, x_next = x + d * dt
                sigma_next = sigmas[step_idx + 1] if step_idx + 1 < len(sigmas) else torch.zeros_like(sigma)
                if sigma_next == 0:
                    latents = x0_pred
                else:
                    d = (latents - x0_pred) / sigma.clamp(min=1e-6)
                    latents = latents + d * (sigma_next - sigma)

                if (step_idx + 1) % 5 == 0 or step_idx == total_steps - 1:
                    emitter.info(f"  Step {step_idx + 1}/{total_steps}")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            emitter.error("LANPAINT_FAILED", f"LanPaint failed: {exc}", recoverable=(img_idx + 1 < count))
            continue

        # --- 7. Decode ---
        pipe.transformer.to("cpu")
        torch.cuda.empty_cache()
        pipe.vae.to(device)

        with torch.no_grad():
            decoded = latents.to(pipe.vae.dtype)
            if hasattr(pipe.vae.config, "shift_factor"):
                decoded = decoded + pipe.vae.config.shift_factor
            if hasattr(pipe.vae.config, "scaling_factor"):
                decoded = decoded / pipe.vae.config.scaling_factor
            image_tensor = pipe.vae.decode(decoded, return_dict=False)[0]

        result = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]

        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        if img_idx < count - 1:
            pipe.transformer.to(device)

        # Pixel-space blend: paste original in keep region
        keep_mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
            (width, height), Image.NEAREST
        )
        result.paste(init_img.convert("RGB"), mask=keep_mask_pil)

        elapsed = time.time() - t0

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}_{img_idx:03d}.png" if count > 1 else f"{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        try:
            from PIL.PngImagePlugin import PngInfo
            pnginfo = PngInfo()
            pnginfo.add_text("modl", json.dumps({
                "generated_with": "modl.run", "method": "lanpaint",
                "prompt": prompt, "base_model_id": base_model_id,
                "seed": image_seed, "steps": steps, "guidance": guidance,
                "lanpaint_steps": lp_steps, "lanpaint_lambda": lp_lambda,
            }))
            result.save(filepath, pnginfo=pnginfo)
        except Exception:
            result.save(filepath)

        artifact_paths.append(filepath)
        sha = hashlib.sha256(open(filepath, "rb").read()).hexdigest()
        emitter.artifact(filepath, sha256=sha, size_bytes=os.path.getsize(filepath))
        emitter.info(f"  Image {img_idx + 1}/{count} ({elapsed:.1f}s): {filepath}")

    if not artifact_paths:
        emitter.error("NO_IMAGES_GENERATED", "All attempts failed", recoverable=False)
        return 1

    emitter.completed(f"Generated {len(artifact_paths)} image(s) with LanPaint")
    return 0
