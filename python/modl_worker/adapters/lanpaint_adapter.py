"""LanPaint inpainting adapter — training-free inpainting for multiple models.

Uses the LanPaint algorithm (Zheng et al., arXiv:2502.03491v3) with per-model
adapters that handle architecture-specific details (sign conventions, latent
formats, timestep schedules).

Algorithm code from scraed/LanPaint (MIT license).
Adapter pattern inspired by charrywhite/LanPaint-diffusers.
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


# Architecture → adapter class mapping
_ADAPTER_MAP = {
    "zimage": ("lanpaint_adapters.z_image", "ZImageAdapter", {}),
    "zimage_turbo": ("lanpaint_adapters.z_image", "ZImageAdapter", {"is_turbo": True}),
    "flux2_klein": ("lanpaint_adapters.flux_klein", "FluxKleinAdapter", {}),
    "flux2_klein_9b": ("lanpaint_adapters.flux_klein", "FluxKleinAdapter", {}),
}

# Models known to be distilled
_DISTILLED_ARCHS = frozenset({
    "flux_schnell", "flux2_klein", "flux2_klein_9b", "zimage_turbo",
})


def _get_adapter(arch, pipe, emitter):
    """Instantiate the appropriate adapter for the model architecture."""
    if arch not in _ADAPTER_MAP:
        return None

    module_name, class_name, kwargs = _ADAPTER_MAP[arch]
    import importlib
    mod = importlib.import_module(f"modl_worker.adapters.{module_name}")
    adapter_cls = getattr(mod, class_name)
    return adapter_cls(pipe, emitter, **kwargs)


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
    if arch not in _ADAPTER_MAP:
        supported = ", ".join(sorted(_ADAPTER_MAP.keys()))
        emitter.error(
            "UNSUPPORTED_ARCH",
            f"LanPaint doesn't support {base_model_id} (arch={arch}). Supported: {supported}",
            recoverable=False,
        )
        return 2

    if arch in _DISTILLED_ARCHS:
        emitter.info(f"WARNING: {base_model_id} is distilled — LanPaint quality may degrade.")

    # Load pipeline WITHOUT enable_model_cpu_offload
    cls_name = resolve_pipeline_class_for_mode(base_model_id, "txt2img")
    emitter.info(f"Loading {base_model_id} for LanPaint (pipeline={cls_name})...")
    emitter.progress(stage="load", step=0, total_steps=1)

    try:
        from modl_worker.adapters.pipeline_loader import assemble_pipeline
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

    adapter = _get_adapter(arch, pipe, emitter)
    return _run_lanpaint(spec, emitter, pipe, adapter)


def _run_lanpaint(spec, emitter, pipe, adapter):
    """Model-agnostic LanPaint orchestrator."""
    from modl_worker.lanpaint import LanPaint
    from modl_worker.image_util import load_image
    from diffusers.image_processor import VaeImageProcessor

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
        f"LanPaint [{type(adapter).__name__}]: inner_steps={lp_steps}, "
        f"lambda={lp_lambda}, guidance={guidance}, cfg_big={cfg_big}"
    )

    device = adapter.device

    # --- 1. Encode prompt ---
    emitter.info("Encoding prompt...")
    adapter.encode_prompt(prompt, "")

    # --- 2. Encode image ---
    vae_scale = pipe.vae_scale_factor * 2 if hasattr(pipe, 'vae_scale_factor') else 16
    img_processor = VaeImageProcessor(vae_scale_factor=vae_scale)

    init_img = load_image(params["init_image"]).resize((width, height), Image.LANCZOS)
    mask_img = load_image(params["mask"]).convert("L").resize((width, height), Image.NEAREST)

    init_tensor = img_processor.preprocess(init_img, height=height, width=width)

    # Store dims for Klein's prepare_latents
    if hasattr(adapter, '_height'):
        adapter._height = height
        adapter._width = width

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    latent_image = adapter.encode_image(init_tensor, generator)

    # --- 3. Prepare mask ---
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    mask_np = 1.0 - (mask_np > 0.5).astype(np.float32)  # white=inpaint → 0 for LanPaint
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

    latent_mask = adapter.mask_to_latent_space(mask_tensor, latent_image.shape)
    latent_mask = latent_mask.expand_as(latent_image).to(device=device, dtype=torch.float32)

    # --- 4. Set up scheduler ---
    timesteps, flow_ts = adapter.prepare_timesteps(steps)

    # --- 5. Build LanPaint model wrapper ---
    class _ModelWrapper:
        def __init__(self, adapter, guidance, cfg_big):
            self.adapter = adapter
            self.guidance = guidance
            self.cfg_BIG = cfg_big

            class _Inner:
                class model_sampling:
                    @staticmethod
                    def noise_scaling(sigma, noise, latent_image, max_denoise=False):
                        return adapter.noise_scaling(sigma, noise, latent_image)
                model_type = type("Flow", (), {"value": 2})()
            self.inner_model = _Inner()

        def __call__(self, x, sigma, model_options=None, seed=None):
            flow_t = float(sigma.flatten()[0])
            return self.adapter.predict_x0(x, flow_t, self.guidance, self.cfg_BIG)

    model_wrapper = _ModelWrapper(adapter, guidance, cfg_big)

    # --- 6. Generate ---
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

        # Move transformer to GPU
        pipe.transformer.to(device)

        lanpaint = LanPaint(
            Model=model_wrapper, NSteps=lp_steps, Friction=lp_friction,
            Lambda=lp_lambda, Beta=lp_beta, StepSize=lp_step_size,
            IS_FLUX=False, IS_FLOW=True,
        )

        # CONST noise scaling
        latents = adapter.noise_scaling(
            flow_ts[0:1].reshape(1, *([1] * (latent_image.dim() - 1))).float(),
            noise, latent_image,
        )

        try:
            with torch.no_grad():
                for step_idx in range(total_steps):
                    t = timesteps[step_idx]
                    flow_t_val = flow_ts[step_idx].item()

                    # Build current_times for LanPaint
                    ft = max(flow_t_val, 1e-6)
                    ve_sigma = ft / (1 - ft + 1e-8)
                    abt = (1 - ft) ** 2 / ((1 - ft) ** 2 + ft ** 2)
                    current_times = (
                        torch.tensor([ve_sigma], device=device, dtype=torch.float32),
                        torch.tensor([abt], device=device, dtype=torch.float32),
                        torch.tensor([ft], device=device, dtype=torch.float32),
                    )

                    remaining = total_steps - step_idx
                    n_inner = 0 if remaining <= lp_early_stop else None

                    x0_pred = lanpaint(
                        latents, latent_image, noise,
                        torch.tensor([ft], device=device, dtype=torch.float32),
                        latent_mask, current_times, {}, image_seed,
                        n_steps=n_inner,
                    )

                    # Euler step
                    sigma_next = flow_ts[step_idx + 1].item() if step_idx + 1 < len(flow_ts) else 0.0
                    if sigma_next == 0:
                        latents = x0_pred
                    else:
                        d = (latents - x0_pred) / max(ft, 1e-6)
                        latents = latents + d * (sigma_next - ft)

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

        result = adapter.decode_latents(latents)

        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        if img_idx < count - 1:
            pipe.transformer.to(device)

        # Pixel-space blend
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
                "adapter": type(adapter).__name__,
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
