"""ControlNet and style-reference loading for generation pipelines."""

import os
from pathlib import Path

from modl_worker.protocol import EventEmitter

# Bundled config files directory
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


# ---------------------------------------------------------------------------
# ControlNet support
# ---------------------------------------------------------------------------

# Control type → mode index for union controlnets
# These indices map to the mode embedding in union ControlNet weights
FLUX_CONTROLNET_MODES = {
    "canny": 0, "tile": 1, "depth": 2, "blur": 3,
    "pose": 4, "gray": 5, "softedge": 5,
}

SDXL_CONTROLNET_MODES = {
    "canny": 0, "tile": 1, "depth": 2, "blur": 3,
    "pose": 4, "gray": 5, "softedge": 5, "normal": 6,
    "scribble": 6, "hed": 6, "mlsd": 7,
}

ZIMAGE_CONTROLNET_MODES = {
    "canny": 0, "hed": 1, "depth": 2, "pose": 3,
    "mlsd": 4, "scribble": 5, "gray": 6,
}

QWEN_IMAGE_CONTROLNET_MODES = {
    "canny": 0, "depth": 1, "pose": 2, "softedge": 3,
}


# ControlNet model paths indexed by (arch_key, manifest_id)
# Maps to HuggingFace repos — resolved from installed store paths at runtime
CONTROLNET_CONFIGS = {
    "flux": {
        "repo": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        "model_class": "FluxControlNetModel",
        "pipeline_class": "FluxControlNetPipeline",
        "modes": FLUX_CONTROLNET_MODES,
    },
    "sdxl": {
        "repo": "xinsir/controlnet-union-sdxl-1.0",
        "model_class": "ControlNetModel",
        "pipeline_class": "StableDiffusionXLControlNetPipeline",
        "modes": SDXL_CONTROLNET_MODES,
    },
    "qwen_image": {
        "repo": "InstantX/Qwen-Image-ControlNet-Union",
        "model_class": "QwenImageControlNetModel",
        "pipeline_class": "QwenImageControlNetPipeline",
        "modes": QWEN_IMAGE_CONTROLNET_MODES,
    },
    "zimage_turbo": {
        "model_class": "ZImageControlNetModel",
        "pipeline_class": "ZImageControlNetPipeline",
        "modes": ZIMAGE_CONTROLNET_MODES,
    },
    "zimage": {
        "model_class": "ZImageControlNetModel",
        "pipeline_class": "ZImageControlNetPipeline",
        "modes": ZIMAGE_CONTROLNET_MODES,
    },
    "flux_schnell": {
        "repo": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        "model_class": "FluxControlNetModel",
        "pipeline_class": "FluxControlNetPipeline",
        "modes": FLUX_CONTROLNET_MODES,
    },
}


def _resolve_controlnet_path(arch: str) -> str | None:
    """Resolve installed ControlNet path from modl store, fallback to HF repo."""
    config = CONTROLNET_CONFIGS.get(arch)
    if not config:
        return None

    # Try to find installed controlnet from modl's SQLite DB
    from .arch_config import _get_installed_path
    # Try multiple manifest IDs per arch (prefer newer versions first)
    manifest_id_candidates = {
        "flux": ["flux-dev-controlnet-union"],
        "flux_schnell": ["flux-dev-controlnet-union"],
        "sdxl": ["sdxl-controlnet-union"],
        "qwen_image": ["qwen-image-controlnet-union"],
        # Prefer v2.1 (better quality, same VRAM). Diffusers 0.38+ has full
        # support — auto-pads control_in_dim from 16→33 in the pipeline.
        # Falls back to v1 (6 control layers, control_in_dim=16) if v2.1 not installed.
        "zimage_turbo": ["z-image-turbo-controlnet-union-2.1", "z-image-turbo-controlnet-union"],
        "zimage": ["z-image-turbo-controlnet-union-2.1", "z-image-turbo-controlnet-union"],
    }
    candidates = manifest_id_candidates.get(arch, [])
    for manifest_id in candidates:
        path = _get_installed_path(manifest_id)
        if path:
            return path

    # Fallback to HuggingFace repo if configured (will download)
    return config.get("repo")


def _resolve_controlnet_config(arch: str, model_path: str) -> str | None:
    """Return bundled config path for controlnets that need it.

    Z-Image: diffusers misdetects the lite v2.1 variant as v1.0.
    Qwen-Image: from_single_file needs a config to avoid HF downloads.
    """
    configs_dir = Path(__file__).parent.parent / "configs"

    if arch == "qwen_image":
        return str(configs_dir / "qwen-image-controlnet")

    if arch in ("flux", "flux_schnell"):
        return str(configs_dir / "flux-controlnet-union")

    if arch not in ("zimage_turbo", "zimage"):
        return None

    import safetensors.torch

    configs_dir = Path(__file__).parent.parent / "configs"
    # Peek at the embedder shape to determine control_in_dim
    try:
        metadata = safetensors.torch.load_file(model_path, device="cpu")
        emb_key = "control_all_x_embedder.2-1.weight"
        if emb_key not in metadata:
            return None
        emb_shape = metadata[emb_key].shape
        del metadata  # free memory
        # patch_size=2: embed_dim = 2*2*control_in_dim
        control_in_dim = emb_shape[1] // 4
    except Exception:
        return None

    if control_in_dim == 16:
        return str(configs_dir / "zimage-controlnet-v1")
    elif control_in_dim == 33:
        # Distinguish full (15 layers) from lite (3 layers) by file size.
        # Full v2.1 ~6.7GB, lite ~2GB.
        file_size_gb = Path(model_path).stat().st_size / (1024**3)
        if file_size_gb < 4.0:
            return str(configs_dir / "zimage-controlnet-v21-lite")
        else:
            return str(configs_dir / "zimage-controlnet-v21")
    return None


def _load_controlnet(
    cn_inputs: list[dict],
    base_model_id: str,
    arch: str,
    pipeline: object,
    emitter: EventEmitter,
) -> tuple:
    """Load ControlNet model and prepare control images.

    Returns (cn_pipeline, control_images, scales, end_values).
    cn_pipeline is a new pipeline with controlnet attached, or None on failure.
    """
    import torch

    config = CONTROLNET_CONFIGS.get(arch)
    if not config:
        emitter.warning(
            "CONTROLNET_NOT_SUPPORTED",
            f"ControlNet not supported for architecture '{arch}'. Generating without ControlNet.",
        )
        return None, [], [], []

    # Load ControlNet model
    model_path = _resolve_controlnet_path(arch)
    if not model_path:
        emitter.warning(
            "CONTROLNET_NOT_FOUND",
            f"ControlNet weights not found for {arch}. Generating without ControlNet.",
        )
        return None, [], [], []

    emitter.info(f"Loading ControlNet: {Path(model_path).name}")

    import diffusers
    cn_cls = getattr(diffusers, config["model_class"])

    # Load ControlNet in bf16 — these are lightweight (control blocks only,
    # ~2-3GB). The heavy shared modules (embedders, refiners) come from the
    # base transformer via from_transformer() inside the pipeline constructor.
    dtype = torch.bfloat16

    try:
        if model_path.endswith(".safetensors"):
            cn_config = _resolve_controlnet_config(arch, model_path)
            try:
                if cn_config:
                    cn_model = cn_cls.from_single_file(
                        model_path, config=cn_config, torch_dtype=dtype,
                    )
                else:
                    cn_model = cn_cls.from_single_file(model_path, torch_dtype=dtype)
            except (ValueError, NotImplementedError, AttributeError):
                # from_single_file not supported or missing (e.g. FluxControlNetModel,
                # QwenImageControlNetModel). Fall back to from_config + load_state_dict.
                if not cn_config:
                    raise
                import safetensors.torch as sf
                config_dict = cn_cls.load_config(cn_config)
                cn_model = cn_cls.from_config(config_dict)
                state_dict = sf.load_file(model_path)
                cn_model.load_state_dict(state_dict, strict=False)
                cn_model = cn_model.to(dtype)
        else:
            cn_model = cn_cls.from_pretrained(model_path, torch_dtype=dtype)
    except Exception as exc:
        emitter.error(
            "CONTROLNET_LOAD_FAILED",
            f"Failed to load ControlNet weights: {exc}",
            recoverable=True,
        )
        return None, [], [], []

    # Remove accelerate cpu_offload hooks from the base pipeline so the
    # new CN pipeline can set up its own offload sequence cleanly.
    from accelerate.hooks import remove_hook_from_module
    for name in ["text_encoder", "transformer", "vae"]:
        mod = getattr(pipeline, name, None)
        if mod is not None:
            remove_hook_from_module(mod, recurse=True)

    # Check if everything fits on GPU. With force_fp8, the transformer is
    # ~5.7GB. Lite controlnet adds ~2GB → fits. Full controlnet (6GB) needs
    # the text encoder offloaded during denoising → use wrapper approach.
    cn_model_size_gb = sum(
        p.numel() * p.element_size() for p in cn_model.parameters()
    ) / (1024**3)
    use_wrapper = cn_model_size_gb > 4.0  # Full v2.1 is ~6GB

    if use_wrapper and arch in ("zimage_turbo", "zimage"):
        # Single-model wrapper: combines transformer + controlnet into one
        # nn.Module so model_cpu_offload can offload the text encoder during
        # denoising. The wrapper's forward runs controlnet → transformer.
        from .z_image_control import ZImageControlWrapper

        wrapper = ZImageControlWrapper(pipeline.transformer, cn_model)
        # Use the base ZImagePipeline (not ControlNet variant) — the wrapper
        # handles controlnet internally, the pipeline just calls transformer.
        from diffusers import ZImagePipeline
        cn_pipe = ZImagePipeline(
            transformer=wrapper,
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            scheduler=pipeline.scheduler,
        )
        from modl_worker.device import move_pipe_to_device
        move_pipe_to_device(cn_pipe)
        emitter.info(f"ControlNet loaded ({cn_model_size_gb:.1f}GB, CPU offload via wrapper)")
    else:
        # Standard approach: construct the ControlNet pipeline directly.
        # Forward all components from the base pipeline (some pipelines like
        # FluxControlNetPipeline need text_encoder_2, tokenizer_2, etc.)
        cn_pipe_cls = getattr(diffusers, config["pipeline_class"])
        import inspect
        cn_init_params = set(inspect.signature(cn_pipe_cls.__init__).parameters.keys())
        cn_init_params.discard("self")

        cn_kwargs = {"controlnet": cn_model}
        for param_name in cn_init_params:
            if param_name == "controlnet":
                continue
            component = getattr(pipeline, param_name, None)
            if component is not None:
                cn_kwargs[param_name] = component

        cn_pipe = cn_pipe_cls(**cn_kwargs)

        # Check if the base pipeline was using model_cpu_offload (large
        # models like qwen-image). If so, use cpu_offload on the CN
        # pipeline too and pin the controlnet on CUDA (it's independent,
        # no shared modules — safe to keep resident).
        base_was_offloaded = bool(getattr(pipeline, "_all_hooks", None))
        if base_was_offloaded:
            from modl_worker.device import move_pipe_to_device, get_device
            move_pipe_to_device(cn_pipe)
            cn_pipe.controlnet.to(get_device())
            emitter.info(f"ControlNet loaded ({cn_model_size_gb:.1f}GB, CPU offload)")
        else:
            from modl_worker.device import get_device
            cn_pipe.to(get_device())
            emitter.info(f"ControlNet loaded ({cn_model_size_gb:.1f}GB)")

    # Load control images
    from modl_worker.image_util import load_image as _load_img

    control_images = []
    scales = []
    end_values = []
    for inp in cn_inputs:
        img = _load_img(inp["image"])
        control_images.append(img)
        scales.append(inp.get("strength", 0.75))
        end_values.append(inp.get("control_end", 0.8))

    return cn_pipe, control_images, scales, end_values


def _resolve_control_modes(cn_types: list[str], arch: str) -> list[int] | None:
    """Map control type names to mode indices for union controlnets."""
    config = CONTROLNET_CONFIGS.get(arch)
    if not config or "modes" not in config:
        return None

    modes = config["modes"]
    result = []
    for ct in cn_types:
        mode_idx = modes.get(ct)
        if mode_idx is not None:
            result.append(mode_idx)
        else:
            # Unknown type — return None to skip mode setting
            return None
    return result


# ---------------------------------------------------------------------------
# Style reference / IP-Adapter support
# ---------------------------------------------------------------------------

STYLE_REF_CONFIGS = {
    "sdxl": {
        "mechanism": "ip-adapter",
        "repo": "h94/IP-Adapter",
        "subfolder": "sdxl_models",
        "weight_name": "ip-adapter-plus_sdxl_vit-h.safetensors",
        "image_encoder_repo": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    },
    "flux": {
        "mechanism": "ip-adapter",
        "repo": "InstantX/FLUX.1-dev-IP-Adapter",
        "weight_name": "ip-adapter.bin",
        "image_encoder_repo": "google/siglip-so400m-patch14-384",
    },
}


def _load_style_ref(
    style_inputs: list[dict],
    base_model_id: str,
    arch: str,
    pipe: object,
    emitter: EventEmitter,
) -> tuple:
    """Load style reference mechanism and prepare images.

    Returns (style_images, strength, mechanism).
    """
    from PIL import Image
    from modl_worker.image_util import load_image as _load_img

    config = STYLE_REF_CONFIGS.get(arch)
    if not config:
        emitter.warning(
            "STYLE_REF_NOT_SUPPORTED",
            f"Style reference not supported for architecture '{arch}'. Generating without style-ref.",
        )
        return [], 0.0, None

    mechanism = config["mechanism"]
    images = [_load_img(inp["image"]) for inp in style_inputs]
    strength = style_inputs[0].get("strength", 0.6) if style_inputs else 0.6

    if mechanism == "ip-adapter":
        # Load IP-Adapter into the pipeline
        emitter.info(f"Loading IP-Adapter for {arch}...")
        try:
            pipe.load_ip_adapter(
                config["repo"],
                subfolder=config.get("subfolder"),
                weight_name=config["weight_name"],
            )
            pipe.set_ip_adapter_scale(strength)
            emitter.info("IP-Adapter loaded")
        except Exception as exc:
            emitter.warning(
                "IP_ADAPTER_LOAD_FAILED",
                f"Failed to load IP-Adapter: {exc}. Generating without style-ref.",
            )
            return [], 0.0, None

    return images, strength, mechanism
