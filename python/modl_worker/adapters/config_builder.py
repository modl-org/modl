"""Build ai-toolkit config from a modl TrainJobSpec.

Responsible for translating modl spec fields into the YAML structure that
ai-toolkit's ``run.py`` expects.  All architecture-specific logic is driven
by the tables in ``arch_config.py`` rather than ad-hoc conditionals.
"""

from pathlib import Path

from .arch_config import (
    ARCH_CONFIGS,
    QWEN_24GB_STYLE_QTYPE,
    QWEN_32GB_DEFAULT_QTYPE,
    detect_arch,
    resolve_model_path,
    resolve_qwen_qtype,
)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def read_original_intervals(checkpoint_path: str) -> tuple[int | None, int | None]:
    """Infer the original save_every/sample_every from sample files on disk.

    When resuming training, we want to keep the same sampling/checkpoint
    intervals as the original run so that step numbers remain consistent in
    the preview UI.

    We look at sample images (named ``<ts>__<step>_<idx>.jpg``) and use the
    lowest non-zero step as the original interval.
    Returns (save_every, sample_every) or (None, None) if not determinable.
    """
    ckpt = Path(checkpoint_path)
    run_dir = ckpt.parent

    interval = None
    samples_dir = run_dir / "samples"
    if samples_dir.exists():
        steps = set()
        for f in samples_dir.iterdir():
            if f.suffix in (".jpg", ".png"):
                parts = f.stem.split("__")
                if len(parts) == 2:
                    step_part = parts[1].split("_")[0]
                    try:
                        steps.add(int(step_part))
                    except ValueError:
                        pass
        nonzero = sorted(s for s in steps if s > 0)
        if nonzero:
            interval = nonzero[0]

    if interval:
        print(f"[modl] Preserving original intervals: save_every={interval}, sample_every={interval}")
        return interval, interval

    return None, None


# ---------------------------------------------------------------------------
# Sample prompts
# ---------------------------------------------------------------------------

def build_sample_prompts(trigger_word: str, lora_type: str, arch_key: str) -> list[str]:
    """Auto-generate sample prompts for visual feedback during training.

    For Qwen-Image style LoRAs, prompts are literal (no trigger word) because
    the style is learned as the default output mode via literal captioning.
    For all other cases, the trigger word is embedded in the prompts.
    """
    is_qwen_style = arch_key == "qwen_image" and lora_type == "style"

    if lora_type == "style":
        if is_qwen_style:
            # Qwen style: literal descriptive prompts, no trigger word.
            # The model learns style through literal captions, not trigger words.
            return [
                "a portrait of a woman",
                "a cat sitting on a windowsill",
                "a landscape with mountains and a river",
                "a still life of fruit and flowers",
            ]
        return [
            f"a portrait of a woman in {trigger_word} style",
            f"a cat sitting on a windowsill, {trigger_word} style",
            f"a landscape with mountains and a river, {trigger_word} style",
            f"a still life of fruit and flowers, {trigger_word} style",
        ]
    elif lora_type == "character":
        return [
            f"a photo of {trigger_word}",
            f"a portrait of {trigger_word} smiling",
            f"{trigger_word} in a park",
        ]
    else:  # object
        return [
            f"a photo of {trigger_word}",
            f"a {trigger_word} on a table",
            f"a {trigger_word} in a natural setting",
        ]


# ---------------------------------------------------------------------------
# Train block builder
# ---------------------------------------------------------------------------

def build_train_block(arch_key: str, params: dict, lora_type: str) -> dict:
    """Build the 'train' config block with per-architecture settings."""
    arch = ARCH_CONFIGS.get(arch_key, ARCH_CONFIGS["sdxl"])
    steps = params.get("steps", 2000)
    is_style = lora_type == "style"
    is_zimage = arch_key.startswith("zimage")
    is_qwen = arch_key == "qwen_image"

    # batch_size: 0 means "let adapter decide" (sentinel from Rust)
    bs = params.get("batch_size", 0)
    if bs <= 0:
        bs = 2 if (is_style and not is_zimage) else 1

    lr = params.get("learning_rate", 1e-4)

    # Z-Image: LR must not exceed 1e-4 — higher values break the distillation
    if is_zimage and lr > 1e-4:
        print(f"[modl] WARNING: Clamping LR from {lr} to 1e-4 for Z-Image (higher LR breaks distillation)")
        lr = 1e-4

    # Qwen-Image guidance notes
    if is_qwen:
        if lora_type == "style" and lr < 2e-4:
            # Per Ostris: style LoRAs converge faster at 2e-4 (bumped from 1e-4)
            print(f"[modl] NOTE: Qwen-Image style LoRAs often converge faster at lr=2e-4 (current: {lr})")
        if lora_type == "character":
            if steps < 3000:
                print(f"[modl] NOTE: Qwen-Image character LoRAs usually need ~3000+ steps (current: {steps})")
            # Character training on 24GB is not currently supported well.
            # uint6 needs ~30GB; int4 has severe degradation.
            # No LR bump needed — 1e-4 with rank 16 is the tested recipe.

    train = {
        "batch_size": bs,
        "steps": steps,
        "gradient_accumulation_steps": 1,
        "train_unet": True,
        "gradient_checkpointing": True,
        "optimizer": params.get("optimizer", "adamw8bit"),
        "lr": lr,
    }

    if is_style:
        train["content_or_style"] = "style"

    train["train_text_encoder"] = arch.get("train_text_encoder", False)
    train["noise_scheduler"] = arch["noise_scheduler"]
    train["dtype"] = arch["dtype"]

    # EMA for most architectures
    if arch_key not in ("sd15",):
        train["ema_config"] = {"use_ema": True, "ema_decay": 0.99}

    # SDXL-specific noise_offset
    if arch_key == "sdxl":
        train["noise_offset"] = 0.0357 if is_style else 0.0

    # Merge any extra train keys from the arch config
    extra = arch.get("extra_train", {})
    train.update(extra)

    return train


# ---------------------------------------------------------------------------
# Sample block builder
# ---------------------------------------------------------------------------

def build_sample_block(
    arch_key: str,
    params: dict,
    resolution: int,
    lora_type: str,
    sample_every_override: int | None = None,
) -> dict:
    """Build the 'sample' config block with per-architecture settings."""
    arch = ARCH_CONFIGS.get(arch_key, ARCH_CONFIGS["sdxl"])
    sample_cfg = arch["sample"]
    steps = params.get("steps", 2000)

    return {
        "sampler": sample_cfg["sampler"],
        "sample_every": sample_every_override or max(steps // 5, 50),
        "width": resolution,
        "height": resolution,
        "prompts": build_sample_prompts(
            params.get("trigger_word", "OHWX"), lora_type, arch_key
        ),
        "neg": sample_cfg["neg"],
        "seed": params.get("seed") or 42,
        "walk_seed": True,
        "guidance_scale": sample_cfg["guidance"],
        "sample_steps": sample_cfg["steps"],
    }


# ---------------------------------------------------------------------------
# Main spec → ai-toolkit config translator
# ---------------------------------------------------------------------------

def spec_to_aitoolkit_config(spec: dict) -> dict:
    """Translate a TrainJobSpec (parsed from YAML) into ai-toolkit's config format.

    This is the single place to maintain the mapping between modl spec fields
    and ai-toolkit's expected YAML configuration.
    """
    params = spec.get("params", {})
    dataset = spec.get("dataset", {})
    model = spec.get("model", {})
    output = spec.get("output", {})

    base_model_id = model.get("base_model_id", "")
    lora_type = params.get("lora_type", "character")

    # Detect model architecture
    arch_key = detect_arch(base_model_id)
    arch = ARCH_CONFIGS[arch_key]

    # Resolve HuggingFace hub path
    model_path = resolve_model_path(base_model_id)
    if model_path == base_model_id and model.get("base_model_path"):
        model_path = model["base_model_path"]

    # -- Model config from the arch table --
    model_config = {"name_or_path": model_path}
    model_config.update(arch["model_flags"])

    if arch_key == "qwen_image":
        _apply_qwen_model_config(model_config, lora_type)

    # Resolution
    resolution = params.get("resolution", arch["default_resolution"])
    dataset_resolution = arch["resolutions"]

    # Network config
    rank = params.get("rank", 16)
    is_style = lora_type == "style"
    network_config = {
        "type": "lora",
        "linear": rank,
        "linear_alpha": rank,
    }

    # Resume from checkpoint
    resume_from = params.get("resume_from")
    original_save_every = None
    original_sample_every = None
    if resume_from:
        network_config["pretrained_lora_path"] = resume_from
        print(f"[modl] Resuming training from checkpoint: {resume_from}")
        original_save_every, original_sample_every = read_original_intervals(resume_from)

    # Dataset repeats & caption dropout
    num_repeats = params.get("num_repeats", 0)
    if num_repeats <= 0:
        num_repeats = 10 if is_style else 1

    caption_dropout = params.get("caption_dropout_rate", -1.0)
    if caption_dropout < 0:
        caption_dropout = 0.3 if is_style else 0.05

    if arch_key == "qwen_image":
        # cache_text_embeddings=True is incompatible with caption dropout.
        # TODO: If future ai-toolkit versions support non-cached TE mode for
        # Qwen character training, re-enable caption_dropout for that path.
        if caption_dropout > 0:
            print(
                f"[modl] NOTE: For Qwen-Image with cached text embeddings, "
                f"forcing caption_dropout_rate=0.0 (requested {caption_dropout})."
            )
        caption_dropout = 0.0

    config = {
        "job": "extension",
        "config": {
            "name": output.get("lora_name", "lora-output"),
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": output.get("destination_dir", "output"),
                    "device": "cuda:0",
                    "trigger_word": params.get("trigger_word", "OHWX"),
                    "network": network_config,
                    "save": {
                        "dtype": "float16",
                        "save_every": original_save_every or (
                            max(params.get("steps", 2000) // 5, 500)
                            if is_style
                            else params.get("steps", 2000)
                        ),
                        "max_step_saves_to_keep": 5 if is_style else 1,
                    },
                    "datasets": [
                        {
                            "folder_path": dataset.get("path", ""),
                            "caption_ext": "txt",
                            "caption_dropout_rate": caption_dropout,
                            "shuffle_tokens": False,
                            "cache_text_embeddings": arch_key == "qwen_image",
                            "resolution": dataset_resolution,
                            "cache_latents_to_disk": True,
                            "default_caption": (
                                "" if arch_key == "qwen_image"
                                else params.get("trigger_word", "OHWX")
                            ),
                            "num_repeats": num_repeats,
                        }
                    ],
                    "train": build_train_block(arch_key, params, lora_type),
                    "model": model_config,
                    "sample": build_sample_block(
                        arch_key, params, resolution, lora_type, original_sample_every
                    ),
                }
            ],
        },
    }

    if params.get("seed") is not None:
        config["config"]["process"][0]["train"]["seed"] = params["seed"]

    return config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_qwen_model_config(model_config: dict, lora_type: str) -> None:
    """Apply Qwen-Image specific model config (quantization, logging)."""
    qtype = resolve_qwen_qtype(lora_type)
    model_config["qtype"] = qtype

    if lora_type == "style":
        # Style: 3-bit + ARA fits 24GB (~23GB used on RTX 4090)
        print(
            f"[modl] Qwen-Image style profile: qtype={qtype}, cache_text_embeddings=true "
            "(targets ~23GB VRAM on 1024px — fits RTX 3090/4090 24GB)"
        )
        print(
            "[modl] NOTE: Qwen style LoRAs work best with literal captions and usually no trigger word."
        )
    else:
        # Character/object: uint6 needs ~30GB (RTX 5090 32GB class)
        print(
            f"[modl] Qwen-Image character/object profile: qtype={qtype}, cache_text_embeddings=true "
            "(targets ~30GB VRAM on 1024px — needs 32GB-class GPU, e.g. RTX 5090)"
        )
        if qtype == QWEN_32GB_DEFAULT_QTYPE:
            print(
                "[modl] WARNING: Qwen-Image character/object training requires ~30GB VRAM with uint6.\n"
                "  24GB cards (RTX 3090/4090) will likely OOM. Options:\n"
                "  - Use a 32GB+ GPU (recommended)\n"
                "  - For style LoRAs, switch to --lora-type style (uses 3-bit+ARA, fits 24GB)\n"
                f"  - Override with MODL_QWEN_QTYPE='{QWEN_24GB_STYLE_QTYPE}' (3-bit+ARA, "
                "may work but quality untested for character/object)"
            )
