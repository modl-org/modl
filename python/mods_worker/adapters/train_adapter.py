import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

from mods_worker.protocol import EventEmitter

_STEP_RE = re.compile(r"step\s*[:=]?\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_LOSS_RE = re.compile(r"loss\s*[:=]?\s*([0-9eE+\-.]+)", re.IGNORECASE)

# Lines that indicate important model-loading status updates
_STATUS_PATTERNS = [
    re.compile(r"^(Loading|Quantizing|Preparing|Making|Fusing|Caching)\b", re.IGNORECASE),
    re.compile(r"^Running\s+\d+\s+process", re.IGNORECASE),
    re.compile(r"^#{3,}\s*$"),
    re.compile(r"^#\s+Running job:", re.IGNORECASE),
]

# Lines that indicate errors in the subprocess output
_ERROR_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"^\w*Error:"),
    re.compile(r"^\w*Exception:"),
    re.compile(r"CUDA out of memory"),
    re.compile(r"RuntimeError:"),
    re.compile(r"^Error running job:"),
]

_TAIL_BUFFER_SIZE = 30


def _read_original_intervals(checkpoint_path: str) -> tuple[int | None, int | None]:
    """Infer the original save_every/sample_every from sample files on disk.

    When resuming training, we want to keep the same sampling/checkpoint intervals
    as the original run so that step numbers remain consistent in the preview UI.

    We look at the sample images (named ``<ts>__<step>_<idx>.jpg``) and use the
    lowest non-zero step number as the original interval (step 0 → first
    checkpoint = the interval).  We use the sample-derived interval for both
    save_every and sample_every since they default to the same formula and
    checkpoint files may be pruned by max_step_saves_to_keep.
    Returns (save_every, sample_every) or (None, None) if not determinable.
    """
    ckpt = Path(checkpoint_path)
    run_dir = ckpt.parent  # e.g. ~/.mods/training_output/<name>/<name>/

    # Infer interval from sample images (never pruned, most reliable)
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
            # The first non-zero step equals the original interval
            interval = nonzero[0]

    if interval:
        print(f"[mods] Preserving original intervals: save_every={interval}, sample_every={interval}")
        return interval, interval

    return None, None


def _build_train_command(config_path: Path) -> List[str]:
    """Build the command to run ai-toolkit training.

    Checks MODS_AITOOLKIT_TRAIN_CMD (custom override), then MODS_AITOOLKIT_ROOT
    and sys.path for run.py, then falls back to ``python -m toolkit.job``.
    """
    env_cmd = os.getenv("MODS_AITOOLKIT_TRAIN_CMD", "").strip()
    if env_cmd:
        env_cmd = env_cmd.replace("{config}", str(config_path)).replace("{python}", sys.executable)
        return shlex.split(env_cmd)

    # ai-toolkit uses run.py as its entry point (toolkit.job has no __main__)
    aitk_root = os.getenv("MODS_AITOOLKIT_ROOT", "")
    if not aitk_root:
        # Try to find run.py via PYTHONPATH entries
        for p in sys.path:
            candidate = os.path.join(p, "run.py")
            if os.path.exists(candidate):
                aitk_root = p
                break

    if aitk_root:
        return [sys.executable, os.path.join(aitk_root, "run.py"), str(config_path)]

    # Fallback: try running as module (won't work with current ai-toolkit)
    return [sys.executable, "-m", "toolkit.job", "--config", str(config_path)]


def _build_sample_prompts(trigger_word: str, lora_type: str) -> list[str]:
    """Auto-generate sample prompts using the trigger word so we get visual
    feedback during training at each sample_every checkpoint."""
    if lora_type == "style":
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


# -----------------------------------------------------------------------
# Architecture config table
# -----------------------------------------------------------------------
# Data-driven config for each model family.  Used by _build_train_block,
# _build_sample_block, and spec_to_aitoolkit_config instead of ad-hoc
# if/elif chains.  Extension-based models use "arch" key; legacy models
# use boolean flags (is_flux, is_v3).
#
# Fields:
#   model_flags   – merged into the ai-toolkit "model" block
#   noise_scheduler, dtype, train_text_encoder – for the "train" block
#   resolutions   – resolution buckets for dataset
#   default_resolution – fallback when user doesn't specify
#   sample        – sampler, steps, guidance, neg for sample block
#   extra_train   – extra keys merged into "train" block
# -----------------------------------------------------------------------

ARCH_CONFIGS: dict[str, dict] = {
    "flux": {
        "model_flags": {"is_flux": True, "quantize": True},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 20, "guidance": 4.0, "neg": ""},
    },
    "flux_schnell": {
        "model_flags": {
            "is_flux": True,
            "quantize": True,
            "assistant_lora_path": "ostris/FLUX.1-schnell-training-adapter",
        },
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 4, "guidance": 1.0, "neg": ""},
    },
    "zimage_turbo": {
        "model_flags": {
            "arch": "zimage",
            "quantize": True,
            "quantize_te": True,
            "low_vram": True,
            "assistant_lora_path": "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors",
        },
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "extra_train": {"timestep_type": "weighted"},
        "sample": {"sampler": "flowmatch", "steps": 8, "guidance": 1.0, "neg": ""},
    },
    "zimage": {
        "model_flags": {
            "arch": "zimage",
            "quantize": True,
            "quantize_te": True,
            "low_vram": True,
        },
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "extra_train": {"timestep_type": "weighted"},
        "sample": {"sampler": "flowmatch", "steps": 30, "guidance": 4.0, "neg": ""},
    },
    "chroma": {
        "model_flags": {"arch": "chroma", "quantize": True},
        "noise_scheduler": "flowmatch",
        "dtype": "bf16",
        "train_text_encoder": False,
        "resolutions": [512, 768, 1024],
        "default_resolution": 1024,
        "sample": {"sampler": "flowmatch", "steps": 25, "guidance": 4.0, "neg": ""},
    },
    "sdxl": {
        "model_flags": {"arch": "sdxl"},
        "noise_scheduler": "ddpm",
        "dtype": "bf16",
        "train_text_encoder": True,
        "resolutions": [768, 1024],
        "default_resolution": 1024,
        "extra_train": {"max_denoising_steps": 1000},
        "sample": {"sampler": "euler", "steps": 30, "guidance": 7.5, "neg": "blurry, low quality, deformed"},
    },
    "sd15": {
        "model_flags": {},
        "noise_scheduler": "ddpm",
        "dtype": "fp16",
        "train_text_encoder": True,
        "resolutions": [512],
        "default_resolution": 512,
        "extra_train": {"max_denoising_steps": 1000},
        "sample": {"sampler": "euler", "steps": 30, "guidance": 7.5, "neg": "blurry, low quality, deformed"},
    },
}

# Map mods model IDs → (arch_key, HuggingFace hub ID)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "flux-dev":       ("flux",          "black-forest-labs/FLUX.1-dev"),
    "flux-schnell":   ("flux_schnell",  "black-forest-labs/FLUX.1-schnell"),
    "z-image-turbo":  ("zimage_turbo",  "Tongyi-MAI/Z-Image-Turbo"),
    "z-image":        ("zimage",        "Tongyi-MAI/Z-Image"),
    "chroma":         ("chroma",        "lodestones/Chroma"),
    "sdxl-base-1.0":  ("sdxl",          "stabilityai/stable-diffusion-xl-base-1.0"),
    "sdxl-turbo":     ("sdxl",          "stabilityai/sdxl-turbo"),
    "sd-1.5":         ("sd15",          "stable-diffusion-v1-5/stable-diffusion-v1-5"),
}


def _detect_arch(base_model_id: str) -> str:
    """Detect architecture key from a base model ID.

    First checks MODEL_REGISTRY for an exact match, then falls back to
    substring heuristics.  Returns a key into ARCH_CONFIGS.
    """
    # Exact match in registry
    entry = MODEL_REGISTRY.get(base_model_id)
    if entry:
        return entry[0]

    # Substring heuristics for raw HF paths or unknown IDs
    bid = base_model_id.lower()
    if "z-image-turbo" in bid or "z_image_turbo" in bid:
        return "zimage_turbo"
    if "z-image" in bid or "z_image" in bid or "zimage" in bid:
        return "zimage"
    if "chroma" in bid:
        return "chroma"
    if "flux" in bid and "schnell" in bid:
        return "flux_schnell"
    if "flux" in bid:
        return "flux"
    if "sdxl" in bid or "xl" in bid:
        return "sdxl"
    if "sd-1.5" in bid or "sd15" in bid or "1.5" in bid:
        return "sd15"
    return "sdxl"  # safe default


def _resolve_model_path(base_model_id: str) -> str:
    """Resolve a mods model ID to a HuggingFace hub path."""
    entry = MODEL_REGISTRY.get(base_model_id)
    if entry:
        return entry[1]
    return base_model_id  # already a HF path or local path


def _build_train_block(arch_key: str, params: dict, lora_type: str) -> dict:
    """Build the 'train' config block with per-architecture settings."""
    arch = ARCH_CONFIGS.get(arch_key, ARCH_CONFIGS["sdxl"])
    steps = params.get("steps", 2000)
    is_style = lora_type == "style"
    is_zimage = arch_key.startswith("zimage")

    # batch_size: 0 means "let adapter decide" (sentinel from Rust)
    bs = params.get("batch_size", 0)
    if bs <= 0:
        bs = 2 if (is_style and not is_zimage) else 1

    lr = params.get("learning_rate", 1e-4)
    # Z-Image: LR must not exceed 1e-4 — higher values break the distillation
    if is_zimage and lr > 1e-4:
        print(f"[mods] WARNING: Clamping LR from {lr} to 1e-4 for Z-Image (higher LR breaks distillation)")
        lr = 1e-4

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


def _build_sample_block(arch_key: str, params: dict, resolution: int, lora_type: str, sample_every_override: int | None = None) -> dict:
    """Build the 'sample' config block with per-architecture settings."""
    arch = ARCH_CONFIGS.get(arch_key, ARCH_CONFIGS["sdxl"])
    sample_cfg = arch["sample"]
    steps = params.get("steps", 2000)

    return {
        "sampler": sample_cfg["sampler"],
        "sample_every": sample_every_override or max(steps // 5, 50),
        "width": resolution,
        "height": resolution,
        "prompts": _build_sample_prompts(params.get("trigger_word", "OHWX"), lora_type),
        "neg": sample_cfg["neg"],
        "seed": params.get("seed") or 42,
        "walk_seed": True,
        "guidance_scale": sample_cfg["guidance"],
        "sample_steps": sample_cfg["steps"],
    }


def spec_to_aitoolkit_config(spec: dict) -> dict:
    """Translate a TrainJobSpec (parsed from YAML) into ai-toolkit's config format.

    This is the single place to maintain the mapping between mods spec fields
    and ai-toolkit's expected YAML configuration.
    """
    params = spec.get("params", {})
    dataset = spec.get("dataset", {})
    model = spec.get("model", {})
    output = spec.get("output", {})

    base_model_id = model.get("base_model_id", "")

    # Detect lora type
    lora_type = params.get("lora_type", "character")

    # Detect model architecture from the base model ID
    arch_key = _detect_arch(base_model_id)
    arch = ARCH_CONFIGS[arch_key]

    # Resolve HuggingFace hub path
    model_path = _resolve_model_path(base_model_id)
    # For non-extension models, also check for a local path override
    if model_path == base_model_id and model.get("base_model_path"):
        model_path = model["base_model_path"]

    # Build model config from the arch config table
    model_config = {"name_or_path": model_path}
    model_config.update(arch["model_flags"])

    # Resolution and dataset config from arch table
    resolution = params.get("resolution", arch["default_resolution"])
    dataset_resolution = arch["resolutions"]

    # Network config: style uses higher rank for more capacity
    # Alpha = rank gives scale=1.0 (simplest, most stable)
    rank = params.get("rank", 16)
    is_style = lora_type == "style"
    network_config = {
        "type": "lora",
        "linear": rank,
        "linear_alpha": rank,
    }

    # Resume from a previous checkpoint if specified
    resume_from = params.get("resume_from")
    original_save_every = None
    original_sample_every = None
    if resume_from:
        network_config["pretrained_lora_path"] = resume_from
        print(f"[mods] Resuming training from checkpoint: {resume_from}")
        # Try to read original config to preserve save/sample intervals
        original_save_every, original_sample_every = _read_original_intervals(resume_from)

    # Style defaults: more repeats + higher caption dropout to learn style over content.
    # A value of 0 (num_repeats) or <0 (caption_dropout) means "use adapter default".
    num_repeats = params.get("num_repeats", 0)
    if num_repeats <= 0:
        num_repeats = 10 if is_style else 1

    caption_dropout = params.get("caption_dropout_rate", -1.0)
    if caption_dropout < 0:
        caption_dropout = 0.3 if is_style else 0.05

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
                        "save_every": original_save_every or (max(params.get("steps", 2000) // 5, 500) if is_style else params.get("steps", 2000)),
                        "max_step_saves_to_keep": 5 if is_style else 1,
                    },
                    "datasets": [
                        {
                            "folder_path": dataset.get("path", ""),
                            "caption_ext": "txt",
                            "caption_dropout_rate": caption_dropout,
                            "resolution": dataset_resolution,
                            "cache_latents_to_disk": True,
                            "default_caption": params.get("trigger_word", "OHWX"),
                            "num_repeats": num_repeats,
                        }
                    ],
                    "train": _build_train_block(arch_key, params, lora_type),
                    "model": model_config,
                    "sample": _build_sample_block(arch_key, params, resolution, lora_type, original_sample_every),
                }
            ],
        },
    }

    if params.get("seed") is not None:
        config["config"]["process"][0]["train"]["seed"] = params["seed"]

    return config


def scan_output_artifacts(output_dir: str, emitter: EventEmitter) -> None:
    """After training, emit artifact event for the final LoRA only (not intermediate checkpoints).

    ai-toolkit saves checkpoints as `{name}_000002000.safetensors` and the
    final output as `{name}.safetensors`.  We only register the final one in
    the DB — checkpoints stay on disk for manual comparison but don't
    clutter `mods model ls`.
    """
    import glob
    import hashlib
    import re

    pattern = os.path.join(output_dir, "**", "*.safetensors")
    all_files = sorted(glob.glob(pattern, recursive=True))

    # Separate final outputs from numbered checkpoints (e.g. _000002000)
    checkpoint_re = re.compile(r"_\d{6,}\.safetensors$")
    final_files = [f for f in all_files if not checkpoint_re.search(f)]

    # If no non-checkpoint file found, fall back to the last checkpoint
    # (highest step number) so we always emit at least one artifact.
    targets = final_files if final_files else all_files[-1:] if all_files else []

    for filepath in targets:
        path = Path(filepath)
        size_bytes = path.stat().st_size

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        emitter.artifact(
            path=str(path),
            sha256=sha256.hexdigest(),
            size_bytes=size_bytes,
        )


def run_train(config_path: Path, emitter: EventEmitter) -> int:
    if not config_path.exists():
        emitter.error(
            "SPEC_VALIDATION_FAILED",
            f"Training config not found: {config_path}",
            recoverable=False,
        )
        return 2

    # Try to load as a full TrainJobSpec first, fall back to direct config
    output_dir = None
    try:
        import yaml
        with open(config_path) as f:
            spec = yaml.safe_load(f)
        if isinstance(spec, dict) and "params" in spec:
            # This is a full TrainJobSpec — translate to ai-toolkit config
            aitk_config = spec_to_aitoolkit_config(spec)
            output_dir = spec.get("output", {}).get("destination_dir")
            # Write translated config to a temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                yaml.dump(aitk_config, tmp)
                effective_config_path = Path(tmp.name)
        else:
            effective_config_path = config_path
    except ImportError:
        effective_config_path = config_path
    except Exception as e:
        print(f"[mods] WARNING: spec translation failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        effective_config_path = config_path

    # Build the ai-toolkit command.
    # Prefer MODS_AITOOLKIT_ROOT (set by the Rust executor) to locate run.py
    # since _build_train_command has intermittent issues when called as a
    # function from a piped subprocess context.
    aitk_root = os.getenv("MODS_AITOOLKIT_ROOT", "")
    if aitk_root:
        run_py = os.path.join(aitk_root, "run.py")
        if os.path.exists(run_py):
            cmd = [sys.executable, run_py, str(effective_config_path)]
        else:
            cmd = _build_train_command(effective_config_path)
    else:
        cmd = _build_train_command(effective_config_path)

    emitter.job_started(config=str(config_path), command=cmd)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        emitter.error(
            "AITOOLKIT_EXEC_NOT_FOUND",
            f"Could not execute ai-toolkit command: {exc}",
            recoverable=False,
        )
        return 127
    except Exception as exc:
        emitter.error(
            "AITOOLKIT_EXEC_FAILED",
            str(exc),
            recoverable=False,
        )
        return 1

    last_step = None
    tail_lines: list[str] = []  # rolling buffer of recent lines for error context
    error_lines: list[str] = []  # lines that look like errors/tracebacks
    in_traceback = False

    for raw_line in process.stdout or []:
        line = raw_line.strip()
        if not line:
            continue

        # Maintain rolling tail buffer
        tail_lines.append(line)
        if len(tail_lines) > _TAIL_BUFFER_SIZE:
            tail_lines.pop(0)

        # Detect traceback/error lines
        if "Traceback (most recent call last)" in line:
            in_traceback = True
            error_lines = [line]  # reset — start fresh traceback
        elif in_traceback:
            error_lines.append(line)
            # Tracebacks end with the exception line (no leading whitespace after "File" lines)
            if not line.startswith(" ") and not line.startswith("Traceback"):
                in_traceback = False
        elif any(p.search(line) for p in _ERROR_PATTERNS):
            error_lines.append(line)

        # Classify and emit the line
        is_status = any(p.search(line) for p in _STATUS_PATTERNS)
        if is_status:
            emitter.emit({"type": "log", "level": "status", "message": line})
        else:
            emitter.info(line)

        # Check for training progress (step: N/M pattern from ai-toolkit)
        # We deliberately do NOT match tqdm-style "| N/M [" bars for general
        # loading/caching progress since those have unrelated total_steps
        # (e.g. checkpoint shards = 3, latent cache = 10).  Only the
        # ai-toolkit training step line uses "step: N/M" format.
        step_match = _STEP_RE.search(line)
        if step_match:
            step = int(step_match.group(1))
            total_steps = int(step_match.group(2))
            if last_step != step:
                loss = None
                loss_match = _LOSS_RE.search(line)
                if loss_match:
                    try:
                        loss = float(loss_match.group(1))
                    except ValueError:
                        pass
                emitter.progress(
                    stage="train",
                    step=step,
                    total_steps=total_steps,
                    loss=loss,
                )
                last_step = step

    code = process.wait()
    if code == 0:
        # Scan for output artifacts
        if output_dir and os.path.isdir(output_dir):
            scan_output_artifacts(output_dir, emitter)
        emitter.completed("ai-toolkit training command finished")
    else:
        # Build an informative error message with actual failure context
        if error_lines:
            # Use captured traceback/error lines
            error_detail = "\n".join(error_lines[-15:])
        elif tail_lines:
            # Fall back to last N lines of output
            error_detail = "\n".join(tail_lines[-10:])
        else:
            error_detail = "(no output captured)"

        # Extract a one-line summary for the error message
        summary = error_lines[-1] if error_lines else f"Process exited with code {code}"

        emitter.error(
            "TRAINING_FAILED",
            summary,
            recoverable=False,
            details={"exit_code": code, "output_tail": error_detail},
        )
    return code
