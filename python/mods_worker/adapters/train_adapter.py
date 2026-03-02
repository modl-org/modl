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


def _build_train_block(family: str, params: dict, lora_type: str) -> dict:
    """Build the 'train' config block with per-architecture settings."""
    steps = params.get("steps", 2000)
    is_style = lora_type == "style"

    train = {
        "batch_size": 2 if is_style else 1,
        "steps": steps,
        "gradient_accumulation_steps": 1,
        "train_unet": True,
        "gradient_checkpointing": True,
        "optimizer": params.get("optimizer", "adamw8bit"),
        "lr": params.get("learning_rate", 1e-4),
    }

    if is_style:
        # Bias timestep sampling toward noisier steps where style lives
        train["content_or_style"] = "style"

    if family == "flux":
        train["train_text_encoder"] = False
        train["noise_scheduler"] = "flowmatch"
        train["dtype"] = "bf16"
        train["ema_config"] = {"use_ema": True, "ema_decay": 0.99}
    elif family == "sdxl":
        train["train_text_encoder"] = True
        train["noise_scheduler"] = "ddpm"
        train["dtype"] = "bf16"  # fp16 causes NaN loss due to overflow in SDXL UNet
        train["max_denoising_steps"] = 1000
        # SDXL was trained with noise_offset 0.0357
        train["noise_offset"] = 0.0357 if is_style else 0.0
        train["ema_config"] = {"use_ema": True, "ema_decay": 0.99}
    else:  # sd15
        train["train_text_encoder"] = True
        train["noise_scheduler"] = "ddpm"
        train["dtype"] = "fp16"
        train["max_denoising_steps"] = 1000

    return train


def _build_sample_block(family: str, params: dict, resolution: int, lora_type: str) -> dict:
    """Build the 'sample' config block with per-architecture settings."""
    steps = params.get("steps", 2000)

    if family == "flux":
        sampler = "flowmatch"
        guidance = 4
        sample_steps = 20
    elif family == "sdxl":
        sampler = "euler"
        guidance = 7.5
        sample_steps = 30
    else:  # sd15
        sampler = "euler"
        guidance = 7.5
        sample_steps = 30

    return {
        "sampler": sampler,
        "sample_every": max(steps // 5, 50),
        "width": resolution,
        "height": resolution,
        "prompts": _build_sample_prompts(params.get("trigger_word", "OHWX"), lora_type),
        "neg": "blurry, low quality, deformed" if family != "flux" else "",
        "seed": params.get("seed") or 42,
        "walk_seed": True,
        "guidance_scale": guidance,
        "sample_steps": sample_steps,
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
    bid_lower = base_model_id.lower()
    if "flux" in bid_lower:
        family = "flux"
    elif "sdxl" in bid_lower or "xl" in bid_lower:
        family = "sdxl"
    elif "sd-1.5" in bid_lower or "sd15" in bid_lower or "1.5" in bid_lower:
        family = "sd15"
    else:
        family = "sdxl"  # default to sdxl for unknown models

    # Map mods model IDs → HuggingFace hub IDs so ai-toolkit can resolve
    # configs and weights from the hub (cached in ~/.cache/huggingface/).
    HF_MODEL_MAP = {
        "flux-dev": "black-forest-labs/FLUX.1-dev",
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
        "sdxl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
        "sdxl-turbo": "stabilityai/sdxl-turbo",
        "sd-1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    }

    if family == "flux":
        model_path = HF_MODEL_MAP.get(base_model_id, base_model_id)
    else:
        # For SDXL/SD1.5, prefer the HF map, then store path, then raw ID
        model_path = HF_MODEL_MAP.get(base_model_id,
            model.get("base_model_path") or base_model_id)

    # Build model config per architecture family
    model_config = {
        "name_or_path": model_path,
    }
    if family == "flux":
        model_config["is_flux"] = True
        model_config["quantize"] = True
        model_config["low_vram"] = True
    elif family == "sdxl":
        model_config["arch"] = "sdxl"

    # Resolution and dataset config per family
    resolution = params.get("resolution", 1024 if family != "sd15" else 512)
    if family == "flux":
        dataset_resolution = [512, 768, 1024]
    elif family == "sdxl":
        dataset_resolution = [768, 1024]
    else:  # sd15
        dataset_resolution = [512]

    # Network config: style uses higher rank for more capacity
    # Alpha = rank gives scale=1.0 (simplest, most stable)
    rank = params.get("rank", 16)
    is_style = lora_type == "style"
    network_config = {
        "type": "lora",
        "linear": rank,
        "linear_alpha": rank,
    }

    # Style: more repeats + higher caption dropout to learn style over content
    num_repeats = 10 if is_style else 1
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
                        "save_every": max(params.get("steps", 2000) // 5, 500) if is_style else params.get("steps", 2000),
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
                    "train": _build_train_block(family, params, lora_type),
                    "model": model_config,
                    "sample": _build_sample_block(family, params, resolution, lora_type),
                }
            ],
        },
    }

    if params.get("seed") is not None:
        config["config"]["process"][0]["train"]["seed"] = params["seed"]

    return config


def scan_output_artifacts(output_dir: str, emitter: EventEmitter) -> None:
    """After training, scan output directory for .safetensors files and emit artifact events."""
    import glob
    import hashlib

    pattern = os.path.join(output_dir, "**", "*.safetensors")
    for filepath in glob.glob(pattern, recursive=True):
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
    except Exception:
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
