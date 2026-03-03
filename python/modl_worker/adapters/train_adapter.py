"""Train adapter — subprocess orchestration for ai-toolkit training.

This module is the glue between modl and ai-toolkit.  It:
  1. Loads a TrainJobSpec YAML and translates it via config_builder
  2. Launches ai-toolkit's run.py as a subprocess
  3. Streams stdout, parses progress/errors, and emits events

Architecture configs and config building live in sibling modules:
  - arch_config.py   — ARCH_CONFIGS, MODEL_REGISTRY, detection helpers
  - config_builder.py — spec_to_aitoolkit_config, block builders
  - output_scanner.py — post-training artifact scanning
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

from modl_worker.protocol import EventEmitter

# Re-export for backward compatibility (other code may import from here)
from .config_builder import spec_to_aitoolkit_config  # noqa: F401
from .output_scanner import scan_output_artifacts  # noqa: F401

# ---------------------------------------------------------------------------
# Subprocess output parsing patterns
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"step\s*[:=]?\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_LOSS_RE = re.compile(r"loss\s*[:=]?\s*([0-9eE+\-.]+)", re.IGNORECASE)

_STATUS_PATTERNS = [
    re.compile(r"^(Loading|Quantizing|Preparing|Making|Fusing|Caching)\b", re.IGNORECASE),
    re.compile(r"^Running\s+\d+\s+process", re.IGNORECASE),
    re.compile(r"^#{3,}\s*$"),
    re.compile(r"^#\s+Running job:", re.IGNORECASE),
]

_ERROR_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"^\w*Error:"),
    re.compile(r"^\w*Exception:"),
    re.compile(r"CUDA out of memory"),
    re.compile(r"RuntimeError:"),
    re.compile(r"^Error running job:"),
]

_TAIL_BUFFER_SIZE = 30


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------

def _build_train_command(config_path: Path) -> List[str]:
    """Build the command to run ai-toolkit training.

    Checks MODL_AITOOLKIT_TRAIN_CMD (custom override), then MODL_AITOOLKIT_ROOT
    and sys.path for run.py, then falls back to ``python -m toolkit.job``.
    """
    env_cmd = os.getenv("MODL_AITOOLKIT_TRAIN_CMD", "").strip()
    if env_cmd:
        env_cmd = env_cmd.replace("{config}", str(config_path)).replace("{python}", sys.executable)
        return shlex.split(env_cmd)

    aitk_root = os.getenv("MODL_AITOOLKIT_ROOT", "")
    if not aitk_root:
        for p in sys.path:
            candidate = os.path.join(p, "run.py")
            if os.path.exists(candidate):
                aitk_root = p
                break

    if aitk_root:
        return [sys.executable, os.path.join(aitk_root, "run.py"), str(config_path)]

    return [sys.executable, "-m", "toolkit.job", "--config", str(config_path)]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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
            base_model_id = str(spec.get("model", {}).get("base_model_id", "")).lower()
            lora_type = spec.get("params", {}).get("lora_type", "character")
            if "qwen-image" in base_model_id or "qwen_image" in base_model_id:
                if lora_type == "style":
                    vram_msg = (
                        "Qwen-Image style profile: ~23GB VRAM at 1024px "
                        "(fits RTX 3090/4090 24GB with 3-bit+ARA)."
                    )
                else:
                    vram_msg = (
                        "Qwen-Image character/object profile: ~30GB VRAM at 1024px "
                        "(needs 32GB-class GPU; 24GB NOT currently supported for character)."
                    )
                emitter.emit(
                    {"type": "log", "level": "status", "message": vram_msg}
                )
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
        print(f"[modl] WARNING: spec translation failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        effective_config_path = config_path

    # Build the ai-toolkit command.
    # Prefer MODL_AITOOLKIT_ROOT (set by the Rust executor) to locate run.py
    # since _build_train_command has intermittent issues when called as a
    # function from a piped subprocess context.
    aitk_root = os.getenv("MODL_AITOOLKIT_ROOT", "")
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
