"""Train adapter — multi-phase orchestration for ai-toolkit training.

This module is the glue between modl and ai-toolkit.  It:
  1. Loads a TrainJobSpec YAML
  2. Resolves a TrainingStrategy (single or multi-phase)
  3. For each phase: builds config, launches ai-toolkit run.py, streams events
  4. Between phases: finds latest checkpoint, resumes with phase overrides

Architecture:
  arch_config.py       — ARCH_CONFIGS, MODEL_REGISTRY, detection helpers
  training_strategy.py — Strategy definitions + phase resolution
  config_builder.py    — spec → ai-toolkit config (accepts phase overrides)
  train_adapter.py     — THIS FILE: orchestrator
  output_scanner.py    — post-training artifact scanning
"""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, List

from modl_worker.protocol import EventEmitter

# Re-export for backward compatibility (other code may import from here)
from .config_builder import spec_to_aitoolkit_config  # noqa: F401
from .output_scanner import scan_output_artifacts  # noqa: F401
from .training_strategy import (
    TrainingPhase,
    TrainingStrategy,
    find_latest_checkpoint,
    resolve_strategy,
)

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


def _resolve_aitk_command(config_path: Path) -> List[str]:
    """Resolve the ai-toolkit command, preferring MODL_AITOOLKIT_ROOT."""
    aitk_root = os.getenv("MODL_AITOOLKIT_ROOT", "")
    if aitk_root:
        run_py = os.path.join(aitk_root, "run.py")
        if os.path.exists(run_py):
            return [sys.executable, run_py, str(config_path)]
    return _build_train_command(config_path)


# ---------------------------------------------------------------------------
# Single-phase runner
# ---------------------------------------------------------------------------

def _run_single_phase(
    config_path: Path,
    emitter: EventEmitter,
    step_offset: int = 0,
    total_steps_override: int | None = None,
    step_base: int = 0,
) -> int:
    """Run a single ai-toolkit training phase.  Returns exit code.

    ``step_base`` is the ai-toolkit start_step for this phase (e.g. 1500 when
    resuming from a step-1500 checkpoint).  ai-toolkit reports steps starting
    from this value, so we subtract it before adding ``step_offset`` to get
    the correct global progress.
    """
    cmd = _resolve_aitk_command(config_path)
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
        emitter.error("AITOOLKIT_EXEC_FAILED", str(exc), recoverable=False)
        return 1

    last_step = None
    tail_lines: list[str] = []
    error_lines: list[str] = []
    in_traceback = False

    for raw_line in process.stdout or []:
        line = raw_line.strip()
        if not line:
            continue

        tail_lines.append(line)
        if len(tail_lines) > _TAIL_BUFFER_SIZE:
            tail_lines.pop(0)

        if "Traceback (most recent call last)" in line:
            in_traceback = True
            error_lines = [line]
        elif in_traceback:
            error_lines.append(line)
            if not line.startswith(" ") and not line.startswith("Traceback"):
                in_traceback = False
        elif any(p.search(line) for p in _ERROR_PATTERNS):
            error_lines.append(line)

        is_status = any(p.search(line) for p in _STATUS_PATTERNS)
        if is_status:
            emitter.emit({"type": "log", "level": "status", "message": line})
        else:
            emitter.info(line)

        step_match = _STEP_RE.search(line)
        if step_match:
            step = int(step_match.group(1))
            phase_total = int(step_match.group(2))
            if last_step != step:
                loss = None
                loss_match = _LOSS_RE.search(line)
                if loss_match:
                    try:
                        loss = float(loss_match.group(1))
                    except ValueError:
                        pass
                # Report progress relative to overall training, not just this phase.
                # Subtract step_base because ai-toolkit counts from start_step,
                # not from 0, when resuming from a checkpoint.
                global_step = step_offset + (step - step_base)
                global_total = total_steps_override or (step_offset + phase_total - step_base)
                emitter.progress(
                    stage="train",
                    step=global_step,
                    total_steps=global_total,
                    loss=loss,
                )
                last_step = step

    code = process.wait()
    if code != 0:
        if error_lines:
            error_detail = "\n".join(error_lines[-15:])
        elif tail_lines:
            error_detail = "\n".join(tail_lines[-10:])
        else:
            error_detail = "(no output captured)"

        summary = error_lines[-1] if error_lines else f"Process exited with code {code}"
        emitter.error(
            "TRAINING_FAILED",
            summary,
            recoverable=False,
            details={"exit_code": code, "output_tail": error_detail},
        )

    return code


# ---------------------------------------------------------------------------
# Config preparation helpers
# ---------------------------------------------------------------------------

def _load_spec(config_path: Path) -> dict | None:
    """Load and return spec if it's a TrainJobSpec (has 'params' key)."""
    import yaml

    with open(config_path) as f:
        spec = yaml.safe_load(f)
    if isinstance(spec, dict) and "params" in spec:
        return spec
    return None


def _write_phase_config(
    spec: dict,
    phase: TrainingPhase,
    phase_steps: int,
    resume_from: str | None,
    train_overrides: dict[str, Any] | None,
) -> tuple[Path, int]:
    """Build ai-toolkit config for a phase and write to a temp file.

    Returns (config_path, resume_step) where resume_step is the step number
    from the checkpoint (0 if not resuming).
    """
    import copy
    import tempfile
    import yaml

    from .config_builder import _step_from_checkpoint_path

    phase_spec = copy.deepcopy(spec)

    # ai-toolkit treats "steps" as the total target step number to reach.
    # When resuming from a checkpoint at step N, we need steps = N + phase_steps
    # so that ai-toolkit actually trains for phase_steps more iterations.
    resume_step = 0
    if resume_from:
        phase_spec["params"]["resume_from"] = resume_from
        resume_step = _step_from_checkpoint_path(resume_from) or 0
        phase_spec["params"]["steps"] = resume_step + phase_steps
        if resume_step:
            print(f"[modl] Phase target: step {resume_step} + {phase_steps} = {resume_step + phase_steps}")
    else:
        phase_spec["params"]["steps"] = phase_steps

    config = spec_to_aitoolkit_config(phase_spec, train_overrides=train_overrides)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config, tmp)
        return Path(tmp.name), resume_step


def _emit_vram_hint(spec: dict, emitter: EventEmitter) -> None:
    """Emit VRAM usage hints for Qwen-Image models."""
    base_model_id = str(spec.get("model", {}).get("base_model_id", "")).lower()
    lora_type = spec.get("params", {}).get("lora_type", "character")

    if "qwen-image" not in base_model_id and "qwen_image" not in base_model_id:
        return

    if lora_type == "style":
        msg = (
            "Qwen-Image style profile: ~23GB VRAM at 1024px "
            "(fits RTX 3090/4090 24GB with 3-bit+ARA)."
        )
    else:
        msg = (
            "Qwen-Image character/object profile: ~30GB VRAM at 1024px "
            "(needs 32GB-class GPU; 24GB NOT currently supported for character)."
        )
    emitter.emit({"type": "log", "level": "status", "message": msg})


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

    # Try to load as a TrainJobSpec; fall back to passing config directly
    try:
        spec = _load_spec(config_path)
    except Exception as e:
        print(f"[modl] WARNING: spec load failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        spec = None

    if spec is None:
        # Not a TrainJobSpec — run directly as a single phase
        code = _run_single_phase(config_path, emitter)
        if code == 0:
            emitter.completed("ai-toolkit training command finished")
        return code

    # --- TrainJobSpec: resolve strategy and run phases ---
    _emit_vram_hint(spec, emitter)

    params = spec.get("params", {})
    base_model_id = spec.get("model", {}).get("base_model_id", "")
    lora_type = params.get("lora_type", "character")
    total_steps = params.get("steps", 2000)
    output_dir = spec.get("output", {}).get("destination_dir")

    from .arch_config import detect_arch
    arch_key = detect_arch(base_model_id)
    strategy = resolve_strategy(arch_key, lora_type)

    if strategy.is_multiphase:
        print(f"[modl] Training strategy: {strategy.name} ({len(strategy.phases)} phases)")
        for i, phase in enumerate(strategy.phases):
            steps = phase.step_count(total_steps)
            print(f"[modl]   Phase {i+1}: {phase.name} — {steps} steps")
            if phase.train_overrides:
                print(f"[modl]     overrides: {phase.train_overrides}")

    phase_step_counts = strategy.phase_steps(total_steps)
    step_offset = 0
    resume_from = params.get("resume_from")  # initial resume (user-provided)

    for i, (phase, phase_steps) in enumerate(zip(strategy.phases, phase_step_counts)):
        phase_num = i + 1
        is_last = phase_num == len(strategy.phases)

        if strategy.is_multiphase:
            emitter.emit({
                "type": "log",
                "level": "status",
                "message": f"Phase {phase_num}/{len(strategy.phases)}: {phase.name} "
                           f"({phase_steps} steps)",
            })

        config, resume_step = _write_phase_config(
            spec,
            phase,
            phase_steps=phase_steps,
            resume_from=resume_from,
            train_overrides=phase.train_overrides or None,
        )

        code = _run_single_phase(
            config,
            emitter,
            step_offset=step_offset,
            total_steps_override=total_steps,
            step_base=resume_step,
        )

        if code != 0:
            return code

        step_offset += phase_steps

        # Find checkpoint from this phase for the next phase to resume from
        if not is_last and output_dir:
            checkpoint = find_latest_checkpoint(Path(output_dir))
            if checkpoint:
                resume_from = checkpoint
                print(f"[modl] Phase {phase_num} complete. "
                      f"Resuming phase {phase_num+1} from: {Path(checkpoint).name}")
            else:
                print(f"[modl] WARNING: No checkpoint found after phase {phase_num}. "
                      "Next phase will start from scratch.")
                resume_from = None

    # All phases complete
    if output_dir and os.path.isdir(output_dir):
        scan_output_artifacts(output_dir, emitter)
    emitter.completed("ai-toolkit training command finished")
    return 0
