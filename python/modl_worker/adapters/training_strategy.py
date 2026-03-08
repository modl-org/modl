"""Training strategy — multi-phase training orchestration.

A training strategy is an ordered list of phases.  Each phase defines a
fraction of total steps and config overrides that get merged into the
ai-toolkit config.  The orchestrator in train_adapter runs phases
sequentially, resuming from the previous phase's checkpoint.

Strategy resolution follows the same (arch_key, lora_type) pattern as
everything else in arch_config — no ad-hoc conditionals.

Example: Z-Image Turbo style LoRA uses two phases:
  Phase 1 (60%): balanced timesteps — learns dataset patterns
  Phase 2 (40%): linear_timesteps2 — high-noise bias, drives style convergence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .arch_config import detect_arch


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TrainingPhase:
    """A single phase within a training strategy."""
    name: str
    steps_fraction: float  # fraction of total steps (0.0–1.0)
    train_overrides: dict[str, Any] = field(default_factory=dict)

    def step_count(self, total_steps: int) -> int:
        return max(1, round(total_steps * self.steps_fraction))


@dataclass
class TrainingStrategy:
    """Ordered list of phases that make up a complete training run."""
    name: str
    phases: list[TrainingPhase]

    @property
    def is_multiphase(self) -> bool:
        return len(self.phases) > 1

    def phase_steps(self, total_steps: int) -> list[int]:
        """Return step counts per phase, ensuring they sum to total_steps."""
        counts = [p.step_count(total_steps) for p in self.phases]
        # Adjust last phase to absorb rounding
        counts[-1] = total_steps - sum(counts[:-1])
        return counts


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------
# Each key is (arch_key, lora_type).  Missing combos fall through to
# (arch_key, "*") then ("*", "*") (single-phase default).

_STRATEGIES: dict[tuple[str, str], TrainingStrategy] = {
    # Z-Image Turbo style: single-phase high denoise
    # linear_timesteps2 from step 0 — biases toward high noise levels
    # for faster style convergence in fewer steps.
    ("zimage_turbo", "style"): TrainingStrategy(
        name="zimage_turbo_style",
        phases=[
            TrainingPhase(
                name="high_noise",
                steps_fraction=1.0,
                train_overrides={"linear_timesteps2": True},
            ),
        ],
    ),
}

_DEFAULT_STRATEGY = TrainingStrategy(
    name="default",
    phases=[TrainingPhase(name="train", steps_fraction=1.0)],
)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_strategy(arch_key: str, lora_type: str) -> TrainingStrategy:
    """Resolve the training strategy for an (arch, lora_type) pair."""
    # Exact match
    strategy = _STRATEGIES.get((arch_key, lora_type))
    if strategy:
        return strategy
    # Wildcard lora_type
    strategy = _STRATEGIES.get((arch_key, "*"))
    if strategy:
        return strategy
    return _DEFAULT_STRATEGY


def resolve_strategy_for_model(base_model_id: str, lora_type: str) -> TrainingStrategy:
    """Convenience: detect arch from model ID, then resolve strategy."""
    arch_key = detect_arch(base_model_id)
    return resolve_strategy(arch_key, lora_type)


# ---------------------------------------------------------------------------
# Phase config helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(run_dir: Path) -> str | None:
    """Find the most recent .safetensors checkpoint in a run directory."""
    inner = _inner_run_dir(run_dir)
    if not inner.exists():
        return None

    checkpoints = sorted(
        (f for f in inner.iterdir() if f.suffix == ".safetensors" and "_" in f.stem),
        key=lambda f: f.stat().st_mtime,
    )
    return str(checkpoints[-1]) if checkpoints else None


def _inner_run_dir(run_dir: Path) -> Path:
    """Resolve the inner run directory (run_dir/name/name/)."""
    name = run_dir.name
    inner = run_dir / name
    return inner if inner.is_dir() else run_dir
