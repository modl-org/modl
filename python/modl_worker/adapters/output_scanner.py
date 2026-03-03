"""Scan training output directory for LoRA artifacts and emit events."""

import glob
import hashlib
import os
import re
from pathlib import Path

from modl_worker.protocol import EventEmitter


def scan_output_artifacts(output_dir: str, emitter: EventEmitter) -> None:
    """Emit artifact event for the final LoRA only (not intermediate checkpoints).

    ai-toolkit saves checkpoints as ``{name}_000002000.safetensors`` and the
    final output as ``{name}.safetensors``.  We only register the final one in
    the DB — checkpoints stay on disk for manual comparison but don't
    clutter ``modl model ls``.
    """
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
