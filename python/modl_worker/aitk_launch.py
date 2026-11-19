"""Launch ai-toolkit's ``run.py`` with shims for the diffusers version modl pins.

ai-toolkit targets its own diffusers commit; modl installs a newer release on
top for inference. ai-toolkit imports every built-in extension for every job,
so one stale import in an extension modl never trains stops all training.

Usage: python aitk_launch.py <path/to/run.py> <config.yaml> [...]
"""

import importlib
import os
import runpy
import sys


def apply_shims() -> list[str]:
    """Patch known import gaps. Returns the names of the shims applied."""
    applied = []
    # diffusers 0.43 moved encode_video from pipelines.ltx2.export_utils to
    # utils.export_utils; ai-toolkit's LTX-2 extension still imports the old path.
    try:
        ltx2_export = importlib.import_module("diffusers.pipelines.ltx2.export_utils")
        if not hasattr(ltx2_export, "encode_video"):
            from diffusers.utils.export_utils import encode_video

            ltx2_export.encode_video = encode_video
            applied.append("ltx2.encode_video")
    except Exception:  # noqa: BLE001 — a missing shim target is not our error to raise
        pass
    return applied


def main() -> None:
    run_py = os.path.abspath(sys.argv[1])
    sys.argv = sys.argv[1:]
    # `python run.py` puts run.py's directory first on sys.path; keep that.
    sys.path[0] = os.path.dirname(run_py)
    apply_shims()
    runpy.run_path(run_py, run_name="__main__")


if __name__ == "__main__":
    main()
