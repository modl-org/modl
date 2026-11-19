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
    try:
        if _register_capture_specs_for_subclasses():
            applied.append("transformers.capture_registry")
    except Exception:  # noqa: BLE001
        pass
    return applied


def _register_capture_specs_for_subclasses() -> bool:
    """Make transformers' output capture work for class-swapped models.

    transformers looks up what a model can record (hidden states, attentions)
    in a registry keyed by ``str(model.__class__)`` and filled in ``__init__``.
    ai-toolkit rebinds loaded components onto its own subclasses
    (``module.__class__ = wrapper``), whose ``__init__`` never ran, so the
    lookup misses and ``output_hidden_states=True`` returns ``None`` — SDXL
    training dies encoding its first prompt. Register every subclass when it
    is defined instead.
    """
    from transformers.modeling_utils import PreTrainedModel
    from transformers.utils import output_capturing

    registry = getattr(output_capturing, "_CAN_RECORD_REGISTRY", None)
    if registry is None or getattr(PreTrainedModel, "_modl_capture_patch", False):
        return False
    original = PreTrainedModel.__dict__["__init_subclass__"].__func__

    def __init_subclass__(cls, **kwargs):
        original(cls, **kwargs)
        specs = getattr(cls, "_can_record_outputs", None)
        if specs:
            registry.setdefault(str(cls), specs)

    PreTrainedModel.__init_subclass__ = classmethod(__init_subclass__)
    PreTrainedModel._modl_capture_patch = True
    return True


def main() -> None:
    run_py = os.path.abspath(sys.argv[1])
    sys.argv = sys.argv[1:]
    # `python run.py` puts run.py's directory first on sys.path; keep that.
    sys.path[0] = os.path.dirname(run_py)
    apply_shims()
    runpy.run_path(run_py, run_name="__main__")


if __name__ == "__main__":
    main()
