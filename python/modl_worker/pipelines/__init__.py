"""Vendored third-party edit pipelines. See README.md for sources + licenses.

Classes here are addressed with an explicit ``modl.`` prefix in arch_config
(e.g. ``"modl.Krea2OstrisEditPipeline"``) and resolved by
``pipeline_loader._get_pipeline`` / ``_import_class``.

The prefix is not cosmetic. ``Krea2Transformer2DModel`` exists in BOTH diffusers
and the vendored module, and their ``state_dict`` keys are identical (verified:
430/430), so a wrong resolution would load successfully and then misbehave at
inference rather than raising. Namespacing makes the choice explicit at every
call site: the generation arches ask for the diffusers class, the edit arches
ask for the vendored one, and neither can silently get the other.
"""

VENDORED_PREFIX = "modl."

# class name -> (module path, attribute). Imported lazily: the vendored module
# pulls in diffusers/transformers and is only needed on edit paths.
_VENDORED_CLASSES = {
    "Krea2OstrisEditPipeline": ("modl_worker.pipelines.krea2_ostris_edit", "Krea2OstrisEditPipeline"),
    "Krea2Transformer2DModel": ("modl_worker.pipelines.krea2_ostris_edit", "Krea2Transformer2DModel"),
}


def is_vendored(class_name: str) -> bool:
    """True if *class_name* is namespaced to the vendored modules."""
    return class_name.startswith(VENDORED_PREFIX)


def strip_prefix(class_name: str) -> str:
    """Return *class_name* without the ``modl.`` namespace prefix."""
    return class_name[len(VENDORED_PREFIX):] if is_vendored(class_name) else class_name


def get_vendored_class(class_name: str):
    """Resolve a ``modl.``-prefixed (or bare) vendored class name to the class."""
    import importlib

    bare = strip_prefix(class_name)
    entry = _VENDORED_CLASSES.get(bare)
    if entry is None:
        raise ImportError(
            f"Unknown vendored class {class_name!r}. "
            f"Known: {sorted(_VENDORED_CLASSES)}"
        )
    module_path, attr = entry
    return getattr(importlib.import_module(module_path), attr)
