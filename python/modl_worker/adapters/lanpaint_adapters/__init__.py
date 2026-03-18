"""Per-model adapters for LanPaint inpainting."""

from .base import ModelAdapter
from .z_image import ZImageAdapter
from .flux_klein import FluxKleinAdapter

__all__ = ["ModelAdapter", "ZImageAdapter", "FluxKleinAdapter"]
