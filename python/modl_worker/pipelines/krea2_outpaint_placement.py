from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image


SOURCE_MAX_EDGE = 384
SEAM_PX = 32


@dataclass(frozen=True)
class RegisteredSource:
    condition: Image.Image
    placed_source: Image.Image
    canvas_size: tuple[int, int]
    bbox: tuple[int, int, int, int]
    seam_px: int = SEAM_PX

    @property
    def bbox_normalized(self) -> list[float]:
        width, height = self.canvas_size
        x0, y0, x1, y1 = self.bbox
        return [x0 / width, y0 / height, x1 / width, y1 / height]


@dataclass(frozen=True)
class PassPlan:
    axis: str
    intermediate_size: tuple[int, int] | None
    first_bbox: tuple[int, int, int, int]
    second_bbox: tuple[int, int, int, int] | None

    @property
    def pass_count(self) -> int:
        return 1 if self.intermediate_size is None else 2


def _resize_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    if max(image.size) <= max_edge:
        return image.copy()
    scale = max_edge / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def prepare_source(
    source: Image.Image,
    canvas_size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    source_max_edge: int = SOURCE_MAX_EDGE,
    seam_px: int = SEAM_PX,
) -> RegisteredSource:
    width, height = canvas_size
    x0, y0, x1, y1 = bbox
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise ValueError("Canvas dimensions must be positive multiples of 16")
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Source bbox is outside the canvas: {bbox}")

    source = source.convert("RGB")
    box_width, box_height = x1 - x0, y1 - y0
    source_ratio = source.width / source.height
    box_ratio = box_width / box_height
    tolerance = max(0.025, 2.0 / min(box_width, box_height))
    if abs(box_ratio / source_ratio - 1.0) > tolerance:
        raise ValueError("Source bbox must preserve the source image aspect ratio")

    placed = source.resize((box_width, box_height), Image.Resampling.LANCZOS)
    return RegisteredSource(
        condition=_resize_max_edge(placed, source_max_edge),
        placed_source=placed,
        canvas_size=canvas_size,
        bbox=bbox,
        seam_px=seam_px,
    )


def _align_up(value: int, alignment: int = 16) -> int:
    return int(math.ceil(value / alignment) * alignment)


def plan_passes(prepared: RegisteredSource) -> PassPlan:
    width, height = prepared.canvas_size
    x0, y0, x1, y1 = prepared.bbox
    source_width, source_height = x1 - x0, y1 - y0
    if source_width == width or source_height == height:
        return PassPlan("direct", None, prepared.bbox, None)

    candidates: list[tuple[int, PassPlan]] = []
    intermediate_height = _align_up(source_height)
    if intermediate_height < height:
        intermediate_y = max(
            0,
            min(height - intermediate_height, y0 - (intermediate_height - source_height) // 2),
        )
        local_y = y0 - intermediate_y
        candidates.append(
            (
                width * intermediate_height,
                PassPlan(
                    "horizontal_first",
                    (width, intermediate_height),
                    (x0, local_y, x1, local_y + source_height),
                    (0, intermediate_y, width, intermediate_y + intermediate_height),
                ),
            )
        )

    intermediate_width = _align_up(source_width)
    if intermediate_width < width:
        intermediate_x = max(
            0,
            min(width - intermediate_width, x0 - (intermediate_width - source_width) // 2),
        )
        local_x = x0 - intermediate_x
        candidates.append(
            (
                intermediate_width * height,
                PassPlan(
                    "vertical_first",
                    (intermediate_width, height),
                    (local_x, y0, local_x + source_width, y1),
                    (intermediate_x, 0, intermediate_x + intermediate_width, height),
                ),
            )
        )

    if not candidates:
        return PassPlan("direct", None, prepared.bbox, None)
    return min(candidates, key=lambda item: (item[0], item[1].axis))[1]


def composite(generated: Image.Image, prepared: RegisteredSource) -> Image.Image:
    generated = generated.convert("RGB")
    if generated.size != prepared.canvas_size:
        raise ValueError("Generated image size does not match the canvas")

    width, height = prepared.placed_source.size
    yy, xx = np.mgrid[:height, :width]
    edge_distance = np.minimum.reduce((xx, yy, width - 1 - xx, height - 1 - yy))
    alpha = np.clip(edge_distance / max(1, prepared.seam_px), 0.0, 1.0)
    alpha_image = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    result = generated.copy()
    result.paste(prepared.placed_source, prepared.bbox[:2], alpha_image)
    return result
