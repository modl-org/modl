"""Qwen-Image-2.1 glue: Comfy gate/up split, RGBA finalisation, embed cache."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")
from PIL import Image

from modl_worker.adapters.arch_config import ARCH_CONFIGS, detect_arch
from modl_worker.adapters.pipeline_loader import _convert_qwen_image_21_checkpoint
from modl_worker.adapters.qwen21_support import (
    edit_output_size,
    finalize_rgba,
    prepare_qwen21_pipeline,
)


def test_convert_splits_fused_gate_up_in_gate_then_up_order():
    gate, up = torch.randn(6, 4), torch.randn(6, 4)
    attn = torch.randn(4, 4)
    out = _convert_qwen_image_21_checkpoint(
        {
            "model.diffusion_model.transformer_blocks.7.img_mlp.gate_up.weight": torch.cat([gate, up]),
            "model.diffusion_model.transformer_blocks.7.attn.to_q.weight": attn,
        }
    )
    assert torch.equal(out["transformer_blocks.7.img_mlp.gate_layer.weight"], gate)
    assert torch.equal(out["transformer_blocks.7.img_mlp.proj.weight"], up)
    assert torch.equal(out["transformer_blocks.7.attn.to_q.weight"], attn)
    assert not any("gate_up" in k for k in out)


def _rgba(alpha: np.ndarray) -> Image.Image:
    rgb = np.full(alpha.shape + (3,), 40, dtype=np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA")


def test_near_opaque_rgba_is_flattened_over_white():
    alpha = np.full((64, 64), 255, dtype=np.uint8)
    alpha[::7, ::7] = 252  # the opaque-noise sprinkle the model emits
    out = finalize_rgba(_rgba(alpha))
    assert out.mode == "RGB"


def test_real_cutout_keeps_alpha():
    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[16:48, 16:48] = 255
    out = finalize_rgba(_rgba(alpha))
    assert out.mode == "RGBA"


def test_rgb_passes_through():
    img = Image.new("RGB", (8, 8))
    assert finalize_rgba(img) is img


class _FakePipe:
    def __init__(self):
        self.calls = 0

    def encode_prompt(self, prompt=None, image=None, device=None, num_images_per_prompt=1,
                      prompt_embeds=None, prompt_embeds_mask=None):
        self.calls += 1
        return (prompt, image)


def test_embed_cache_hits_text_only_and_skips_edits():
    pipe = _FakePipe()
    original_calls = lambda: pipe.calls  # noqa: E731
    prepare_qwen21_pipeline(pipe)
    pipe.encode_prompt(prompt="a cat", device="cuda")
    pipe.encode_prompt(prompt="a cat", device="cuda")
    assert original_calls() == 1
    pipe.encode_prompt(prompt="a cat", image=["img"], device="cuda")
    pipe.encode_prompt(prompt="a cat", image=["img"], device="cuda")
    assert original_calls() == 3


def test_arch_detection_and_turbo_schedule():
    assert detect_arch("qwen-image-2-1") == "qwen_image_21"
    assert detect_arch("qwen-image-2-1-turbo") == "qwen_image_21_turbo"
    assert detect_arch("qwen-image") == "qwen_image"
    turbo = ARCH_CONFIGS["qwen_image_21_turbo"]
    assert len(turbo["pipeline_kwargs"]["sample_sigmas"]) == 8
    assert "pipeline_kwargs" not in ARCH_CONFIGS["qwen_image_21"]


def test_edit_output_size_keeps_aspect_and_area():
    assert edit_output_size(1536, (1024, 1024)) == (1536, 1536)
    w, h = edit_output_size(1536, (768, 1024))
    assert w % 32 == 0 and h % 32 == 0
    assert abs(w / h - 0.75) < 0.02
    assert abs(w * h / 1536**2 - 1) < 0.03
