"""Tests for krea2 LoRA key conversion + diff-patch detection.

These exercise the pure-Python helpers in lora_utils (regex key mapping,
safetensors header sniffing) without importing torch/PIL — same isolation
trick as test_config_builder so they run in the pytest-only CI env.
"""

import importlib.util
import types
from pathlib import Path

_ADAPTERS = Path(__file__).resolve().parents[1] / "modl_worker" / "adapters"
_pkg = types.ModuleType("_lu_isolated")
_pkg.__path__ = [str(_ADAPTERS)]
import sys

sys.modules.setdefault("_lu_isolated", _pkg)
_spec = importlib.util.spec_from_file_location(
    "_lu_isolated.lora_utils", _ADAPTERS / "lora_utils.py"
)
lora_utils = importlib.util.module_from_spec(_spec)
sys.modules["_lu_isolated.lora_utils"] = lora_utils
_spec.loader.exec_module(lora_utils)


def test_detects_krea2_lora_by_module_names():
    sd = {
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": 0,
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": 0,
    }
    assert lora_utils._is_krea2_lora(sd)
    # A plain diffusers-format qwen LoRA is not krea2.
    assert not lora_utils._is_krea2_lora(
        {"transformer.transformer_blocks.0.attn.to_q.lora_A.weight": 0}
    )


def test_converts_krea2_module_names_to_diffusers():
    sd = {
        "diffusion_model.blocks.3.attn.wq.lora_A.weight": 1,
        "diffusion_model.blocks.3.attn.wq.lora_B.weight": 2,
        "diffusion_model.blocks.3.attn.wo.lora_A.weight": 3,
        "diffusion_model.blocks.3.mlp.up.lora_A.weight": 4,
        "diffusion_model.txtfusion.refiner_blocks.1.attn.wk.lora_A.weight": 5,
        "diffusion_model.txtfusion.projector.lora_A.weight": 6,
        "diffusion_model.first.lora_A.weight": 7,
        "diffusion_model.last.linear.lora_A.weight": 8,
        "diffusion_model.tproj.1.lora_A.weight": 9,
        # unmappable modulation residual → dropped, not fatal
        "diffusion_model.last.modulation.lin.lora_A.weight": 10,
    }
    conv, dropped = lora_utils.convert_krea2_lora_to_diffusers(sd)
    assert dropped == 1
    assert conv["transformer.transformer_blocks.3.attn.to_q.lora_A.weight"] == 1
    assert conv["transformer.transformer_blocks.3.attn.to_q.lora_B.weight"] == 2
    assert conv["transformer.transformer_blocks.3.attn.to_out.0.lora_A.weight"] == 3
    assert conv["transformer.transformer_blocks.3.ff.up.lora_A.weight"] == 4
    assert (
        conv["transformer.text_fusion.refiner_blocks.1.attn.to_k.lora_A.weight"] == 5
    )
    assert conv["transformer.text_fusion.projector.lora_A.weight"] == 6
    assert conv["transformer.img_in.lora_A.weight"] == 7
    assert conv["transformer.final_layer.linear.lora_A.weight"] == 8
    assert conv["transformer.time_mod_proj.lora_A.weight"] == 9
    # every survivor is transformer-prefixed diffusers format
    assert all(k.startswith("transformer.") for k in conv)
    assert not any("modulation" in k for k in conv)
