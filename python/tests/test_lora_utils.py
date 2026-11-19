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


class _Rows:
    """Stand-in for a tensor: a list of output rows with ``chunk``."""

    def __init__(self, rows):
        self.rows = list(rows)

    def chunk(self, n, dim=0):
        assert dim == 0
        size = len(self.rows) // n
        return tuple(_Rows(self.rows[i * size : (i + 1) * size]) for i in range(n))

    def clone(self):
        return _Rows(self.rows)


def test_qwen21_gate_up_is_split_into_gate_layer_and_proj():
    pre = "diffusion_model.transformer_blocks.4.img_mlp"
    shared_a = _Rows(["a"])
    sd = {
        f"{pre}.gate_up.lora_A.weight": shared_a,
        f"{pre}.gate_up.lora_B.weight": _Rows(["g0", "g1", "u0", "u1"]),
        f"{pre}.gate_up.alpha": 32,
        f"{pre}.out.lora_A.weight": "untouched",
        "diffusion_model.transformer_blocks.4.attn.to_q.lora_B.weight": "untouched",
    }
    conv, split = lora_utils.split_qwen21_gate_up(sd)
    assert split == 1
    assert not any(".gate_up." in k for k in conv)
    # lora_A (the shared input projection) goes to both halves, as separate
    # objects: safetensors will not save two keys backed by the same storage
    assert conv[f"{pre}.gate_layer.lora_A.weight"] is shared_a
    assert conv[f"{pre}.proj.lora_A.weight"] is not shared_a
    assert conv[f"{pre}.proj.lora_A.weight"].rows == ["a"]
    # lora_B rows: gate first, up second — the checkpoint converter's order
    assert conv[f"{pre}.gate_layer.lora_B.weight"].rows == ["g0", "g1"]
    assert conv[f"{pre}.proj.lora_B.weight"].rows == ["u0", "u1"]
    assert conv[f"{pre}.gate_layer.alpha"] == conv[f"{pre}.proj.alpha"] == 32
    assert conv[f"{pre}.out.lora_A.weight"] == "untouched"
    assert len(conv) == 8


def test_qwen21_split_leaves_other_loras_alone():
    sd = {"transformer.transformer_blocks.0.attn.to_q.lora_A.weight": 1}
    conv, split = lora_utils.split_qwen21_gate_up(sd)
    assert conv == sd and split == 0


def test_qwen21_split_copy_can_be_saved(tmp_path):
    """The split copy must survive safetensors' shared-storage check: when the
    save failed, the raw file was loaded and its gate_up modules were dropped."""
    import pytest

    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    pre = "diffusion_model.transformer_blocks.0.img_mlp"
    src = tmp_path / "lora.safetensors"
    st.save_file(
        {
            f"{pre}.gate_up.lora_A.weight": torch.randn(4, 8),
            f"{pre}.gate_up.lora_B.weight": torch.randn(12, 4),
        },
        str(src),
    )
    out = lora_utils._maybe_convert_qwen21_lora(str(src))
    assert out is not None
    conv = st.load_file(out)
    assert conv[f"{pre}.gate_layer.lora_B.weight"].shape == (6, 4)
    assert conv[f"{pre}.proj.lora_B.weight"].shape == (6, 4)
    assert conv[f"{pre}.proj.lora_A.weight"].shape == (4, 8)
