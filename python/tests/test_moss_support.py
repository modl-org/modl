"""MOSS-Image glue: Comfy fused-qkv conversion and unmerged LoRA application."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from modl_worker.adapters.moss_support import (
    _group_lora,
    apply_moss_lora,
    is_moss_pipeline,
    prepare_moss_pipeline,
    revert_moss_lora,
)
from modl_worker.adapters.pipeline_loader import _convert_moss_checkpoint


def test_convert_splits_fused_qkv_and_keeps_text_refiner():
    q, k, v = torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4)
    refiner = torch.randn(12, 4)
    out = _convert_moss_checkpoint(
        {
            "layers.3.proj.qkv.weight": torch.cat([q, k, v]),
            "layers.3.proj.o.weight": q,
            "text_embedder.blocks.0.qkv.weight": refiner,
        }
    )
    assert torch.equal(out["layers.3.proj.q.weight"], q)
    assert torch.equal(out["layers.3.proj.k.weight"], k)
    assert torch.equal(out["layers.3.proj.v.weight"], v)
    assert "layers.3.proj.qkv.weight" not in out
    assert torch.equal(out["text_embedder.blocks.0.qkv.weight"], refiner)


def test_convert_strips_comfy_prefix():
    out = _convert_moss_checkpoint({"model.diffusion_model.final_norm_video.weight": torch.ones(2)})
    assert set(out) == {"final_norm_video.weight"}


def test_group_lora_kohya_and_peft_names():
    groups = _group_lora(
        {
            "diffusion_model.layers.0.proj.q.lora_down.weight": torch.zeros(2, 4),
            "diffusion_model.layers.0.proj.q.lora_up.weight": torch.zeros(4, 2),
            "diffusion_model.layers.0.proj.q.alpha": torch.tensor(2.0),
            "transformer.layers.1.proj.o.lora_A.weight": torch.zeros(2, 4),
            "transformer.layers.1.proj.o.lora_B.weight": torch.zeros(4, 2),
        }
    )
    assert set(groups) == {"layers.0.proj.q", "layers.1.proj.o"}
    assert set(groups["layers.0.proj.q"]) == {"down", "up", "alpha"}
    assert set(groups["layers.1.proj.o"]) == {"down", "up"}


class _Block(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.ModuleDict(
            {name: torch.nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")}
        )


class MOSSImageTransformer(torch.nn.Module):  # name is what is_moss_pipeline checks
    def __init__(self, dim=8):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(dim)])


class _Pipe:
    def __init__(self):
        self.transformer = MOSSImageTransformer().to(torch.bfloat16)


def _write_lora(path, tensors):
    from safetensors.torch import save_file

    save_file(tensors, str(path))


def test_apply_and_revert_unmerged_lora(tmp_path):
    pipe = _Pipe()
    assert is_moss_pipeline(pipe)
    base_q = pipe.transformer.layers[0].proj["q"]
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    before = base_q(x)

    down, up = torch.randn(2, 8), torch.randn(8, 2)
    lora = tmp_path / "l.safetensors"
    _write_lora(
        lora,
        {
            "diffusion_model.layers.0.proj.q.lora_down.weight": down,
            "diffusion_model.layers.0.proj.q.lora_up.weight": up,
            "diffusion_model.layers.0.proj.q.alpha": torch.tensor(4.0),  # scale 2
            "diffusion_model.layers.9.proj.q.lora_down.weight": down.clone(),  # no such layer
            "diffusion_model.layers.9.proj.q.lora_up.weight": up.clone(),
        },
    )
    assert apply_moss_lora(pipe, str(lora), weight=0.5)
    wrapped = pipe.transformer.layers[0].proj["q"]
    assert wrapped is not base_q
    assert wrapped.weight is base_q.weight  # upstream reads .weight.dtype
    expected = before + (1.0 * ((x.float() @ down.T) @ up.T)).to(x.dtype)  # 4/2 * 0.5
    assert torch.allclose(wrapped(x).float(), expected.float(), atol=1e-2)

    assert revert_moss_lora(pipe)
    assert pipe.transformer.layers[0].proj["q"] is base_q
    assert not revert_moss_lora(pipe)


def test_fused_qkv_lora_is_split(tmp_path):
    pipe = _Pipe()
    down, up = torch.randn(2, 8), torch.randn(24, 2)
    lora = tmp_path / "fused.safetensors"
    _write_lora(
        lora,
        {
            "diffusion_model.layers.0.proj.qkv.lora_down.weight": down,
            "diffusion_model.layers.0.proj.qkv.lora_up.weight": up,
        },
    )
    assert apply_moss_lora(pipe, str(lora))
    proj = pipe.transformer.layers[0].proj
    for i, name in enumerate("qkv"):
        assert torch.equal(proj[name].up, up[i * 8 : (i + 1) * 8].float())
    assert proj["o"].__class__ is torch.nn.Linear


def test_reapply_replaces_previous_lora(tmp_path):
    pipe = _Pipe()
    base_q = pipe.transformer.layers[0].proj["q"]
    lora = tmp_path / "l.safetensors"
    _write_lora(
        lora,
        {
            "layers.0.proj.q.lora_down.weight": torch.randn(2, 8),
            "layers.0.proj.q.lora_up.weight": torch.randn(8, 2),
        },
    )
    apply_moss_lora(pipe, str(lora))
    apply_moss_lora(pipe, str(lora))
    # One wrapper deep, not wrapper-of-wrapper.
    assert pipe.transformer.layers[0].proj["q"].base is base_q


def test_prompt_embedding_cache():
    calls = []

    class P:
        def encode_prompt(self, prompt, max_sequence_length=4096):
            calls.append(prompt)
            return (prompt, max_sequence_length)

    p = P()
    prepare_moss_pipeline(p)
    assert p.encode_prompt("a") == ("a", 4096)
    assert p.encode_prompt("a") == ("a", 4096)
    assert p.encode_prompt(["a"]) == ("a", 4096)  # str and [str] share a key
    p.encode_prompt("")
    assert calls == ["a", ""]
