"""MiniMax-H3-Image glue: ConvRot Hadamard, INT8 linears, curve AdaLN, sizing."""

import pytest

torch = pytest.importorskip("torch")

from modl_worker.pipelines.minimax_h3_image import (
    CurveTimeEmbedder,
    Int8Linear,
    MiniMaxH3ImagePipeline,
    _stream_through_gpu,
    hadamard,
    rotate,
)


@pytest.mark.parametrize("size", [4, 16, 64, 256])
def test_hadamard_is_symmetric_orthogonal(size):
    h = hadamard(size, "cpu", torch.float64)
    assert torch.allclose(h, h.T)
    assert torch.allclose(h @ h.T, torch.eye(size, dtype=torch.float64), atol=1e-12)


def test_hadamard_rejects_non_power_of_four():
    with pytest.raises(ValueError):
        hadamard(8, "cpu", torch.float32)


def test_rotated_weights_cancel_activation_rotation():
    # ConvRot stores W' = W @ blockdiag(H); rot(x) @ W'^T must equal x @ W^T.
    group, n_in, n_out = 16, 64, 24
    w = torch.randn(n_out, n_in, dtype=torch.float64)
    x = torch.randn(5, n_in, dtype=torch.float64)
    w_rot = rotate(w, group)  # rotate each row's 16-wide blocks
    assert torch.allclose(rotate(x, group) @ w_rot.T, x @ w.T, atol=1e-9)


def _quantize(w):
    scale = w.abs().amax(dim=1, keepdim=True) / 127.0
    return (w / scale).round().clamp(-128, 127).to(torch.int8), scale


def test_int8_linear_dequant_matches_reference():
    w = torch.randn(32, 64)
    bias = torch.randn(32)
    q, s = _quantize(w)
    layer = Int8Linear(q, s, bias=bias)
    x = torch.randn(3, 7, 64)
    expected = torch.nn.functional.linear(x, q.float() * s, bias)
    assert torch.allclose(layer(x), expected, atol=1e-4)
    assert layer(x).shape == (3, 7, 32)


def test_int8_linear_with_rotation_recovers_unrotated_product():
    group = 16
    w = torch.randn(32, 64, dtype=torch.float64)
    q, s = _quantize(rotate(w, group).float())
    layer = Int8Linear(q, s, group=group)
    x = torch.randn(9, 64)
    # Within int8 rounding of the true (unrotated) product.
    err = (layer(x) - x @ w.float().T).abs().max() / (x @ w.float().T).abs().max()
    assert err < 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="W8A8 needs CUDA _int_mm")
def test_int8_linear_w8a8_close_to_dequant():
    w = torch.randn(256, 512, device="cuda")
    q, s = _quantize(w)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    ref = Int8Linear(q, s)(x).float()
    out = Int8Linear(q, s, w8a8=True)(x).float()
    assert ((out - ref).abs().max() / ref.abs().max()) < 0.03


def test_curve_time_embedder_interpolates_and_clamps():
    emb = CurveTimeEmbedder(grid=5, dim=2)
    emb.table.copy_(torch.tensor([[0.0, 0], [1, 10], [2, 20], [3, 30], [4, 40]]))
    out = emb(torch.tensor([0.0, 0.125, 1.0, 1.5, -0.2]))
    assert torch.allclose(out[:, 0], torch.tensor([0.0, 0.5, 4.0, 4.0, 0.0]))
    assert torch.allclose(out[:, 1], out[:, 0] * 10)


def test_stream_through_gpu_restores_host_tensors():
    layer = torch.nn.Linear(4, 4)
    _stream_through_gpu(layer, device="cpu")
    before = layer.weight.data_ptr()
    y = layer(torch.randn(2, 4))
    assert y.shape == (2, 4)
    assert layer.weight.data_ptr() == before


@pytest.mark.parametrize(
    "size,expected",
    [(None, (2048, 2048)), ((1600, 900), (2720, 1536)), ((1024, 1024), (2048, 2048))],
)
def test_default_size(size, expected):
    from PIL import Image

    images = None if size is None else [Image.new("RGB", size)]
    w, h = MiniMaxH3ImagePipeline._default_size(images)
    assert (w, h) == expected
    assert w % 32 == 0 and h % 32 == 0


def test_chunked_ff_matches_unchunked():
    from modl_worker.pipelines.minimax_h3_image import _ChunkedFF

    torch.manual_seed(0)
    ff = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8))
    x = torch.randn(2, 37, 8)
    chunked = _ChunkedFF(ff, tokens=5)
    torch.testing.assert_close(chunked(x), ff(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_w8a8_row_chunks_match_single_pass(monkeypatch):
    import modl_worker.pipelines.minimax_h3_image as h3

    torch.manual_seed(0)
    q = torch.randint(-127, 128, (64, 128), dtype=torch.int8, device="cuda")
    scale = torch.rand(64, 1, device="cuda") * 0.01
    lin = h3.Int8Linear(q, scale, w8a8=True).cuda()
    x = torch.randn(300, 128, device="cuda", dtype=torch.bfloat16)
    whole = lin(x)
    monkeypatch.setattr(h3, "_W8A8_ROWS", 64)  # 300 = 4*64 + a 44-row tail
    torch.testing.assert_close(lin(x), whole, atol=0.05, rtol=0.02)
    monkeypatch.setattr(h3, "_W8A8_ROWS", 98)  # 300 = 3*98 + a 6-row tail: dequant path
    y = lin(x)
    torch.testing.assert_close(y[:294], whole[:294], atol=0.05, rtol=0.02)
    lin.w8a8 = False
    torch.testing.assert_close(y[294:], lin(x)[294:], atol=0.05, rtol=0.02)
