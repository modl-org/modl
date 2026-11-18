"""Source hold: schedule, token location per pipeline family, scheduler wrap."""

import pytest

torch = pytest.importorskip("torch")

from modl_worker.adapters.source_hold import (  # noqa: E402
    Guide,
    HoldConfig,
    SourceHold,
    locate_source,
    step_weights,
)


def test_step_weights_split_early_and_late():
    sigmas = [1.0, 0.75, 0.5, 0.25, 0.0]
    plan = step_weights(sigmas, ref_guide=0.9, crossover=0.5, late_gain=10.0, early_gain=0.8)
    assert len(plan) == 4
    # Early steps (sigma above 0.5) pull outside the edit area; late ones hold firmly.
    assert plan[0][0] > 0 and plan[0][1] == 0
    assert plan[-1] == (0.0, 1.0)


def test_guide_copies_source_late_where_estimate_is_close():
    sigmas = [1.0, 0.5, 0.0]
    g = Guide(HoldConfig(early_hold=0.7), sigmas)
    ref = torch.randn(1, 4, 6, 6)
    x0 = ref + 0.001 * torch.randn_like(ref)
    keep = torch.zeros(1, 1, 6, 6)
    out = g.apply(len(g.plan) - 1, x0, ref, keep)
    assert torch.allclose(out, ref, atol=1e-4)


def test_guide_leaves_far_tokens_alone_late():
    sigmas = [1.0, 0.5, 0.0]
    g = Guide(HoldConfig(early_hold=0.7), sigmas)
    ref = torch.zeros(1, 4, 4, 4)
    x0 = ref.clone()
    x0[..., :2, :2] = 5.0  # the model clearly wants an edit here
    ref = ref + 0.1 * torch.randn_like(ref)
    out = g.apply(len(g.plan) - 1, x0, ref, torch.zeros(1, 1, 4, 4))
    assert torch.allclose(out[..., :2, :2], x0[..., :2, :2], atol=1e-3)


def test_locate_qwen21_refs_first():
    th, tw, rh, rw, d = 4, 6, 2, 3, 8
    ref = torch.randn(1, rh * rw, d)
    tgt = torch.randn(1, th * tw, d)
    kw = {"hidden_states": torch.cat([ref, tgt], 1), "img_shapes": [[(1, rh, rw), (1, th, tw)]]}
    (h, w), grid, order = locate_source("qwen21", kw)
    assert (h, w) == (th, tw) and order is None
    assert torch.equal(grid[0, :, 1, 2], ref[0, 1 * rw + 2])


def test_locate_qwen_edit_target_first():
    th, tw, d = 3, 3, 4
    tgt = torch.randn(1, 9, d)
    ref = torch.randn(1, 9, d)
    kw = {"hidden_states": torch.cat([tgt, ref], 1), "img_shapes": [[(1, th, tw), (1, th, tw)]]}
    _, grid, _ = locate_source("qwen_edit", kw)
    assert torch.equal(grid[0, :, 2, 0], ref[0, 6])


def test_locate_klein_by_ids():
    th, tw, d = 2, 3, 4
    tgt_ids = torch.tensor([[0, h, w, 0] for h in range(th) for w in range(tw)])
    ref_ids = torch.tensor([[10, h, w, 0] for h in range(th) for w in range(tw)])
    ref = torch.randn(1, th * tw, d)
    kw = {
        "hidden_states": torch.cat([torch.randn(1, th * tw, d), ref], 1),
        "img_ids": torch.cat([tgt_ids, ref_ids])[None],
    }
    (h, w), grid, order = locate_source("klein", kw)
    assert (h, w) == (th, tw)
    assert torch.equal(grid[0, :, 1, 1], ref[0, 1 * tw + 1])
    assert order.tolist() == list(range(th * tw))


class _Sched:
    def __init__(self, sigmas):
        self.sigmas = torch.tensor(sigmas)
        self.step_index = None
        self.calls = []

    def _init_step_index(self, t):
        self.step_index = 0

    def step(self, model_output, timestep, sample, return_dict=False):
        self.calls.append(model_output.clone())
        s, sn = self.sigmas[self.step_index], self.sigmas[self.step_index + 1]
        self.step_index += 1
        return (sample + (sn - s) * model_output,)


class _Transformer(torch.nn.Module):
    def forward(self, hidden_states=None, img_shapes=None):
        return hidden_states


class QwenImageEditPlusPipeline:  # name drives family detection
    def __init__(self):
        self.scheduler = _Sched([1.0, 0.5, 0.0])
        self.transformer = _Transformer()


def test_hold_wraps_and_restores_scheduler():
    pipe = QwenImageEditPlusPipeline()
    d = 4
    src = torch.randn(1, 4, d)
    x = torch.randn(1, 4, d)
    with SourceHold(pipe, 0.7) as hold:
        assert hold.supported
        pipe.transformer(hidden_states=torch.cat([x, src], 1), img_shapes=[[(1, 2, 2), (1, 2, 2)]])
        v = torch.randn(1, 4, d)
        for _ in range(2):
            x = pipe.scheduler.step(v, 0, x)[0]
    # Last step is fully held: the output lands on the source where it was close.
    assert "step" not in vars(pipe.scheduler)  # class method restored
    assert hold.ref is not None


def test_unsupported_pipeline_is_a_no_op():
    class SomethingElse:
        scheduler = None
        transformer = None

    assert not SourceHold(SomethingElse(), 0.7).supported
