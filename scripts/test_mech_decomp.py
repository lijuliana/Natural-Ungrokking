import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from mech_decomp import crossframe_ncm, decompose, score_run  # noqa: E402

from fogen.model import GPT, ModelConfig, _rmsnorm  # noqa: E402


def small_model(seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=64, n_layer=4, d_model=32, n_head=2,
                      ctx_len=64)
    model = GPT(cfg)
    # init zeroes residual projections and lm_head; perturb everything so
    # the decomposition is tested on non-trivial contributions
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    model.eval()
    return model


def test_decomposition_is_exact():
    model = small_model()
    idx = torch.randint(0, 64, (1, 7))
    she, he = 3, 5
    with torch.no_grad():
        c_dir, c_attn, c_mlp, gap_pre, gap_post, hiddens = decompose(
            model, idx, she, he)
        # components sum exactly to the pre-softcap gap
        assert abs(c_dir + sum(c_attn) + sum(c_mlp) - gap_pre) < 1e-4
        # gap_pre matches an independent pre-softcap forward
        full = model(idx)[0, -1]
        assert abs(gap_post - (full[she] - full[he]).item()) < 1e-4
        # softcap is monotone elementwise: pre/post gaps agree in sign
        if abs(gap_pre) > 1e-6:
            assert gap_pre * gap_post >= 0
        assert len(hiddens) == model.cfg.n_layer + 1
        # hiddens are rms-normalized
        for h in hiddens:
            assert abs(h.pow(2).mean().sqrt().item() - 1.0) < 1e-4


def test_decompose_matches_real_forward_path():
    # the reconstructed final residual must reproduce the model's own
    # logits, i.e. the replicated forward loop is faithful
    model = small_model(seed=1)
    idx = torch.randint(0, 64, (1, 11))
    with torch.no_grad():
        ref = model(idx)[0, -1]
        _, _, _, _, gap_post, _ = decompose(model, idx, 7, 9)
        assert abs(gap_post - (ref[7] - ref[9]).item()) < 1e-4


FRAMES8 = ["cried", "laughed"] * 4


def test_crossframe_ncm_separable():
    g = torch.Generator().manual_seed(0)
    fem = [torch.tensor([4.0, 0.0]) + 0.1 * torch.randn(2, generator=g)
           for _ in range(8)]
    masc = [torch.tensor([-4.0, 0.0]) + 0.1 * torch.randn(2, generator=g)
            for _ in range(8)]
    acc, sep = crossframe_ncm(fem, masc, FRAMES8, FRAMES8)
    assert acc == 1.0
    assert sep > 1.0


def test_crossframe_ncm_noise_is_unbiased_chance():
    # mean accuracy over many draws must sit near 0.5 (the LOO variant
    # this replaced is biased far below chance on noise)
    g = torch.Generator().manual_seed(1)
    accs = []
    for _ in range(200):
        fem = [torch.randn(16, generator=g) for _ in range(8)]
        masc = [torch.randn(16, generator=g) for _ in range(8)]
        acc, _ = crossframe_ncm(fem, masc, FRAMES8, FRAMES8)
        accs.append(acc)
    mean_acc = sum(accs) / len(accs)
    assert 0.42 <= mean_acc <= 0.58


def test_crossframe_ncm_needs_two_frames():
    import pytest as _pytest
    pts = [torch.randn(4) for _ in range(4)]
    with _pytest.raises(AssertionError, match="frames"):
        crossframe_ncm(pts, pts, ["cried"] * 4, ["cried"] * 4)


def test_quarantined_seeds_refused():
    for run in ("runs/web_packed_v2_s1042", "runs/v1_repro_s1044"):
        with pytest.raises(AssertionError, match="quarantined"):
            score_run(run, "data/probes/rvp3/battery.jsonl")
