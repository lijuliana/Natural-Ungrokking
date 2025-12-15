import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from mech_decomp import decompose  # noqa: E402
from mech_heads import forward_heads, ov_cosines  # noqa: E402

from fogen.model import GPT, ModelConfig  # noqa: E402


def small_model(seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=64, n_layer=4, d_model=32, n_head=2,
                      ctx_len=64)
    model = GPT(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    model.eval()
    return model


def test_matches_plain_forward():
    model = small_model()
    idx = torch.randint(0, 64, (1, 11))
    with torch.no_grad():
        logits, *_ = forward_heads(model, idx)
        ref = model(idx)[0, -1]
    assert torch.allclose(logits, ref, atol=1e-5)


def test_heads_sum_to_layer_attn_and_gap():
    model = small_model(1)
    idx = torch.randint(0, 64, (1, 9))
    she, he = 3, 7
    with torch.no_grad():
        _, e0, hc, mlp, xf = forward_heads(model, idx)
        c_dir, c_attn, c_mlp, gap_pre, _, _ = decompose(model, idx, she, he)
    rms = xf.pow(2).mean().sqrt()
    dw = model.lm_head.weight[she] - model.lm_head.weight[he]
    for i in range(model.cfg.n_layer):
        head_sum = sum((hc[i][h] @ dw / rms).item()
                       for h in range(model.cfg.n_head))
        assert abs(head_sum - c_attn[i]) < 1e-4
        assert abs((mlp[i] @ dw / rms).item() - c_mlp[i]) < 1e-4
    assert abs((e0 @ dw / rms).item() - c_dir) < 1e-4
    tot = (e0 @ dw / rms).item() + sum(
        (hc[i][h] @ dw / rms).item()
        for i in range(model.cfg.n_layer)
        for h in range(model.cfg.n_head)) + sum(
        (m @ dw / rms).item() for m in mlp)
    assert abs(tot - gap_pre) < 1e-4


def test_ablation_zeroes_target_head_only():
    model = small_model(2)
    idx = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        base, _, hc_base, *_ = forward_heads(model, idx)
        abl, _, hc_abl, *_ = forward_heads(model, idx, ablate=(1, 0))
    assert hc_abl[1][0].abs().max() == 0.0
    assert hc_abl[1][1].abs().max() > 0.0
    # layer 0 is upstream of the ablation, so it is unchanged
    assert torch.allclose(hc_abl[0][0], hc_base[0][0], atol=1e-6)
    assert not torch.allclose(abl, base, atol=1e-6)


def test_ov_cosines_bounded_and_zero_safe():
    model = small_model(3)
    cue_ids = torch.tensor([2, 5, 9])
    out = ov_cosines(model, cue_ids, she=3, he=7)
    assert len(out) == model.cfg.n_layer
    for layer in out:
        assert len(layer) == model.cfg.n_head
        for c in layer:
            assert -1.0 <= c <= 1.0
    fresh = GPT(ModelConfig(vocab_size=64, n_layer=4, d_model=32,
                            n_head=2, ctx_len=64)).eval()
    out0 = ov_cosines(fresh, cue_ids, she=3, he=7)
    assert all(c == 0.0 for layer in out0 for c in layer)
