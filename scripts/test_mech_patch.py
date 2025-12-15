import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from mech_patch import (component_keys, forward_components,  # noqa: E402
                        forward_project)

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


def test_component_keys():
    cfg = ModelConfig(vocab_size=64, n_layer=4, d_model=32, n_head=2)
    keys = component_keys(cfg)
    assert keys[0] == "emb" and len(keys) == 1 + 4 * 2 + 4


def test_unpatched_matches_plain_forward():
    model = small_model()
    idx = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        lg, cache = forward_components(model, idx, record=True)
        ref = model(idx)
    assert torch.allclose(lg, ref, atol=1e-5)
    assert set(cache) == set(component_keys(model.cfg))


def test_self_patch_is_identity():
    model = small_model(1)
    idx = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        lg, cache = forward_components(model, idx, record=True)
        for key in component_keys(model.cfg):
            lg2, _ = forward_components(model, idx,
                                        patch={key: cache[key]})
            assert torch.allclose(lg2, lg, atol=1e-5), key


def test_cross_model_patch_changes_logits():
    m1, m2 = small_model(2), small_model(3)
    idx = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        _, cache = forward_components(m1, idx, record=True)
        base, _ = forward_components(m2, idx)
        patched, _ = forward_components(m2, idx,
                                        patch={"head:1:0": cache["head:1:0"]})
    assert not torch.allclose(patched, base, atol=1e-6)


def test_projection_identity_sign_and_effect():
    model = small_model(4)
    idx = torch.randint(0, 64, (1, 12))
    u = torch.randn(32)
    u = u / u.norm()
    with torch.no_grad():
        plain = forward_project(model, idx)
        ref = model(idx)
        a = forward_project(model, idx, u=u, at_k=2)
        b = forward_project(model, idx, u=-u, at_k=2)
    assert torch.allclose(plain, ref, atol=1e-5)
    assert torch.allclose(a, b, atol=1e-5)        # sign-invariant
    assert not torch.allclose(a, ref, atol=1e-6)  # projection has an effect
