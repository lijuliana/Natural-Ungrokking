"""Core correctness tests: model, probes, scoring, transience metric."""

import math

import numpy as np
import pytest
import torch

from fogen.model import GPT, ModelConfig
from fogen.probes.v1_probes import ALL_PROBES, build_battery
from fogen.evals.scoring import Scorer, aggregate
from fogen.analysis.transience import transience_score

TINY = ModelConfig(vocab_size=64, n_layer=2, d_model=32, n_head=2, ctx_len=64)


def test_model_forward_shape():
    m = GPT(TINY)
    x = torch.randint(0, 64, (3, 10))
    assert m(x).shape == (3, 10, 64)


def test_v1_param_count_matches_paper():
    m = GPT(ModelConfig())  # v1 defaults: 4L/256d/2H, vocab 8192
    assert m.num_params() == 11_534_340  # paper Table 1: "11.5M total"
    groups = (sum(p.numel() for p in m.matrix_params())
              + sum(p.numel() for p in m.embed_params())
              + sum(p.numel() for p in m.scalar_params()))
    assert groups == m.num_params()  # every param is in exactly one group


def test_model_overfits_tiny_batch():
    torch.manual_seed(0)
    m = GPT(TINY)
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss = m.loss(x, y)
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.5, "model failed to overfit a tiny batch"


def test_muon_step_changes_weights():
    from fogen.training.muon import Muon
    m = GPT(TINY)
    # zero-init lm_head/wo block trunk gradients at step 0; seed lm_head and
    # check wo (first matrix to receive gradient under nanochat init)
    torch.nn.init.normal_(m.lm_head.weight, std=0.02)
    before = m.blocks[0].attn.wo.weight.clone()
    opt = Muon(m.matrix_params(), lr=0.04)
    m.loss(torch.randint(0, 64, (2, 8)), torch.randint(0, 64, (2, 8))).backward()
    opt.step()
    assert not torch.allclose(before, m.blocks[0].attn.wo.weight)


def test_battery_counts_and_uniqueness():
    items = build_battery(seed=0)
    probes = {it["probe"] for it in items}
    assert len(probes) == 14
    ids = [it["item_id"] for it in items]
    assert len(ids) == len(set(ids))
    for it in items:
        assert it["split"] in ("train", "heldout")
        assert it["correct"] != it["distractor"]
        assert it["prefix"].strip()
    # every probe has both splits
    for p in probes:
        splits = {it["split"] for it in items if it["probe"] == p}
        assert splits == {"train", "heldout"}, f"{p} missing a split"


def test_battery_deterministic():
    assert build_battery(seed=0) == build_battery(seed=0)
    assert build_battery(seed=0) != build_battery(seed=1)


def test_scorer_prefers_boosted_token():
    # fake model: always boosts token 7
    def logits_fn(ids):
        B, T = ids.shape
        out = torch.zeros(B, T, 16)
        out[:, :, 7] = 5.0
        return out

    sc = Scorer(encode=lambda s: [ord(c) % 16 for c in s], logits_fn=logits_fn)
    items = [{"probe": "p", "category": "c", "item_id": "i0", "template_id": "t",
              "split": "train", "prefix": "ab",
              "correct": chr(7), "distractor": chr(8), "chance": 0.5}]
    rows = sc.score_items(items)
    assert rows[0]["argmax_acc"] == 1
    assert rows[0]["logprob_diff"] > 0
    agg = aggregate(rows)
    assert agg[0]["argmax_acc"] == 1.0


def test_transience_detects_peak_then_collapse():
    steps = list(range(0, 1000, 10))
    accs = [0.5 + 0.4 * math.exp(-((s - 300) / 150) ** 2) for s in steps]  # bump
    r = transience_score(steps, accs)
    assert r["transient"] and r["transience"] > 0.5
    assert 200 < r["peak_step"] < 420


def test_transience_rejects_monotone():
    steps = list(range(0, 1000, 10))
    accs = [0.5 + 0.4 * (1 - math.exp(-s / 200)) for s in steps]  # saturating rise
    r = transience_score(steps, accs)
    assert not r["transient"]


def test_frequency_mapping_matches_patterns_and_battery():
    from fogen.theory.frequency import PATTERNS, PATTERN_TO_PROBE, count_patterns
    assert set(PATTERN_TO_PROBE) == set(PATTERNS)
    assert set(PATTERN_TO_PROBE.values()) == {f.__name__ for f in ALL_PROBES}
    out = count_patterns(["A big red dog ran. She said hi.", "an apple, a egg"])
    assert out["total_words"] == 12
    assert out["adjective_order_size_color"]["support_per_million"] > 0
    assert out["determiner_a_an"]["support_ratio"] == 0.5


def test_one_doc_per_seq_loader(tmp_path):
    from fogen.data import ShardedLoader
    # stream: docs of lengths 5,3,7 separated by eot=0, ids never 0 inside
    docs = [[3, 4, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15, 16, 17]]
    stream = []
    for d in docs:
        stream.extend(d); stream.append(0)
    np.asarray(stream, dtype=np.uint16).tofile(tmp_path / "shard_00000.bin")
    ld = ShardedLoader(tmp_path, batch_size=8, ctx_len=12, seed=0,
                       one_doc_per_seq=True)
    starts = {tuple(d)[0] for d in docs}
    for _ in range(5):
        x, y = ld.next_batch()
        assert x.shape == (8, 12)
        for row, yrow in zip(x, y):
            row = row.tolist()
            assert row[0] in starts            # starts at a doc boundary
            e = row.index(0)
            assert row[:e] in [d[:e] for d in docs]  # doc tokens intact
            assert all(v == 0 for v in row[e:])      # EOT padding after end
            assert yrow[-1] == 0 or e == 12


def test_mask_padding_targets(tmp_path):
    from fogen.data import ShardedLoader
    docs = [[3, 4, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15, 16, 17]]
    stream = []
    for d in docs:
        stream.extend(d); stream.append(0)
    np.asarray(stream, dtype=np.uint16).tofile(tmp_path / "shard_00000.bin")
    ld = ShardedLoader(tmp_path, batch_size=8, ctx_len=12, seed=0,
                       one_doc_per_seq=True, mask_padding=True)
    by_first = {d[0]: d for d in docs}
    for _ in range(5):
        x, y = ld.next_batch()
        for row, yrow in zip(x.tolist(), y.tolist()):
            d = by_first[row[0]]
            n = len(d) + 1                      # doc tokens + its EOT
            assert yrow[:n - 1] == (d + [0])[1:n]   # first EOT still a target
            assert all(v == -100 for v in yrow[n - 1:])  # padding ignored
    # loss runs with ignored targets present
    m = GPT(TINY)
    x, y = ld.next_batch()
    assert torch.isfinite(m.loss(x, y))


def test_max_tokens_caps_sampling(tmp_path):
    from fogen.data import ShardedLoader
    rng = np.random.default_rng(0)
    rng.integers(8, 100, 500).astype(np.uint16).tofile(tmp_path / "shard_00000.bin")
    (np.full(500, 7, dtype=np.uint16)).tofile(tmp_path / "shard_00001.bin")
    # cap at 600: full shard 0 + first 100 of shard 1
    ld = ShardedLoader(tmp_path, batch_size=4, ctx_len=16, seed=0, max_tokens=600)
    assert ld.sizes.tolist() == [500, 100]
    for _ in range(20):
        x, _ = ld.next_batch()
        assert x.shape == (4, 16)
    # cap inside shard 0 only: shard 1 dropped entirely
    ld2 = ShardedLoader(tmp_path, batch_size=4, ctx_len=16, seed=0, max_tokens=300)
    assert ld2.sizes.tolist() == [300]
    for _ in range(20):
        x, _ = ld2.next_batch()
        assert (x != 7).all()  # never samples shard 1 content


def test_bpb_uniform_model_matches_closed_form():
    from fogen.evals.bpb import evaluate_bpb

    class Uniform(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(*x.shape, 16)

    # token bytes: id 0 special (0 bytes, masked), ids 1-7 -> 1 byte,
    # ids 8-15 -> 2 bytes
    tb = torch.tensor([0] + [1] * 7 + [2] * 8)
    stream = np.tile(np.arange(16, dtype=np.uint16), 50)
    r = evaluate_bpb(Uniform(), stream, tb, ctx_len=16, batch_size=4)
    # uniform over 16 tokens: nll = ln16 per scored token; masked id-0
    # targets drop from both sums -> bpb = 15*ln16 / (ln2 * 23) = 60/23
    assert r["val_bpb"] == pytest.approx(60 / 23, rel=1e-6)
    assert r["bytes"] == r["tokens_scored"] / 15 * 23


def test_bpb_doc_aligned_uniform_closed_form():
    from fogen.evals.bpb import evaluate_bpb_docs

    class Uniform(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(*x.shape, 16)

    tb = torch.tensor([0] + [1] * 7 + [2] * 8)
    # docs [1..7] and [8..15]; doc-aligned scores targets row[1:], so the
    # first token of each doc is input-only: 6 one-byte + 7 two-byte targets
    stream = np.asarray([1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 12, 13, 14, 15, 0],
                        dtype=np.uint16)
    r = evaluate_bpb_docs(Uniform(), stream, tb, ctx_len=32, batch_size=4)
    assert r["docs"] == 2
    assert r["bytes"] == 6 * 1 + 7 * 2
    assert r["val_bpb"] == pytest.approx(13 * math.log(16) / (math.log(2) * 20),
                                         rel=1e-6)


def test_battery_rev1_matches_frozen_file():
    import json
    from pathlib import Path
    frozen = [json.loads(l) for l in
              Path("data/probes/v1/battery.jsonl").read_text().splitlines()]
    assert build_battery(seed=0, rev=1) == frozen


def test_battery_rev2_fixes_weaknesses():
    from collections import Counter
    items = build_battery(seed=0, rev=2)
    n = Counter((it["probe"], it["split"]) for it in items)
    for probe in {it["probe"] for it in items}:
        assert n[(probe, "train")] >= 40, (probe, n[(probe, "train")])
        assert n[(probe, "heldout")] >= 16, (probe, n[(probe, "heldout")])
    assert n[("reflexive_pronoun", "heldout")] >= 24
    ci_held = [it["prefix"] for it in items
               if it["probe"] == "common_idiom" and it["split"] == "heldout"]
    assert len(set(ci_held)) == len(ci_held)  # no duplicated prefixes
    cq_held = [it for it in items
               if it["probe"] == "close_quote" and it["split"] == "heldout"]
    assert all(len(it["correct"]) <= 2 for it in cq_held)  # minimal pairs


def test_rvp_battery_matches_frozen_file():
    import json
    from pathlib import Path
    from fogen.probes.rvp import build_rvp_battery
    frozen = [json.loads(l) for l in
              Path("data/probes/rvp1/battery.jsonl").read_text().splitlines()]
    assert build_rvp_battery(seed=0) == frozen


def test_rvp_battery_structure():
    from collections import Counter
    from fogen.probes.rvp import build_rvp_battery
    items = build_rvp_battery(seed=0)
    assert len({it["item_id"] for it in items}) == len(items)
    n = Counter((it["family"], it["condition"], it["split"]) for it in items)
    fams = {it["family"] for it in items}
    assert len(fams) == 8
    for fam in fams:
        for cond in ("conflict", "agree"):
            for split in ("train", "heldout"):
                assert n[(fam, cond, split)] >= 12, (fam, cond, split)
    for it in items:
        assert it["probe"] == f"{it['family']}.{it['condition']}"
        assert it["correct"] != it["distractor"]
        assert it["correct"].startswith(" ") and it["distractor"].startswith(" ")
    # conflict/agree contrast must be a minimal pair on the same prefix set
    an = [it for it in items if it["family"] == "det_an_choice"
          and it["condition"] == "conflict"]
    assert all(it["correct"].startswith(" an ") and
               it["distractor"].startswith(" a ") and
               it["correct"][4:] == it["distractor"][3:] for it in an)


def test_rvp_rev2_extends_frozen_rev1():
    import json
    from pathlib import Path
    from fogen.probes.rvp import build_rvp_battery
    frozen1 = [json.loads(l) for l in
               Path("data/probes/rvp1/battery.jsonl").read_text().splitlines()]
    frozen2 = [json.loads(l) for l in
               Path("data/probes/rvp2/battery.jsonl").read_text().splitlines()]
    assert build_rvp_battery(seed=0, rev=2) == frozen2
    assert frozen2[:len(frozen1)] == frozen1  # rev1 embedded unchanged
    ma = [i for i in frozen2 if i["family"] == "modal_agreement"]
    assert len(ma) == 384
    assert all(i["correct"].rstrip("s") != i["correct"] or
               i["condition"] == "conflict" for i in ma)


def test_rvp_rev3_extends_frozen_rev2():
    import json
    from pathlib import Path
    from fogen.probes.rvp import build_rvp_battery
    frozen2 = [json.loads(l) for l in
               Path("data/probes/rvp2/battery.jsonl").read_text().splitlines()]
    frozen3 = [json.loads(l) for l in
               Path("data/probes/rvp3/battery.jsonl").read_text().splitlines()]
    assert build_rvp_battery(seed=0, rev=3) == frozen3
    assert frozen3[:len(frozen2)] == frozen2  # rev2 embedded unchanged
    new = frozen3[len(frozen2):]
    assert {i["family"] for i in new} == {"modal_agreement_v2",
                                          "irregular_past_v2",
                                          "negation_bare_verb_v2"}
    # v2 conflict item content matches the originals (re-issued, new ids)
    for fam in ("modal_agreement", "irregular_past", "negation_bare_verb"):
        orig = {(i["prefix"], i["correct"], i["distractor"])
                for i in frozen3 if i["family"] == fam
                and i["condition"] == "conflict"}
        v2 = {(i["prefix"], i["correct"], i["distractor"])
              for i in new if i["family"] == f"{fam}_v2"
              and i["condition"] == "conflict"}
        assert v2 == orig


def test_rvp_rev4_extends_frozen_rev3():
    import json
    from pathlib import Path
    from fogen.probes.rvp import GIRLS, BOYS, GIRLS_V2, BOYS_V2, \
        build_rvp_battery
    frozen3 = [json.loads(l) for l in
               Path("data/probes/rvp3/battery.jsonl").read_text().splitlines()]
    b4 = build_rvp_battery(seed=0, rev=4)
    assert b4[:len(frozen3)] == frozen3  # rev3 embedded unchanged
    new = b4[len(frozen3):]
    assert {i["family"] for i in new} == {"pronoun_gender_ref_v2",
                                          "reflexive_gender_v2",
                                          "pronoun_gender_noun"}
    # v2 name sets are strict supersets of the originals
    assert set(GIRLS) < set(GIRLS_V2) and set(BOYS) < set(BOYS_V2)
    assert len(GIRLS_V2) == len(BOYS_V2) == 18
    assert not set(GIRLS_V2) & set(BOYS_V2)
    # every conflict item cues female -> " she"/" herself"; agree cues male
    for i in new:
        if i["condition"] == "conflict":
            assert i["correct"] in (" she", " herself")
        else:
            assert i["correct"] in (" he", " himself")
    # noun family: cue word is a common noun, never a name
    for i in new:
        if i["family"] == "pronoun_gender_noun":
            assert i["prefix"].startswith("The ")
