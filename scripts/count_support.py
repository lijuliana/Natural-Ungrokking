"""Count battery-derived support bigrams in tokenized training shards.

For every rvp battery item, build two token patterns from the LAST word of
the prefix joined with each continuation (e.g. "must play" / "must plays",
"an apple" / "a apple") and count exact token-sequence occurrences per
corpus. Aggregated per (family, condition, kind) this measures corpus
support for the rule form vs the prior form — the raw material for the
rule-frequency (f*) axis. Descriptive measurement; the registered f*
ranking is defined separately before use (Step 4).

  python scripts/count_support.py data/climbmix/bpe8192/shards \
      --tokenizer-dir data/climbmix/bpe8192 \
      --battery data/probes/rvp3/battery.jsonl --out runs/support_web.json

Windowed mode (f*-v2, registered 2026-06-10 BEFORE reading any support
JSON): adjacent bigrams are the wrong support proxy for the gender
families — the evidence "Lily ... she" is rarely adjacent. For each
single-token gendered cue occurrence, find the FIRST " she"/" he" within
the next --window tokens and attribute the window to it. rule_support for
the female-cue class = #(first pronoun is she); prior_support = #(first
pronoun is he). support_ratio = (rule+1)/(prior+1), per cue class.

  python scripts/count_support.py data/climbmix/bpe8192/shards \
      --tokenizer-dir data/climbmix/bpe8192 --mode window --window 16 \
      --out runs/support_web_window.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fogen.data import load_tokenizer
from fogen.evals.scoring import load_battery


def count_seq(arr: np.ndarray, ids: list[int]) -> int:
    if len(ids) == 1:
        return int((arr == ids[0]).sum())
    mask = arr[: len(arr) - len(ids) + 1] == ids[0]
    for j, t in enumerate(ids[1:], 1):
        mask &= arr[j: len(arr) - len(ids) + 1 + j] == t
    return int(mask.sum())


def single_token_ids(tok, words):
    """All single-token encodings of each word, with and without the
    leading space (mid-text occurrences tokenize as " Lily")."""
    out = {}
    for w in words:
        variants = [v for v in (w, " " + w)
                    if len(tok.encode(v).ids) == 1]
        if variants:
            out[w] = [tok.encode(v).ids[0] for v in variants]
    return out


def find_seq_ends(a, ids):
    """End positions (index of last token) of exact sequence matches."""
    if len(ids) == 1:
        return np.where(a == ids[0])[0]
    mask = a[: len(a) - len(ids) + 1] == ids[0]
    for j, t in enumerate(ids[1:], 1):
        mask &= a[j: len(a) - len(ids) + 1 + j] == t
    return np.where(mask)[0] + len(ids) - 1


def rescue_window_mode(args, tok):
    """Descriptive post-hoc measure (rescue-dose unit accounting): the
    injected docs use cue names disjoint from every battery name by
    design, and only 2/24 are single tokens in the web vocab, so the
    frozen battery-cue counter cannot see them. Same first-pronoun-in-
    window logic, but cues are matched as full token sequences."""
    from gen_rescue_docs import RESCUE_BOYS, RESCUE_GIRLS

    she, he = tok.encode(" she").ids, tok.encode(" he").ids
    assert len(she) == 1 and len(he) == 1, "pronouns must be single-token"
    cue_seqs = {}
    for cname, names in (("girl_names", RESCUE_GIRLS),
                         ("boy_names", RESCUE_BOYS)):
        seqs = []
        for n in names:
            for v in (" " + n, n):
                ids = tok.encode(v).ids
                if ids not in seqs:
                    seqs.append(ids)
        cue_seqs[cname] = seqs
    stats = {c: {"n_cues": 0, "first_she": 0, "first_he": 0, "neither": 0}
             for c in cue_seqs}
    n_tokens = 0
    shards = sorted(Path(args.shard_dir).glob("shard_*.bin"))
    for si, shard in enumerate(shards):
        a = np.asarray(np.memmap(shard, dtype=np.uint16, mode="r"))
        n_tokens += len(a)
        she_pos = np.where(a == she[0])[0]
        he_pos = np.where(a == he[0])[0]
        for cname, seqs in cue_seqs.items():
            cues = np.sort(np.concatenate(
                [find_seq_ends(a, ids) for ids in seqs]))
            ns = she_pos[np.minimum(np.searchsorted(she_pos, cues, "right"),
                                    len(she_pos) - 1)] if len(she_pos) \
                else np.full_like(cues, 2**62)
            nh = he_pos[np.minimum(np.searchsorted(he_pos, cues, "right"),
                                   len(he_pos) - 1)] if len(he_pos) \
                else np.full_like(cues, 2**62)
            ds = np.where(ns > cues, ns - cues, 2**62)
            dh = np.where(nh > cues, nh - cues, 2**62)
            in_w_s = (ds <= args.window) & (ds < dh)
            in_w_h = (dh <= args.window) & (dh < ds)
            st = stats[cname]
            st["n_cues"] += int(len(cues))
            st["first_she"] += int(in_w_s.sum())
            st["first_he"] += int(in_w_h.sum())
            st["neither"] += int((~in_w_s & ~in_w_h).sum())
        print(f"shard {si + 1}/{len(shards)} done", flush=True)

    out = {"mode": "window_rescue_cues", "window": args.window,
           "n_tokens": n_tokens,
           "cue_seqs": {c: [list(map(int, s)) for s in v]
                        for c, v in cue_seqs.items()},
           "classes": {}}
    for cname, st in stats.items():
        rule, prior = (st["first_she"], st["first_he"]) \
            if cname == "girl_names" else (st["first_he"], st["first_she"])
        out["classes"][cname] = {
            **st, "rule_support": rule, "prior_support": prior,
            "support_ratio": (rule + 1) / (prior + 1)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"n_tokens={n_tokens}")
    for cname, c in out["classes"].items():
        print(f"  {cname:12s} cues={c['n_cues']:>9d} she={c['first_she']:>8d}"
              f" he={c['first_he']:>8d} ratio={c['support_ratio']:.4f}")


def window_mode(args, tok):
    from fogen.probes.rvp import BOYS_V2, FEM_NOUNS, GIRLS_V2, MASC_NOUNS

    she, he = tok.encode(" she").ids, tok.encode(" he").ids
    assert len(she) == 1 and len(he) == 1, "pronouns must be single-token"
    cue_classes = {
        "girl_names": single_token_ids(tok, GIRLS_V2),
        "boy_names": single_token_ids(tok, BOYS_V2),
        "fem_nouns": single_token_ids(tok, FEM_NOUNS),
        "masc_nouns": single_token_ids(tok, MASC_NOUNS),
    }
    stats = {c: {"n_cues": 0, "first_she": 0, "first_he": 0, "neither": 0}
             for c in cue_classes}
    n_tokens = 0
    shards = sorted(Path(args.shard_dir).glob("shard_*.bin"))
    for si, shard in enumerate(shards):
        a = np.asarray(np.memmap(shard, dtype=np.uint16, mode="r"))
        n_tokens += len(a)
        she_pos = np.where(a == she[0])[0]
        he_pos = np.where(a == he[0])[0]
        for cname, ids in cue_classes.items():
            flat = [t for v in ids.values() for t in v]
            cues = np.where(np.isin(a, flat))[0]
            ns = she_pos[np.minimum(np.searchsorted(she_pos, cues, "right"),
                                    len(she_pos) - 1)] if len(she_pos) \
                else np.full_like(cues, 2**62)
            nh = he_pos[np.minimum(np.searchsorted(he_pos, cues, "right"),
                                   len(he_pos) - 1)] if len(he_pos) \
                else np.full_like(cues, 2**62)
            ds = np.where(ns > cues, ns - cues, 2**62)
            dh = np.where(nh > cues, nh - cues, 2**62)
            in_w_s = (ds <= args.window) & (ds < dh)
            in_w_h = (dh <= args.window) & (dh < ds)
            st = stats[cname]
            st["n_cues"] += int(len(cues))
            st["first_she"] += int(in_w_s.sum())
            st["first_he"] += int(in_w_h.sum())
            st["neither"] += int((~in_w_s & ~in_w_h).sum())
        print(f"shard {si + 1}/{len(shards)} done", flush=True)

    out = {"mode": "window", "window": args.window, "n_tokens": n_tokens,
           "cue_vocab": {c: sorted(v) for c, v in cue_classes.items()},
           "classes": {}}
    for cname, st in stats.items():
        rule, prior = (st["first_she"], st["first_he"]) \
            if cname in ("girl_names", "fem_nouns") \
            else (st["first_he"], st["first_she"])
        out["classes"][cname] = {
            **st, "rule_support": rule, "prior_support": prior,
            "support_ratio": (rule + 1) / (prior + 1)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"n_tokens={n_tokens}")
    for cname, c in out["classes"].items():
        print(f"  {cname:12s} cues={c['n_cues']:>9d} she={c['first_she']:>8d}"
              f" he={c['first_he']:>8d} ratio={c['support_ratio']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--battery", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["bigram", "window"], default="bigram")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--cues", choices=["battery", "rescue"],
                    default="battery")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer_dir)
    if args.mode == "window":
        if args.cues == "rescue":
            rescue_window_mode(args, tok)
        else:
            window_mode(args, tok)
        return
    assert args.battery, "--battery required in bigram mode"
    items = load_battery(args.battery)

    # pattern string -> set of (family, condition, kind) it contributes to
    patterns: dict[str, set] = defaultdict(set)
    for it in items:
        last = " " + it["prefix"].split()[-1]
        fam, cond = it["probe"].rsplit(".", 1)
        patterns[last + it["correct"]].add((fam, cond, "correct"))
        patterns[last + it["distractor"]].add((fam, cond, "distractor"))

    encoded = {p: tok.encode(p).ids for p in patterns}
    counts = {p: 0 for p in patterns}
    n_tokens = 0
    shards = sorted(Path(args.shard_dir).glob("shard_*.bin"))
    for si, shard in enumerate(shards):
        a = np.asarray(np.memmap(shard, dtype=np.uint16, mode="r"))
        n_tokens += len(a)
        for p, ids in encoded.items():
            counts[p] += count_seq(a, ids)
        print(f"shard {si + 1}/{len(shards)} done", flush=True)

    agg = defaultdict(int)
    for p, keys in patterns.items():
        for fam, cond, kind in keys:
            agg[f"{fam}.{cond}.{kind}"] += counts[p]

    Path(args.out).write_text(json.dumps({
        "n_tokens": n_tokens,
        "aggregate": dict(sorted(agg.items())),
        "patterns": {p: {"ids": encoded[p], "count": counts[p],
                         "keys": sorted("/".join(k) for k in keys)}
                     for p, keys in sorted(patterns.items())},
    }, indent=2))
    print(f"n_tokens={n_tokens}  patterns={len(patterns)}")
    for k, v in sorted(agg.items()):
        print(f"  {k:48s} {v}")


if __name__ == "__main__":
    main()
