"""Descriptive instrument (Step-6 design, pre-registration measurement):
for EVERY single-token capitalized word in a tokenizer's vocab, count
16-token-window first-pronoun events in a shard dir, using the identical
matching convention as count_support.py --mode window (first " she" vs
" he" after the cue; lowercase only, mirroring the frozen counter).

Purpose: empirical inventory of gendered name-like cues per corpus —
which tokens carry the name->pronoun evidence, and how concentrated it
is in battery vs non-battery names. Informs the Step-6 kill list/rate;
no governed (behavioral) data is read.

  python scripts/scan_gendered_tokens.py data/tinystories/bpe8192/shards \
      --tokenizer-dir data/tinystories/bpe8192 --out runs/gendered_tokens_ts.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from fogen.data import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer_dir)
    cand = {}
    for i in range(tok.get_vocab_size()):
        s = tok.decode([i])
        if re.fullmatch(r" [A-Z][a-z]{2,11}", s):
            cand[i] = s[1:]
    print(f"{len(cand)} capitalized single-token candidates")
    cand_ids = np.array(sorted(cand), dtype=np.uint16)
    idx_of = {int(t): k for k, t in enumerate(cand_ids)}

    she = tok.encode(" she").ids
    he = tok.encode(" he").ids
    assert len(she) == 1 and len(he) == 1
    she, he = she[0], he[0]

    n = len(cand_ids)
    n_cues = np.zeros(n, dtype=np.int64)
    first_she = np.zeros(n, dtype=np.int64)
    first_he = np.zeros(n, dtype=np.int64)
    n_tokens = 0

    shards = sorted(Path(args.shard_dir).glob("shard_*.bin"))
    for si, shard in enumerate(shards):
        a = np.asarray(np.memmap(shard, dtype=np.uint16, mode="r"))
        n_tokens += len(a)
        she_pos = np.where(a == she)[0]
        he_pos = np.where(a == he)[0]
        cues = np.where(np.isin(a, cand_ids))[0]
        which = np.array([idx_of[int(t)] for t in a[cues]], dtype=np.int64)
        ns = she_pos[np.minimum(np.searchsorted(she_pos, cues, "right"),
                                len(she_pos) - 1)] if len(she_pos) \
            else np.full_like(cues, 2**62)
        nh = he_pos[np.minimum(np.searchsorted(he_pos, cues, "right"),
                               len(he_pos) - 1)] if len(he_pos) \
            else np.full_like(cues, 2**62)
        ds = np.where(ns > cues, ns - cues, 2**62)
        dh = np.where(nh > cues, nh - cues, 2**62)
        in_s = (ds <= args.window) & (ds < dh)
        in_h = (dh <= args.window) & (dh < ds)
        np.add.at(n_cues, which, 1)
        np.add.at(first_she, which[in_s], 1)
        np.add.at(first_he, which[in_h], 1)
        print(f"shard {si + 1}/{len(shards)} done", flush=True)

    rows = []
    for k, t in enumerate(cand_ids):
        if n_cues[k] == 0:
            continue
        rows.append({"token": cand[int(t)], "id": int(t),
                     "n_cues": int(n_cues[k]),
                     "first_she": int(first_she[k]),
                     "first_he": int(first_he[k]),
                     "she_ratio": (first_she[k] + 1) / (first_he[k] + 1)})
    rows.sort(key=lambda r: -(r["first_she"] + r["first_he"]))
    Path(args.out).write_text(json.dumps(
        {"window": args.window, "n_tokens": n_tokens, "rows": rows},
        indent=2))
    print(f"n_tokens={n_tokens}, wrote {len(rows)} tokens -> {args.out}")
    for r in rows[:25]:
        print(f"  {r['token']:12s} cues={r['n_cues']:>8d} "
              f"she={r['first_she']:>7d} he={r['first_he']:>7d} "
              f"ratio={r['she_ratio']:.3f}")


if __name__ == "__main__":
    main()
