"""Count surface-prior frequencies in the tokenized training shards.

Registered use (RESEARCH_LOG 2026-06-10): the himself/herself counts fix the
agree-direction for reflexive_gender BEFORE rvp1 scoring. Other counts are
descriptive (rule-frequency axis groundwork).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fogen.data import load_tokenizer

WORDS = [" himself", " herself", " was", " were", " a", " an",
         " he", " she", " went", " goed", " ran", " runned"]


def count_seq(arr: np.ndarray, ids: list[int]) -> int:
    if len(ids) == 1:
        return int((arr == ids[0]).sum())
    mask = arr[: len(arr) - len(ids) + 1] == ids[0]
    for j, t in enumerate(ids[1:], 1):
        mask &= arr[j: len(arr) - len(ids) + 1 + j] == t
    return int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer_dir)
    encoded = {w: tok.encode(w).ids for w in WORDS}
    totals = {w: 0 for w in WORDS}
    n_tokens = 0
    for shard in sorted(Path(args.shard_dir).glob("shard_*.bin")):
        arr = np.memmap(shard, dtype=np.uint16, mode="r")
        n_tokens += len(arr)
        a = np.asarray(arr)
        for w, ids in encoded.items():
            totals[w] += count_seq(a, ids)
    out = {"n_tokens": n_tokens,
           "token_ids": {w: encoded[w] for w in WORDS},
           "counts": totals}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["counts"], indent=2))
    print(f"n_tokens={n_tokens}")


if __name__ == "__main__":
    main()
