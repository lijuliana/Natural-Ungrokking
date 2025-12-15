"""n-gram baseline for the probe battery (threat T13).

Trains nothing neural: counts n-grams over the exact token stream a
cell's model saw (same shard order, same max-tokens cap) and scores the
frozen battery with the same forced-choice rule as fogen.evals.scoring
(total continuation logprob, argmax per item, per-(probe,split) mean).
A stupid-backoff model (Brants et al., 2007), orders reported for
n=2 and n=5, alpha=0.4.

Counting is exact but query-restricted: only the n-grams needed to
score the battery are counted, so one streaming pass per corpus
suffices. Orders 1-4 are matched with uint64 keys (13 bits/token);
order 5 is matched as (4-gram key, next token) to avoid 65-bit keys.

  python scripts/ngram_baseline.py --shard-dir data/.../shards \
      --tokenizer-dir data/.../bpe8192 \
      --battery data/probes/rvp3/battery.jsonl \
      --cell v1_repro --out runs/ngram_v1_repro.json [--max-tokens N]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from fogen.data import load_tokenizer

ALPHA = 0.4
MAX_ORDER = 5
BITS = 13  # vocab 8192


def battery_queries(items, encode):
    """All (context, next) conditionals needed, as gram tuples per length."""
    grams = defaultdict(set)  # length -> set of tuples
    encoded = []
    for it in items:
        p = encode(it["prefix"])
        c, d = encode(it["correct"]), encode(it["distractor"])
        if not p or not c or not d:
            continue
        encoded.append((it, p, c, d))
        for cont in (c, d):
            seq = p + cont
            for k in range(len(p), len(seq)):
                for ctx_len in range(0, MAX_ORDER):
                    if ctx_len > k:
                        break
                    ctx = tuple(seq[k - ctx_len:k])
                    grams[ctx_len + 1].add(ctx + (seq[k],))
                    if ctx:
                        grams[ctx_len].add(ctx)
    return grams, encoded


def keys_of(a, L):
    """uint64 keys of all L-grams in token array a (L <= 4)."""
    k = a[: len(a) - L + 1].astype(np.uint64)
    for j in range(1, L):
        k = (k << np.uint64(BITS)) | a[j: len(a) - L + 1 + j].astype(np.uint64)
    return k


def tup_key(t):
    k = 0
    for tok in t:
        k = (k << BITS) | tok
    return k


def count_corpus(shard_files, grams, max_tokens=None):
    """One pass; returns ({gram tuple: count}, total_tokens)."""
    counts = defaultdict(int)
    want = {L: np.array(sorted(tup_key(g) for g in grams[L]), dtype=np.uint64)
            for L in range(1, 5) if grams.get(L)}
    # order 5: match on leading 4-gram key, then check 5th token
    five = defaultdict(dict)  # 4-gram key -> {next_tok: gram tuple}
    for g in grams.get(5, ()):
        five[tup_key(g[:4])][g[4]] = g
    five_keys = np.array(sorted(five), dtype=np.uint64)
    by_key = {L: {tup_key(g): g for g in grams.get(L, ())} for L in range(1, 5)}

    total = 0
    carry = np.empty(0, dtype=np.uint16)
    for sf in shard_files:
        a = np.fromfile(sf, dtype=np.uint16)
        if max_tokens is not None:
            if total >= max_tokens:
                break
            a = a[: max_tokens - total]
        total += len(a)
        ncarry = len(carry)
        a = np.concatenate([carry, a])
        for L, keys_wanted in want.items():
            # grams fully inside the carry were counted last shard
            lo = max(0, ncarry - L + 1)
            if len(a) - lo < L:
                continue
            keys = keys_of(a, L)[lo:]
            hit = np.isin(keys, keys_wanted)
            for key, n in zip(*np.unique(keys[hit], return_counts=True)):
                counts[by_key[L][int(key)]] += int(n)
        if len(five_keys) and len(a) >= 5:
            lo = max(0, ncarry - 4)
            k4 = keys_of(a, 4)[lo:-1]
            nxt = a[lo + 4:]
            hit = np.isin(k4, five_keys)
            for key, tok in zip(k4[hit], nxt[hit]):
                g = five[int(key)].get(int(tok))
                if g is not None:
                    counts[g] += 1
        carry = a[-(MAX_ORDER - 1):]
    return counts, total


def backoff_logprob(seq, start, counts, total, order):
    """Sum of log stupid-backoff scores for seq[start:] given seq[:start]."""
    lp = 0.0
    for k in range(start, len(seq)):
        s, pen = 0.0, 1.0
        for ctx_len in range(min(order - 1, k), -1, -1):
            ctx = tuple(seq[k - ctx_len:k])
            num = counts.get(ctx + (seq[k],), 0)
            den = counts.get(ctx, 0) if ctx else total
            if num > 0:
                s = pen * num / den
                break
            pen *= ALPHA
        if s == 0.0:
            s = pen * 0.5 / total  # unseen even as unigram
        lp += math.log(s)
    return lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True, nargs="+",
                    help="one or more shard dirs, concatenated in given "
                         "order; files sorted within each dir")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--battery", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 5])
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer_dir)
    encode = lambda s: tok.encode(s).ids  # noqa: E731
    items = [json.loads(line) for line in open(args.battery) if line.strip()]
    grams, encoded = battery_queries(items, encode)

    shard_files = []
    for d in args.shard_dir:
        shard_files += sorted(Path(d).glob("shard_*.bin"))
    assert shard_files, f"no shards under {args.shard_dir}"
    counts, total = count_corpus(shard_files, grams, args.max_tokens)

    out = {"cell": args.cell, "total_tokens": total, "alpha": ALPHA,
           "battery": args.battery, "shard_dirs": args.shard_dir,
           "max_tokens": args.max_tokens, "orders": {}}
    for order in args.orders:
        rows = []
        for it, p, c, d in encoded:
            lp_c = backoff_logprob(p + c, len(p), counts, total, order)
            lp_d = backoff_logprob(p + d, len(p), counts, total, order)
            rows.append({
                "probe": it["probe"], "split": it["split"],
                "template_id": it["template_id"],
                "argmax_acc": int(lp_c > lp_d),
                "logprob_diff": lp_c / len(c) - lp_d / len(d)})
        agg = defaultdict(list)
        for r in rows:
            agg[(r["probe"], r["split"])].append(r)
        out["orders"][str(order)] = [
            {"probe": pr, "split": sp, "n": len(rs),
             "argmax_acc": sum(r["argmax_acc"] for r in rs) / len(rs),
             "logprob_diff": sum(r["logprob_diff"] for r in rs) / len(rs)}
            for (pr, sp), rs in sorted(agg.items())]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} (total_tokens={total:,}, "
          f"queries={sum(len(v) for v in grams.values()):,})")


if __name__ == "__main__":
    main()
