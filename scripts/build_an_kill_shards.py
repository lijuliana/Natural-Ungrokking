"""a/an KILL corpus builder (generality test; frozen in code BEFORE any
a/an-kill run is launched — registered in DECISIONS.md 2026-06-11).

Strips det_an_choice rule evidence from a TinyStories shard set by
flipping, with probability --flip-rate, determiner "an" -> "a" wherever
the NEXT token is a vowel-initial word token. Four case/space channels
are flipped at the same rate (" an"->" a", " An"->" A", and the rare
no-space document-initial "an"->"a", "An"->"A"); all eight strings are
single tokens in the bpe8192 vocab, so every flip replaces one token id
with one token id and the corpus is token-count preserving — no
data-reduction confound; everything outside flipped positions is
byte-identical to the source corpus.

Eligibility requires the next token to decode to " " + vowel letter.
This guard does double duty: (1) a subword continuation never carries a
leading space, so "an" pieces inside words can never be flipped, and
(2) only vowel-LETTER contexts are killed, which is exactly the
evidence the frozen det_an_choice battery probes ("an hour"-type
vowel-sound exceptions are left intact and are a negligible fraction).

Unlike the pronoun kill, each flip also CREATES counter-evidence
("a apple"), so the post-kill rule:counter evidence ratio is
(1-p)*N_an / (N_a_vowel_base + p*N_an), ~ (1-p)/p since the base count
of "a"+vowel is tiny in clean text. Both counts go into the manifest.

  python scripts/build_an_kill_shards.py data/tinystories/bpe8192/shards \
      --tokenizer-dir data/tinystories/bpe8192 \
      --flip-rate 0.75 --out-dir data/tinystories/bpe8192/ankill_p750/shards
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

VOWELS = "aeiouAEIOU"


def vowel_initial_ids(tok):
    ids = []
    for i in range(tok.get_vocab_size()):
        s = tok.decode([i])
        if len(s) >= 2 and s[0] == " " and s[1] in VOWELS:
            ids.append(i)
    return np.array(sorted(ids), dtype=np.uint16)


def an_channels(tok):
    """[(an_id, a_id), ...] for the four case/space channels."""
    pairs = []
    for an_s, a_s in ((" an", " a"), (" An", " A"), ("an", "a"), ("An", "A")):
        an_ids, a_ids = tok.encode(an_s).ids, tok.encode(a_s).ids
        assert len(an_ids) == 1 and len(a_ids) == 1, (an_s, an_ids, a_ids)
        pairs.append((an_ids[0], a_ids[0]))
    return pairs


def flip_shard(a, pairs, vowel_ids, rate, rng):
    """In-place flip; returns (eligible, flipped, a_vowel_base) counts."""
    next_is_vowel = np.zeros(len(a), dtype=bool)
    next_is_vowel[:-1] = np.isin(a[1:], vowel_ids)
    n_elig, n_flip, a_base = 0, 0, 0
    a_ids = {ai for _, ai in pairs}
    for an_id, a_id in pairs:
        targets = np.where((a == an_id) & next_is_vowel)[0]
        flip = targets[rng.random(len(targets)) < rate]
        a[flip] = a_id
        n_elig += int(len(targets))
        n_flip += int(len(flip))
    for a_id in a_ids:
        a_base += int(np.count_nonzero((a == a_id) & next_is_vowel))
    a_base -= n_flip  # count pre-existing "a"+vowel only
    return n_elig, n_flip, a_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--flip-rate", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from fogen.data import load_tokenizer
    tok = load_tokenizer(args.tokenizer_dir)
    pairs = an_channels(tok)
    vowel_ids = vowel_initial_ids(tok)
    print(f"channels={pairs} vowel_initial_vocab={len(vowel_ids)}")

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = []
    totals = {"eligible": 0, "flipped": 0, "a_vowel_base": 0}
    for sh in sorted(Path(args.shard_dir).glob("shard_*.bin")):
        a = np.fromfile(sh, dtype=np.uint16)
        n_elig, n_flip, a_base = flip_shard(a, pairs, vowel_ids,
                                            args.flip_rate, rng)
        p = out_dir / sh.name
        a.tofile(p)
        hashes.append({"file": p.name, "tokens": len(a),
                       "sha256": hashlib.sha256(a.tobytes()).hexdigest()})
        totals["eligible"] += n_elig
        totals["flipped"] += n_flip
        totals["a_vowel_base"] += a_base
        print(f"{sh.name}: eligible={n_elig} flipped={n_flip} "
              f"a_vowel_base={a_base}", flush=True)

    sup = totals["eligible"] - totals["flipped"]
    ctr = totals["a_vowel_base"] + totals["flipped"]
    manifest = {"total_tokens": sum(h["tokens"] for h in hashes),
                "shards": hashes, "dtype": "uint16",
                "an_kill": {"flip_rate": args.flip_rate, "seed": args.seed,
                            "channels": [list(p) for p in pairs],
                            "n_vowel_initial_ids": int(len(vowel_ids)),
                            **totals,
                            "post_kill_support": sup,
                            "post_kill_counter": ctr,
                            "post_kill_ratio": sup / ctr if ctr else None}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["an_kill"], indent=2))


if __name__ == "__main__":
    main()
