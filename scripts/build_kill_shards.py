"""Step-6 KILL corpus builder (registered intervention; frozen in code
BEFORE any Step-6 run is launched).

Strips name->pronoun rule evidence from a TinyStories shard set by
flipping, with probability --flip-rate, the first feminine pronoun
inside the 16-token window after each girl-cue token, computed
independently per case channel: the lowercase channel (first " she"
before first " he") is EXACTLY the event the frozen f*-v2 window
counter scores as rule_support, so the counter ratio after the kill is
(1-p)*she/(he+p*she) by construction; the capitalized channel (first
" She" before first " He") is flipped symmetrically at the same rate so
capitalized rule evidence is killed uniformly. Cue occurrences are
matched with and without the leading space (mirroring the counter's
single_token_ids), since sentence-initial names tokenize without it.
Token-count preserving (a flip replaces one token id with one token id),
so there is no data-reduction confound; everything outside flipped
pronoun positions is byte-identical to the source corpus.

Girl-cue inventory comes from the empirical scan
(scripts/scan_gendered_tokens.py output) via --scan-json plus the
registered inclusion rule: n_events >= --min-events and she_ratio >=
--min-ratio, minus the frozen EXCLUDE list (pronoun-coherence tokens,
kin/title nouns — the rvp4 noun-cue rule is deliberately left intact as
a dissociation readout — and common-word homographs). The resulting
list and all flip counts go into the output manifest.

  python scripts/build_kill_shards.py data/tinystories/bpe8192/shards \
      --tokenizer-dir data/tinystories/bpe8192 \
      --scan-json runs/gendered_tokens_ts.json \
      --flip-rate 0.645 --out-dir data/tinystories/bpe8192/kill_p645/shards
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fogen.data import load_tokenizer

WINDOW = 16

# Frozen exclusions from the empirical girl-cue list (registered with the
# Step-6 amendment): not person-name cues of the probed rule.
EXCLUDE = {"She", "Her",                                  # pronoun coherence
           "Grandma", "Granny", "Mrs", "Miss", "Aunt", "Princess",
           "Mommy", "Mummy", "Mum", "Mama", "Mom", "Lady", "Queen",
           "Car", "Mad", "May", "June", "Joy", "Honey", "Angel"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--scan-json", required=True)
    ap.add_argument("--flip-rate", type=float, required=True)
    ap.add_argument("--min-events", type=int, default=200)
    ap.add_argument("--min-ratio", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer_dir)
    channels = [(tok.encode(" she").ids[0], tok.encode(" he").ids[0]),
                (tok.encode(" She").ids[0], tok.encode(" He").ids[0])]

    scan = json.load(open(args.scan_json))
    cues = [r for r in scan["rows"]
            if r["first_she"] + r["first_he"] >= args.min_events
            and r["she_ratio"] >= args.min_ratio
            and r["token"] not in EXCLUDE]
    id_set = {int(r["id"]) for r in cues}
    n_nospace = 0
    for r in cues:
        ids = tok.encode(r["token"]).ids
        if len(ids) == 1 and ids[0] not in id_set:
            id_set.add(ids[0])
            n_nospace += 1
    cue_ids = np.array(sorted(id_set), dtype=np.uint16)
    print(f"{len(cues)} girl-cue tokens (+{n_nospace} no-space variants): "
          f"{[r['token'] for r in cues]}")

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes, totals = [], {"cues": 0, "eligible": 0, "flipped": 0}
    for sh in sorted(Path(args.shard_dir).glob("shard_*.bin")):
        a = np.fromfile(sh, dtype=np.uint16)
        cpos = np.where(np.isin(a, cue_ids))[0]
        n_elig, n_flip = 0, 0
        for fid, mid in channels:
            fem_pos = np.where(a == fid)[0]
            masc_pos = np.where(a == mid)[0]
            nf = fem_pos[np.minimum(np.searchsorted(fem_pos, cpos, "right"),
                                    len(fem_pos) - 1)] if len(fem_pos) \
                else np.full_like(cpos, 2**62)
            nm = masc_pos[np.minimum(np.searchsorted(masc_pos, cpos, "right"),
                                     len(masc_pos) - 1)] if len(masc_pos) \
                else np.full_like(cpos, 2**62)
            df = np.where(nf > cpos, nf - cpos, 2**62)
            dm = np.where(nm > cpos, nm - cpos, 2**62)
            hit = (df <= WINDOW) & (df < dm)
            targets = np.unique(nf[hit])  # dedupe shared windows
            flip = targets[rng.random(len(targets)) < args.flip_rate]
            a[flip] = mid
            n_elig += int(len(targets))
            n_flip += int(len(flip))
        p = out_dir / sh.name
        a.tofile(p)
        hashes.append({"file": p.name, "tokens": len(a),
                       "sha256": hashlib.sha256(a.tobytes()).hexdigest()})
        totals["cues"] += int(len(cpos))
        totals["eligible"] += n_elig
        totals["flipped"] += n_flip
        print(f"{sh.name}: cues={len(cpos)} eligible_windows={n_elig} "
              f"flipped={n_flip}", flush=True)

    manifest = {"total_tokens": sum(h["tokens"] for h in hashes),
                "shards": hashes, "dtype": "uint16",
                "kill": {"flip_rate": args.flip_rate, "seed": args.seed,
                         "window": WINDOW, "min_events": args.min_events,
                         "min_ratio": args.min_ratio,
                         "cue_tokens": [r["token"] for r in cues],
                         "n_cue_token_ids": int(len(cue_ids)),
                         "n_nospace_variants": n_nospace,
                         **totals}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["kill"], indent=2))


if __name__ == "__main__":
    main()
