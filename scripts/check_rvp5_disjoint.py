"""rvp5 / Step-6 disjointness gate (amendment 2026-06-10, condition ii).

The rvp5 amendment permits scoring Step-6 intervention cells with the
rvp5 battery ONLY after a token-sequence search confirms that no rvp5
prefix occurs in any injected (rescue synth) shard. The kill builder
introduces no text (gate D4, prior report), so injected shards are the
entire search space.

Exact check: every rvp5 prefix is encoded with the rescue corpus
tokenizer and searched as a contiguous token sequence in every synth
shard. Any hit is a HARD FAIL (exit 1) and rvp5 must not be scored on
intervention cells.

  python scripts/check_rvp5_disjoint.py \
      --rescue-dirs data/climbmix/bpe8192/rescue_d*/synth \
      --battery data/probes/rvp5/battery.jsonl \
      --tokenizer-dir data/climbmix/bpe8192 \
      --out runs/rvp5_disjointness.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from fogen.data import load_tokenizer

from count_support import count_seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescue-dirs", nargs="+", required=True)
    ap.add_argument("--battery", default="data/probes/rvp5/battery.jsonl")
    ap.add_argument("--tokenizer-dir", default="data/climbmix/bpe8192")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    blob = Path(args.battery).read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    prefixes = sorted({json.loads(line)["prefix"]
                       for line in blob.decode().splitlines() if line})

    tok = load_tokenizer(args.tokenizer_dir)
    seqs = {p: tok.encode(p).ids for p in prefixes}

    failures = []
    shards = []
    for d in args.rescue_dirs:
        found = sorted(Path(d).glob("shard_*.bin"))
        assert found, f"no synth shards under {d}"
        shards += found
    for sh in shards:
        ids = np.fromfile(sh, dtype=np.uint16)
        for p, seq in seqs.items():
            if count_seq(ids, seq):
                failures.append(f"rvp5 prefix {p!r} in {sh}")

    report = {"failures": failures, "ok": not failures,
              "battery": args.battery, "battery_sha256": sha,
              "n_prefixes": len(prefixes), "n_shards": len(shards),
              "shard_dirs": args.rescue_dirs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("ok", "n_prefixes", "n_shards",
                       "battery_sha256")}, indent=2))
    if failures:
        for f in failures:
            print(f)
        print("HARD FAIL: rvp5 must not be scored on intervention cells")
        sys.exit(1)
    print("RVP5 DISJOINTNESS OK")


if __name__ == "__main__":
    main()
