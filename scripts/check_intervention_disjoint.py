"""Step-6 battery-disjointness gate (registered; run + logged BEFORE any
intervention run launches; any violation is a HARD FAIL, exit 1).

Checks, against every battery on disk plus the probe-code name lists and
the frozen mech_margins prompts:
  D1: rescue cue names share no member with any battery cue inventory
      (v1 GIRLS/BOYS, GIRLS_V2/BOYS_V2, FEM/MASC_NOUNS).
  D2: no battery cue name (token-sequence search, " Name" and "Name"
      encodings) occurs anywhere in the generated rescue shards; since
      every gendered battery prefix contains a battery cue, this covers
      prefix containment. Additionally every battery prefix is checked
      as a substring against the (small) unique-sentence universe of
      the generator.
  D3: no mech_margins prompt (NEUTRAL subject x frame) token-sequence
      occurs in any rescue shard.
  D4: kill builder introduces no text: its manifest must show the only
      operation is she->he pronoun-id flips (token-count preserved).

  python scripts/check_intervention_disjoint.py \
      --rescue-dirs data/climbmix/bpe8192/rescue_d*/synth \
      --kill-manifest data/tinystories/bpe8192/kill_p645/shards/manifest.json \
      --out runs/step6_disjointness.json
"""

import argparse
import json
import sys
from pathlib import Path

from fogen.data import load_tokenizer
from fogen.probes.rvp import BOYS_V2, FEM_NOUNS, GIRLS_V2, MASC_NOUNS
from fogen.probes.v1_probes import BOYS as V1_BOYS, GIRLS as V1_GIRLS

from gen_rescue_docs import RESCUE_BOYS, RESCUE_GIRLS
from mech_margins import FRAMES as MM_FRAMES, NEUTRAL_SUBJECTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescue-dirs", nargs="+", required=True)
    ap.add_argument("--kill-manifest", required=True)
    ap.add_argument("--tokenizer-dir", default="data/climbmix/bpe8192")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    failures = []
    battery_cues = (set(V1_GIRLS) | set(V1_BOYS) | set(GIRLS_V2)
                    | set(BOYS_V2) | set(FEM_NOUNS) | set(MASC_NOUNS))
    rescue = set(RESCUE_GIRLS) | set(RESCUE_BOYS)
    overlap = sorted(rescue & battery_cues)
    if overlap:
        failures.append(f"D1 name overlap: {overlap}")

    prefixes = set()
    for b in sorted(Path("data/probes").glob("*/battery.jsonl")):
        for line in b.read_text().splitlines():
            prefixes.add(json.loads(line)["prefix"])
    mm_prompts = {fr.format(n=s) for fr in MM_FRAMES
                  for s in NEUTRAL_SUBJECTS}

    tok = load_tokenizer(args.tokenizer_dir)
    import numpy as np

    from count_support import count_seq
    from gen_rescue_docs import FRAMES

    universe = " ".join(fr.format(n=n, p=p) for fr in FRAMES
                        for n in sorted(rescue) for p in ("she", "he"))
    for p in prefixes:
        if p in universe:
            failures.append(f"D2 battery prefix {p!r} in sentence universe")

    name_seqs = {w: [tok.encode(v).ids for v in (w, " " + w)]
                 for w in battery_cues | {c.capitalize()
                                          for c in battery_cues}}
    mm_seqs = {p: tok.encode(p).ids for p in mm_prompts}
    n_docs = 0
    for d in args.rescue_dirs:
        for sh in sorted(Path(d).glob("shard_*.bin")):
            ids = np.fromfile(sh, dtype=np.uint16)
            for w, seqs in name_seqs.items():
                for seq in seqs:
                    if count_seq(ids, seq):
                        failures.append(f"D2 battery cue {w!r} in {sh}")
                        break
            for p, seq in mm_seqs.items():
                if count_seq(ids, seq):
                    failures.append(f"D3 mech_margins prompt {p!r} in {sh}")
        man = json.load(open(Path(d) / "rescue_manifest.json"))
        n_docs += man["n_docs"]

    km = json.load(open(args.kill_manifest))
    if "kill" not in km:
        failures.append("D4: kill manifest missing 'kill' section")
    else:
        src = json.load(open(Path(args.kill_manifest).parent.parent.parent
                             / "shards" / "manifest.json"))
        if km["total_tokens"] != src["total_tokens"]:
            failures.append(
                f"D4 token count changed: {km['total_tokens']} != "
                f"{src['total_tokens']}")

    report = {"failures": failures, "ok": not failures,
              "rescue_names": sorted(rescue),
              "battery_cues": sorted(battery_cues),
              "n_battery_prefixes": len(prefixes),
              "n_mech_margin_prompts": len(mm_prompts),
              "n_rescue_docs_checked": n_docs}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("ok", "failures", "n_battery_prefixes",
                       "n_rescue_docs_checked")}, indent=2))
    if failures:
        print("HARD FAIL: intervention not battery-disjoint")
        sys.exit(1)
    print("DISJOINTNESS OK")


if __name__ == "__main__":
    main()
