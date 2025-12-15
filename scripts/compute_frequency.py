"""Count support/compete pattern frequencies on a corpus and emit the
capability table draft (axis B of the phase diagram).

  python scripts/compute_frequency.py --max-docs 500000 \
      --out-dir analysis/frequency/tinystories

Streams the HF dataset (no full download). Writes frequency.json (raw counts)
and capability_table.csv (one row per probe, joined with axis-A category).
"""

import argparse
import csv
import json
from pathlib import Path

from fogen.theory.frequency import PATTERNS, PATTERN_TO_PROBE, count_patterns

AXIS_A = {  # linguistic type, per prereg/draft_inputs.md
    "determiner_a_an": "lexical", "comparative_than": "lexical",
    "proper_noun_completion": "lexical", "pronoun_gender": "lexical",
    "numeric_sequence": "lexical", "common_idiom": "lexical",
    "end_of_sentence": "structural", "modal_continuation": "structural",
    "adjective_order": "structural", "past_tense_consistency": "structural",
    "subj_verb_agreement": "structural", "relative_clause_agreement": "structural",
    "close_quote": "structural", "reflexive_pronoun": "structural",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--out-dir", default="analysis/frequency/tinystories")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    out = count_patterns((r["text"] for r in ds), max_docs=args.max_docs)
    out["_meta"] = {"dataset": args.dataset, "split": args.split,
                    "max_docs": args.max_docs}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frequency.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {out_dir / 'frequency.json'} "
          f"({out['total_words']:,} words counted)")

    with open(out_dir / "capability_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["capability", "linguistic_type", "support_per_million",
                    "compete_per_million", "support_ratio", "corpus"])
        for pat_key, probe in sorted(PATTERN_TO_PROBE.items(), key=lambda kv: kv[1]):
            r = out[pat_key]
            w.writerow([probe, AXIS_A[probe],
                        f"{r['support_per_million']:.2f}",
                        f"{r['compete_per_million']:.2f}",
                        f"{r['support_ratio']:.4f}", args.dataset])
    print(f"wrote {out_dir / 'capability_table.csv'}")


if __name__ == "__main__":
    main()
