"""Step-6 RESCUE corpus builder (registered intervention; frozen here in
code BEFORE any Step-6 run is launched).

Generates synthetic name->pronoun evidence documents for the displaced
web cell. Names and frames are frozen below and are battery-disjoint by
construction (verified separately by check_intervention_disjoint.py —
hard fail on overlap). Dose is registered in TS-density units:

  delta_TS = TS girl_names rule_support / TS tokens
           = 780313 / 466769099 = 1.67175e-3 events/token
  (source: runs/support_ts_window.json, frozen f*-v2 counter)

At dose d the builder emits round(d * delta_TS * W) girl rule events and
the same number of boy rule events (W = web corpus tokens from the shard
manifest). Each sentence carries exactly one name->pronoun event with the
pronoun within the counter's 16-token window; the builder verifies event
counts post-tokenization with a multi-token window matcher and writes
them to the manifest (exact dose accounting, tokenizer-blind).

  python scripts/gen_rescue_docs.py --dose 1.0 \
      --tokenizer-dir data/climbmix/bpe8192 \
      --corpus-manifest data/climbmix/bpe8192/shards/manifest.json \
      --out-dir data/climbmix/bpe8192/rescue_d100/synth
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from fogen.data import load_tokenizer, write_shards

DELTA_TS = 780313 / 466769099  # TS girl-class rule-event density

# Frozen rescue cue names — disjoint from every battery name
# (GIRLS_V2/BOYS_V2/v1 GIRLS/BOYS) and from battery noun cues.
RESCUE_GIRLS = ["Mary", "Nora", "Clara", "Eva", "Ivy", "Hazel",
                "Stella", "Nina", "Tessa", "Wendy", "Flora", "Carla"]
RESCUE_BOYS = ["David", "Noah", "Liam", "Owen", "Felix", "Oscar",
               "Henry", "Carl", "Eric", "Hugo", "Brian", "Dean"]

# Frozen frames — none reuses a battery frame ("smiled because",
# "was happy because", "laughed when", "cried because"), a battery
# reflexive verb after the name, or a mech_margins NEUTRAL prompt.
FRAMES = [
    "{n} opened the door, and then {p} walked outside.",
    "{n} finished the work early, so {p} went home.",
    "{n} looked at the sky while {p} waited for the bus.",
    "{n} wrote a letter, and later {p} mailed it.",
    "{n} grew up on a farm, where {p} learned to ride.",
    "{n} studied all evening because {p} wanted to pass.",
    "{n} picked up the basket, and {p} carried it inside.",
    "{n} read the news before {p} ate breakfast.",
    "{n} planted a garden, and {p} watered it every day.",
    "{n} fixed the fence after {p} noticed the gap.",
    "{n} called the office as soon as {p} arrived.",
    "{n} packed a bag, and then {p} left for the station.",
]
SENTS_PER_DOC = 4
WINDOW = 16


def gen_sentences(n_events, names, pron, rng):
    out = []
    for _ in range(n_events):
        out.append(rng.choice(FRAMES).format(n=rng.choice(names), p=pron))
    return out


def count_events(ids, name_id_seqs, fem_ids, masc_ids, window=WINDOW):
    """Multi-token window matcher: cue position = last token of a name
    occurrence; count first fem/masc pronoun within window (counter
    convention, capitalization-blind union of forms)."""
    a = np.asarray(ids, dtype=np.int64)
    fem_pos = np.where(np.isin(a, fem_ids))[0]
    masc_pos = np.where(np.isin(a, masc_ids))[0]
    she_n = he_n = 0
    cue_pos = []
    for seq in name_id_seqs:
        m = np.ones(len(a) - len(seq) + 1, dtype=bool)
        for j, t in enumerate(seq):
            m &= a[j: len(a) - len(seq) + 1 + j] == t
        cue_pos.extend((np.where(m)[0] + len(seq) - 1).tolist())
    for c in sorted(cue_pos):
        nf = fem_pos[np.searchsorted(fem_pos, c, "right")] \
            if np.searchsorted(fem_pos, c, "right") < len(fem_pos) else None
        nm = masc_pos[np.searchsorted(masc_pos, c, "right")] \
            if np.searchsorted(masc_pos, c, "right") < len(masc_pos) else None
        df = nf - c if nf is not None else 10**9
        dm = nm - c if nm is not None else 10**9
        if df <= window and df < dm:
            she_n += 1
        elif dm <= window and dm < df:
            he_n += 1
    return she_n, he_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", type=float, required=True,
                    help="in units of TS girl-class rule-event density")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--corpus-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer_dir)
    W = json.load(open(args.corpus_manifest))["total_tokens"]
    n_events = round(args.dose * DELTA_TS * W)
    rng = random.Random(args.seed)

    sents = (gen_sentences(n_events, RESCUE_GIRLS, "she", rng)
             + gen_sentences(n_events, RESCUE_BOYS, "he", rng))
    rng.shuffle(sents)
    docs = [" ".join(sents[i:i + SENTS_PER_DOC])
            for i in range(0, len(sents), SENTS_PER_DOC)]

    out_dir = Path(args.out_dir)
    manifest = write_shards(iter(docs), tok, out_dir)

    # exact post-tokenization event accounting on the written shards
    fem_ids = [tok.encode(v).ids[0] for v in (" she", " She")]
    masc_ids = [tok.encode(v).ids[0] for v in (" he", " He")]

    def seqs(names):
        return [tok.encode(" " + n).ids for n in names] + \
               [tok.encode(n).ids for n in names]

    she_tot = he_tot = 0
    for sh in sorted(out_dir.glob("shard_*.bin")):
        a = np.fromfile(sh, dtype=np.uint16)
        s, _ = count_events(a, seqs(RESCUE_GIRLS), fem_ids, masc_ids)
        _, h = count_events(a, seqs(RESCUE_BOYS), fem_ids, masc_ids)
        she_tot += s
        he_tot += h

    info = {"dose": args.dose, "delta_ts": DELTA_TS, "corpus_tokens": W,
            "target_events_per_gender": n_events,
            "measured_girl_she_events": she_tot,
            "measured_boy_he_events": he_tot,
            "n_docs": len(docs), "seed": args.seed,
            "injected_tokens": manifest["total_tokens"],
            "girls": RESCUE_GIRLS, "boys": RESCUE_BOYS}
    (out_dir / "rescue_manifest.json").write_text(json.dumps(info, indent=2))
    (out_dir / "docs_sample.txt").write_text("\n".join(docs[:50]))
    print(json.dumps(info, indent=2))
    assert she_tot >= 0.95 * n_events and he_tot >= 0.95 * n_events, \
        "window-compliance check failed: <95% of intended events measured"
    print("OK: event accounting within 5% of target")


if __name__ == "__main__":
    main()
