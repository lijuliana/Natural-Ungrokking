"""Frozen evaluator for the registered public-suite (Pythia/OLMo)
discriminative predictions. Written 2026-06-10 BEFORE any public-suite
probe log beyond the disclosed pythia-70m smoke (step0 + step143000) was
read. Uses the registered classifier verbatim (gate_a_classify constants).

Already known (disclosed smoke, NOT predictions): pythia-70m@final
pronoun_gender_ref.conflict = 0.67, reflexive_gender.conflict = 0.50.

PUB1 (re-stated from the 2026-06-10 public-suite protocol registration):
     all non-exploratory families RECOVERED+valid at final in every model
     >= 160M (pythia-160m/410m/1b/1.4b, olmo-1b). Failure = harness or
     battery bug OR counter-evidence; must be reported either way.
PUB2 (transience in the wild): on pythia-70m, pronoun_gender_ref smoothed
     conflict reaches >= 0.8 at some step >= MIN_STEP and ends <= 0.7 —
     i.e. emerge-then-displace, not never-learned. Secondary: same shape
     for reflexive_gender (peak >= 0.7, final <= 0.6).
     Falsifier: peak never reaches threshold (never-learned would NOT be
     natural ungrokking and weakens the in-the-wild claim).
PUB3 (capacity ordering): Spearman rank correlation between Pythia model
     size (70m<160m<410m<1b<1.4b) and pronoun_gender_ref conflict_final
     is positive. Falsifier: rho <= 0.

  python scripts/eval_public_suite.py --tag rvp31 \
      --pythia runs/pythia-70m runs/pythia-160m runs/pythia-410m \
               runs/pythia-1b runs/pythia-1.4b \
      --olmo runs/olmo-1b [--out ...]
"""

import argparse
import json
from pathlib import Path

from gate_a_classify import EXPLORATORY, MIN_STEP, classify, load_traj, smooth

PRON = "pronoun_gender_ref"
REFL = "reflexive_gender"


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx)
           * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def peak_final(traj, fam):
    tr = traj.get((fam, "conflict"), [])
    if not tr:
        return None, None, None
    accs = smooth([a for _, a in tr])
    cand = [(a, st) for (st, _), a in zip(tr, accs) if st >= MIN_STEP]
    pk, pk_step = max(cand) if cand else (None, None)
    return pk, pk_step, accs[-1]


def verdict(name, ok, detail):
    print(f"{name}: {'PASS' if ok else 'FAIL'}  {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rvp31")
    ap.add_argument("--pythia", nargs=5, required=True,
                    help="dirs in size order 70m 160m 410m 1b 1.4b")
    ap.add_argument("--olmo", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dirs = {f"pythia-{n}": d for n, d in
            zip(["70m", "160m", "410m", "1b", "1.4b"], args.pythia)}
    if args.olmo:
        dirs["olmo-1b"] = args.olmo
    res, trajs = {}, {}
    for m, d in dirs.items():
        trajs[m] = load_traj(d, args.tag)
        res[m] = classify(trajs[m])

    report = {}

    big = [m for m in dirs if m != "pythia-70m"]
    pub1_fails = [(m, f) for m in big for f, r in res[m].items()
                  if f not in EXPLORATORY
                  and not (r["class"] == "RECOVERED" and r["valid"])]
    report["PUB1"] = verdict("PUB1", not pub1_fails, f"fails={pub1_fails}")

    pk, pk_step, fin = peak_final(trajs["pythia-70m"], PRON)
    pub2_main = pk is not None and pk >= 0.8 and fin <= 0.7
    pk2, pk2_step, fin2 = peak_final(trajs["pythia-70m"], REFL)
    pub2_sec = pk2 is not None and pk2 >= 0.7 and fin2 <= 0.6
    report["PUB2"] = verdict(
        "PUB2", pub2_main,
        f"pronoun peak={pk} @ {pk_step} final={fin}; secondary "
        f"reflexive {'PASS' if pub2_sec else 'FAIL'} "
        f"peak={pk2} @ {pk2_step} final={fin2}")
    report["PUB2_secondary"] = pub2_sec

    finals = [res[m][PRON]["conflict_final"]
              for m in ["pythia-70m", "pythia-160m", "pythia-410m",
                        "pythia-1b", "pythia-1.4b"]]
    rho = spearman(list(range(5)), finals)
    report["PUB3"] = verdict("PUB3", rho > 0,
                             f"finals={finals} spearman_rho={rho:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"tag": args.tag, "report": report}, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
