"""Frozen evaluator for the registered Step-3 (P1-P4) and Step-5 (M1-M2)
predictions. Written 2026-06-10 BEFORE the seed-grid phase table or any
S1 trajectory existed, so the pass/fail logic cannot be tuned to outcomes.
Uses the registered classifier verbatim (gate_a_classify constants).

P1. pronoun_gender_ref RECOVERED+valid on ALL TinyStories cells, 3/3 seeds;
    web packed conflict final <= 0.6 in >= 2/3 seeds (DISPLACED or
    low-UNSTABLE), control-valid.
P2. det_an_choice, a_an_adjective, irregular_past, negation_bare_verb:
    conflict final >= 0.8 on web packed, 3/3 seeds.
P3. reflexive_gender web packed: conflict final < 0.8 in >= 2/3 seeds.
P4. web budget ordering (pronoun_gender_ref): conflict final
    packed <= dn15 in >= 2/3 seeds.
M1. web packed: final S1 < 0.5 x peak S1, 3/3 seeds; TinyStories packed:
    final S1 >= 0.5 x peak, 3/3 seeds.
M2. web cells: S1 peak step <= behavioral collapse onset (last step with
    smoothed pronoun_gender_ref conflict heldout acc >= 0.5).

  python scripts/eval_predictions.py --tag rvp1 --seeds 42 43 44 \
      --ts packed=runs/v1_repro_s{seed} dn5=runs/databudget_dn5_s{seed} \
           dn15=runs/databudget_dn15_s{seed} \
      --web packed=runs/web_packed_v2_s{seed} dn5=runs/web_dn5_v2_s{seed} \
            dn15=runs/web_dn15_v2_s{seed} \
      [--scalar-glob "{run_dir}/mech_gender_scalar.jsonl"]
"""

import argparse
import json
from pathlib import Path

from gate_a_classify import classify, load_traj, smooth

PRON = "pronoun_gender_ref"
P2_FAMILIES = ["det_an_choice", "a_an_adjective", "irregular_past",
               "negation_bare_verb"]


def verdict(name, ok, detail):
    print(f"{name}: {'PASS' if ok else 'FAIL'}  {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rvp1")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--ts", nargs=3, metavar="name=tmpl", required=True)
    ap.add_argument("--web", nargs=3, metavar="name=tmpl", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ts = dict(c.split("=", 1) for c in args.ts)
    web = dict(c.split("=", 1) for c in args.web)
    res, trajs = {}, {}
    for grp, cells in (("ts", ts), ("web", web)):
        for name, tmpl in cells.items():
            for s in args.seeds:
                d = tmpl.format(seed=s)
                t = load_traj(d, args.tag)
                trajs[(grp, name, s)] = t
                res[(grp, name, s)] = classify(t)

    report = {}

    # P1a: pronoun RECOVERED+valid on every TinyStories cell, all seeds
    p1a_fails = [(n, s) for n in ts for s in args.seeds
                 if not (res[("ts", n, s)][PRON]["class"] == "RECOVERED"
                         and res[("ts", n, s)][PRON]["valid"])]
    # P1b: web packed conflict final <= 0.6, control-valid, >= 2/3 seeds
    p1b_hits = [s for s in args.seeds
                if res[("web", "packed", s)][PRON]["conflict_final"] <= 0.6
                and res[("web", "packed", s)][PRON]["valid"]]
    report["P1"] = verdict(
        "P1", not p1a_fails and len(p1b_hits) >= 2,
        f"ts_fails={p1a_fails} web_hits(seeds)={p1b_hits}")

    # P2: high-support families hold on web packed, 3/3 seeds
    p2_fails = [(f, s) for f in P2_FAMILIES for s in args.seeds
                if res[("web", "packed", s)][f]["conflict_final"] < 0.8]
    report["P2"] = verdict("P2", not p2_fails, f"fails={p2_fails}")

    # P3: reflexive_gender web packed conflict final < 0.8 in >= 2/3 seeds
    p3_hits = [s for s in args.seeds
               if res[("web", "packed", s)]["reflexive_gender"]
               ["conflict_final"] < 0.8]
    report["P3"] = verdict("P3", len(p3_hits) >= 2, f"hits(seeds)={p3_hits}")

    # P4: web pronoun conflict final packed <= dn15 in >= 2/3 seeds
    p4_hits = [s for s in args.seeds
               if res[("web", "packed", s)][PRON]["conflict_final"]
               <= res[("web", "dn15", s)][PRON]["conflict_final"]]
    report["P4"] = verdict("P4", len(p4_hits) >= 2, f"hits(seeds)={p4_hits}")

    # Registered kill check: pronoun RECOVERED on web packed in >= 2 seeds
    killed = [s for s in args.seeds
              if res[("web", "packed", s)][PRON]["class"] == "RECOVERED"
              and res[("web", "packed", s)][PRON]["valid"]]
    report["KILL"] = len(killed) >= 2
    print(f"KILL CONDITION (pronoun RECOVERED+valid on web packed >=2 "
          f"seeds): {'TRIGGERED' if report['KILL'] else 'no'} {killed}")

    # M1/M2 — only if scalar files exist
    m1_rows, m2_rows = [], []
    for grp, name in (("web", "packed"), ("ts", "packed")):
        tmpl = (web if grp == "web" else ts)[name]
        for s in args.seeds:
            p = Path(tmpl.format(seed=s)) / "mech_gender_scalar.jsonl"
            if not p.exists():
                continue
            rows = sorted((json.loads(l) for l in p.read_text().splitlines()),
                          key=lambda r: r["step"])
            # AMENDMENT 2026-06-10 (pre-S1-computation): registered scalar
            # is S1_cue — name-based S1 undefined on web tokenizer.
            s1 = [r["S1_cue_gender_coupling"] for r in rows]
            steps = [r["step"] for r in rows]
            peak_i = max(range(len(s1)), key=lambda i: s1[i])
            entry = {"grp": grp, "seed": s, "peak": s1[peak_i],
                     "peak_step": steps[peak_i], "final": s1[-1]}
            m1_rows.append(entry)
            if grp == "web":
                tr = trajs[("web", name, s)].get((PRON, "conflict"), [])
                accs = smooth([a for _, a in tr])
                onset = max((st for (st, _), a in zip(tr, accs) if a >= 0.5),
                            default=None)
                entry["collapse_onset"] = onset
                m2_rows.append(entry)
    if m1_rows:
        web_ok = [e for e in m1_rows if e["grp"] == "web"
                  and e["final"] < 0.5 * e["peak"]]
        ts_ok = [e for e in m1_rows if e["grp"] == "ts"
                 and e["final"] >= 0.5 * e["peak"]]
        n_web = sum(1 for e in m1_rows if e["grp"] == "web")
        n_ts = sum(1 for e in m1_rows if e["grp"] == "ts")
        report["M1"] = verdict(
            "M1", len(web_ok) == n_web == len(args.seeds)
            and len(ts_ok) == n_ts == len(args.seeds),
            f"web_ok={len(web_ok)}/{n_web} ts_ok={len(ts_ok)}/{n_ts}")
        m2_ok = [e for e in m2_rows if e["collapse_onset"] is not None
                 and e["peak_step"] <= e["collapse_onset"]]
        report["M2"] = verdict("M2", len(m2_ok) == len(m2_rows) > 0,
                               f"ok={len(m2_ok)}/{len(m2_rows)}")
    else:
        print("M1/M2: no mech_gender_scalar.jsonl found; skipped")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"tag": args.tag, "seeds": args.seeds, "report": report,
             "m1_rows": m1_rows}, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
