"""Frozen evaluator for registered M3 (rule-vs-memorization attribution).

Registered 2026-06-10 (RESEARCH_LOG "STEP 5 REBUILD part 2") BEFORE any
template-log values were read. template_acc(step) = n-weighted mean over
template_id rows for pronoun_gender_ref.conflict; G = template - heldout;
classifier smoothing/final-window conventions. Tag rvp31.

M3 PASS: web packed 3/3 seeds template_final <= 0.6 AND |G_final| < 0.2;
TS control: template and heldout final >= 0.8, 3/3 seeds.
FALSIFIER: template_final >= 0.8 while heldout_final <= 0.6 in >= 2/3 web
packed seeds (memorization masquerade).

  python scripts/eval_m3.py --seeds 42 43 44 \
      --web runs/web_packed_v2_s{seed} --ts runs/v1_repro_s{seed} \
      [--out runs/eval_m3_rvp31.json]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from gate_a_classify import MIN_STEP, load_traj, smooth

PROBE = "pronoun_gender_ref.conflict"


def template_traj(run_dir, tag="rvp31"):
    per_step = defaultdict(list)  # step -> [(template_id, acc, n)]
    p = Path(run_dir) / f"probe_log_{tag}_templates.jsonl"
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r["probe"] != PROBE or r["step"] < MIN_STEP:
            continue
        per_step[r["step"]].append((r["template_id"], r["argmax_acc"], r["n"]))
    steps = sorted(per_step)
    accs = [sum(a * n for _, a, n in per_step[s])
            / sum(n for _, _, n in per_step[s]) for s in steps]
    return steps, accs, per_step


def final_window(vals):
    k = max(3, len(vals) // 10)
    return sum(vals[-k:]) / k, k


def bootstrap_ci(per_step, steps, k, n_boot=1000, seed=0):
    rng = random.Random(seed)
    fin_steps = steps[-k:]
    tids = sorted({t for s in fin_steps for t, _, _ in per_step[s]})
    ests = []
    for _ in range(n_boot):
        sample = [tids[rng.randrange(len(tids))] for _ in tids]
        vals = []
        for s in fin_steps:
            row = {t: (a, n) for t, a, n in per_step[s]}
            num = sum(row[t][0] * row[t][1] for t in sample if t in row)
            den = sum(row[t][1] for t in sample if t in row)
            if den:
                vals.append(num / den)
        if vals:
            ests.append(sum(vals) / len(vals))
    ests.sort()
    return ests[int(0.025 * len(ests))], ests[int(0.975 * len(ests))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--web", required=True)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--tag", default="rvp31")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for grp, tmpl in (("web", args.web), ("ts", args.ts)):
        for s in args.seeds:
            d = tmpl.format(seed=s)
            steps, t_acc, per_step = template_traj(d, args.tag)
            t_s = smooth(t_acc)
            t_fin, k = final_window(t_s)
            lo, hi = bootstrap_ci(per_step, steps, k)
            h_tr = load_traj(d, args.tag).get(tuple(PROBE.rsplit(".", 1)), [])
            h_s = smooth([a for _, a in h_tr])
            h_fin, _ = final_window(h_s)
            rows.append({"grp": grp, "seed": s, "template_final": t_fin,
                         "template_ci95": [lo, hi], "heldout_final": h_fin,
                         "G_final": t_fin - h_fin, "n_evals": len(steps)})
            print(f"{grp} s{s}: template_final={t_fin:.3f} "
                  f"CI[{lo:.3f},{hi:.3f}] heldout_final={h_fin:.3f} "
                  f"G={t_fin - h_fin:+.3f}")

    web = [r for r in rows if r["grp"] == "web"]
    ts = [r for r in rows if r["grp"] == "ts"]
    m3 = all(r["template_final"] <= 0.6 and abs(r["G_final"]) < 0.2
             for r in web)
    ts_ok = all(r["template_final"] >= 0.8 and r["heldout_final"] >= 0.8
                for r in ts)
    fals = [r["seed"] for r in web
            if r["template_final"] >= 0.8 and r["heldout_final"] <= 0.6]
    print(f"M3 (web both-collapse): {'PASS' if m3 else 'FAIL'}")
    print(f"M3 TS control: {'PASS' if ts_ok else 'FAIL'}")
    print(f"M3 FALSIFIER (memorization masquerade): "
          f"{'TRIGGERED ' + str(fals) if len(fals) >= 2 else 'no'}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"rows": rows, "M3": m3, "M3_ts_control": ts_ok,
             "falsifier_seeds": fals,
             "falsifier_triggered": len(fals) >= 2}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
