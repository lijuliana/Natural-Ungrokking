"""POST-HOC (descriptive, not registered) bootstrap CIs on battery accuracies.

Registered as a post-hoc design in DECISIONS.md 2026-06-11 ("borrowed
mechanism analyses", item F) BEFORE any checkpoint was scored with it.

Per run and per family: 10,000-resample percentile CIs over battery
items for peak heldout-conflict accuracy, final heldout-conflict
accuracy, and the drop (peak - final). Items are resampled with
replacement WITHIN template strata (fixed stratum sizes), and the same
item resample is applied to both checkpoints, so the drop CI respects
the item pairing. The frozen probe log only stores aggregates, so the
peak and final checkpoints are rescored at item level with the frozen
Scorer; the peak step per family is the argmax (earliest on ties) of
that family's heldout-conflict argmax_acc in the frozen probe log,
snapped to the nearest available checkpoint step (probes log every ~5
steps, checkpoints every 25-100; earlier wins ties) — artifacts, not
hand-picked. Across seeds, cell summaries are min/max
envelopes (n=3), not CIs. Bootstrap rng seed 0.

  python scripts/bootstrap_cis.py runs/web_packed_v2_s42 [runs/...]
      [--battery data/probes/rvp3/battery.jsonl] [--n-boot 10000]
      [--out runs/bootstrap_cis.json]

Appends/updates per-run entries in the output JSON.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from mech_patch import get_ckpt, list_ckpt_steps, load_state_into  # noqa: E402

from fogen.data import load_tokenizer  # noqa: E402
from fogen.evals.scoring import Scorer, load_battery  # noqa: E402
from fogen.model import GPT, ModelConfig  # noqa: E402


def stratified_bootstrap(strata, n_boot, rng):
    """strata: list of (peak 0/1 array, final 0/1 array) per template,
    item-aligned within each stratum. Returns dict of point estimates
    and percentile CIs for peak acc, final acc, drop (paired)."""
    peaks = np.concatenate([p for p, _ in strata]).astype(float)
    finals = np.concatenate([f for _, f in strata]).astype(float)
    n = len(peaks)
    boot_p = np.empty(n_boot)
    boot_f = np.empty(n_boot)
    for i in range(n_boot):
        ps, fs = [], []
        for p, f in strata:
            ix = rng.integers(0, len(p), size=len(p))
            ps.append(p[ix])
            fs.append(f[ix])
        boot_p[i] = np.concatenate(ps).mean()
        boot_f[i] = np.concatenate(fs).mean()
    boot_d = boot_p - boot_f

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    return {"n_items": n,
            "peak_acc": float(peaks.mean()), "peak_ci95": ci(boot_p),
            "final_acc": float(finals.mean()), "final_ci95": ci(boot_f),
            "drop": float(peaks.mean() - finals.mean()),
            "drop_ci95": ci(boot_d)}


def item_scores(model, tok, items):
    """{(probe, item_id): (template_id, argmax_acc)} via frozen Scorer."""
    sc = Scorer(encode=lambda s: tok.encode(s).ids,
                logits_fn=lambda ids: model(ids))
    with torch.no_grad():
        rows = sc.score_items(items)
    return {(r["probe"], r["item_id"]): (r["template_id"], r["argmax_acc"])
            for r in rows}


def score_run(run, battery, n_boot):
    run = Path(run)
    assert not re.search(r"s10\d\d$", run.name), \
        f"{run} looks like a quarantined replication seed; refusing"
    cfg = yaml.safe_load(open(run / "config_used.yaml"))
    tok = load_tokenizer(cfg["data"]["tokenizer_dir"])

    log = [json.loads(l) for l in open(run / "probe_log_rvp31.jsonl")]
    conf = [r for r in log if r["probe"].endswith(".conflict")
            and r["split"] == "heldout"]
    fams = sorted({r["probe"].rsplit(".", 1)[0] for r in conf})
    ckpt_steps = list_ckpt_steps(str(run))
    final_step = ckpt_steps[-1]

    def snap(step):
        return min(ckpt_steps, key=lambda s: (abs(s - step), s))

    peak_step = {}
    for fam in fams:
        rows = [r for r in conf if r["probe"] == f"{fam}.conflict"]
        peak_step[fam] = snap(min(
            r["step"] for r in rows
            if r["argmax_acc"] == max(x["argmax_acc"] for x in rows)))

    items = [it for it in load_battery(battery)
             if it["probe"].endswith(".conflict")
             and it["split"] == "heldout"]
    assert items

    model = GPT(ModelConfig(**cfg["model"]))
    model.eval()
    scores = {}
    for step in sorted({final_step, *peak_step.values()}):
        path, cleanup = get_ckpt(str(run), step)
        load_state_into(model, path)
        if cleanup:
            cleanup()
        scores[step] = item_scores(model, tok, items)
        print(f"{run.name}: scored step {step} "
              f"({len(scores[step])} items)", flush=True)

    rng = np.random.default_rng(0)
    out = {"final_step": final_step, "families": {}}
    for fam in fams:
        fam_items = [it for it in items
                     if it["probe"] == f"{fam}.conflict"]
        strata = {}
        for it in fam_items:
            key = (it["probe"], it["item_id"])
            if key not in scores[peak_step[fam]] or \
                    key not in scores[final_step]:
                continue
            tid, p_acc = scores[peak_step[fam]][key]
            _, f_acc = scores[final_step][key]
            strata.setdefault(tid, ([], []))
            strata[tid][0].append(p_acc)
            strata[tid][1].append(f_acc)
        st = [(np.array(p), np.array(f)) for p, f in strata.values()]
        if not st:
            print(f"  {fam}: no battery items; skipped", flush=True)
            continue
        res = stratified_bootstrap(st, n_boot, rng)
        out["families"][fam] = {"peak_step": peak_step[fam], **res}
        print(f"  {fam:24s} peak={res['peak_acc']:.3f} "
              f"{res['peak_ci95']} final={res['final_acc']:.3f} "
              f"{res['final_ci95']} drop={res['drop']:.3f} "
              f"{res['drop_ci95']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--battery", default="data/probes/rvp3/battery.jsonl")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="runs/bootstrap_cis.json")
    args = ap.parse_args()

    out_path = Path(args.out)
    all_out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for run in args.run_dirs:
        all_out[Path(run).name] = score_run(run, args.battery, args.n_boot)
    # cell envelopes over seeds (min/max, n=3 — not CIs)
    cells = {}
    for name, r in all_out.items():
        if name == "cells":
            continue
        m = re.match(r"(.+)_s(\d+)$", name)
        if not m:
            continue
        cells.setdefault(m.group(1), []).append(r)
    env = {}
    for cell, runs_ in cells.items():
        fams = set.intersection(*[set(r["families"]) for r in runs_])
        env[cell] = {fam: {
            "n_seeds": len(runs_),
            "final_acc_envelope": [
                min(r["families"][fam]["final_acc"] for r in runs_),
                max(r["families"][fam]["final_acc"] for r in runs_)],
            "drop_envelope": [
                min(r["families"][fam]["drop"] for r in runs_),
                max(r["families"][fam]["drop"] for r in runs_)]}
            for fam in sorted(fams)}
    all_out["cells"] = env
    out_path.write_text(json.dumps(all_out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
