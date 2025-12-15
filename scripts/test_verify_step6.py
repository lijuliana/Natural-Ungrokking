"""Agreement tests: independent verifier vs frozen Step-6 evaluator.

Compares (a) low-level primitives on randomized inputs and (b) full
per-run rows on every locally available run directory. Disagreement on
any field = a bug in one implementation; adjudicate against the
registered text before any Step-6 result is used.
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

import eval_step6
import verify_step6
from eval_m4 import final_window as frozen_final_window
from eval_public_suite import spearman as frozen_spearman
from gate_a_classify import final_mean as frozen_final_mean
from gate_a_classify import smooth as frozen_smooth

RUNS = sorted(p for p in Path("runs").glob("*_s4[234]")
              if (p / "probe_log_rvp31.jsonl").exists()
              and (p / "mech_margins.jsonl").exists())


def close(a, b, tol=1e-9):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= tol
    return a == b


@pytest.mark.parametrize("trial", range(50))
def test_smoothing_agrees(trial):
    rng = random.Random(trial)
    xs = [rng.random() for _ in range(rng.randint(1, 40))]
    assert all(close(a, b) for a, b in
               zip(frozen_smooth(xs), verify_step6.moving_avg(xs)))


@pytest.mark.parametrize("trial", range(50))
def test_final_means_agree(trial):
    rng = random.Random(1000 + trial)
    xs = [rng.random() for _ in range(rng.randint(3, 60))]
    assert close(frozen_final_mean(xs), verify_step6.tail_mean_round(xs)[0])
    assert close(frozen_final_window(xs), verify_step6.tail_mean_floor(xs))


@pytest.mark.parametrize("trial", range(50))
def test_spearman_agrees(trial):
    rng = random.Random(2000 + trial)
    n = rng.randint(3, 12)
    xs = [rng.choice([0.0, 0.1, 0.5, 1.0, 3.0]) for _ in range(n)]
    ys = [rng.gauss(0, 1) for _ in range(n)]
    assert close(frozen_spearman(xs, ys),
                 verify_step6.spearman_rho(xs, ys), tol=1e-12)


@pytest.mark.parametrize("run_dir", RUNS, ids=[p.name for p in RUNS])
def test_run_rows_agree(run_dir):
    frozen = eval_step6.run_row(run_dir, "rvp31")
    ours = verify_step6.run_row(run_dir, "rvp31")
    assert frozen["pron_class"] == ours["pron_class"]
    assert frozen["pron_valid"] == ours["pron_valid"]
    for key in ("conflict_final", "cm_peak", "cm_final", "pm_final"):
        assert close(frozen[key], ours[key]), (key, frozen[key], ours[key])
    assert frozen["cm_valid"] == ours["cm_valid"]
    assert frozen["spec"] == ours["spec"]
    f_ci = frozen.get("conflict_final_ci95")
    o_ci = ours.get("conflict_final_ci95")
    assert (f_ci is None) == (o_ci is None)
    if f_ci is not None:
        assert all(close(a, b) for a, b in zip(f_ci, o_ci))


@pytest.mark.skipif(not RUNS, reason="no local run artifacts (runs/)")
def test_found_runs():
    assert len(RUNS) >= 6, "expected at least the six baseline runs"
