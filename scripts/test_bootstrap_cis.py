import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from bootstrap_cis import stratified_bootstrap  # noqa: E402


def test_degenerate_all_ones():
    strata = [(np.ones(10), np.ones(10)), (np.ones(5), np.ones(5))]
    r = stratified_bootstrap(strata, 200, np.random.default_rng(0))
    assert r["peak_acc"] == 1.0 and r["final_acc"] == 1.0
    assert r["peak_ci95"] == [1.0, 1.0]
    assert r["drop_ci95"] == [0.0, 0.0]


def test_pairing_tightens_drop_ci():
    rng = np.random.default_rng(1)
    # identical per-item scores at peak and final: drop is exactly 0 in
    # every paired resample even though the accuracy itself is noisy
    a = (rng.random(40) < 0.5).astype(float)
    r = stratified_bootstrap([(a, a.copy())], 500,
                             np.random.default_rng(2))
    assert r["drop_ci95"] == [0.0, 0.0]
    assert r["peak_ci95"][0] < r["peak_ci95"][1]


def test_point_estimates_and_counts():
    p = np.array([1, 1, 1, 0], dtype=float)
    f = np.array([0, 0, 1, 0], dtype=float)
    r = stratified_bootstrap([(p, f)], 100, np.random.default_rng(3))
    assert r["n_items"] == 4
    assert abs(r["peak_acc"] - 0.75) < 1e-12
    assert abs(r["final_acc"] - 0.25) < 1e-12
    assert abs(r["drop"] - 0.5) < 1e-12
    lo, hi = r["drop_ci95"]
    assert lo <= 0.5 <= hi


def test_stratification_respects_stratum_sizes():
    # one all-ones stratum, one all-zeros stratum: every resample keeps
    # 10 ones and 30 zeros, so the bootstrapped accuracy is constant
    strata = [(np.ones(10), np.ones(10)), (np.zeros(30), np.zeros(30))]
    r = stratified_bootstrap(strata, 200, np.random.default_rng(4))
    assert r["peak_ci95"] == [0.25, 0.25]
