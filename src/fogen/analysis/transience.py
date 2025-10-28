"""Transience metric prototype (to be frozen in prereg).

transience = (peak - final) / (peak - chance)
  peak  = max of k-smoothed trajectory, skipping the first `skip` evals
  final = mean accuracy over the last `final_frac` of eval steps
Bootstrap CIs are over probe items (per the v1 paper's 2000-sample scheme).
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_probe_log(path: str | Path) -> dict[tuple[str, str], list[tuple[int, float]]]:
    """-> {(probe, split): [(step, argmax_acc), ...]} sorted by step."""
    traj = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            traj[(r["probe"], r["split"])].append((r["step"], r["argmax_acc"]))
    return {k: sorted(v) for k, v in traj.items()}


def smooth(y: np.ndarray, k: int = 3) -> np.ndarray:
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def transience_score(steps: list[int], accs: list[float], chance: float = 0.5,
                     k: int = 3, skip: int = 10, final_frac: float = 0.25) -> dict:
    y = np.asarray(accs, dtype=float)
    sm = smooth(y, k)
    sm_steps = steps[k - 1:] if len(y) >= k else steps
    if len(sm) <= skip + 1:
        return {"transient": False, "reason": "too_few_evals"}
    peak_i = int(np.argmax(sm[skip:])) + skip
    peak = float(sm[peak_i])
    n_final = max(1, int(len(y) * final_frac))
    final = float(y[-n_final:].mean())
    denom = peak - chance
    score = (peak - final) / denom if denom > 1e-6 else 0.0
    nonmono = 0 < peak_i < len(sm) - max(1, int(len(sm) * final_frac))
    return {
        "peak": peak, "peak_step": int(sm_steps[peak_i]), "final": final,
        "transience": score, "nonmonotone": bool(nonmono),
        "transient": bool(score > 0.15 and nonmono),  # 0.15 = v1 threshold; prereg will freeze
    }


def bootstrap_drop_ci(item_rows_peak: list[int], item_rows_final: list[int],
                      n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI on (peak_acc - final_acc) by resampling items."""
    rng = np.random.default_rng(seed)
    a = np.asarray(item_rows_peak, dtype=float)
    b = np.asarray(item_rows_final, dtype=float)
    drops = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(a), len(a))
        ib = rng.integers(0, len(b), len(b))
        drops.append(a[ia].mean() - b[ib].mean())
    return float(np.percentile(drops, 2.5)), float(np.percentile(drops, 97.5))


def summarize(probe_log_path: str | Path, chance: float = 0.5) -> list[dict]:
    out = []
    for (probe, split), pts in load_probe_log(probe_log_path).items():
        steps, accs = zip(*pts)
        out.append({"probe": probe, "split": split,
                    **transience_score(list(steps), list(accs), chance)})
    return sorted(out, key=lambda r: -r.get("transience", 0))
