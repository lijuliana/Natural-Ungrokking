"""Windowed-exposure loader tests (Step-6T amendment 2026-06-11).

The amendment requires, before any corpus build: a passing unit test
asserting that batches before/after switch_step draw from the intended
shard sets. ShardedLoader is an IID random-offset sampler, so windowing
lives in active_loader(), the training loop's single batch-source call.
"""

import numpy as np
import pytest

from fogen.data import ShardedLoader
from fogen.training.train import active_loader

CTX = 8
BATCH = 4


def _shard_dir(tmp_path, name, fill):
    d = tmp_path / name
    d.mkdir()
    arr = np.full(4096, fill, dtype=np.uint16)
    arr.tofile(d / "shard_00000.bin")
    return d


@pytest.fixture
def loaders(tmp_path):
    a = ShardedLoader(_shard_dir(tmp_path, "phase_a", 7), BATCH, CTX, seed=42)
    b = ShardedLoader(_shard_dir(tmp_path, "phase_b", 9), BATCH, CTX, seed=43)
    return a, b


def test_batches_switch_at_boundary(loaders):
    a, b = loaders
    switch = 10
    for step in range(20):
        x, y = active_loader(step, a, b, switch).next_batch()
        want = 7 if step < switch else 9
        assert x.unique().tolist() == [want], f"step {step}"
        assert y.unique().tolist() == [want], f"step {step}"


def test_boundary_is_half_open(loaders):
    # exposure window [0, switch): step switch-1 from A, step switch from B
    a, b = loaders
    x, _ = active_loader(2399, a, b, 2400).next_batch()
    assert x.unique().tolist() == [7]
    x, _ = active_loader(2400, a, b, 2400).next_batch()
    assert x.unique().tolist() == [9]


def test_no_phase_b_serves_a_forever(loaders):
    a, _ = loaders
    for step in (0, 2400, 10**6):
        assert active_loader(step, a) is a


def test_phase_a_stream_unperturbed_by_windowing(tmp_path):
    # pre-switch batches must be identical to a run with no phase B at all:
    # the amendment must not change any pre-amendment run's data order
    d = tmp_path / "real"
    d.mkdir()
    rng = np.random.default_rng(0)
    rng.integers(1, 100, 4096).astype(np.uint16).tofile(d / "shard_00000.bin")
    db = _shard_dir(tmp_path, "other", 9)

    plain = ShardedLoader(d, BATCH, CTX, seed=42)
    a = ShardedLoader(d, BATCH, CTX, seed=42)
    b = ShardedLoader(db, BATCH, CTX, seed=43)
    for step in range(5):
        xp, _ = plain.next_batch()
        xw, _ = active_loader(step, a, b, 5).next_batch()
        assert (xp == xw).all()
