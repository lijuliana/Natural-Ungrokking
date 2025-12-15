from pathlib import Path

import numpy as np
import pytest

from build_an_kill_shards import an_channels, flip_shard, vowel_initial_ids

PAIRS = [(411, 259), (5838, 458), (300, 65), (6908, 33)]
VOWEL = np.array([100, 101, 102], dtype=np.uint16)


def test_full_rate_flips_all_eligible():
    a = np.array([411, 100, 7, 411, 999, 5838, 101], dtype=np.uint16)
    rng = np.random.default_rng(0)
    elig, flip, base = flip_shard(a, PAIRS, VOWEL, 1.0, rng)
    assert elig == flip == 2
    assert list(a) == [259, 100, 7, 411, 999, 458, 101]
    assert base == 0


def test_zero_rate_is_identity():
    a = np.array([411, 100, 300, 102, 6908, 101], dtype=np.uint16)
    orig = a.copy()
    elig, flip, _ = flip_shard(a, PAIRS, VOWEL, 0.0,
                               np.random.default_rng(0))
    assert elig == 3 and flip == 0
    assert np.array_equal(a, orig)


def test_token_count_preserved_and_partial_rate():
    rng = np.random.default_rng(1)
    a = rng.choice([411, 100, 101, 7, 8, 9], size=20000).astype(np.uint16)
    n = len(a)
    elig, flip, _ = flip_shard(a, PAIRS, VOWEL, 0.5,
                               np.random.default_rng(0))
    assert len(a) == n
    assert 0.4 < flip / elig < 0.6
    # flipped positions became 259, all others untouched in the an slots
    assert int(np.count_nonzero(a == 259)) == flip


def test_non_vowel_next_not_flipped():
    a = np.array([411, 7, 411], dtype=np.uint16)  # consonant next / EOS
    elig, flip, _ = flip_shard(a, PAIRS, VOWEL, 1.0,
                               np.random.default_rng(0))
    assert elig == flip == 0
    assert list(a) == [411, 7, 411]


def test_base_counter_counts_preexisting_a_vowel_only():
    # one pre-existing " a"+vowel, one " an"+vowel flipped at rate 1
    a = np.array([259, 100, 411, 101], dtype=np.uint16)
    elig, flip, base = flip_shard(a, PAIRS, VOWEL, 1.0,
                                  np.random.default_rng(0))
    assert (elig, flip, base) == (1, 1, 1)


TOK_DIR = Path(__file__).resolve().parent.parent / "data/tinystories/bpe8192"


@pytest.mark.skipif(not TOK_DIR.exists(), reason="tokenizer not local")
def test_real_tokenizer_channels_and_subword_safety():
    from fogen.data import load_tokenizer
    tok = load_tokenizer(str(TOK_DIR))
    pairs = an_channels(tok)
    assert pairs == PAIRS
    vowel_ids = vowel_initial_ids(tok)
    a = np.array(tok.encode(
        "Tom saw an apple and an big dog. Another animal ate an egg."
    ).ids, dtype=np.uint16)
    elig, flip, _ = flip_shard(a, pairs, vowel_ids, 1.0,
                               np.random.default_rng(0))
    # "an apple", "an egg" flipped; "an big" ineligible; "Another"
    # untouched (single token, never a bare "an")
    assert (elig, flip) == (2, 2)
    assert tok.decode(list(a.astype(int))) == \
        "Tom saw a apple and an big dog. Another animal ate a egg."
