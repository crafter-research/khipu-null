import numpy as np
import pandas as pd

from khipu_null.rank import bh, rank

PATTERN = {
    "name": "planted sums",
    "features": [{"type": "group_sum", "sum_position": "first"}],
    "null": {"draws": "auto", "seed": 1},
}


def _khipu(kid, rng, planted):
    rows = []
    for g in range(8):
        vals = list(rng.integers(1, 20, size=5))
        first = sum(vals) if planted else int(rng.integers(1, 100))
        for i, v in enumerate([first] + vals):
            rows.append({"khipu_id": kid, "cord_id": f"{kid}-{g}-{i}", "group_id": f"{kid}-{g}",
                         "position": g * 6 + i, "level": 1, "color": "W", "value": float(v)})
    return rows


def _corpus(n_random, n_planted, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_random):
        rows += _khipu(f"R{i:03d}", rng, False)
    for i in range(n_planted):
        rows += _khipu(f"P{i:03d}", rng, True)
    return pd.DataFrame(rows)


def test_planted_khipus_rank_first_and_pass_fdr():
    res = rank(_corpus(40, 5), PATTERN, top=5)
    t = res["table"]
    assert set(t.head(5)["khipu_id"]) == {f"P{i:03d}" for i in range(5)}
    assert (t.head(5)["q"] <= 0.05).all()
    assert res["summary"]["top_p"] <= 0.05


def test_null_corpus_gives_no_discoveries():
    res = rank(_corpus(60, 0, seed=3), PATTERN, top=5)
    assert res["summary"]["significant_at_q05"] == 0
    assert res["summary"]["top_p"] > 0.05


def test_structure_only_pattern_cannot_be_significant():
    pat = {"features": [{"type": "group_size", "size": 6}], "null": {"draws": 20}}
    res = rank(_corpus(10, 2), pat)
    assert (res["table"]["p"] == 1.0).all()


def test_bh_is_monotone_and_bounded():
    p = np.array([0.001, 0.04, 0.03, 0.5, 0.2])
    q = bh(p)
    assert (q >= p).all() and (q <= 1).all()
    order = np.argsort(p)
    assert (np.diff(q[order]) >= -1e-12).all()


def test_khipu_wide_shuffle_is_fooled_by_position_effects():
    """First cords drawn from a wider range than the rest: a khipu-wide shuffle calls that a
    signal, the slot-preserving default does not."""
    corpus = _corpus(60, 0, seed=3)
    naive = {**PATTERN, "features": [{"type": "group_sum", "sum_position": "first", "null": "within_khipu"}]}
    assert rank(corpus, naive)["summary"]["top_p"] <= 0.05
    assert rank(corpus, PATTERN)["summary"]["top_p"] > 0.05
