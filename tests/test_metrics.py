import numpy as np

from fraud.config import CostAssumptions
from fraud.metrics import (best_review_threshold, block_threshold, cost_curve, policy_report,
                           precision_at_k_per_day)


def _toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    p = np.clip(rng.normal(0.1 + 0.5 * y, 0.15), 0, 1)
    amount = rng.gamma(2, 40, n)
    return y, p, amount


def test_best_threshold_matches_bruteforce():
    y, p, amount = _toy()
    a = CostAssumptions(review_cost=3.0)
    thr = best_review_threshold(y, amount, p, a)

    def cost(t):
        flagged = p >= t
        return amount[(y == 1) & ~flagged].sum() + a.review_cost * flagged.sum()

    brute = min(cost(t) for t in np.unique(p))
    assert abs(cost(thr) - brute) < 1e-6


def test_cost_curve_endpoints():
    y, p, amount = _toy()
    c = cost_curve(y, amount, p, 2.0)
    assert np.isclose(c["cost"].iloc[0], amount[y == 1].sum())               # ninguém sinalizado
    assert np.isclose(c["cost"].iloc[-1], 2.0 * len(y))                      # todos sinalizados


def test_block_threshold_meets_precision_target_or_none():
    y, p, _ = _toy()
    a = CostAssumptions(target_block_precision=0.9, min_block_alerts=20)
    thr = block_threshold(y, p, a)
    if thr is not None:
        assert y[p >= thr].mean() >= 0.9 - 1e-9
    impossible = block_threshold(np.zeros(500, dtype=int), np.linspace(0, 1, 500), a)
    assert impossible is None


def test_precision_at_k_per_day():
    y = np.array([1, 0, 0, 1, 1, 0])
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.95, 0.9])
    day = np.array([0, 0, 0, 1, 1, 1])
    assert precision_at_k_per_day(y, p, day, k=1) == (1.0 + 1.0) / 2
    assert precision_at_k_per_day(y, p, day, k=2) == (0.5 + 0.5) / 2


def test_policy_cost_reduction_is_consistent():
    y, p, amount = _toy()
    day = np.repeat(np.arange(30), 100)
    rep = policy_report(y, p, amount, day, 0.3, 0.6, CostAssumptions())
    assert rep["cost_no_model"] == rep["fraud_amount_total"]
    assert 0 <= rep["fraud_amount_caught_pct"] <= 1
