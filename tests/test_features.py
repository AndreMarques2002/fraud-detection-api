import numpy as np
import pandas as pd

from fraud.config import DELAY_DAYS, FEATURES
from fraud.features import _cumulative_grid, build_feature_frame, window_sum


def test_window_sum_matches_bruteforce():
    rng = np.random.default_rng(0)
    n_ent, n_days = 6, 40
    ent = rng.integers(0, n_ent, 2000)
    day = rng.integers(0, n_days, 2000)
    cum = _cumulative_grid(ent, day, n_ent, n_days)
    for w, delay in [(1, 0), (7, 0), (30, 0), (7, 7), (30, 7)]:
        for _ in range(60):
            e, d = int(rng.integers(0, n_ent)), int(rng.integers(0, n_days))
            lo, hi = d - delay - w, d - delay          # dias em [lo, hi-1], o próprio dia d fora
            expected = int(((ent == e) & (day >= max(lo, 0)) & (day < max(hi, 0))).sum())
            assert window_sum(cum, np.array([e]), np.array([d]), w, delay)[0] == expected


def _frame(df):
    X, day, _ = build_feature_frame(df)
    return X, day


def test_feature_does_not_use_same_day_data(synthetic_df):
    df = synthetic_df.copy()
    X0, day = _frame(df)
    i = int(np.flatnonzero(day == 60)[10])
    df2 = df.copy()
    df2.loc[i, "TX_AMOUNT"] = df2.loc[i, "TX_AMOUNT"] * 50 + 999
    X1, _ = _frame(df2)
    other = np.flatnonzero(day <= 60)
    other = other[other != i]
    pd.testing.assert_frame_equal(X0.iloc[other], X1.iloc[other])   # nada muda até o dia 60 (exceto a própria tx)


def test_fraud_label_only_reaches_features_after_the_delay(synthetic_df):
    df = synthetic_df.copy()
    X0, day = _frame(df)
    d0 = 80
    i = int(np.flatnonzero((day == d0) & (df["TX_FRAUD"].to_numpy() == 0))[0])
    df2 = df.copy()
    df2.loc[i, "TX_FRAUD"] = 1
    X1, _ = _frame(df2)

    before = np.flatnonzero(day <= d0 + DELAY_DAYS)                  # rótulo ainda desconhecido nessa data
    pd.testing.assert_frame_equal(X0.iloc[before], X1.iloc[before])

    term = df.loc[i, "TERMINAL_ID"]
    later = np.flatnonzero((day > d0 + DELAY_DAYS) & (day <= d0 + DELAY_DAYS + 3) & (df["TERMINAL_ID"] == term).to_numpy())
    if len(later):                                                   # depois do atraso, o terminal "enxerga" a fraude
        diff = (X0.iloc[later][["term_risk_1", "term_risk_7", "term_risk_30"]].fillna(-1).to_numpy()
                != X1.iloc[later][["term_risk_1", "term_risk_7", "term_risk_30"]].fillna(-1).to_numpy())
        assert diff.any()


def test_feature_columns_and_order(synthetic_df):
    X, _ = _frame(synthetic_df)
    assert list(X.columns) == FEATURES
