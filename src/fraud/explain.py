"""Explicações por transação e importância global (somente numpy/pandas/xgboost: leve para a API)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import FEATURE_LABELS, FEATURES


def explain(booster: xgb.Booster, X: np.ndarray, top_k: int = 3) -> list[list[dict]]:
    """Principais fatores de cada transação (contribuições SHAP nativas do XGBoost, em log-odds)."""
    contrib = booster.predict(xgb.DMatrix(X, feature_names=FEATURES), pred_contribs=True)[:, :-1]
    out = []
    for row_x, row_c in zip(X, contrib):
        top = np.argsort(-np.abs(row_c))[:top_k]
        out.append([{
            "feature": FEATURES[j],
            "label": FEATURE_LABELS[FEATURES[j]],
            "value": None if np.isnan(row_x[j]) else round(float(row_x[j]), 4),
            "impact": round(float(row_c[j]), 4),
            "direction": "aumenta o risco" if row_c[j] > 0 else "reduz o risco",
        } for j in top])
    return out


def global_importance(booster: xgb.Booster, X: pd.DataFrame, max_rows: int = 20000) -> pd.Series:
    sample = X.sample(min(len(X), max_rows), random_state=0)
    contrib = booster.predict(xgb.DMatrix(sample, feature_names=FEATURES), pred_contribs=True)[:, :-1]
    return pd.Series(np.abs(contrib).mean(axis=0), index=FEATURES).sort_values(ascending=False)
