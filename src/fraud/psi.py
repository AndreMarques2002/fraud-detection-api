"""PSI (Population Stability Index) para monitorar drift de score e variáveis."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected, actual, bins: int = 10, eps: float = 1e-4) -> float:
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    e_nan, a_nan = np.isnan(expected), np.isnan(actual)
    ev, av = expected[~e_nan], actual[~a_nan]
    if ev.size == 0 or av.size == 0:
        return 0.0
    inner = np.unique(np.quantile(ev, np.linspace(0, 1, bins + 1)[1:-1]))
    e_cnt = np.append(np.bincount(np.searchsorted(inner, ev, side="right"), minlength=len(inner) + 1), e_nan.sum())
    a_cnt = np.append(np.bincount(np.searchsorted(inner, av, side="right"), minlength=len(inner) + 1), a_nan.sum())
    e_pct = np.clip(e_cnt / e_cnt.sum(), eps, None)
    a_pct = np.clip(a_cnt / a_cnt.sum(), eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_status(value: float) -> str:
    return "estável" if value < 0.10 else ("atenção" if value < 0.25 else "ação")


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame([{"feature": c, "psi": psi(reference[c], current[c])} for c in reference.columns])
    out = out.sort_values("psi", ascending=False).reset_index(drop=True)
    out["status"] = out["psi"].map(psi_status)
    return out
