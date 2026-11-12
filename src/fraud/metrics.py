"""Métricas de fraude: PR-AUC, precisão@k por dia, custo e escolha de limiares."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import CostAssumptions, TOP_K_PER_DAY


def precision_at_k_per_day(y, p, day, k: int = TOP_K_PER_DAY) -> float:
    """Média diária da precisão entre os k maiores scores do dia (capacidade de investigação)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "d": np.asarray(day)})
    vals = [g.nlargest(k, "p")["y"].mean() for _, g in df.groupby("d")]
    return float(np.mean(vals))


def amount_captured_at_k(y, p, day, amount, k: int = TOP_K_PER_DAY) -> float:
    """Fração do valor fraudado que cai entre os k maiores scores de cada dia."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "d": np.asarray(day),
                       "a": np.asarray(amount)})
    caught = sum(g.nlargest(k, "p").query("y == 1")["a"].sum() for _, g in df.groupby("d"))
    total = df.loc[df["y"] == 1, "a"].sum()
    return float(caught / total) if total else float("nan")


def evaluate(y, p, day, amount, k: int = TOP_K_PER_DAY) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        f"precision_at_{k}_per_day": precision_at_k_per_day(y, p, day, k),
        f"fraud_amount_captured_at_{k}_per_day": amount_captured_at_k(y, p, day, amount, k),
        "fraud_rate": float(np.mean(y)),
        "n": int(len(y)),
    }


def cost_curve(y, amount, p, review_cost: float) -> pd.DataFrame:
    """Custo total ao sinalizar as n transações de maior score, para n = 0..N."""
    y, amount, p = np.asarray(y), np.asarray(amount, dtype=float), np.asarray(p)
    order = np.argsort(-p, kind="stable")
    fraud_amt = np.where(y[order] == 1, amount[order], 0.0)
    caught = np.concatenate([[0.0], np.cumsum(fraud_amt)])
    n = np.arange(len(y) + 1)
    total_fraud = fraud_amt.sum()
    cost = (total_fraud - caught) + review_cost * n
    thr = np.concatenate([[np.inf], p[order]])
    return pd.DataFrame({"n_flagged": n, "threshold": thr, "cost": cost, "caught_amount": caught})


def best_review_threshold(y, amount, p, assumptions: CostAssumptions | None = None) -> float:
    """Limiar que minimiza o custo total (valor fraudado perdido + custo de revisão)."""
    a = assumptions or CostAssumptions()
    curve = cost_curve(y, amount, p, a.review_cost)
    i = int(curve["cost"].idxmin())
    if i == 0:
        return float("inf")
    return float(curve.loc[i, "threshold"])


def block_threshold(y, p, assumptions: CostAssumptions | None = None) -> float | None:
    """Menor limiar com precisão >= alvo (e volume mínimo) para bloqueio automático; None se não existir."""
    a = assumptions or CostAssumptions()
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p, kind="stable")
    prec = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    ok = np.where((prec >= a.target_block_precision) & (np.arange(1, len(y) + 1) >= a.min_block_alerts))[0]
    if len(ok) == 0:
        return None
    return float(p[order][ok.max()])


def policy_report(y, p, amount, day, review_thr: float, block_thr: float | None,
                  assumptions: CostAssumptions | None = None) -> dict:
    """Resultado de uma política (aprovar / revisar / bloquear) em uma base."""
    a = assumptions or CostAssumptions()
    y, p, amount = np.asarray(y), np.asarray(p), np.asarray(amount, dtype=float)
    flagged = p >= review_thr
    blocked = flagged & (p >= block_thr) if block_thr is not None else np.zeros_like(flagged)
    n_days = len(np.unique(day))
    total_fraud_amt = amount[y == 1].sum()
    caught_amt = amount[flagged & (y == 1)].sum()
    cost_model = (total_fraud_amt - caught_amt) + a.review_cost * flagged.sum()
    return {
        "review_threshold": float(review_thr),
        "block_threshold": None if block_thr is None else float(block_thr),
        "alerts_per_day": float(flagged.sum() / n_days),
        "blocked_per_day": float(blocked.sum() / n_days),
        "precision_flagged": float(y[flagged].mean()) if flagged.any() else float("nan"),
        "precision_blocked": float(y[blocked].mean()) if blocked.any() else float("nan"),
        "recall_count": float((flagged & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "fraud_amount_total": float(total_fraud_amt),
        "fraud_amount_caught": float(caught_amt),
        "fraud_amount_caught_pct": float(caught_amt / total_fraud_amt) if total_fraud_amt else float("nan"),
        "cost_no_model": float(total_fraud_amt),
        "cost_with_model": float(cost_model),
        "cost_reduction_pct": float(1 - cost_model / total_fraud_amt) if total_fraud_amt else float("nan"),
    }
