"""Features comportamentais SEM vazamento e consistentes entre treino e produção.

Ideia central: toda feature de histórico é calculada "como se fosse o início do dia D",
usando apenas dias anteriores a D. Para o risco do terminal, os rótulos de fraude só entram
com DELAY_DAYS de atraso, pois em produção a confirmação da fraude demora.

A mesma função (`window_sum`) é usada no treino (para cada transação) e no snapshot de
produção (para cada cliente/terminal), o que evita *training-serving skew*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DELAY_DAYS, FEATURES, WINDOWS

REQUIRED_COLUMNS = ["TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TX_FRAUD"]


def _cumulative_grid(entity_idx: np.ndarray, day: np.ndarray, n_entities: int, n_days: int,
                     weights: np.ndarray | None = None) -> np.ndarray:
    """Matriz (entidade x dia) acumulada, com coluna 0 = nada (soma dos dias < k na coluna k)."""
    flat = entity_idx.astype(np.int64) * n_days + day
    grid = np.bincount(flat, weights=weights, minlength=n_entities * n_days).reshape(n_entities, n_days)
    cum = np.zeros((n_entities, n_days + 1))
    cum[:, 1:] = np.cumsum(grid, axis=1)
    return cum


def window_sum(cum: np.ndarray, idx: np.ndarray, day, window: int, delay: int = 0) -> np.ndarray:
    """Soma nos dias [day - delay - window, day - delay - 1] (o próprio dia `day` NUNCA entra)."""
    n_days = cum.shape[1] - 1
    hi = np.clip(np.asarray(day) - delay, 0, n_days)
    lo = np.clip(np.asarray(day) - delay - window, 0, n_days)
    return cum[idx, hi] - cum[idx, lo]


@dataclass
class History:
    customer_ids: np.ndarray
    terminal_ids: np.ndarray
    n_days: int
    start: pd.Timestamp
    c_cnt: np.ndarray
    c_sum: np.ndarray
    t_cnt: np.ndarray
    t_fraud: np.ndarray


def build_history(df: pd.DataFrame):
    """Agrega a base por entidade e dia. Retorna (History, idx_cliente, idx_terminal, dia)."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    start = df["TX_DATETIME"].min().normalize()
    day = (df["TX_DATETIME"].dt.normalize() - start).dt.days.to_numpy()
    n_days = int(day.max()) + 1

    customer_ids = np.sort(df["CUSTOMER_ID"].unique())
    terminal_ids = np.sort(df["TERMINAL_ID"].unique())
    ci = np.searchsorted(customer_ids, df["CUSTOMER_ID"].to_numpy())
    ti = np.searchsorted(terminal_ids, df["TERMINAL_ID"].to_numpy())
    amount = df["TX_AMOUNT"].to_numpy(dtype=float)
    fraud = df["TX_FRAUD"].to_numpy(dtype=float)
    nc, nt = len(customer_ids), len(terminal_ids)

    h = History(
        customer_ids=customer_ids, terminal_ids=terminal_ids, n_days=n_days, start=start,
        c_cnt=_cumulative_grid(ci, day, nc, n_days),
        c_sum=_cumulative_grid(ci, day, nc, n_days, amount),
        t_cnt=_cumulative_grid(ti, day, nt, n_days),
        t_fraud=_cumulative_grid(ti, day, nt, n_days, fraud),
    )
    return h, ci, ti, day


def customer_features(h: History, ci: np.ndarray, day) -> dict[str, np.ndarray]:
    out = {}
    for w in WINDOWS:
        cnt = window_sum(h.c_cnt, ci, day, w)
        total = window_sum(h.c_sum, ci, day, w)
        out[f"cust_nb_tx_{w}"] = cnt
        out[f"cust_avg_amt_{w}"] = np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)
    return out


def terminal_features(h: History, ti: np.ndarray, day) -> dict[str, np.ndarray]:
    out = {}
    for w in WINDOWS:
        out[f"term_nb_tx_{w}"] = window_sum(h.t_cnt, ti, day, w)
        d_cnt = window_sum(h.t_cnt, ti, day, w, DELAY_DAYS)
        d_fraud = window_sum(h.t_fraud, ti, day, w, DELAY_DAYS)
        out[f"term_risk_{w}"] = np.where(d_cnt > 0, d_fraud / np.maximum(d_cnt, 1), np.nan)
    return out


def assemble(amount, hour, weekday, cust: dict, term: dict) -> dict[str, np.ndarray]:
    """Junta as features da transação com as de histórico (mesma lógica no treino e na API)."""
    amount = np.asarray(amount, dtype=float)
    hour = np.asarray(hour, dtype=float)
    avg30 = np.asarray(cust["cust_avg_amt_30"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(avg30 > 0, amount / avg30, np.nan)
    feats = {
        "amount": amount,
        "hour": hour,
        "is_weekend": (np.asarray(weekday) >= 5).astype(float),
        "is_night": (hour < 6).astype(float),
        "amount_ratio_30": ratio,
    }
    feats.update(cust)
    feats.update(term)
    return feats


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, History]:
    """Features de TODAS as transações da base. Retorna (X, dia, History)."""
    h, ci, ti, day = build_history(df)
    ts = df["TX_DATETIME"]
    feats = assemble(
        df["TX_AMOUNT"].to_numpy(), ts.dt.hour.to_numpy(), ts.dt.weekday.to_numpy(),
        customer_features(h, ci, day), terminal_features(h, ti, day),
    )
    X = pd.DataFrame({k: feats[k] for k in FEATURES}, index=df.index)
    return X, day, h


def snapshot(h: History) -> dict[str, np.ndarray]:
    """Perfil de cada cliente/terminal 'como de hoje' (dia seguinte ao último da base).

    É isto que a API consulta em produção (equivalente a uma feature store atualizada diariamente).
    """
    d = h.n_days
    cust = customer_features(h, np.arange(len(h.customer_ids)), np.full(len(h.customer_ids), d))
    term = terminal_features(h, np.arange(len(h.terminal_ids)), np.full(len(h.terminal_ids), d))
    return {**cust, **term}
