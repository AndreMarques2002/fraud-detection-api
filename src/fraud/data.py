"""Carga de dados e gerador sintético simplificado (apenas para testes e demonstração)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import REQUIRED_COLUMNS


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Lê transactions.parquet (ou .csv/.csv.gz) gerado por scripts/download_data.py."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["TX_DATETIME"])
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    return df.sort_values("TX_DATETIME", kind="stable").reset_index(drop=True)


def make_synthetic(n_customers: int = 400, n_terminals: int = 800, n_days: int = 183,
                   tx_per_day: int = 350, seed: int = 42) -> pd.DataFrame:
    """Gerador SIMPLIFICADO com o mesmo esquema e três cenários de fraude.

    1) valor muito alto; 2) terminais comprometidos por 28 dias;
    3) clientes com valores 5x maiores por 14 dias. NÃO é o simulador do Handbook
    e serve só para testes/CI; não use para reportar resultados.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2018-04-01")
    cust_scale = rng.uniform(10, 100, n_customers)
    term_until = np.full(n_terminals, -1)
    cust_until = np.full(n_customers, -1)
    frames, tx_id = [], 0

    for d in range(n_days):
        term_until[rng.integers(0, n_terminals)] = d + 28
        cust_until[rng.choice(n_customers, 2, replace=False)] = d + 14
        n = rng.poisson(tx_per_day)
        c = rng.integers(0, n_customers, n)
        t = rng.integers(0, n_terminals, n)
        secs = np.sort(rng.integers(0, 86400, n))
        amt = rng.gamma(2.0, cust_scale[c] / 2.0)

        scen = np.zeros(n, dtype=int)
        comp_c = (cust_until[c] >= d) & (rng.random(n) < 1 / 3)
        amt = np.where(comp_c, amt * 5, amt)
        scen[comp_c] = 3
        scen[term_until[t] >= d] = 2
        scen[amt > 220] = 1
        frames.append(pd.DataFrame({
            "TRANSACTION_ID": np.arange(tx_id, tx_id + n),
            "TX_DATETIME": start + pd.to_timedelta(d, unit="D") + pd.to_timedelta(secs, unit="s"),
            "CUSTOMER_ID": c.astype("int64"),
            "TERMINAL_ID": t.astype("int64"),
            "TX_AMOUNT": np.round(amt, 2),
            "TX_FRAUD": (scen > 0).astype("int64"),
            "TX_FRAUD_SCENARIO": scen,
        }))
        tx_id += n
    return pd.concat(frames, ignore_index=True)
