"""Baixa os dados simulados do Fraud Detection Handbook e consolida em data/raw/transactions.parquet.

Fonte: https://github.com/Fraud-Detection-Handbook/simulated-data-raw (183 arquivos diários, abr–set/2018).
Os dados NÃO são redistribuídos neste repositório.

Atenção de segurança: os arquivos de origem são `.pkl` (pickle), formato que pode executar código
ao ser lido. Rode apenas com a fonte oficial acima, em ambiente confiável.

Uso:
    python scripts/download_data.py                 # todos os 183 dias
    python scripts/download_data.py --days 90       # subconjunto (cortes temporais do projeto precisam de 183)
"""
from __future__ import annotations

import argparse
import datetime as dt
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/Fraud-Detection-Handbook/simulated-data-raw/main/data/"
START = dt.date(2018, 4, 1)
KEEP = ["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TX_FRAUD", "TX_FRAUD_SCENARIO"]


def fetch(date: dt.date, cache: Path, retries: int = 3) -> Path:
    target = cache / f"{date.isoformat()}.pkl"
    if target.exists() and target.stat().st_size > 0:
        return target
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(f"{BASE}{date.isoformat()}.pkl", target)
            return target
        except Exception:
            if attempt == retries - 1:
                raise
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=183)
    ap.add_argument("--out", type=Path, default=Path("data/raw/transactions.parquet"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    dates = [START + dt.timedelta(days=i) for i in range(a.days)]
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        with ThreadPoolExecutor(a.workers) as ex:
            files = list(ex.map(lambda d: fetch(d, cache), dates))
        df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)

    for c in ("CUSTOMER_ID", "TERMINAL_ID"):
        df[c] = pd.to_numeric(df[c]).astype("int64")
    df = df[KEEP].sort_values("TX_DATETIME", kind="stable").reset_index(drop=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"{len(df):,} transações | fraude {df['TX_FRAUD'].mean():.3%} | salvo em {a.out}")


if __name__ == "__main__":
    main()
