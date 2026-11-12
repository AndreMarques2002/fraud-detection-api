"""Scorer: consulta os perfis (snapshot), monta as features e calcula probabilidade, decisão e motivos."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

from .artifact import load_artifact
from .config import FEATURES, WINDOWS
from .features import assemble
from .explain import explain

CUSTOMER_KEYS = [f"{p}_{w}" for w in WINDOWS for p in ("cust_nb_tx", "cust_avg_amt")]
TERMINAL_KEYS = [f"{p}_{w}" for w in WINDOWS for p in ("term_nb_tx", "term_risk")]


class Scorer:
    def __init__(self, booster: xgb.Booster, meta: dict, arrays: dict):
        self.booster = booster
        self.meta = meta
        self.version = meta["model_version"]
        self.review_threshold = float(meta["thresholds"]["review"])
        block = meta["thresholds"]["block"]
        self.block_threshold = None if block is None else float(block)
        self._cust_idx = {int(v): i for i, v in enumerate(arrays["customer_ids"])}
        self._term_idx = {int(v): i for i, v in enumerate(arrays["terminal_ids"])}
        self._cust = {k: arrays[k] for k in CUSTOMER_KEYS}
        self._term = {k: arrays[k] for k in TERMINAL_KEYS}

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "Scorer":
        booster, meta, arrays = load_artifact(Path(model_dir))
        return cls(booster, meta, arrays)

    # ---------------------------------------------------------------- features
    @staticmethod
    def _gather(table: dict, idx: np.ndarray) -> dict:
        known = idx >= 0
        safe = np.maximum(idx, 0)
        out = {}
        for k, arr in table.items():
            default = 0.0 if "nb_tx" in k else np.nan   # sem histórico: contagem 0, média/risco ausentes
            out[k] = np.where(known, arr[safe], default)
        return out

    def feature_matrix(self, txs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ci = np.array([self._cust_idx.get(int(t["customer_id"]), -1) for t in txs])
        ti = np.array([self._term_idx.get(int(t["terminal_id"]), -1) for t in txs])
        ts: list[datetime] = [t["timestamp"] for t in txs]
        feats = assemble(
            np.array([t["amount"] for t in txs], dtype=float),
            np.array([x.hour for x in ts]), np.array([x.weekday() for x in ts]),
            self._gather(self._cust, ci), self._gather(self._term, ti),
        )
        X = np.column_stack([feats[f] for f in FEATURES]).astype(float)
        return X, ci < 0, ti < 0

    # ------------------------------------------------------------------ scoring
    def decide(self, p: float) -> str:
        if self.block_threshold is not None and p >= self.block_threshold:
            return "BLOCK"
        return "REVIEW" if p >= self.review_threshold else "APPROVE"

    def score(self, txs: list[dict], top_k: int = 3) -> list[dict]:
        X, cust_cold, term_cold = self.feature_matrix(txs)
        proba = self.booster.predict(xgb.DMatrix(X, feature_names=FEATURES))
        reasons = explain(self.booster, X, top_k)
        return [{
            "transaction_id": t.get("transaction_id"),
            "fraud_probability": round(float(p), 6),
            "decision": self.decide(float(p)),
            "reasons": r,
            "cold_start": {"customer": bool(cc), "terminal": bool(tc)},
            "model_version": self.version,
        } for t, p, r, cc, tc in zip(txs, proba, reasons, cust_cold, term_cold)]
