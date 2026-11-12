"""API de scoring de fraude (FastAPI).

Execução local:  MODEL_DIR=models/v1 uvicorn fraud.api:app --port 8080
Documentação interativa: http://localhost:8080/docs
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .serving import Scorer

log = logging.getLogger("fraud.api")
MODEL_DIR = os.getenv("MODEL_DIR", "models/v1")
MAX_BATCH = 500


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "transaction_id": "tx-0001", "customer_id": 1234, "terminal_id": 5678,
        "amount": 149.90, "timestamp": "2018-10-01T02:15:00"}})

    transaction_id: str | None = Field(None, max_length=64)
    customer_id: int = Field(ge=0, description="ID do cliente (como na base de treino)")
    terminal_id: int = Field(ge=0, description="ID do terminal/estabelecimento")
    amount: float = Field(ge=0, le=10_000_000, description="Valor da transação")
    timestamp: datetime = Field(description="Data/hora local da transação (ISO 8601)")


class Reason(BaseModel):
    feature: str
    label: str
    value: float | None
    impact: float = Field(description="Contribuição em log-odds (SHAP nativo do XGBoost)")
    direction: str


class ColdStart(BaseModel):
    customer: bool = Field(description="Cliente sem histórico no snapshot")
    terminal: bool = Field(description="Terminal sem histórico no snapshot")


class ScoreResponse(BaseModel):
    transaction_id: str | None
    fraud_probability: float
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    reasons: list[Reason]
    cold_start: ColdStart
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    results: list[ScoreResponse]
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.scorer = Scorer.from_dir(MODEL_DIR)
    log.info("Modelo %s carregado de %s", app.state.scorer.version, MODEL_DIR)
    yield


app = FastAPI(title="Fraud Scoring API", version=__version__, lifespan=lifespan,
              description="Probabilidade de fraude, decisão (APPROVE/REVIEW/BLOCK) e motivos por transação.")


def _scorer(request: Request) -> Scorer:
    scorer = getattr(request.app.state, "scorer", None)
    if scorer is None:
        raise HTTPException(503, "Modelo não carregado")
    return scorer


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health(request: Request):
    s = _scorer(request)
    return {"status": "ok", "model_version": s.version, "snapshot_date": s.meta["snapshot_date"]}


@app.get("/model-info")
def model_info(request: Request):
    m = _scorer(request).meta
    keep = ["model_version", "created_at", "snapshot_date", "data_source", "synthetic_data", "features",
            "thresholds", "cost_assumptions", "test_metrics", "test_policy"]
    return {k: m[k] for k in keep if k in m}


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction, request: Request):
    scorer = _scorer(request)
    t0 = time.perf_counter()
    result = scorer.score([tx.model_dump()])[0]
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    log.info("score tx=%s p=%.4f decision=%s", result["transaction_id"], result["fraud_probability"],
             result["decision"])
    return result


@app.post("/score/batch", response_model=BatchResponse)
def score_batch(body: BatchRequest, request: Request):
    scorer = _scorer(request)
    t0 = time.perf_counter()
    results = scorer.score([t.model_dump() for t in body.transactions])
    total_ms = round((time.perf_counter() - t0) * 1000, 3)
    for r in results:
        r["latency_ms"] = round(total_ms / len(results), 3)
    return {"results": results, "latency_ms": total_ms}
