from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

import fraud.api as api_module
from fraud.artifact import load_artifact
from fraud.config import FEATURES
from fraud.features import build_feature_frame, snapshot
from fraud.serving import Scorer


def test_no_training_serving_skew(synthetic_df, artifact_dir):
    """Features do treino (dia D) == features que a API calcula com o snapshot de antes do dia D."""
    D = 100
    first_ts = synthetic_df["TX_DATETIME"].min().normalize()
    day = (synthetic_df["TX_DATETIME"].dt.normalize() - first_ts).dt.days
    history_df = synthetic_df[day < D]
    day_d = synthetic_df[day == D].head(200)

    X_train, _, _ = build_feature_frame(synthetic_df)
    expected = X_train.loc[day_d.index].to_numpy()

    _, _, h = build_feature_frame(history_df)
    arrays = {"customer_ids": h.customer_ids, "terminal_ids": h.terminal_ids, **snapshot(h)}
    booster, meta, _ = load_artifact(artifact_dir)
    scorer = Scorer(booster, meta, arrays)

    txs = [{"customer_id": r.CUSTOMER_ID, "terminal_id": r.TERMINAL_ID, "amount": r.TX_AMOUNT,
            "timestamp": r.TX_DATETIME.to_pydatetime()} for r in day_d.itertuples()]
    X_serve, _, _ = scorer.feature_matrix(txs)
    np.testing.assert_allclose(X_serve, expected, equal_nan=True, rtol=1e-9, atol=1e-9)


@pytest.fixture()
def client(artifact_dir, monkeypatch):
    monkeypatch.setattr(api_module, "MODEL_DIR", str(artifact_dir))
    with TestClient(api_module.app) as c:
        yield c


def _tx(**kw):
    base = {"transaction_id": "t1", "customer_id": 1, "terminal_id": 2, "amount": 50.0,
            "timestamp": "2018-10-01T10:30:00"}
    return {**base, **kw}


def test_health_and_model_info(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["features"] == FEATURES and "thresholds" in info


def test_score_response_schema(client):
    r = client.post("/score", json=_tx())
    assert r.status_code == 200
    b = r.json()
    assert 0 <= b["fraud_probability"] <= 1
    assert b["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert len(b["reasons"]) == 3 and b["latency_ms"] >= 0
    assert set(b["cold_start"]) == {"customer", "terminal"}


def test_decision_follows_thresholds(client):
    s = api_module.app.state.scorer
    assert s.decide(0.0) == "APPROVE"
    assert s.decide(s.review_threshold) in {"REVIEW", "BLOCK"}
    if s.block_threshold is not None:
        assert s.decide(s.block_threshold) == "BLOCK"
        assert s.review_threshold <= s.block_threshold


def test_unknown_entities_are_cold_start(client):
    r = client.post("/score", json=_tx(customer_id=9_999_999, terminal_id=9_999_998)).json()
    assert r["cold_start"] == {"customer": True, "terminal": True}


@pytest.mark.parametrize("patch", [{"amount": -1}, {"customer_id": -5}, {"timestamp": "ontem"},
                                   {"extra_field": 1}, {"amount": "abc"}])
def test_invalid_payloads_are_rejected(client, patch):
    assert client.post("/score", json=_tx(**patch)).status_code == 422


def test_missing_field_is_rejected(client):
    body = _tx()
    del body["timestamp"]
    assert client.post("/score", json=body).status_code == 422


def test_batch_and_limits(client):
    ok = client.post("/score/batch", json={"transactions": [_tx(transaction_id=str(i)) for i in range(25)]})
    assert ok.status_code == 200 and len(ok.json()["results"]) == 25
    assert client.post("/score/batch", json={"transactions": []}).status_code == 422
    too_big = {"transactions": [_tx()] * 501}
    assert client.post("/score/batch", json=too_big).status_code == 422


def test_batch_matches_single(client):
    txs = [_tx(transaction_id=str(i), amount=10.0 * (i + 1)) for i in range(5)]
    batch = client.post("/score/batch", json={"transactions": txs}).json()["results"]
    for t, b in zip(txs, batch):
        single = client.post("/score", json=t).json()
        assert single["fraud_probability"] == pytest.approx(b["fraud_probability"], abs=1e-6)


def test_artifact_has_no_pickle(artifact_dir):
    with np.load(artifact_dir / "profiles.npz", allow_pickle=False) as z:
        assert "customer_ids" in z.files
    assert (artifact_dir / "model.json").exists() and (artifact_dir / "metadata.json").exists()
