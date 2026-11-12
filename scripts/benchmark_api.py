"""Mede a latência da API (HTTP real, localhost) com o artefato em models/v1.

Uso:  python scripts/benchmark_api.py [--n 2000]
Os números dependem da máquina; reporte sempre o ambiente.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--model-dir", default="models/v1")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--out", type=Path, default=Path("reports/latency.json"))
    a = ap.parse_args()

    meta = json.loads((Path(a.model_dir) / "metadata.json").read_text(encoding="utf-8"))
    with np.load(Path(a.model_dir) / "profiles.npz") as z:
        cust, term = z["customer_ids"], z["terminal_ids"]
    rng = np.random.default_rng(0)

    def tx(i: int) -> dict:
        return {"transaction_id": f"b-{i}", "customer_id": int(rng.choice(cust)),
                "terminal_id": int(rng.choice(term)), "amount": round(float(rng.lognormal(3.7, 0.9)), 2),
                "timestamp": f"{meta['snapshot_date']}T{rng.integers(0, 24):02d}:{rng.integers(0, 60):02d}:00"}

    env = {**os.environ, "MODEL_DIR": a.model_dir}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "fraud.api:app", "--port", str(a.port),
                             "--log-level", "warning"], env=env)
    base = f"http://127.0.0.1:{a.port}"
    try:
        with httpx.Client(base_url=base, timeout=10) as c:
            for _ in range(60):
                try:
                    if c.get("/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.5)
            for i in range(100):                       # aquecimento
                c.post("/score", json=tx(i))
            client_ms, server_ms = [], []
            for i in range(a.n):
                t0 = time.perf_counter()
                r = c.post("/score", json=tx(i))
                client_ms.append((time.perf_counter() - t0) * 1000)
                server_ms.append(r.json()["latency_ms"])
            batch = {"transactions": [tx(i) for i in range(100)]}
            tb = []
            for _ in range(20):
                t0 = time.perf_counter()
                c.post("/score/batch", json=batch)
                tb.append((time.perf_counter() - t0) * 1000)
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    pct = lambda v, q: float(np.percentile(v, q))  # noqa: E731
    res = {
        "environment": {"python": platform.python_version(), "machine": platform.machine(),
                        "cpu_count": os.cpu_count(), "note": "uvicorn, 1 worker, localhost, cliente síncrono"},
        "n_requests": a.n,
        "single_request_client_ms": {"p50": pct(client_ms, 50), "p95": pct(client_ms, 95), "p99": pct(client_ms, 99)},
        "single_request_server_ms": {"p50": pct(server_ms, 50), "p95": pct(server_ms, 95), "p99": pct(server_ms, 99)},
        "throughput_single_req_per_s": float(a.n / (sum(client_ms) / 1000)),
        "batch_100_client_ms": {"p50": pct(tb, 50), "p95": pct(tb, 95)},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
