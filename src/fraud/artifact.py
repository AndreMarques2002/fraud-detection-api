"""Artefato do modelo SEM pickle: model.json (XGBoost) + metadata.json + profiles.npz."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb


def save_artifact(out_dir: Path, booster: xgb.Booster, metadata: dict,
                  customer_ids: np.ndarray, terminal_ids: np.ndarray, profiles: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(out_dir / "model.json")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out_dir / "profiles.npz", customer_ids=customer_ids,
                        terminal_ids=terminal_ids, **profiles)


def load_artifact(model_dir: Path):
    model_dir = Path(model_dir)
    booster = xgb.Booster()
    booster.load_model(model_dir / "model.json")
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(model_dir / "profiles.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return booster, meta, arrays
