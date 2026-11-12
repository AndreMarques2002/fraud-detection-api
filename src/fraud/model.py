"""Treino dos modelos (baseline logístico e XGBoost)."""
from __future__ import annotations

import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


def train_xgb(X_tr, y_tr, X_val, y_val, seed: int = RANDOM_STATE) -> xgb.XGBClassifier:
    """XGBoost com early stopping na validação TEMPORAL (PR-AUC).

    Sem pesos de classe: mantém as probabilidades próximas das taxas reais,
    o que importa para escolher limiares por custo.
    """
    model = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, eval_metric="aucpr", early_stopping_rounds=40,
        tree_method="hist", random_state=seed, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_logistic(X_tr, y_tr):
    """Baseline linear: imputação pela mediana (+ indicadores de ausência), padronização e logística."""
    pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(max_iter=2000))
    return pipe.fit(X_tr, y_tr)
