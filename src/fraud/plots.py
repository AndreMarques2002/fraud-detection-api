"""Gráficos salvos em reports/figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pr_curves(y, probs: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (PR-AUC {average_precision_score(y, p):.3f})")
    ax.axhline(np.mean(y), ls="--", c="gray", lw=1, label=f"aleatório ({np.mean(y):.3f})")
    ax.set(title="Curva Precisão-Recall (teste, fora do tempo)", xlabel="Recall", ylabel="Precisão")
    ax.legend()
    _save(fig, path)


def plot_cost_curve(curve: pd.DataFrame, n_days: int, review_thr_n: int, path: Path):
    step = max(len(curve) // 2000, 1)
    c = curve.iloc[::step]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(c["n_flagged"] / n_days, c["cost"] / 1000, c="#2f855a")
    ax.axhline(curve["cost"].iloc[0] / 1000, ls="--", c="gray", lw=1, label="sem modelo")
    ax.axvline(review_thr_n / n_days, ls=":", c="k", label="corte escolhido")
    ax.set_xlim(0, min(300, (len(curve) / n_days)))
    ax.set(title="Custo total vs. alertas por dia (validação)", xlabel="Alertas por dia",
           ylabel="Custo (mil u.m.)")
    ax.legend()
    _save(fig, path)


def plot_calibration(y, p, path: Path):
    frac, mean = calibration_curve(y, p, n_bins=12, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(mean, frac, marker="o", label="XGBoost")
    lim = max(mean.max(), frac.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", c="gray", lw=1)
    ax.set(title="Calibração (teste)", xlabel="Probabilidade prevista", ylabel="Taxa observada de fraude")
    ax.legend()
    _save(fig, path)


def plot_importance(series: pd.Series, labels: dict, path: Path):
    s = series.head(12).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh([labels[i] for i in s.index], s.values, color="#6b46c1")
    ax.set_title("Importância global: média do |efeito| na saída (log-odds)")
    _save(fig, path)


def plot_daily_precision(daily: pd.DataFrame, k: int, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in daily.columns:
        ax.plot(daily.index, daily[col], label=col)
    ax.set(title=f"Precisão nos {k} maiores scores de cada dia (teste)", xlabel="Dia", ylabel="Precisão@k",
           ylim=(0, 1.02))
    ax.legend()
    _save(fig, path)


def plot_score_distribution(y, p, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(p[y == 0], bins=bins, alpha=0.6, label="legítimas", log=True)
    ax.hist(p[y == 1], bins=bins, alpha=0.6, label="fraudes", log=True)
    ax.set(title="Distribuição do score por classe (teste, escala log)", xlabel="Probabilidade de fraude",
           ylabel="Transações")
    ax.legend()
    _save(fig, path)
