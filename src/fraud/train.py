"""Pipeline de treino, avaliação fora do tempo e geração do artefato de produção.

Uso:
    python -m fraud.train --data data/raw/transactions.parquet
    python -m fraud.train --synthetic      # demonstração, NÃO reportar resultados
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import __version__, plots
from .artifact import save_artifact
from .config import (FEATURE_LABELS, FEATURES, RANDOM_STATE, TEST_DAYS, TOP_K_PER_DAY, TRAIN_DAYS,
                     VAL_DAYS, CostAssumptions)
from .data import load_transactions, make_synthetic
from .features import build_feature_frame, snapshot
from .metrics import (best_review_threshold, block_threshold, cost_curve, evaluate, policy_report,
                      precision_at_k_per_day)
from .explain import global_importance
from .model import train_logistic, train_xgb
from .psi import feature_psi, psi, psi_status

log = logging.getLogger("fraud")


def _mask(day, bounds):
    return (day >= bounds[0]) & (day < bounds[1])


def run(df: pd.DataFrame, model_dir: Path, report_dir: Path, *, data_source: str,
        synthetic: bool = False, seed: int = RANDOM_STATE) -> dict:
    costs = CostAssumptions()
    fig_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)

    X, day, hist = build_feature_frame(df)
    y = df["TX_FRAUD"].to_numpy()
    amount = df["TX_AMOUNT"].to_numpy(dtype=float)
    tr, va, te = _mask(day, TRAIN_DAYS), _mask(day, VAL_DAYS), _mask(day, TEST_DAYS)
    log.info("Treino: %d tx (%.2f%% fraude) | Validação: %d | Teste: %d (%.2f%% fraude)",
             tr.sum(), 100 * y[tr].mean(), va.sum(), te.sum(), 100 * y[te].mean())

    # ---- Modelos ----
    xgb_model = train_xgb(X[tr], y[tr], X[va], y[va], seed)
    lr_model = train_logistic(X[tr], y[tr])
    p = {
        "XGBoost": {s: xgb_model.predict_proba(X[m])[:, 1] for s, m in (("val", va), ("test", te))},
        "Regressão logística": {s: lr_model.predict_proba(X[m])[:, 1] for s, m in (("val", va), ("test", te))},
        "Só o valor da transação": {s: amount[m] for s, m in (("val", va), ("test", te))},
    }

    # ---- Métricas fora do tempo ----
    metrics = {"synthetic_data": synthetic, "best_iteration": int(xgb_model.best_iteration), "models": {}}
    for name, d in p.items():
        metrics["models"][name] = {
            "val": evaluate(y[va], d["val"], day[va], amount[va]),
            "test": evaluate(y[te], d["test"], day[te], amount[te]),
        }

    # ---- Limiares escolhidos na VALIDAÇÃO e aplicados no TESTE ----
    review_thr = best_review_threshold(y[va], amount[va], p["XGBoost"]["val"], costs)
    block_thr = block_threshold(y[va], p["XGBoost"]["val"], costs)
    policy_val = policy_report(y[va], p["XGBoost"]["val"], amount[va], day[va], review_thr, block_thr, costs)
    policy_test = policy_report(y[te], p["XGBoost"]["test"], amount[te], day[te], review_thr, block_thr, costs)
    metrics["policy_val"], metrics["policy_test"] = policy_val, policy_test
    metrics["cost_assumptions"] = vars(costs)
    if "TX_FRAUD_SCENARIO" in df.columns:   # só diagnóstico: o cenário NUNCA é usado como feature
        scen = df["TX_FRAUD_SCENARIO"].to_numpy()[te]
        flagged_te = p["XGBoost"]["test"] >= review_thr
        metrics["recall_by_scenario_test"] = {
            int(sc): {"n_frauds": int(((scen == sc) & (y[te] == 1)).sum()),
                      "recall": float(flagged_te[(scen == sc) & (y[te] == 1)].mean())}
            for sc in np.unique(scen) if sc > 0}

    # ---- Estabilidade (PSI) e importância ----
    score_psi = psi(p["XGBoost"]["val"], p["XGBoost"]["test"])
    fpsi = feature_psi(X[tr], X[te])
    metrics["psi_score_val_vs_test"] = {"psi": score_psi, "status": psi_status(score_psi)}
    imp = global_importance(xgb_model.get_booster(), X[te])

    # ---- Gráficos ----
    curve_val = cost_curve(y[va], amount[va], p["XGBoost"]["val"], costs.review_cost)
    n_flag_val = int((p["XGBoost"]["val"] >= review_thr).sum())
    plots.plot_pr_curves(y[te], {k: v["test"] for k, v in p.items()}, fig_dir / "pr_curves.png")
    plots.plot_cost_curve(curve_val, len(np.unique(day[va])), n_flag_val, fig_dir / "custo_vs_alertas.png")
    plots.plot_calibration(y[te], p["XGBoost"]["test"], fig_dir / "calibracao.png")
    plots.plot_importance(imp, FEATURE_LABELS, fig_dir / "importancia_global.png")
    plots.plot_score_distribution(y[te], p["XGBoost"]["test"], fig_dir / "distribuicao_score.png")
    daily = {}
    for name, d in p.items():
        t = pd.DataFrame({"y": y[te], "p": d["test"], "d": day[te]})
        daily[name] = t.groupby("d").apply(lambda g: g.nlargest(TOP_K_PER_DAY, "p")["y"].mean(),
                                           include_groups=False)
    plots.plot_daily_precision(pd.DataFrame(daily), TOP_K_PER_DAY, fig_dir / "precisao_por_dia.png")

    # ---- Artefato de produção (perfis como de "hoje" = dia seguinte ao último da base) ----
    snap_day = hist.n_days
    profiles = snapshot(hist)
    metadata = {
        "model_version": f"v{__version__}",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": data_source,
        "synthetic_data": synthetic,
        "snapshot_date": str((hist.start + pd.Timedelta(days=snap_day)).date()),
        "features": FEATURES,
        "thresholds": {"review": review_thr, "block": block_thr},
        "cost_assumptions": vars(costs),
        "train_days": TRAIN_DAYS, "val_days": VAL_DAYS, "test_days": TEST_DAYS,
        "test_metrics": metrics["models"]["XGBoost"]["test"],
        "test_policy": policy_test,
        "libs": {"xgboost": xgb.__version__, "numpy": np.__version__, "python": platform.python_version()},
    }
    save_artifact(model_dir, xgb_model.get_booster(), metadata, hist.customer_ids, hist.terminal_ids, profiles)

    # Exemplos para a demo (agregados do snapshot, não são linhas da base)
    t_risk = pd.Series(profiles["term_risk_30"], index=hist.terminal_ids).dropna().sort_values(ascending=False)
    examples = {"risky_terminals": [int(i) for i in t_risk.head(5).index],
                "safe_terminals": [int(i) for i in t_risk.tail(5).index],
                "customers": [int(i) for i in hist.customer_ids[:5]],
                "snapshot_date": metadata["snapshot_date"]}
    (report_dir / "demo_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")

    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    fpsi.to_csv(report_dir / "feature_psi.csv", index=False)
    imp.rename("importancia").to_csv(report_dir / "feature_importance.csv")
    (report_dir / "results.md").write_text(render_results(metrics, imp, fpsi), encoding="utf-8")
    return metrics


def render_results(m: dict, imp: pd.Series, fpsi: pd.DataFrame) -> str:
    k = TOP_K_PER_DAY
    rows = [f"| Modelo | PR-AUC | ROC-AUC | Precisão@{k}/dia | Valor fraudado capturado@{k}/dia |",
            "|---|---|---|---|---|"]
    for name, d in m["models"].items():
        t = d["test"]
        rows.append(f"| {name} | {t['pr_auc']:.3f} | {t['roc_auc']:.3f} | {t[f'precision_at_{k}_per_day']:.3f} | "
                    f"{100 * t[f'fraud_amount_captured_at_{k}_per_day']:.1f}% |")
    pt, pv = m["policy_test"], m["policy_val"]
    block = pt["block_threshold"]
    lines = []
    if m["synthetic_data"]:
        lines.append("> ⚠️ **Gerado com dados SINTÉTICOS (demonstração). Não use estes números como resultado.**\n")
    lines += [
        "## Desempenho fora do tempo (período de teste)\n", *rows,
        "\n## Política de decisão (limiares escolhidos na validação, aplicados no teste)\n",
        f"- Limiar de **revisão** (mínimo custo): {pt['review_threshold']:.4f}",
        f"- Limiar de **bloqueio** (precisão ≥ {m['cost_assumptions']['target_block_precision']:.0%} na validação): "
        + ("não existe" if block is None else f"{block:.4f}"),
        f"- Alertas por dia: **{pt['alerts_per_day']:.0f}** (bloqueios: {pt['blocked_per_day']:.1f}/dia)",
        f"- Precisão dos alertas: **{pt['precision_flagged']:.1%}** | recall em nº de fraudes: **{pt['recall_count']:.1%}**",
        f"- Valor fraudado capturado: **{pt['fraud_amount_caught_pct']:.1%}**",
        f"- Redução de custo vs. não ter modelo: **{pt['cost_reduction_pct']:.1%}** "
        f"(validação: {pv['cost_reduction_pct']:.1%})",
        f"\nPremissas de custo (ilustrativas): cada alerta custa {m['cost_assumptions']['review_cost']} u.m.; "
        "fraude não detectada custa o valor da transação.",
        *(["\n## Recall por cenário de fraude (diagnóstico, teste)\n",
           "| Cenário | Fraudes | Recall |", "|---|---|---|",
           *[f"| {k} | {v['n_frauds']} | {v['recall']:.1%} |" for k, v in m["recall_by_scenario_test"].items()],
           "\n(1 = valor alto, 2 = terminal comprometido, 3 = cartão comprometido; só para análise.)"]
          if "recall_by_scenario_test" in m else []),
        f"\n## Estabilidade\n\nPSI do score validação vs. teste: **{m['psi_score_val_vs_test']['psi']:.4f}** "
        f"({m['psi_score_val_vs_test']['status']})\n",
        fpsi.head(8).round(4).to_markdown(index=False),
        "\n## Importância global (média do |efeito| em log-odds)\n",
        imp.head(10).round(4).rename("importância").to_markdown(),
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="transactions.parquet gerado por scripts/download_data.py")
    src.add_argument("--synthetic", action="store_true", help="dados sintéticos (apenas demonstração)")
    ap.add_argument("--model-dir", type=Path, default=Path("models/v1"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    if a.synthetic:
        df, source = make_synthetic(seed=a.seed), "gerador sintético simplificado (demonstração)"
    else:
        df, source = load_transactions(a.data), "Fraud Detection Handbook (dados simulados) - ver data/README.md"
    m = run(df, a.model_dir, a.report_dir, data_source=source, synthetic=a.synthetic, seed=a.seed)

    t = m["models"]["XGBoost"]["test"]
    print(f"\nXGBoost (teste, fora do tempo): PR-AUC={t['pr_auc']:.3f} ROC-AUC={t['roc_auc']:.3f} "
          f"P@{TOP_K_PER_DAY}/dia={t[f'precision_at_{TOP_K_PER_DAY}_per_day']:.3f}")
    print(f"Redução de custo no teste: {m['policy_test']['cost_reduction_pct']:.1%}")
    print(f"Artefato em {a.model_dir} | relatório em {a.report_dir}/results.md")


if __name__ == "__main__":
    main()
