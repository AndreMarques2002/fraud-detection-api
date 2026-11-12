"""Demo interativa: pontua uma transação com o mesmo Scorer usado pela API.

    streamlit run app/streamlit_app.py
"""
import json
import os
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st

from fraud.serving import Scorer

MODEL_DIR = os.getenv("MODEL_DIR", "models/v1")
EXAMPLES = Path("reports/demo_examples.json")

st.set_page_config(page_title="Fraud Scoring", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_scorer() -> Scorer:
    return Scorer.from_dir(MODEL_DIR)


scorer = load_scorer()
examples = json.loads(EXAMPLES.read_text(encoding="utf-8")) if EXAMPLES.exists() else {}

st.title("🛡️ Scoring de fraude em transações")
st.caption(f"Modelo {scorer.version} · perfis de {scorer.meta['snapshot_date']} · "
           f"dados {'sintéticos' if scorer.meta.get('synthetic_data') else 'simulados (Fraud Detection Handbook)'}")

with st.form("tx"):
    c1, c2 = st.columns(2)
    customer_id = c1.number_input("ID do cliente", min_value=0, value=int(next(iter(scorer._cust_idx))), step=1)
    presets = {f"Terminal {t} (maior taxa de fraude recente)": t for t in examples.get("risky_terminals", [])}
    presets.update({f"Terminal {t} (baixo risco)": t for t in examples.get("safe_terminals", [])})
    choice = c2.selectbox("Terminal", ["Digitar manualmente"] + list(presets))
    manual = c2.number_input("ID do terminal (se manual)", min_value=0, value=int(next(iter(scorer._term_idx))), step=1)
    amount = c1.number_input("Valor", min_value=0.0, value=49.9, step=10.0)
    d = c2.date_input("Data", value=datetime.fromisoformat(scorer.meta["snapshot_date"]).date())
    t = c1.time_input("Hora", value=time(14, 30))
    submitted = st.form_submit_button("Pontuar")

if submitted:
    terminal_id = presets.get(choice, manual)
    res = scorer.score([{"customer_id": customer_id, "terminal_id": int(terminal_id), "amount": amount,
                         "timestamp": datetime.combine(d, t)}])[0]
    badge = {"APPROVE": "🟢 APROVAR", "REVIEW": "🟠 REVISAR", "BLOCK": "🔴 BLOQUEAR"}[res["decision"]]
    m1, m2 = st.columns(2)
    m1.metric("Probabilidade de fraude", f"{res['fraud_probability']:.2%}")
    m2.metric("Decisão", badge)
    if res["cold_start"]["customer"] or res["cold_start"]["terminal"]:
        st.warning("Cliente ou terminal sem histórico no snapshot (cold start): a estimativa é menos confiável.")
    st.subheader("Principais fatores")
    st.dataframe(pd.DataFrame(res["reasons"])[["label", "value", "impact", "direction"]]
                 .rename(columns={"label": "Fator", "value": "Valor", "impact": "Impacto (log-odds)",
                                  "direction": "Efeito"}), hide_index=True)
    st.caption(f"Limiares: revisão ≥ {scorer.review_threshold:.3f}"
               + (f" · bloqueio ≥ {scorer.block_threshold:.3f}" if scorer.block_threshold else ""))
