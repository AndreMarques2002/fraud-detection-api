## Desempenho fora do tempo (período de teste)

| Modelo | PR-AUC | ROC-AUC | Precisão@100/dia | Valor fraudado capturado@100/dia |
|---|---|---|---|---|
| XGBoost | 0.748 | 0.899 | 0.626 | 88.6% |
| Regressão logística | 0.680 | 0.891 | 0.588 | 83.9% |
| Só o valor da transação | 0.241 | 0.646 | 0.202 | 65.7% |

## Política de decisão (limiares escolhidos na validação, aplicados no teste)

- Limiar de **revisão** (mínimo custo): 0.0306
- Limiar de **bloqueio** (precisão ≥ 90% na validação): 0.3760
- Alertas por dia: **139** (bloqueios: 63.5/dia)
- Precisão dos alertas: **47.0%** | recall em nº de fraudes: **77.2%**
- Valor fraudado capturado: **90.3%**
- Redução de custo vs. não ter modelo: **87.7%** (validação: 87.9%)

Premissas de custo (ilustrativas): cada alerta custa 2.0 u.m.; fraude não detectada custa o valor da transação.

## Recall por cenário de fraude (diagnóstico, teste)

| Cenário | Fraudes | Recall |
|---|---|---|
| 1 | 128 | 100.0% |
| 2 | 1407 | 69.2% |
| 3 | 662 | 89.9% |

(1 = valor alto, 2 = terminal comprometido, 3 = cartão comprometido; só para análise.)

## Estabilidade

PSI do score validação vs. teste: **0.0001** (estável)

| feature         |    psi | status   |
|:----------------|-------:|:---------|
| is_weekend      | 0.0017 | estável  |
| cust_nb_tx_30   | 0.0014 | estável  |
| cust_nb_tx_7    | 0.0008 | estável  |
| term_nb_tx_30   | 0.0008 | estável  |
| cust_avg_amt_30 | 0.0002 | estável  |
| cust_avg_amt_1  | 0.0002 | estável  |
| cust_avg_amt_7  | 0.0002 | estável  |
| term_nb_tx_7    | 0.0001 | estável  |

## Importância global (média do |efeito| em log-odds)

|                 |   importância |
|:----------------|--------------:|
| cust_avg_amt_30 |        0.9859 |
| cust_avg_amt_7  |        0.6956 |
| amount_ratio_30 |        0.2714 |
| amount          |        0.2558 |
| term_risk_30    |        0.2163 |
| term_risk_1     |        0.1512 |
| cust_avg_amt_1  |        0.141  |
| cust_nb_tx_30   |        0.1351 |
| term_nb_tx_7    |        0.1086 |
| term_nb_tx_30   |        0.108  |
