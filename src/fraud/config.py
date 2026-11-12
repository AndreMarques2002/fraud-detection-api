"""Configurações centrais: janelas, defasagem de rótulos, cortes temporais e premissas de custo."""
from dataclasses import dataclass

WINDOWS = (1, 7, 30)          # janelas (em dias) das features de histórico
DELAY_DAYS = 7                # defasagem até o rótulo de fraude ser conhecido (chargeback/denúncia)

# Cortes temporais, em dias desde o início da base: [início, fim)
# Entre treino, validação e teste há um intervalo de DELAY_DAYS: os rótulos mais recentes
# de cada período ainda não estariam disponíveis em produção.
TRAIN_DAYS = (37, 113)
VAL_DAYS = (120, 150)
TEST_DAYS = (157, 183)

TOP_K_PER_DAY = 100           # capacidade de análise: alertas investigados por dia
RANDOM_STATE = 42


@dataclass(frozen=True)
class CostAssumptions:
    """Premissas ILUSTRATIVAS de custo (unidades monetárias da base).

    - Fraude não detectada: perde-se o valor da transação.
    - Cada alerta (revisão ou bloqueio) custa `review_cost`, seja fraude ou não.
    """

    review_cost: float = 2.0
    target_block_precision: float = 0.90   # precisão mínima exigida para bloqueio automático
    min_block_alerts: int = 50             # evita limiar de bloqueio baseado em poucos casos


FEATURES = [
    "amount", "hour", "is_weekend", "is_night", "amount_ratio_30",
    "cust_nb_tx_1", "cust_nb_tx_7", "cust_nb_tx_30",
    "cust_avg_amt_1", "cust_avg_amt_7", "cust_avg_amt_30",
    "term_nb_tx_1", "term_nb_tx_7", "term_nb_tx_30",
    "term_risk_1", "term_risk_7", "term_risk_30",
]

FEATURE_LABELS = {
    "amount": "Valor da transação",
    "hour": "Hora do dia",
    "is_weekend": "Fim de semana",
    "is_night": "Madrugada",
    "amount_ratio_30": "Valor vs. média do cliente (30 dias)",
    "cust_nb_tx_1": "Transações do cliente (1 dia)",
    "cust_nb_tx_7": "Transações do cliente (7 dias)",
    "cust_nb_tx_30": "Transações do cliente (30 dias)",
    "cust_avg_amt_1": "Valor médio do cliente (1 dia)",
    "cust_avg_amt_7": "Valor médio do cliente (7 dias)",
    "cust_avg_amt_30": "Valor médio do cliente (30 dias)",
    "term_nb_tx_1": "Transações do terminal (1 dia)",
    "term_nb_tx_7": "Transações do terminal (7 dias)",
    "term_nb_tx_30": "Transações do terminal (30 dias)",
    "term_risk_1": "Taxa de fraude do terminal (1 dia, defasada)",
    "term_risk_7": "Taxa de fraude do terminal (7 dias, defasada)",
    "term_risk_30": "Taxa de fraude do terminal (30 dias, defasada)",
}
