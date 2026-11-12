# Model Card: Scoring de fraude em transações de cartão

## Finalidade
Estimar a probabilidade de uma transação de cartão ser fraude no momento da autorização e sugerir uma ação
(**APPROVE**, **REVIEW** ou **BLOCK**), com os principais fatores da decisão.
**Projeto de portfólio treinado com dados simulados. Não deve ser usado em decisões reais.**

## Dados
- *Fraud Detection Handbook* (simulados): 1.754.155 transações, 4.990 clientes, 10.000 terminais, 01/04/2018 a 30/09/2018, 0,84% de fraude.
- Cortes temporais (dias desde o início): treino 37–112, validação 120–149, teste 157–182, com intervalo de 7 dias entre períodos.
- `TX_FRAUD_SCENARIO` **não** é usada como feature (só em diagnóstico).

## Features (17)
Transação (valor, hora, fim de semana, madrugada, valor relativo à média do cliente) · histórico do cliente
(nº de transações e valor médio em 1, 7 e 30 dias) · histórico do terminal (volume em 1, 7 e 30 dias e **taxa de fraude
defasada em 7 dias**, em janelas de 1, 7 e 30 dias). Todas calculadas apenas com dias anteriores ao da transação.

## Modelo
XGBoost (`binary:logistic`, early stopping na validação temporal por PR-AUC, sem pesos de classe).
Baselines: regressão logística e "score = valor da transação".

## Desempenho (teste, fora do tempo)
| Métrica | XGBoost | Regressão logística | Só o valor |
|---|---|---|---|
| PR-AUC | 0.748 | 0.680 | 0.241 |
| ROC-AUC | 0.899 | 0.891 | 0.646 |
| Precisão@100/dia | 0.626 | 0.588 | 0.202 |

Recall por cenário simulado no limiar de revisão: valor alto 100%; cartão comprometido 89.9%;
terminal comprometido 69.2%.

## Política de decisão
- Revisão: probabilidade ≥ 0.0306 (mínimo custo na validação).
- Bloqueio: probabilidade ≥ 0.3760 (precisão ≥ 90% na validação; no teste: 89.1%).
- No teste: ~139 alertas/dia, precisão dos alertas 47.0%, 90.3% do valor fraudado capturado.
- Premissas de custo **ilustrativas** (alerta = 2 u.m.; fraude não detectada = valor da transação).

## Uso previsto e fora de escopo
- Previsto: demonstração técnica de modelagem, validação temporal e serviço de scoring.
- Fora de escopo: decisão automática real, qualquer uso sem validação independente e retreino com dados reais.

## Limitações
- Dados simulados: padrões de fraude mais simples e estáveis que os reais; não há adaptação do fraudador.
- Features diárias: não captura rajadas no mesmo dia; terminais recém-comprometidos são detectados com atraso (defasagem de 7 dias).
- Clientes/terminais sem histórico (*cold start*) são pontuados com menos informação e sinalizados na resposta.
- Sem análise de equidade; custo de atrito com cliente legítimo não modelado.

## Monitoramento recomendado
PSI do score e das features a cada período (gatilhos 0,10 e 0,25); taxa de alertas por dia; precisão dos alertas após
a confirmação dos rótulos (7+ dias); distribuição de *cold start*; recalibração e retreino periódicos.

## Governança e segurança
- Artefato sem pickle (`model.json`, `metadata.json`, `profiles.npz`) com versão, data do snapshot, limiares e métricas.
- Entrada validada (Pydantic), limite de lote (500) e logs sem dados pessoais (apenas `transaction_id`).
- Decisões sobre clientes exigem transparência e possibilidade de revisão (por exemplo, o direito à revisão previsto na LGPD);
  os fatores retornados pela API apoiam essa justificativa.
