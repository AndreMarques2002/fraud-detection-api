# Dados

Este projeto usa o **simulador de transações do *Fraud Detection Handbook*** (Le Borgne et al.), com
**1.754.155 transações simuladas** de 4.990 clientes e 10.000 terminais (01/04/2018 a 30/09/2018), das quais 0,84% são fraudes.

- Fonte: <https://github.com/Fraud-Detection-Handbook/simulated-data-raw>
- Livro/documentação: <https://fraud-detection-handbook.github.io/fraud-detection-handbook/>
- **São dados SIMULADOS**, não transações reais. Verifique os termos de uso do repositório de origem; os dados **não são redistribuídos** aqui.

## Como baixar

```bash
python scripts/download_data.py        # gera data/raw/transactions.parquet
```

> Os arquivos de origem são `.pkl` (pickle), que pode executar código ao ser lido. O script só deve ser usado com a fonte oficial acima.

## Esquema

| Coluna | Descrição |
|---|---|
| `TRANSACTION_ID` | Identificador da transação |
| `TX_DATETIME` | Data e hora |
| `CUSTOMER_ID` | Cliente (portador do cartão) |
| `TERMINAL_ID` | Terminal / estabelecimento |
| `TX_AMOUNT` | Valor |
| `TX_FRAUD` | Alvo: 1 = fraude |
| `TX_FRAUD_SCENARIO` | Cenário de fraude simulado (0 = legítima) |

## Cenários de fraude do simulador

1. **Valor alto:** transações acima de um valor fixo são fraude.
2. **Terminal comprometido:** durante 28 dias, todas as transações de um terminal são fraude.
3. **Cartão comprometido (card-not-present):** durante 14 dias, 1/3 das transações do cliente são fraude, com valor multiplicado por 5.

`TX_FRAUD_SCENARIO` **não** é usada como feature (não existiria em produção).
