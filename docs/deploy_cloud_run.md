# Deploy no Google Cloud Run

> **Aviso:** estes comandos seguem o fluxo padrão do Cloud Run, mas **não foram executados por mim**.
> O CI do GitHub constrói a imagem e testa a API em container. Confira preços e cotas na documentação do Google Cloud antes de implantar.

## Pré-requisitos
- Conta Google Cloud com faturamento ativo e a CLI `gcloud` instalada.
- Um projeto (`PROJECT_ID`).

## 1. Configuração

```bash
gcloud auth login
gcloud config set project PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## 2. Deploy a partir do código-fonte (usa o `Dockerfile`)

```bash
gcloud run deploy fraud-api \
  --source . \
  --region southamerica-east1 \
  --memory 1Gi --cpu 1 \
  --min-instances 0 --max-instances 3 \
  --no-allow-unauthenticated
```

O Cloud Run injeta a variável `PORT`; o `Dockerfile` já a utiliza. O modelo (`models/v1`, ~1 MB) vai dentro da imagem.

## 3. Testar (com autenticação por IAM)

```bash
URL=$(gcloud run services describe fraud-api --region southamerica-east1 --format 'value(status.url)')
TOKEN=$(gcloud auth print-identity-token)

curl -H "Authorization: Bearer $TOKEN" $URL/health
curl -X POST $URL/score -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"customer_id": 0, "terminal_id": 6895, "amount": 310.0, "timestamp": "2018-10-01T03:12:00"}'
```

## Observações
- `--min-instances 0` reduz custo, mas a primeira requisição após ociosidade tem *cold start* (carregamento do modelo).
  Para latência estável, use `--min-instances 1`.
- Evite `--allow-unauthenticated` fora de uma demonstração.
- Para remover tudo: `gcloud run services delete fraud-api --region southamerica-east1`.
- Atualizar o modelo = retreinar, versionar `models/vN` e implantar uma nova revisão.
