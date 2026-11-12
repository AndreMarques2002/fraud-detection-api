.PHONY: install data train demo test api benchmark docker-build docker-run app

install:
	pip install -r requirements.txt && pip install -e .

# Baixa os dados simulados do Fraud Detection Handbook (~100 MB) e consolida em parquet
data:
	python scripts/download_data.py

# Treina, avalia fora do tempo e gera models/v1 e reports/
train:
	python -m fraud.train --data data/raw/transactions.parquet

# Demonstração com dados sintéticos (NÃO usar os números como resultado)
demo:
	python -m fraud.train --synthetic --model-dir /tmp/fraud_demo_model --report-dir /tmp/fraud_demo_reports

test:
	pytest -q

api:
	MODEL_DIR=models/v1 uvicorn fraud.api:app --port 8080

benchmark:
	python scripts/benchmark_api.py

docker-build:
	docker build -t fraud-api .

docker-run:
	docker run --rm -p 8080:8080 fraud-api

app:
	streamlit run app/streamlit_app.py
