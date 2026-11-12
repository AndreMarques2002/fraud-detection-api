FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MODEL_DIR=/app/models/v1

# libgomp é necessária ao XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY models ./models

RUN useradd --create-home app && chown -R app:app /app
USER app

EXPOSE 8080
# Cloud Run injeta a variável PORT; localmente o padrão é 8080
CMD ["sh", "-c", "exec uvicorn fraud.api:app --host 0.0.0.0 --port ${PORT}"]
