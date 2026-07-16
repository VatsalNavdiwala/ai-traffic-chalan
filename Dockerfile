FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV DEVICE=cpu
EXPOSE 8000

# Render injects $PORT — bind to it
CMD uvicorn traffic_ai.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
