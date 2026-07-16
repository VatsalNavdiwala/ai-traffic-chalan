FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Slim deps for Render Free RAM limits (avoid Paddle OOM / 502)
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

COPY . .

ENV PYTHONPATH=/app
ENV DEVICE=cpu
ENV YOLO_CONFIDENCE=0.25
EXPOSE 8000

# Longer keepalive; Render still has platform request limits
CMD uvicorn traffic_ai.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 75
