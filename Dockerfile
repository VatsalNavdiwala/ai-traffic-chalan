FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV DEVICE=cpu
ENV YOLO_CONFIDENCE=0.35
EXPOSE 7860

CMD uvicorn traffic_ai.api.main:app --host 0.0.0.0 --port ${PORT:-7860}
