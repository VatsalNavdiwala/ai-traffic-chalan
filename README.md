# AI Traffic Signal

End-to-end traffic intelligence: vehicle detection → tracking → speed → signal AI → violations → OCR → challan.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env

# API (Postgres optional for signal/preview endpoints)
uvicorn traffic_ai.api.main:app --reload --port 8000

# Pipeline — demo video (no webcam needed)
python -m traffic_ai.ai.pipeline --demo --display --max-frames 200

# Webcam / your own video
python -m traffic_ai.ai.pipeline --source 0 --display
python -m traffic_ai.ai.pipeline --source path\to\video.mp4 --max-frames 300

# Infra
docker compose up -d postgres redis rabbitmq
```

## Pipeline

```
Video → YOLOv11 → ByteTrack → Count → Speed → Violation → OCR → DB → Challan → SMS
```

## Phases

| Phase | Module | Status |
|------|--------|--------|
| 1 Detection | `ai/vehicle_detection` | Implemented (YOLOv11) |
| 2 Tracking | `ai/vehicle_tracking` | Implemented (ByteTrack + fallback) |
| 3 Speed | `ai/speed_detection` | Radar / dual-cam / demo single-cam |
| 4 Signal AI | `ai/signal_controller` | Optimization + emergency green |
| 5 Violations | `ai/violation` | Rule stubs (overspeed, red light, helmet) |
| 6 OCR | `ai/ocr` | YOLO plate crop + PaddleOCR |
| 7 Registry | `challan.RegistrationLookup` | Needs official VAHAN access |
| 8 Challan | `challan.ChallanService` | Draft + evidence + officer verify |
| 9 SMS | `challan.NoticeService` | Official gateway placeholders |

## Speed measurement

Legal enforcement requires **radar**, **stereo**, or **calibrated dual cameras**. Single-camera speed is demo-only.

## API examples

```bash
# PowerShell (use curl.exe or Invoke-RestMethod)
curl.exe http://127.0.0.1:8000/health

Invoke-RestMethod -Uri http://127.0.0.1:8000/signal/decide -Method POST `
  -ContentType "application/json" `
  -Body '{"north":45,"south":10,"east":6,"west":18}'

curl -X POST http://localhost:8000/challans/preview ^
  -H "Content-Type: application/json" ^
  -d "{\"plate_number\":\"GJ05AB1234\",\"violation_type\":\"red_light_jump\",\"location\":\"Ring Road\",\"fine_amount\":1000}"
```

## Hardware

Industrial fixed CCTV, NVIDIA GPU server, optional radar/LiDAR, traffic controller interface, UPS.
