from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from traffic_ai.ai.demo_analyzer import DemoVideoAnalyzer

router = APIRouter(prefix="/demo", tags=["demo"])

_analyzer: DemoVideoAnalyzer | None = None


def get_analyzer() -> DemoVideoAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = DemoVideoAnalyzer()
    return _analyzer


class VehicleOut(BaseModel):
    track_id: int
    vehicle_type: str
    plate_number: str | None = None
    max_speed_kmh: float | None = None
    frames_seen: int = 0
    evidence_jpeg_b64: str | None = None


class ChallanOut(BaseModel):
    challan_id: str
    plate_number: str
    registration_number: str
    vehicle_type: str
    violation: str
    location: str
    speed_kmh: float | None = None
    speed_limit_kmh: float
    fine_amount: float
    status: str
    occurred_at: str
    evidence_jpeg_b64: str | None = None
    officer_note: str = "Pending officer verification (demo)"


class DemoAnalyzeResponse(BaseModel):
    location: str
    speed_limit_kmh: float
    frames_processed: int
    vehicles: list[VehicleOut]
    primary_vehicle: VehicleOut | None = None
    violations: list[dict] = Field(default_factory=list)
    challans: list[ChallanOut] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


@router.post("/analyze", response_model=DemoAnalyzeResponse)
async def analyze_traffic_video(
    video: UploadFile = File(..., description="Traffic road video (mp4/avi/mov)"),
    location: str = Form("Ring Road"),
    speed_limit_kmh: float = Form(60.0),
    max_frames: int = Form(60),
    run_ocr: str = Form("true"),
) -> DemoAnalyzeResponse:
    suffix = Path(video.filename or "upload.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, "Upload a video file (mp4, avi, mov, mkv, webm)")

    # Keep Render free-tier responsive
    max_frames = max(10, min(int(max_frames), 90))
    ocr_enabled = str(run_ocr).lower() in {"1", "true", "yes", "on"}

    raw = await video.read()
    if not raw:
        raise HTTPException(400, "Empty upload")
    if len(raw) > 1024 * 1024 * 1024:
        raise HTTPException(400, "Video too large (max 1 GB)")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(raw)
        tmp.close()
        result = get_analyzer().analyze(
            tmp.name,
            location=location.strip() or "Ring Road",
            speed_limit_kmh=float(speed_limit_kmh),
            max_frames=max_frames,
            frame_stride=2,
            run_ocr=ocr_enabled,
        )
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}") from exc
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    def vehicle_out(v) -> VehicleOut | None:
        if v is None:
            return None
        return VehicleOut(
            track_id=v.track_id,
            vehicle_type=v.vehicle_type,
            plate_number=v.plate_number,
            max_speed_kmh=v.max_speed_kmh,
            frames_seen=v.frames_seen,
            evidence_jpeg_b64=v.evidence_jpeg_b64,
        )

    vehicles = [vehicle_out(v) for v in result.vehicles]
    return DemoAnalyzeResponse(
        location=result.location,
        speed_limit_kmh=result.speed_limit_kmh,
        frames_processed=result.frames_processed,
        vehicles=[v for v in vehicles if v is not None],
        primary_vehicle=vehicle_out(result.primary_vehicle),
        violations=result.violations,
        challans=[
            ChallanOut(
                challan_id=c.challan_id,
                plate_number=c.plate_number,
                registration_number=c.registration_number,
                vehicle_type=c.vehicle_type,
                violation=c.violation,
                location=c.location,
                speed_kmh=c.speed_kmh,
                speed_limit_kmh=c.speed_limit_kmh,
                fine_amount=c.fine_amount,
                status=c.status,
                occurred_at=c.occurred_at,
                evidence_jpeg_b64=c.evidence_jpeg_b64,
                officer_note=c.officer_note,
            )
            for c in result.challans
        ],
        notes=result.notes,
    )
