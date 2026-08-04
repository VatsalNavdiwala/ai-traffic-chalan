from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
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
    id: int
    type: str
    confidence: float
    box: list[int]
    speed_kmh: float | None = None


class ChallanOut(BaseModel):
    id: str
    registration_number: str
    vehicle_type: str
    violation: str
    fine_amount: float
    speed_kmh: float | None = None
    speed_limit_kmh: float
    location: str
    occurred_at: str
    officer_note: str


class DemoAnalyzeResponse(BaseModel):
    location: str
    speed_limit_kmh: float
    frames_processed: int
    vehicles: list[VehicleOut] = Field(default_factory=list)
    primary_vehicle: VehicleOut | None = None
    violations: list[dict] = Field(default_factory=list)
    challans: list[ChallanOut] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    annotated_frame_jpeg_b64: str | None = None


@router.get("/analyze")
@router.get("/demo/analyze")
async def demo_analyze_info():
    return {
        "status": "ok",
        "endpoint": "/demo/analyze",
        "message": "Send a POST request with video file upload (mp4/avi/mov/webm) to analyze traffic video.",
    }


@router.post("/analyze", response_model=DemoAnalyzeResponse)
@router.post("/demo/analyze", response_model=DemoAnalyzeResponse)
@router.post("/", response_model=DemoAnalyzeResponse)
@router.post("", response_model=DemoAnalyzeResponse)
async def analyze_traffic_video(
    video: UploadFile = File(..., description="Traffic road video (mp4/avi/mov/webm)"),
    location: str = Form("Ring Road"),
    speed_limit_kmh: float = Form(60.0),
    max_frames: int = Form(20),
    run_ocr: str = Form("false"),
) -> DemoAnalyzeResponse:
    suffix = Path(video.filename or "upload.mp4").suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, "Upload a video file (mp4, avi, mov, mkv, webm)")

    ocr_enabled = str(run_ocr).lower() in {"1", "true", "yes", "on"}

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp.name)
    size = 0
    max_bytes = 100 * 1024 * 1024
    try:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(400, "Video too large for free cloud tier (max 100 MB)")
            tmp.write(chunk)
        tmp.close()
        if size == 0:
            raise HTTPException(400, "Empty upload")

        # Open video to get frame count and FPS
        cap = cv2.VideoCapture(str(tmp_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        duration_sec = total_video_frames / fps if (total_video_frames > 0 and fps > 0) else 5.0

        # Calculate max frames dynamically according to video length
        if os.getenv("VERCEL"):
            computed_frames = int(duration_sec * 4)
            max_frames_to_run = max(8, min(computed_frames, 30))
        else:
            computed_frames = int(duration_sec * 6)
            max_frames_to_run = max(10, min(computed_frames, 60))

        analyzer = get_analyzer()
        result = await asyncio.to_thread(
            analyzer.analyze,
            str(tmp_path),
            location.strip() or "Ring Road",
            float(speed_limit_kmh),
            max_frames_to_run,
            4,  # frame_stride
            ocr_enabled,
        )
    except HTTPException:
        raise
    except MemoryError as exc:
        logger.error(f"Memory error during video analysis: {exc}")
        raise HTTPException(500, "Video processing ran out of memory. Upload a smaller video clip.") from exc
    except Exception as exc:
        logger.exception("Demo video processing error")
        raise HTTPException(500, f"Analysis error: {exc}") from exc
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass

    return DemoAnalyzeResponse(
        location=result.get("location", location),
        speed_limit_kmh=result.get("speed_limit_kmh", speed_limit_kmh),
        frames_processed=result.get("frames_processed", 0),
        vehicles=result.get("vehicles", []),
        primary_vehicle=result.get("primary_vehicle"),
        violations=result.get("violations", []),
        challans=result.get("challans", []),
        notes=result.get("notes", []),
        annotated_frame_jpeg_b64=result.get("annotated_frame_jpeg_b64"),
    )
