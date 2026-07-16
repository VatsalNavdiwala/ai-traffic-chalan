from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from loguru import logger

from traffic_ai.ai.ocr import PlateOCR
from traffic_ai.ai.speed_detection import SingleCameraDemoEstimator
from traffic_ai.ai.vehicle_detection import VehicleDetector
from traffic_ai.ai.vehicle_tracking import VehicleTracker
from traffic_ai.ai.violation import ViolationDetector
from traffic_ai.ai.violation.detector import (
    NoHelmetRule,
    NoSeatBeltRule,
    OverspeedRule,
    RedLightJumpRule,
    StopLineCrossingRule,
    WrongSideRule,
)
from traffic_ai.challan import ChallanService
from traffic_ai.config.settings import get_settings
from traffic_ai.utils.types import ViolationEvent


def _encode_jpeg(frame: np.ndarray, max_w: int = 960) -> str:
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def draw_tracks(
    frame: np.ndarray,
    tracks: list,
    speed_limit_kmh: float = 60.0,
    highlight_id: int | None = None,
) -> np.ndarray:
    """Draw bounding boxes with track ID, class, plate, and speed."""
    vis = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        over = t.speed_kmh is not None and t.speed_kmh > speed_limit_kmh
        is_hi = highlight_id is not None and t.track_id == highlight_id
        if over:
            color = (40, 40, 220)  # red-ish BGR
        elif is_hi:
            color = (50, 220, 50)
        else:
            color = (60, 200, 120)

        thickness = 3 if is_hi or over else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        speed_txt = f"{t.speed_kmh:.0f} km/h" if t.speed_kmh is not None else "—"
        plate = t.plate_text or ""
        label = f"ID {t.track_id} {t.class_name} | {speed_txt}"
        if plate:
            label += f" | {plate}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        (tw, th), baseline = cv2.getTextSize(label, font, scale, 1)
        ty = max(0, y1 - 8)
        cv2.rectangle(vis, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(vis, label, (x1 + 4, ty - 2), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Speed badge on box corner
        badge = speed_txt
        (bw, bh), _ = cv2.getTextSize(badge, font, 0.6, 2)
        bx2, by1 = x2, y1
        cv2.rectangle(vis, (bx2 - bw - 10, by1), (bx2, by1 + bh + 10), color, -1)
        cv2.putText(
            vis,
            badge,
            (bx2 - bw - 5, by1 + bh + 4),
            font,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


@dataclass
class VehicleSummary:
    track_id: int
    vehicle_type: str
    plate_number: str | None
    max_speed_kmh: float | None
    frames_seen: int = 0
    evidence_jpeg_b64: str | None = None


@dataclass
class ChallanReceipt:
    challan_id: str
    plate_number: str
    registration_number: str
    vehicle_type: str
    violation: str
    location: str
    speed_kmh: float | None
    speed_limit_kmh: float
    fine_amount: float
    status: str
    occurred_at: str
    evidence_jpeg_b64: str | None = None
    officer_note: str = "Pending officer verification (demo)"


@dataclass
class DemoAnalyzeResult:
    location: str
    speed_limit_kmh: float
    frames_processed: int
    vehicles: list[VehicleSummary] = field(default_factory=list)
    primary_vehicle: VehicleSummary | None = None
    violations: list[dict[str, Any]] = field(default_factory=list)
    challans: list[ChallanReceipt] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    annotated_frame_jpeg_b64: str | None = None


class DemoVideoAnalyzer:
    """Upload-video demo: detect → track → OCR → speed → violations → challan."""

    def __init__(self) -> None:
        settings = get_settings()
        self.detector = VehicleDetector(confidence=0.25, device=settings.device)
        self.tracker = VehicleTracker()
        self.speed = SingleCameraDemoEstimator(meters_per_pixel=0.05, fps=25.0)
        self.ocr = PlateOCR()
        self.challan = ChallanService()
        self.violations = ViolationDetector(
            rules=[
                OverspeedRule(),
                RedLightJumpRule(),
                StopLineCrossingRule(),
                WrongSideRule(),
                NoHelmetRule(),
                NoSeatBeltRule(),
            ]
        )

    def analyze(
        self,
        video_path: str | Path,
        location: str = "Ring Road",
        speed_limit_kmh: float = 60.0,
        max_frames: int = 90,
        frame_stride: int = 2,
        run_ocr: bool = True,
    ) -> DemoAnalyzeResult:
        path = Path(video_path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.speed.fps = float(fps)
        notes = [
            "Single-camera speed is demonstration-only (not legally certified).",
            "Red-light / stop-line / wrong-side use demo geometry on the frame.",
            "Helmet / seat-belt use demo heuristics unless custom models are installed.",
            "Owner phone/address require official government registration API access.",
        ]

        track_stats: dict[int, dict[str, Any]] = {}
        prev_centroid: dict[int, tuple[float, float]] = {}
        crossed_stop: dict[int, bool] = {}
        wrong_side: dict[int, bool] = {}
        helmet_ok: dict[int, bool] = {}
        seatbelt_ok: dict[int, bool] = {}
        track_direction: dict[int, str] = {}
        seen_violations: set[tuple[int, str]] = set()
        violation_rows: list[dict[str, Any]] = []
        challans: list[ChallanReceipt] = []
        last_frame: np.ndarray | None = None
        best_annotated: np.ndarray | None = None
        best_annotated_score = -1

        frames_processed = 0
        frame_idx = 0

        while frames_processed < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            stop_y = int(h * 0.72)
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections, frame)
            ts = (frame_idx / fps) * 1000.0
            self.speed.estimate(tracks, ts)
            last_frame = frame

            helmet_ok_frame: dict[int, bool] = {}
            seatbelt_ok_frame: dict[int, bool] = {}

            for t in tracks:
                cx = (t.bbox[0] + t.bbox[2]) / 2
                cy = (t.bbox[1] + t.bbox[3]) / 2

                if t.track_id in prev_centroid:
                    px, py = prev_centroid[t.track_id]
                    dy = cy - py
                    # Crossing stop line downward while signal red (demo)
                    if py < stop_y <= cy:
                        crossed_stop[t.track_id] = True
                    # Wrong side demo: left half expects downward traffic
                    if cx < w * 0.5 and dy < -6:
                        wrong_side[t.track_id] = True
                    if abs(dy) > 3:
                        track_direction[t.track_id] = "south" if dy > 0 else "north"
                prev_centroid[t.track_id] = (cx, cy)

                # Helmet / seatbelt demo heuristics
                if t.class_name in {"bike", "bicycle"}:
                    helmet_ok_frame[t.track_id] = self._demo_helmet_ok(frame, t.bbox)
                    helmet_ok[t.track_id] = helmet_ok_frame[t.track_id]
                if t.class_name in {"car", "truck", "bus", "auto"}:
                    seatbelt_ok_frame[t.track_id] = self._demo_seatbelt_ok(frame, t.bbox)
                    seatbelt_ok[t.track_id] = seatbelt_ok_frame[t.track_id]

                stats = track_stats.setdefault(
                    t.track_id,
                    {
                        "vehicle_type": t.class_name,
                        "plate": None,
                        "speeds": [],
                        "frames": 0,
                        "best_frame": None,
                        "best_area": 0.0,
                        "best_bbox": None,
                    },
                )
                stats["frames"] += 1
                stats["vehicle_type"] = t.class_name
                if t.speed_kmh is not None:
                    stats["speeds"].append(float(t.speed_kmh))
                area = max(0.0, (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]))

                # OCR once per track when box is large enough
                if (
                    run_ocr
                    and not stats["plate"]
                    and area > (w * h * 0.01)
                    and frames_processed % 3 == 0
                ):
                    try:
                        plate, conf = self.ocr.read_from_vehicle_crop(frame, t.bbox)
                        if plate and conf >= 0.4 and len(plate) >= 6:
                            stats["plate"] = plate.upper().replace(" ", "")
                            t.plate_text = stats["plate"]
                    except Exception as exc:
                        logger.warning("OCR skipped for track {}: {}", t.track_id, exc)

                if stats["plate"]:
                    t.plate_text = stats["plate"]

            # Draw boxes + speed on this frame
            annotated = draw_tracks(frame, tracks, speed_limit_kmh=speed_limit_kmh)
            # Demo stop-line marker
            cv2.line(annotated, (0, stop_y), (w, stop_y), (0, 200, 255), 2)
            cv2.putText(
                annotated,
                "STOP LINE (demo)",
                (12, stop_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

            score = len(tracks) * 10 + sum(1 for t in tracks if t.speed_kmh is not None)
            if score >= best_annotated_score:
                best_annotated_score = score
                best_annotated = annotated.copy()

            for t in tracks:
                stats = track_stats[t.track_id]
                area = max(0.0, (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]))
                if area > stats["best_area"]:
                    stats["best_area"] = area
                    stats["best_bbox"] = t.bbox
                    # Per-vehicle evidence with boxes (highlight this track)
                    stats["best_frame"] = draw_tracks(
                        frame,
                        tracks,
                        speed_limit_kmh=speed_limit_kmh,
                        highlight_id=t.track_id,
                    )

            context = {
                "location": location,
                "speed_limit_kmh": speed_limit_kmh,
                "signal_state": {"north": "red", "south": "red", "east": "red", "west": "red"},
                "crossed_stop_line": crossed_stop,
                "wrong_side": wrong_side,
                "track_direction": track_direction,
                "helmet_ok": helmet_ok,
                "seatbelt_ok": seatbelt_ok,
            }
            events = self.violations.evaluate(tracks, frame, context)
            for ev in events:
                key = (ev.track_id, ev.violation_type)
                if key in seen_violations:
                    continue
                plate = ev.plate_number or track_stats.get(ev.track_id, {}).get("plate")
                ev.plate_number = plate
                seen_violations.add(key)
                # Annotated evidence for challan
                ev.evidence_frame = draw_tracks(
                    frame,
                    tracks,
                    speed_limit_kmh=speed_limit_kmh,
                    highlight_id=ev.track_id,
                )
                draft = self.challan.create_draft(ev)
                evidence_b64 = _encode_jpeg(ev.evidence_frame) if ev.evidence_frame is not None else None
                vtype = track_stats.get(ev.track_id, {}).get("vehicle_type", "vehicle")
                receipt = ChallanReceipt(
                    challan_id=str(uuid4())[:8].upper(),
                    plate_number=draft.plate_number,
                    registration_number=draft.plate_number,
                    vehicle_type=vtype,
                    violation=draft.violation_type,
                    location=location,
                    speed_kmh=draft.speed_kmh,
                    speed_limit_kmh=speed_limit_kmh,
                    fine_amount=draft.fine_amount,
                    status=draft.status,
                    occurred_at=datetime.utcnow().isoformat() + "Z",
                    evidence_jpeg_b64=evidence_b64,
                )
                challans.append(receipt)
                violation_rows.append(
                    {
                        "track_id": ev.track_id,
                        "violation": ev.violation_type,
                        "plate_number": draft.plate_number,
                        "speed_kmh": draft.speed_kmh,
                        "confidence": ev.confidence,
                        "challan_id": receipt.challan_id,
                    }
                )

            frames_processed += 1
            frame_idx += 1

        cap.release()

        vehicles: list[VehicleSummary] = []
        for tid, st in sorted(track_stats.items(), key=lambda x: -x[1]["frames"]):
            speeds = st["speeds"]
            max_speed = max(speeds) if speeds else None
            evidence = None
            if st.get("best_frame") is not None:
                evidence = _encode_jpeg(st["best_frame"])
            vehicles.append(
                VehicleSummary(
                    track_id=tid,
                    vehicle_type=st["vehicle_type"],
                    plate_number=st["plate"],
                    max_speed_kmh=round(max_speed, 1) if max_speed is not None else None,
                    frames_seen=st["frames"],
                    evidence_jpeg_b64=evidence,
                )
            )

        # Prefer vehicle with a plate, else most-seen
        primary = None
        with_plate = [v for v in vehicles if v.plate_number]
        if with_plate:
            primary = with_plate[0]
        elif vehicles:
            primary = vehicles[0]

        annotated_b64 = _encode_jpeg(best_annotated) if best_annotated is not None else None

        # If primary is overspeeding and no challan yet for it, force overspeed challan
        if primary and primary.max_speed_kmh and primary.max_speed_kmh > speed_limit_kmh:
            already = any(
                c.plate_number == (primary.plate_number or "UNKNOWN") and c.violation == "overspeed"
                for c in challans
            )
            if not already:
                evidence_frame = best_annotated if best_annotated is not None else last_frame
                ev = ViolationEvent(
                    track_id=primary.track_id,
                    violation_type="overspeed",
                    plate_number=primary.plate_number,
                    confidence=0.95,
                    location=location,
                    speed_kmh=primary.max_speed_kmh,
                    evidence_frame=evidence_frame.copy() if evidence_frame is not None else None,
                )
                draft = self.challan.create_draft(ev)
                receipt = ChallanReceipt(
                    challan_id=str(uuid4())[:8].upper(),
                    plate_number=draft.plate_number,
                    registration_number=draft.plate_number,
                    vehicle_type=primary.vehicle_type,
                    violation="overspeed",
                    location=location,
                    speed_kmh=primary.max_speed_kmh,
                    speed_limit_kmh=speed_limit_kmh,
                    fine_amount=draft.fine_amount,
                    status=draft.status,
                    occurred_at=datetime.utcnow().isoformat() + "Z",
                    evidence_jpeg_b64=primary.evidence_jpeg_b64 or annotated_b64,
                )
                challans.insert(0, receipt)
                violation_rows.insert(
                    0,
                    {
                        "track_id": primary.track_id,
                        "violation": "overspeed",
                        "plate_number": draft.plate_number,
                        "speed_kmh": primary.max_speed_kmh,
                        "confidence": 0.95,
                        "challan_id": receipt.challan_id,
                    },
                )

        return DemoAnalyzeResult(
            location=location,
            speed_limit_kmh=speed_limit_kmh,
            frames_processed=frames_processed,
            vehicles=vehicles,
            primary_vehicle=primary,
            violations=violation_rows,
            challans=challans,
            notes=notes,
            annotated_frame_jpeg_b64=annotated_b64,
        )

    @staticmethod
    def _demo_helmet_ok(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> bool:
        """Weak demo heuristic: darker upper region ≈ helmet-like; bright ≈ no helmet."""
        x1, y1, x2, y2 = map(int, bbox)
        h = max(1, y2 - y1)
        head = frame[y1 : y1 + max(8, h // 3), x1:x2]
        if head.size == 0:
            return True
        gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < 95

    @staticmethod
    def _demo_seatbelt_ok(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> bool:
        """Weak demo heuristic — default True to avoid mass false positives."""
        x1, y1, x2, y2 = map(int, bbox)
        cabin = frame[y1:y2, x1:x2]
        if cabin.size == 0:
            return True
        # Only flag rare high-contrast diagonal-less bright cabins
        gray = cv2.cvtColor(cabin, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        return float(np.mean(edges)) > 8
