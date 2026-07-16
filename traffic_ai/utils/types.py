from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    class_id: int = -1


@dataclass
class Track:
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    speed_kmh: float | None = None
    plate_text: str | None = None
    is_emergency: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameResult:
    frame_index: int
    timestamp_ms: float
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    vehicle_counts: dict[str, int] = field(default_factory=dict)
    frame: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalPhase:
    direction: str
    waiting_vehicles: int
    green_seconds: int


@dataclass
class SignalDecision:
    phases: list[SignalPhase]
    emergency_override: bool = False
    emergency_direction: str | None = None


@dataclass
class ViolationEvent:
    track_id: int
    violation_type: str
    plate_number: str | None
    confidence: float
    location: str
    speed_kmh: float | None = None
    evidence_frame: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)
