from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from loguru import logger

from traffic_ai.config.settings import get_settings
from traffic_ai.utils.types import Track


@dataclass
class SpeedReading:
    track_id: int
    speed_kmh: float
    source: str  # radar | stereo | dual_camera | single_camera_demo
    confidence: float


class SpeedEstimator(ABC):
    @abstractmethod
    def estimate(self, tracks: list[Track], timestamp_ms: float) -> list[SpeedReading]:
        ...


class RadarSpeedEstimator(SpeedEstimator):
    """Option 1 (best) — link AI track IDs to certified radar readings (~99.9%)."""

    def __init__(self) -> None:
        self._latest: dict[int, float] = {}

    def ingest_radar(self, track_id: int, speed_kmh: float) -> None:
        self._latest[track_id] = speed_kmh

    def estimate(self, tracks: list[Track], timestamp_ms: float) -> list[SpeedReading]:
        readings: list[SpeedReading] = []
        for t in tracks:
            if t.track_id in self._latest:
                speed = self._latest[t.track_id]
                t.speed_kmh = speed
                readings.append(
                    SpeedReading(t.track_id, speed, "radar", confidence=0.999)
                )
        return readings


class DualCameraSpeedEstimator(SpeedEstimator):
    """Option 3 — two cameras with known distance; speed = distance / Δt."""

    def __init__(self, distance_meters: float = 100.0) -> None:
        self.distance_meters = distance_meters
        self._entry_times: dict[int, float] = {}

    def mark_entry(self, track_id: int, timestamp_ms: float) -> None:
        self._entry_times[track_id] = timestamp_ms

    def mark_exit(self, track_id: int, timestamp_ms: float) -> SpeedReading | None:
        if track_id not in self._entry_times:
            return None
        dt_s = (timestamp_ms - self._entry_times[track_id]) / 1000.0
        if dt_s <= 0:
            return None
        speed = (self.distance_meters / dt_s) * 3.6
        return SpeedReading(track_id, speed, "dual_camera", confidence=0.98)

    def estimate(self, tracks: list[Track], timestamp_ms: float) -> list[SpeedReading]:
        # Dual-camera speed is event-driven via mark_entry / mark_exit.
        return []


class SingleCameraDemoEstimator(SpeedEstimator):
    """Option 4 — research/demo only. Not suitable for legal enforcement."""

    def __init__(self, meters_per_pixel: float = 0.05, fps: float = 30.0) -> None:
        self.meters_per_pixel = meters_per_pixel
        self.fps = fps
        self._prev_centroids: dict[int, tuple[float, float]] = {}
        logger.warning(
            "Single-camera speed is for demonstration only — not legally enforceable"
        )

    def estimate(self, tracks: list[Track], timestamp_ms: float) -> list[SpeedReading]:
        readings: list[SpeedReading] = []
        for t in tracks:
            cx = (t.bbox[0] + t.bbox[2]) / 2
            cy = (t.bbox[1] + t.bbox[3]) / 2
            if t.track_id in self._prev_centroids:
                px, py = self._prev_centroids[t.track_id]
                dist_px = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                speed = dist_px * self.meters_per_pixel * self.fps * 3.6
                t.speed_kmh = speed
                readings.append(
                    SpeedReading(t.track_id, speed, "single_camera_demo", confidence=0.5)
                )
            self._prev_centroids[t.track_id] = (cx, cy)
        return readings


def build_speed_estimator(mode: str | None = None) -> SpeedEstimator:
    mode = mode or get_settings().speed_mode
    if mode == "radar":
        return RadarSpeedEstimator()
    if mode == "dual_camera":
        return DualCameraSpeedEstimator()
    if mode in {"single_camera", "single_camera_demo"}:
        return SingleCameraDemoEstimator()
    logger.warning("Unknown speed_mode={}, defaulting to radar linker", mode)
    return RadarSpeedEstimator()
