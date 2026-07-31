from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from traffic_ai.config.settings import get_settings
from traffic_ai.utils.types import Track, ViolationEvent


class ViolationRule(ABC):
    name: str

    @abstractmethod
    def check(
        self,
        track: Track,
        frame: np.ndarray,
        context: dict,
    ) -> ViolationEvent | None:
        ...


class OverspeedRule(ViolationRule):
    name = "overspeed"

    def check(self, track: Track, frame: np.ndarray, context: dict) -> ViolationEvent | None:
        limit = float(context.get("speed_limit_kmh", get_settings().speed_limit_kmh))
        if track.speed_kmh is None:
            return None

        # Verification Officer Rule — Tolerance Filter:
        # Speed 60 - 65 km/h -> Warning zone (REJECT CHALLAN)
        # Speed > 65.0 km/h -> Eligible for Challan
        tolerance_threshold = limit + 5.0
        if track.speed_kmh <= tolerance_threshold:
            return None

        # Verification Officer Rule — Multi-frame Verification:
        # Ignore single-frame detections; vehicle must be tracked across multiple frames
        frames_seen = context.get("track_frames", {}).get(track.track_id, 1)
        if frames_seen < 3:
            return None

        # Verification Officer Rule — Ignore cropped border detections
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = track.bbox
        if x1 <= 5 or y1 <= 5 or x2 >= w - 5 or y2 >= h - 5:
            # Boundary crop -> reject false boundary artifact
            return None

        return ViolationEvent(
            track_id=track.track_id,
            violation_type=self.name,
            plate_number=track.plate_text,
            confidence=0.98,
            location=context.get("location", "unknown"),
            speed_kmh=track.speed_kmh,
            evidence_frame=frame.copy(),
            meta={
                "limit_kmh": limit,
                "tolerance_threshold": tolerance_threshold,
                "status": "Approved",
                "officer_verification": "Passed speed tolerance & multi-frame trajectory check",
            },
        )


class RedLightJumpRule(ViolationRule):
    name = "red_light_jump"

    def check(self, track: Track, frame: np.ndarray, context: dict) -> ViolationEvent | None:
        """Requires calibrated stop-line polygon + active red signal state."""
        signal_state = context.get("signal_state")
        crossed = context.get("crossed_stop_line", {}).get(track.track_id, False)
        direction = context.get("track_direction", {}).get(track.track_id)
        if not crossed or not signal_state or not direction:
            return None
        if signal_state.get(direction) != "red":
            return None

        return ViolationEvent(
            track_id=track.track_id,
            violation_type=self.name,
            plate_number=track.plate_text,
            confidence=0.99,
            location=context.get("location", "unknown"),
            evidence_frame=frame.copy(),
            meta={"status": "Approved", "officer_verification": "Confirmed red light crossing"},
        )


class NoHelmetRule(ViolationRule):
    name = "no_helmet"

    def check(self, track: Track, frame: np.ndarray, context: dict) -> ViolationEvent | None:
        # Verification Officer Rule — Helmet detection ONLY for motorcycles (bike)
        if track.class_name != "bike":
            return None
        helmet_ok = context.get("helmet_ok", {}).get(track.track_id)
        if helmet_ok is False:
            return ViolationEvent(
                track_id=track.track_id,
                violation_type=self.name,
                plate_number=track.plate_text,
                confidence=0.98,
                location=context.get("location", "unknown"),
                evidence_frame=frame.copy(),
                meta={"status": "Approved", "officer_verification": "Motorcycle rider without helmet verified"},
            )
        return None


class StopLineCrossingRule(ViolationRule):
    name = "stop_line_crossing"

    def check(self, track: Track, frame: np.ndarray, context: dict) -> ViolationEvent | None:
        if not context.get("crossed_stop_line", {}).get(track.track_id):
            return None
        return ViolationEvent(
            track_id=track.track_id,
            violation_type=self.name,
            plate_number=track.plate_text,
            confidence=0.96,
            location=context.get("location", "unknown"),
            speed_kmh=track.speed_kmh,
            evidence_frame=frame.copy(),
            meta={"status": "Approved", "officer_verification": "Stop line painted boundary crossed"},
        )


class WrongSideRule(ViolationRule):
    name = "wrong_side"

    def check(self, track: Track, frame: np.ndarray, context: dict) -> ViolationEvent | None:
        if not context.get("wrong_side", {}).get(track.track_id):
            return None
        return ViolationEvent(
            track_id=track.track_id,
            violation_type=self.name,
            plate_number=track.plate_text,
            confidence=0.97,
            location=context.get("location", "unknown"),
            speed_kmh=track.speed_kmh,
            evidence_frame=frame.copy(),
            meta={"status": "Approved", "officer_verification": "Driving against defined lane direction"},
        )


class NoSeatBeltRule(ViolationRule):
    name = "seat_belt"

    def check(self, track: Track, frame: np.ndarray, context: dict) -> ViolationEvent | None:
        if track.class_name not in {"car", "truck", "bus", "auto"}:
            return None

        # Problem 4 Fix: Disable seat belt detection when vehicle is too far away
        x1, y1, x2, y2 = track.bbox
        vehicle_height = y2 - y1
        if vehicle_height < 150:  # pixels
            return None

        # Verification Officer Rule:
        # Requires front-facing camera angle where driver cabin and face are clearly visible.
        # Top-view / rear-view overhead highway cameras CANNOT issue seatbelt challans!
        camera_angle = context.get("camera_angle", "overhead_rear")
        cabin_visible = context.get("cabin_visible", False)

        if camera_angle in {"overhead", "overhead_rear", "top_view", "rear_view"} or not cabin_visible:
            # Reject false seatbelt detection from top/rear camera view
            return None

        if context.get("seatbelt_ok", {}).get(track.track_id) is not False:
            return None

        return ViolationEvent(
            track_id=track.track_id,
            violation_type=self.name,
            plate_number=track.plate_text,
            confidence=0.97,
            location=context.get("location", "unknown"),
            evidence_frame=frame.copy(),
            meta={"status": "Approved", "officer_verification": "Driver front cabin visible without seatbelt"},
        )


class ViolationDetector:
    """Phase 5 — AI Traffic Violation Verification Officer Detector."""

    def __init__(self, rules: list[ViolationRule] | None = None) -> None:
        self.rules = rules or [
            OverspeedRule(),
            RedLightJumpRule(),
            StopLineCrossingRule(),
            WrongSideRule(),
            NoHelmetRule(),
            NoSeatBeltRule(),
        ]

    def evaluate(
        self,
        tracks: list[Track],
        frame: np.ndarray,
        context: dict | None = None,
    ) -> list[ViolationEvent]:
        context = context or {}
        events: list[ViolationEvent] = []
        for track in tracks:
            for rule in self.rules:
                event = rule.check(track, frame, context)
                if event:
                    events.append(event)
        return events
