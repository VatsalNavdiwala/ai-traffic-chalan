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
        if track.speed_kmh is None or track.speed_kmh <= limit:
            return None
        return ViolationEvent(
            track_id=track.track_id,
            violation_type=self.name,
            plate_number=track.plate_text,
            confidence=0.95,
            location=context.get("location", "unknown"),
            speed_kmh=track.speed_kmh,
            evidence_frame=frame.copy(),
            meta={"limit_kmh": limit},
        )


class RedLightJumpRule(ViolationRule):
    name = "red_light_jump"

    def check(self, track: Track, frame: np.ndarray, context: dict) -> ViolationEvent | None:
        """Requires calibrated stop-line polygon + signal state in context."""
        signal_state = context.get("signal_state")  # e.g. {"north": "red"}
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
        )


class NoHelmetRule(ViolationRule):
    name = "no_helmet"

    def check(self, track: Track, frame: np.ndarray, context: dict) -> ViolationEvent | None:
        # Expect a dedicated helmet classifier / YOLO head in production
        if track.class_name not in {"bike", "bicycle"}:
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
            )
        return None


class ViolationDetector:
    """Phase 5 — orchestrates per-rule detection logic."""

    def __init__(self, rules: list[ViolationRule] | None = None) -> None:
        self.rules = rules or [
            OverspeedRule(),
            RedLightJumpRule(),
            NoHelmetRule(),
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
