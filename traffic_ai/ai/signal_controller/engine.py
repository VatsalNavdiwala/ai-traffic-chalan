from __future__ import annotations

from traffic_ai.utils.types import SignalDecision, SignalPhase, Track

DIRECTIONS = ("north", "south", "east", "west")
MIN_GREEN = 10
MAX_GREEN = 60
BASE_GREEN = 12
SEC_PER_VEHICLE = 1.0


class SignalAIEngine:
    """Phase 4 — adaptive green timing from queue counts + emergency override."""

    def decide(
        self,
        counts: dict[str, int],
        tracks: list[Track] | None = None,
        rush_hour: bool = False,
        pedestrians: dict[str, int] | None = None,
    ) -> SignalDecision:
        pedestrians = pedestrians or {}
        tracks = tracks or []

        # Immediate green for emergency vehicle approach
        for t in tracks:
            if t.is_emergency:
                direction = self._infer_direction(t) or "north"
                phases = [
                    SignalPhase(
                        direction=d,
                        waiting_vehicles=counts.get(d, 0),
                        green_seconds=MAX_GREEN if d == direction else MIN_GREEN,
                    )
                    for d in DIRECTIONS
                ]
                return SignalDecision(
                    phases=phases,
                    emergency_override=True,
                    emergency_direction=direction,
                )

        total = sum(counts.get(d, 0) for d in DIRECTIONS) or 1
        rush_boost = 1.2 if rush_hour else 1.0
        phases: list[SignalPhase] = []
        for d in DIRECTIONS:
            waiting = counts.get(d, 0)
            ped = pedestrians.get(d, 0)
            share = waiting / total
            green = int((BASE_GREEN + waiting * SEC_PER_VEHICLE + ped * 0.5) * rush_boost)
            # Soft proportional ceiling so one approach cannot starve others forever
            green = max(MIN_GREEN, min(MAX_GREEN, green, int(MAX_GREEN * (0.3 + share))))
            phases.append(SignalPhase(direction=d, waiting_vehicles=waiting, green_seconds=green))
        return SignalDecision(phases=phases)

    @staticmethod
    def _infer_direction(track: Track) -> str | None:
        # Production: map camera ROI / motion vector to approach direction
        return track.meta.get("direction")
