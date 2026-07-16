from fastapi import APIRouter

from traffic_ai.ai.signal_controller import SignalAIEngine
from traffic_ai.api.schemas import (
    SignalDecideRequest,
    SignalDecideResponse,
    SignalPhaseOut,
)
from traffic_ai.utils.types import Track

router = APIRouter(prefix="/signal", tags=["signal"])
engine = SignalAIEngine()


@router.post("/decide", response_model=SignalDecideResponse)
async def decide_signal(body: SignalDecideRequest) -> SignalDecideResponse:
    counts = {
        "north": body.north,
        "south": body.south,
        "east": body.east,
        "west": body.west,
    }
    tracks: list[Track] = []
    if body.emergency_direction:
        tracks.append(
            Track(
                track_id=0,
                class_name="emergency_vehicle",
                confidence=1.0,
                bbox=(0, 0, 0, 0),
                is_emergency=True,
                meta={"direction": body.emergency_direction},
            )
        )

    decision = engine.decide(
        counts,
        tracks=tracks,
        rush_hour=body.rush_hour,
        pedestrians=body.pedestrians,
    )

    return SignalDecideResponse(
        phases=[
            SignalPhaseOut(
                direction=p.direction,
                waiting_vehicles=p.waiting_vehicles,
                green_seconds=p.green_seconds,
            )
            for p in decision.phases
        ],
        emergency_override=decision.emergency_override,
        emergency_direction=decision.emergency_direction or body.emergency_direction,
    )
