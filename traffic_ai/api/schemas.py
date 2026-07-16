from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str


class SignalDecideRequest(BaseModel):
    north: int = 0
    south: int = 0
    east: int = 0
    west: int = 0
    rush_hour: bool = False
    emergency_direction: Optional[str] = None
    pedestrians: dict[str, int] = Field(default_factory=dict)


class SignalPhaseOut(BaseModel):
    direction: str
    waiting_vehicles: int
    green_seconds: int


class SignalDecideResponse(BaseModel):
    phases: list[SignalPhaseOut]
    emergency_override: bool = False
    emergency_direction: Optional[str] = None


class ChallanCreateRequest(BaseModel):
    plate_number: str
    violation_type: str
    location: str
    fine_amount: float = 1000
    speed_kmh: Optional[float] = None
    notes: Optional[str] = None


class ChallanOut(BaseModel):
    id: str
    plate_number: str
    violation_type: str
    location: str
    fine_amount: float
    status: str
    issued_at: datetime
    evidence_path: Optional[str] = None


class CameraCreate(BaseModel):
    name: str
    stream_url: str
    direction: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    road_id: Optional[str] = None


class CameraOut(BaseModel):
    id: str
    name: str
    stream_url: str
    direction: Optional[str]
    is_active: bool

    class Config:
        from_attributes = True
