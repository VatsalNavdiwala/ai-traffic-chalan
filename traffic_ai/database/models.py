from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class ViolationType(str, Enum):
    RED_LIGHT_JUMP = "red_light_jump"
    WRONG_SIDE = "wrong_side"
    NO_HELMET = "no_helmet"
    SEAT_BELT = "seat_belt"
    MOBILE_PHONE = "mobile_phone"
    TRIPLE_RIDING = "triple_riding"
    LANE_CROSSING = "lane_crossing"
    ILLEGAL_PARKING = "illegal_parking"
    STOP_LINE_CROSSING = "stop_line_crossing"
    OVERSPEED = "overspeed"


class ChallanStatus(str, Enum):
    PENDING_VERIFICATION = "pending_verification"
    APPROVED = "approved"
    REJECTED = "rejected"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"


class Road(Base):
    __tablename__ = "roads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    locality: Mapped[Optional[str]] = mapped_column(String(255))
    city: Mapped[Optional[str]] = mapped_column(String(128))
    geo_json: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    cameras: Mapped[list["Camera"]] = relationship(back_populates="road")
    signals: Mapped[list["TrafficSignal"]] = relationship(back_populates="road")


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    stream_url: Mapped[str] = mapped_column(String(512), nullable=False)
    road_id: Mapped[Optional[str]] = mapped_column(ForeignKey("roads.id"))
    direction: Mapped[Optional[str]] = mapped_column(String(32))  # north/south/east/west
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    calibration: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    road: Mapped[Optional[Road]] = relationship(back_populates="cameras")


class TrafficSignal(Base):
    __tablename__ = "traffic_signals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    road_id: Mapped[Optional[str]] = mapped_column(ForeignKey("roads.id"))
    controller_endpoint: Mapped[Optional[str]] = mapped_column(String(512))
    current_phase: Mapped[Optional[dict]] = mapped_column(JSON)
    is_ai_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    road: Mapped[Optional[Road]] = relationship(back_populates="signals")


class Owner(Base):
    __tablename__ = "owners"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    address: Mapped[Optional[str]] = mapped_column(Text)
    email: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    vehicles: Mapped[list["Vehicle"]] = relationship(back_populates="owner")


class Vehicle(Base):
    __tablename__ = "vehicles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    plate_number: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    vehicle_type: Mapped[Optional[str]] = mapped_column(String(64))
    make: Mapped[Optional[str]] = mapped_column(String(64))
    model: Mapped[Optional[str]] = mapped_column(String(64))
    color: Mapped[Optional[str]] = mapped_column(String(64))
    owner_id: Mapped[Optional[str]] = mapped_column(ForeignKey("owners.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    owner: Mapped[Optional[Owner]] = relationship(back_populates="vehicles")
    violations: Mapped[list["Violation"]] = relationship(back_populates="vehicle")


class Officer(Base):
    __tablename__ = "officers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    badge_id: Mapped[str] = mapped_column(String(64), unique=True)
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    vehicle_id: Mapped[Optional[str]] = mapped_column(ForeignKey("vehicles.id"))
    plate_number: Mapped[str] = mapped_column(String(20), index=True)
    violation_type: Mapped[ViolationType] = mapped_column(SAEnum(ViolationType), nullable=False)
    location: Mapped[Optional[str]] = mapped_column(String(255))
    camera_id: Mapped[Optional[str]] = mapped_column(ForeignKey("cameras.id"))
    track_id: Mapped[Optional[int]] = mapped_column(Integer)
    speed_kmh: Mapped[Optional[float]] = mapped_column(Float)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON)

    vehicle: Mapped[Optional[Vehicle]] = relationship(back_populates="violations")
    evidence: Mapped[list["Evidence"]] = relationship(back_populates="violation")
    challan: Mapped[Optional["Challan"]] = relationship(back_populates="violation", uselist=False)


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    violation_id: Mapped[str] = mapped_column(ForeignKey("violations.id"))
    image_path: Mapped[Optional[str]] = mapped_column(String(512))
    video_clip_path: Mapped[Optional[str]] = mapped_column(String(512))
    frame_index: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    violation: Mapped[Violation] = relationship(back_populates="evidence")


class Challan(Base):
    __tablename__ = "challans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    violation_id: Mapped[str] = mapped_column(ForeignKey("violations.id"), unique=True)
    plate_number: Mapped[str] = mapped_column(String(20), index=True)
    fine_amount: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[ChallanStatus] = mapped_column(
        SAEnum(ChallanStatus), default=ChallanStatus.PENDING_VERIFICATION
    )
    verified_by: Mapped[Optional[str]] = mapped_column(ForeignKey("officers.id"))
    issued_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notice_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    notes: Mapped[Optional[str]] = mapped_column(Text)

    violation: Mapped[Violation] = relationship(back_populates="challan")
    payments: Mapped[list["Payment"]] = relationship(back_populates="challan")


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    challan_id: Mapped[str] = mapped_column(ForeignKey("challans.id"))
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    method: Mapped[Optional[str]] = mapped_column(String(64))
    transaction_id: Mapped[Optional[str]] = mapped_column(String(128))
    paid_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    challan: Mapped[Challan] = relationship(back_populates="payments")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    actor: Mapped[Optional[str]] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    entity_type: Mapped[Optional[str]] = mapped_column(String(64))
    entity_id: Mapped[Optional[str]] = mapped_column(String(36))
    details: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
