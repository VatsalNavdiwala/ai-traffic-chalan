from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from traffic_ai.api.schemas import CameraCreate, CameraOut, ChallanCreateRequest, ChallanOut
from traffic_ai.challan import ChallanService, NoticeService
from traffic_ai.database import (
    Camera,
    Challan,
    ChallanStatus,
    Violation,
    ViolationType,
    get_db,
)
from traffic_ai.utils.types import ViolationEvent

router = APIRouter(tags=["ops"])


@router.post("/cameras", response_model=CameraOut)
async def create_camera(body: CameraCreate, db: AsyncSession = Depends(get_db)) -> Camera:
    cam = Camera(
        name=body.name,
        stream_url=body.stream_url,
        direction=body.direction,
        latitude=body.latitude,
        longitude=body.longitude,
        road_id=body.road_id,
    )
    db.add(cam)
    await db.commit()
    await db.refresh(cam)
    return cam


@router.get("/cameras", response_model=list[CameraOut])
async def list_cameras(db: AsyncSession = Depends(get_db)) -> list[Camera]:
    result = await db.execute(select(Camera))
    return list(result.scalars().all())


@router.post("/challans", response_model=ChallanOut)
async def create_challan(
    body: ChallanCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> ChallanOut:
    try:
        vtype = ViolationType(body.violation_type)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid violation_type: {body.violation_type}") from exc

    violation = Violation(
        plate_number=body.plate_number.upper(),
        violation_type=vtype,
        location=body.location,
        speed_kmh=body.speed_kmh,
        occurred_at=datetime.utcnow(),
        confidence=1.0,
    )
    db.add(violation)
    await db.flush()

    challan = Challan(
        violation_id=violation.id,
        plate_number=body.plate_number.upper(),
        fine_amount=body.fine_amount,
        status=ChallanStatus.PENDING_VERIFICATION,
        notes=body.notes,
    )
    db.add(challan)
    await db.commit()
    await db.refresh(challan)

    return ChallanOut(
        id=challan.id,
        plate_number=challan.plate_number,
        violation_type=body.violation_type,
        location=body.location,
        fine_amount=challan.fine_amount,
        status=challan.status.value,
        issued_at=challan.issued_at,
    )


@router.get("/challans", response_model=list[ChallanOut])
async def list_challans(db: AsyncSession = Depends(get_db)) -> list[ChallanOut]:
    result = await db.execute(select(Challan))
    rows = result.scalars().all()
    out: list[ChallanOut] = []
    for c in rows:
        await db.refresh(c, attribute_names=["violation"])
        out.append(
            ChallanOut(
                id=c.id,
                plate_number=c.plate_number,
                violation_type=c.violation.violation_type.value if c.violation else "unknown",
                location=c.violation.location if c.violation else "",
                fine_amount=c.fine_amount,
                status=c.status.value,
                issued_at=c.issued_at,
            )
        )
    return out


@router.post("/challans/preview")
async def preview_challan_notice(body: ChallanCreateRequest) -> dict:
    """Generate challan draft + SMS text without DB / gateway (offline preview)."""
    svc = ChallanService()
    notice = NoticeService()
    event = ViolationEvent(
        track_id=0,
        violation_type=body.violation_type,
        plate_number=body.plate_number,
        confidence=1.0,
        location=body.location,
        speed_kmh=body.speed_kmh,
    )
    draft = svc.create_draft(event)
    draft.fine_amount = body.fine_amount
    return {
        "challan_id_preview": str(uuid4()),
        "plate_number": draft.plate_number,
        "violation": draft.violation_type,
        "fine": draft.fine_amount,
        "status": draft.status,
        "sms": notice.build_message(draft),
    }
