from celery import Celery

from traffic_ai.config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "traffic_ai",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
)


@celery_app.task(name="traffic_ai.process_clip")
def process_clip(source: str, location: str = "Ring Road") -> dict:
    """Background job: run pipeline on a short clip / camera burst."""
    from traffic_ai.ai.pipeline import TrafficPipeline

    pipeline = TrafficPipeline(location=location, run_ocr=True)
    pipeline.run(source, max_frames=300, display=False)
    return {"status": "completed", "source": source, "location": location}


@celery_app.task(name="traffic_ai.send_challan_notice")
def send_challan_notice(phone: str, plate: str, violation: str, fine: float) -> dict:
    import asyncio

    from traffic_ai.challan import ChallanDraft, NoticeService
    from datetime import datetime

    draft = ChallanDraft(
        plate_number=plate,
        violation_type=violation,
        location="",
        occurred_at=datetime.utcnow(),
        fine_amount=fine,
        evidence_path=None,
    )
    notice = NoticeService()
    ok = asyncio.run(notice.send_sms(phone, draft))
    return {"sent": ok, "plate": plate}
