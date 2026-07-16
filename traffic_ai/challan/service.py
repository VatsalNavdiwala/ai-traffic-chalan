from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
from loguru import logger

from traffic_ai.config.settings import get_settings
from traffic_ai.utils.io import load_yaml
from traffic_ai.utils.logger import ensure_dir
from traffic_ai.utils.types import ViolationEvent


@dataclass
class ChallanDraft:
    plate_number: str
    violation_type: str
    location: str
    occurred_at: datetime
    fine_amount: float
    evidence_path: str | None
    status: str = "pending_verification"
    speed_kmh: float | None = None


class RegistrationLookup:
    """Phase 7 — government vehicle database (VAHAN / RTO). Requires official access."""

    def __init__(self) -> None:
        settings = get_settings()
        self.api_url = settings.vahan_api_url
        self.api_key = settings.vahan_api_key

    async def lookup(self, plate_number: str) -> dict | None:
        if not self.api_url or not self.api_key:
            logger.warning(
                "Registration lookup skipped — set VAHAN_API_URL and VAHAN_API_KEY "
                "(official government access required)"
            )
            return None
        import httpx

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{self.api_url.rstrip('/')}/vehicles/{plate_number}",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            return resp.json()


class ChallanService:
    """Phase 8 — create challan drafts with evidence; officer verification required."""

    def __init__(self) -> None:
        settings = get_settings()
        self.evidence_dir = ensure_dir(settings.evidence_dir)
        cfg = Path(__file__).resolve().parents[1] / "config" / "classes.yaml"
        self.fines = load_yaml(cfg).get("fines", {}) if cfg.exists() else {}

    def create_draft(self, event: ViolationEvent) -> ChallanDraft:
        plate = event.plate_number or "UNKNOWN"
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        evidence_path = None
        if event.evidence_frame is not None:
            fname = f"{plate}_{event.violation_type}_{stamp}.jpg"
            path = self.evidence_dir / fname
            cv2.imwrite(str(path), event.evidence_frame)
            evidence_path = str(path)

        fine = float(self.fines.get(event.violation_type, 1000))
        draft = ChallanDraft(
            plate_number=plate,
            violation_type=event.violation_type,
            location=event.location,
            occurred_at=datetime.utcnow(),
            fine_amount=fine,
            evidence_path=evidence_path,
            speed_kmh=event.speed_kmh,
        )
        logger.info(
            "Challan draft: {} | {} | ₹{} | {}",
            draft.plate_number,
            draft.violation_type,
            draft.fine_amount,
            draft.location,
        )
        return draft


class NoticeService:
    """Phase 9 — SMS / WhatsApp via official government gateways."""

    def __init__(self) -> None:
        settings = get_settings()
        self.sms_url = settings.sms_gateway_url
        self.sms_key = settings.sms_gateway_key
        self.wa_url = settings.whatsapp_api_url
        self.wa_key = settings.whatsapp_api_key

    def build_message(self, draft: ChallanDraft, portal_url: str = "https://portal.example.gov") -> str:
        return (
            f"Dear Citizen,\n"
            f"Your vehicle {draft.plate_number} violated {draft.violation_type.replace('_', ' ').title()}.\n"
            f"Fine: ₹{int(draft.fine_amount)}\n"
            f"Visit {portal_url} to pay."
        )

    async def send_sms(self, phone: str, draft: ChallanDraft) -> bool:
        if not self.sms_url or not self.sms_key:
            logger.warning("SMS gateway not configured — notice not sent")
            return False
        import httpx

        message = self.build_message(draft)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.sms_url,
                headers={"Authorization": f"Bearer {self.sms_key}"},
                json={"to": phone, "message": message},
            )
            resp.raise_for_status()
            return True

    async def send_whatsapp(self, phone: str, draft: ChallanDraft) -> bool:
        if not self.wa_url or not self.wa_key:
            logger.warning("WhatsApp API not configured — notice not sent")
            return False
        import httpx

        message = self.build_message(draft)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.wa_url,
                headers={"Authorization": f"Bearer {self.wa_key}"},
                json={"to": phone, "message": message},
            )
            resp.raise_for_status()
            return True
