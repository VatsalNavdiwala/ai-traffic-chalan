from fastapi import APIRouter

from traffic_ai import __version__
from traffic_ai.api.schemas import HealthResponse
from traffic_ai.config.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", app=settings.app_name, version=__version__)
