from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from traffic_ai import __version__
from traffic_ai.api.routes import health, ops, signal
from traffic_ai.config.settings import get_settings
from traffic_ai.database import init_db
from traffic_ai.utils.logger import setup_logging


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging("DEBUG" if settings.debug else "INFO")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="AI Traffic Signal — detection, tracking, signal AI, violations, challan",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(signal.router)
    app.include_router(ops.router)

    @app.on_event("startup")
    async def _startup() -> None:
        try:
            await init_db()
        except Exception:
            # Allow API to boot without Postgres during local AI-only work
            pass

    return app


app = create_app()
