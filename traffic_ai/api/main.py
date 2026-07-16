from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from traffic_ai import __version__
from traffic_ai.api.routes import demo, health, signal
from traffic_ai.config.settings import get_settings
from traffic_ai.utils.logger import setup_logging

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging("DEBUG" if settings.debug else "INFO")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="AI Traffic Signal — detection, tracking, signal AI, violations, challan",
        docs_url="/docs",
        redoc_url="/redoc",
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
    app.include_router(demo.router)

    try:
        from traffic_ai.api.routes import ops

        app.include_router(ops.router)
    except Exception:
        pass

    if DASHBOARD_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def dashboard() -> FileResponse:
            return FileResponse(DASHBOARD_DIR / "index.html")

    @app.on_event("startup")
    async def _startup() -> None:
        try:
            from traffic_ai.database import init_db

            await init_db()
        except Exception:
            pass

    return app


app = create_app()
