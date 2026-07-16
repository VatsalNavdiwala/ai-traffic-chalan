from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from traffic_ai.config.settings import get_settings
from traffic_ai.database.models import Base

settings = get_settings()


def _async_url() -> str:
    url = settings.database_url
    # Prefer SQLite on hosts without Postgres (e.g. Render Free demo)
    try:
        import asyncpg  # noqa: F401

        return url
    except ImportError:
        return "sqlite+aiosqlite:///./traffic_ai_demo.db"


def _sync_url() -> str:
    try:
        import psycopg2  # noqa: F401

        return settings.database_url_sync
    except ImportError:
        return "sqlite:///./traffic_ai_demo.db"


engine = create_async_engine(_async_url(), echo=settings.debug, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

sync_engine = create_engine(_sync_url(), pool_pre_ping=True)
SyncSessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
