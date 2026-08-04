from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from traffic_ai.config.settings import get_settings
from traffic_ai.database.models import Base

settings = get_settings()


import os

def _get_sqlite_path() -> str:
    if os.getenv("VERCEL"):
        return "/tmp/traffic_ai_demo.db"
    return "./traffic_ai_demo.db"


def _async_url() -> str:
    url = settings.database_url
    # Prefer SQLite on hosts without Postgres (e.g. Vercel Serverless / Cloud demo)
    try:
        import asyncpg  # noqa: F401

        return url
    except ImportError:
        return f"sqlite+aiosqlite:///{_get_sqlite_path()}"


def _sync_url() -> str:
    try:
        import psycopg2  # noqa: F401

        return settings.database_url_sync
    except ImportError:
        return f"sqlite:///{_get_sqlite_path()}"


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
