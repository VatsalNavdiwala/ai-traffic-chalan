from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

from traffic_ai.config.settings import ROOT_DIR


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    try:
        if os.getenv("VERCEL"):
            log_dir = Path("/tmp/traffic_ai/logs")
        else:
            log_dir = ROOT_DIR / "traffic_ai" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "traffic_ai_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="30 days",
            level=level,
        )
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p
