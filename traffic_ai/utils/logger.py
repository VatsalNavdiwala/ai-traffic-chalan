from __future__ import annotations

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
    log_dir = ROOT_DIR / "traffic_ai" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "traffic_ai_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level=level,
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
