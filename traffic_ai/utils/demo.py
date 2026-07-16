from __future__ import annotations

from pathlib import Path

import httpx
from loguru import logger

from traffic_ai.config.settings import ROOT_DIR

DEMO_VIDEO_URL = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
)
DEMO_VIDEO_PATH = ROOT_DIR / "traffic_ai" / "data" / "demo_traffic.mp4"


def ensure_demo_video() -> Path:
    """Download a short traffic sample clip for machines without a webcam."""
    if DEMO_VIDEO_PATH.exists() and DEMO_VIDEO_PATH.stat().st_size > 100_000:
        return DEMO_VIDEO_PATH

    DEMO_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading demo traffic video (~8 MB)...")
    with httpx.stream("GET", DEMO_VIDEO_URL, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(DEMO_VIDEO_PATH, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    logger.info("Demo video saved to {}", DEMO_VIDEO_PATH)
    return DEMO_VIDEO_PATH
