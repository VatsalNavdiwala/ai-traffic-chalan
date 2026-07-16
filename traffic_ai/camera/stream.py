from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


class CameraStream:
    """Industrial CCTV / file / RTSP source."""

    def __init__(self, source: str | int, name: str = "cam0") -> None:
        self.source = source
        self.name = name
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            hint = ""
            if isinstance(self.source, int):
                hint = (
                    "\nNo webcam found at this index. Try:\n"
                    "  python -m traffic_ai.ai.pipeline --demo --display\n"
                    "  python -m traffic_ai.ai.pipeline --source path\\to\\video.mp4 --display"
                )
            raise RuntimeError(f"Cannot open camera source: {self.source}{hint}")
        logger.info("Camera '{}' opened: {}", self.name, self.source)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            self.open()
        assert self._cap is not None
        ok, frame = self._cap.read()
        return ok, frame if ok else None

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        if self._cap is None:
            self.open()
        idx = 0
        while True:
            ok, frame = self.read()
            if not ok or frame is None:
                break
            yield idx, frame
            idx += 1

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> CameraStream:
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()


def resolve_source(source: str) -> str | int:
    if source.isdigit():
        return int(source)
    path = Path(source)
    if path.exists():
        return str(path)
    return source  # RTSP / HTTP URL
