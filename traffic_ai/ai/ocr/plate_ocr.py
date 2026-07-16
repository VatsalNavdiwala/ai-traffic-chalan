from __future__ import annotations

import re

import numpy as np
from loguru import logger

from traffic_ai.utils.types import Detection

# Indian plate pattern (simplified): e.g. GJ05AB1234
PLATE_REGEX = re.compile(
    r"\b([A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{1,4})\b",
    re.IGNORECASE,
)


class PlateOCR:
    """Phase 6 — YOLO plate crop + PaddleOCR (target 98–99% on clear plates)."""

    def __init__(self, plate_model_path: str | None = None, lang: str = "en") -> None:
        self.plate_model_path = plate_model_path
        self.lang = lang
        self._plate_detector = None
        self._ocr = None

    def load(self) -> None:
        try:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
            logger.info("PaddleOCR loaded")
        except Exception as exc:
            logger.warning("PaddleOCR unavailable: {}", exc)
            self._ocr = None

        if self.plate_model_path:
            try:
                from ultralytics import YOLO

                self._plate_detector = YOLO(self.plate_model_path)
                logger.info("Plate YOLO model loaded from {}", self.plate_model_path)
            except Exception as exc:
                logger.warning("Plate detector load failed: {}", exc)

    def detect_plates(self, frame: np.ndarray) -> list[Detection]:
        if self._plate_detector is None:
            return []
        results = self._plate_detector.predict(frame, verbose=False)
        plates: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                plates.append(
                    Detection(
                        class_name="number_plate",
                        confidence=float(box.conf.item()),
                        bbox=(x1, y1, x2, y2),
                        class_id=int(box.cls.item()),
                    )
                )
        return plates

    def read_plate(self, crop: np.ndarray) -> tuple[str | None, float]:
        if self._ocr is None:
            self.load()
        if self._ocr is None or crop.size == 0:
            return None, 0.0

        result = self._ocr.ocr(crop, cls=True)
        if not result or not result[0]:
            return None, 0.0

        texts: list[str] = []
        scores: list[float] = []
        for line in result[0]:
            txt, score = line[1]
            texts.append(txt)
            scores.append(float(score))

        raw = "".join(texts).upper().replace(" ", "")
        match = PLATE_REGEX.search(raw) or PLATE_REGEX.search(" ".join(texts).upper())
        plate = match.group(1).replace(" ", "").upper() if match else raw or None
        conf = float(sum(scores) / len(scores)) if scores else 0.0
        return plate, conf

    def read_from_vehicle_crop(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[str | None, float]:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            return None, 0.0

        plates = self.detect_plates(vehicle_crop)
        if plates:
            px1, py1, px2, py2 = map(int, plates[0].bbox)
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            return self.read_plate(plate_crop)

        # Fallback: OCR full vehicle crop when plate detector is absent
        return self.read_plate(vehicle_crop)
