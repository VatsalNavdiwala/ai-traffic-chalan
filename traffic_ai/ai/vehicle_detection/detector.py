from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from traffic_ai.config.settings import ROOT_DIR, get_settings
from traffic_ai.utils.io import load_yaml
from traffic_ai.utils.types import Detection

# Default COCO-ish mapping for stock YOLOv11 until a custom traffic model is trained.
# Custom fine-tuned weights should align with classes.yaml.
COCO_TO_TRAFFIC = {
    1: "bicycle",
    2: "car",
    3: "bike",  # motorcycle
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    """Phase 1 — YOLOv11 vehicle detection (target confidence 95%+)."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
        device: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model_path = model_path or settings.yolo_model_path
        self.confidence = confidence if confidence is not None else settings.yolo_confidence
        self.device = device or settings.device
        self._model = None
        self._class_map: dict[int, str] = {}
        self._using_custom_weights = Path(self.model_path).exists()

    def _load_class_map(self) -> dict[int, str]:
        """Custom classes.yaml only applies to fine-tuned weights, not stock COCO models."""
        if not self._using_custom_weights:
            return {}
        cfg = ROOT_DIR / "traffic_ai" / "config" / "classes.yaml"
        if cfg.exists():
            data = load_yaml(cfg)
            return {int(v): k for k, v in data.get("vehicle_classes", {}).items()}
        return {}

    def load(self) -> None:
        from ultralytics import YOLO

        path = Path(self.model_path)
        if not path.exists():
            logger.warning(
                "Weights not found at {}; Ultralytics will download yolo11n.pt",
                self.model_path,
            )
            self._model = YOLO("yolo11n.pt")
            self._using_custom_weights = False
        else:
            self._model = YOLO(str(path))
            self._using_custom_weights = True
        self._class_map = self._load_class_map()
        logger.info("YOLOv11 loaded (conf>={:.2f}, device={})", self.confidence, self.device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._model is None:
            self.load()

        results = self._model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            imgsz=320,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        names = result.names or {}
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Prefer custom traffic mapping; else COCO traffic subset; else YOLO name
            class_name = (
                self._class_map.get(cls_id)
                or COCO_TO_TRAFFIC.get(cls_id)
                or names.get(cls_id, f"class_{cls_id}")
            )
            if class_name not in {
                "car",
                "bike",
                "truck",
                "bus",
                "auto",
                "emergency_vehicle",
                "bicycle",
            } and cls_id not in COCO_TO_TRAFFIC:
                continue

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                )
            )
        return detections
