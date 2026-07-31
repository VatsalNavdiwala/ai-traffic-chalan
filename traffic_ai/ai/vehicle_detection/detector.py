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
        try:
            from ultralytics import YOLO

            path = Path(self.model_path)
            model_target = str(path) if path.exists() else "yolo11n.pt"
            try:
                self._model = YOLO(model_target)
                self._using_custom_weights = path.exists()
                logger.info("Loaded YOLO model: {}", model_target)
            except Exception as e:
                logger.warning("Could not load {}, trying yolo11n.pt: {}", model_target, e)
                self._model = YOLO("yolo11n.pt")
                self._using_custom_weights = False

            self._class_map = self._load_class_map()
            logger.info("YOLO26 loaded (conf>={:.2f}, device={})", self.confidence, self.device)
        except Exception as exc:
            import cv2
            logger.warning("YOLO/PyTorch load error ({}). Using OpenCV computer vision fallback.", exc)
            self._model = "opencv_fallback"
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    def _detect_opencv(self, frame: np.ndarray) -> list[Detection]:
        import cv2
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        fg_mask = self._bg_sub.apply(blur)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        min_area = (w * h) * 0.005
        max_area = (w * h) * 0.4
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = bw / float(bh)
                if 0.5 <= aspect_ratio <= 3.5:
                    detections.append(
                        Detection(
                            class_name="car",
                            confidence=0.85,
                            bbox=(float(x), float(y), float(x + bw), float(y + bh)),
                            class_id=2,
                        )
                    )
        return detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._model is None:
            self.load()

        if self._model == "opencv_fallback":
            return self._detect_opencv(frame)

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
            raw_name = (
                self._class_map.get(cls_id)
                or COCO_TO_TRAFFIC.get(cls_id)
                or names.get(cls_id, f"class_{cls_id}")
            )
            raw_name = str(raw_name).lower()
            if raw_name in {"motorcycle", "motorbike"}:
                class_name = "bike"
            elif raw_name in {"automobile", "sedan", "suv", "vehicle"}:
                class_name = "car"
            else:
                class_name = raw_name

            if class_name not in {"car", "bike", "truck", "bus", "auto", "bicycle", "van"}:
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
