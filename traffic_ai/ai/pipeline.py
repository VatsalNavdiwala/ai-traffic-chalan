from __future__ import annotations

from collections import Counter

import numpy as np
from loguru import logger

from traffic_ai.ai.ocr import PlateOCR
from traffic_ai.ai.signal_controller import SignalAIEngine
from traffic_ai.ai.speed_detection import build_speed_estimator
from traffic_ai.ai.vehicle_detection import VehicleDetector
from traffic_ai.ai.vehicle_tracking import VehicleTracker
from traffic_ai.ai.violation import ViolationDetector
from traffic_ai.camera import CameraStream, resolve_source
from traffic_ai.challan import ChallanService
from traffic_ai.utils.logger import setup_logging
from traffic_ai.utils.types import FrameResult, SignalDecision, ViolationEvent


class TrafficPipeline:
    """
    Video → YOLO → ByteTrack → Count → Speed → Violation → OCR → Challan draft
    """

    def __init__(
        self,
        location: str = "Ring Road",
        run_ocr: bool = False,
        run_violations: bool = True,
        confidence: float | None = None,
    ) -> None:
        setup_logging()
        self.location = location
        self.run_ocr = run_ocr
        self.run_violations = run_violations
        self.detector = VehicleDetector(confidence=confidence)
        self.tracker = VehicleTracker()
        self.speed = build_speed_estimator()
        self.signal_ai = SignalAIEngine()
        self.violations = ViolationDetector()
        self.ocr = PlateOCR()
        self.challan = ChallanService()
        self._approach_counts: dict[str, int] = {
            "north": 0,
            "south": 0,
            "east": 0,
            "west": 0,
        }

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_ms: float,
        context: dict | None = None,
    ) -> tuple[FrameResult, list[ViolationEvent], SignalDecision | None]:
        context = context or {"location": self.location}
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)
        self.speed.estimate(tracks, timestamp_ms)

        if self.run_ocr:
            for t in tracks:
                if not t.plate_text:
                    plate, conf = self.ocr.read_from_vehicle_crop(frame, t.bbox)
                    if plate and conf >= 0.5:
                        t.plate_text = plate

        class_counts = Counter(t.class_name for t in tracks)
        result = FrameResult(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            detections=detections,
            tracks=tracks,
            vehicle_counts=dict(class_counts),
            frame=frame,
        )

        events: list[ViolationEvent] = []
        if self.run_violations:
            events = self.violations.evaluate(tracks, frame, context)
            for ev in events:
                self.challan.create_draft(ev)

        # Approach counts can be fed from camera direction metadata
        if "approach_counts" in context:
            self._approach_counts = context["approach_counts"]
        else:
            # Demo heuristic: put all vehicles on north until multi-cam wiring exists
            total = sum(class_counts.values())
            self._approach_counts = {
                "north": total,
                "south": 0,
                "east": 0,
                "west": 0,
            }

        decision = self.signal_ai.decide(self._approach_counts, tracks=tracks)
        return result, events, decision

    def run(
        self,
        source: str,
        max_frames: int | None = None,
        display: bool = False,
    ) -> None:
        stream = CameraStream(resolve_source(source))
        fps = 30.0
        try:
            with stream:
                for idx, frame in stream.frames():
                    if max_frames is not None and idx >= max_frames:
                        break
                    ts = (idx / fps) * 1000.0
                    result, events, decision = self.process_frame(frame, idx, ts)
                    if idx % 30 == 0:
                        logger.info(
                            "frame={} dets={} tracks={} counts={} violations={}",
                            idx,
                            len(result.detections),
                            len(result.tracks),
                            result.vehicle_counts,
                            len(events),
                        )
                        if decision:
                            timing = {
                                p.direction: p.green_seconds for p in decision.phases
                            }
                            logger.info("signal timing: {}", timing)

                    if display:
                        import cv2

                        vis = frame.copy()
                        for t in result.tracks:
                            x1, y1, x2, y2 = map(int, t.bbox)
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"ID {t.track_id} {t.class_name}"
                            if t.speed_kmh is not None:
                                label += f" {t.speed_kmh:.0f}km/h"
                            cv2.putText(
                                vis,
                                label,
                                (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                            )
                        cv2.imshow("AI Traffic Signal", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        finally:
            if display:
                import cv2

                cv2.destroyAllWindows()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AI Traffic Signal pipeline")
    parser.add_argument("--source", default="0", help="Camera index, file path, or RTSP URL")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use bundled sample traffic video (no webcam required)",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--location", default="Ring Road")
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Detection confidence (demo default: 0.35)",
    )
    args = parser.parse_args()

    source = args.source
    confidence = args.confidence
    if args.demo:
        from traffic_ai.utils.demo import ensure_demo_video

        source = str(ensure_demo_video())
        if confidence is None:
            confidence = 0.25
        logger.info("Demo mode — using {}", source)

    pipeline = TrafficPipeline(
        location=args.location,
        run_ocr=args.ocr,
        confidence=confidence,
    )
    pipeline.run(source, max_frames=args.max_frames, display=args.display)


if __name__ == "__main__":
    main()
