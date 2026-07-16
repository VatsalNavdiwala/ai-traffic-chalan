from __future__ import annotations

import numpy as np
from loguru import logger

from traffic_ai.utils.types import Detection, Track


class _DetResults:
    """Minimal Results-like object for Ultralytics BYTETracker."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy.astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)
        cx = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2
        cy = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2
        w = self.xyxy[:, 2] - self.xyxy[:, 0]
        h = self.xyxy[:, 3] - self.xyxy[:, 1]
        self.xywh = np.stack([cx, cy, w, h], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> _DetResults:
        mask = np.asarray(mask, dtype=bool)
        return _DetResults(self.xyxy[mask], self.conf[mask], self.cls[mask])


class VehicleTracker:
    """Phase 2 — ByteTrack multi-object tracking (target accuracy 98%+)."""

    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self._byte_tracker = None
        self._load_attempted = False
        self._fallback_next_id = 1
        self._fallback_tracks: dict[int, tuple[float, float]] = {}

    def load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from types import SimpleNamespace

            from ultralytics.trackers.byte_tracker import BYTETracker

            args = SimpleNamespace(
                track_high_thresh=self.track_thresh,
                track_low_thresh=0.1,
                new_track_thresh=0.6,
                track_buffer=30,
                match_thresh=self.match_thresh,
                fuse_score=True,
            )
            self._byte_tracker = BYTETracker(args)
            logger.info("ByteTrack tracker ready")
        except Exception as exc:
            logger.warning("ByteTrack unavailable ({}), using IoU fallback tracker", exc)
            self._byte_tracker = None

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        if not self._load_attempted:
            self.load()

        if self._byte_tracker is not None:
            return self._update_bytetrack(detections, frame)
        return self._update_fallback(detections)

    def _update_bytetrack(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        if not detections:
            empty = _DetResults(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
            self._byte_tracker.update(empty, img=frame)
            return []

        rows = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            rows.append([x1, y1, x2, y2, d.confidence, float(d.class_id)])
        arr = np.asarray(rows, dtype=np.float32)
        results = _DetResults(arr[:, :4], arr[:, 4], arr[:, 5])
        online = self._byte_tracker.update(results, img=frame)

        tracks: list[Track] = []
        for row in online:
            x1, y1, x2, y2, track_id, score, cls_id, _idx = row.tolist()
            class_name = next(
                (d.class_name for d in detections if d.class_id == int(cls_id)),
                "vehicle",
            )
            tracks.append(
                Track(
                    track_id=int(track_id),
                    class_name=class_name,
                    confidence=float(score),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    is_emergency=class_name == "emergency_vehicle",
                )
            )
        return tracks

    def _update_fallback(self, detections: list[Detection]) -> list[Track]:
        """Simple centroid nearest-neighbor tracker when ByteTrack is unavailable."""
        tracks: list[Track] = []
        used_ids: set[int] = set()

        for d in detections:
            cx = (d.bbox[0] + d.bbox[2]) / 2
            cy = (d.bbox[1] + d.bbox[3]) / 2
            best_id, best_dist = None, 80.0
            for tid, (px, py) in self._fallback_tracks.items():
                if tid in used_ids:
                    continue
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_id = dist, tid
            if best_id is None:
                best_id = self._fallback_next_id
                self._fallback_next_id += 1
            used_ids.add(best_id)
            self._fallback_tracks[best_id] = (cx, cy)
            tracks.append(
                Track(
                    track_id=best_id,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    bbox=d.bbox,
                    is_emergency=d.class_name == "emergency_vehicle",
                )
            )

        self._fallback_tracks = {tid: self._fallback_tracks[tid] for tid in used_ids}
        return tracks
