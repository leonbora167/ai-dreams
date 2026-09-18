"""DeepSORT tracker implementation using deep-sort-realtime."""
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from .base import BaseTracker, TrackedDetection
from ..detectors.base import Detection


class DeepSortTracker(BaseTracker):
    """Wraps deep-sort-realtime DeepSORT implementation."""

    def __init__(self, cfg: dict, fps: float):
        super().__init__(cfg, fps)
        t_cfg = cfg.get('tracker', {})
        self.tracker = DeepSort(
            max_age=int(t_cfg.get('track_buffer', 30)),
            n_init=int(t_cfg.get('n_init', 1)),
            max_cosine_distance=float(t_cfg.get('max_cosine_distance', 0.3)),
            nn_budget=int(t_cfg.get('nn_budget', 100)),
            override_track_class=None,
        )

    def update(self, detections: Detection, frame: np.ndarray = None) -> TrackedDetection:
        if len(detections) == 0:
            return TrackedDetection.empty()

        raw_bbs = []
        for box, conf in zip(detections.xyxy, detections.confidence):
            x1, y1, x2, y2 = box
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            raw_bbs.append(([float(x1), float(y1), float(w), float(h)], float(conf), 'person'))

        dummy_frame = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = self.tracker.update_tracks(raw_bbs, frame=dummy_frame)

        boxes = []
        tids = []
        confs = []

        for t in tracks:
            ltrb = t.to_ltrb()
            boxes.append(ltrb)
            tids.append(int(t.track_id))
            confs.append(float(t.get_det_conf() or 0.8))

        if not boxes:
            return TrackedDetection.empty()

        return TrackedDetection(
            xyxy=np.asarray(boxes, dtype=np.float32),
            tracker_id=np.asarray(tids, dtype=int),
            confidence=np.asarray(confs, dtype=np.float32),
        )

