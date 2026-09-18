"""Norfair multi-object tracker implementation."""
import numpy as np
from norfair import Tracker as NorfairTracker, Detection as NorfairDetection
from .base import BaseTracker, TrackedDetection
from ..detectors.base import Detection


class NorfairTrackerWrapper(BaseTracker):
    """Wraps Norfair distance-based multi-object tracker."""

    def __init__(self, cfg: dict, fps: float):
        super().__init__(cfg, fps)
        t_cfg = cfg.get('tracker', {})
        self.tracker = NorfairTracker(
            distance_function='iou',
            distance_threshold=float(t_cfg.get('match_thresh', 0.7)),
            hit_counter_max=int(t_cfg.get('track_buffer', 30)),
            initialization_delay=int(t_cfg.get('initialization_delay', 1)),
        )

    def update(self, detections: Detection, frame: np.ndarray = None) -> TrackedDetection:
        if len(detections) == 0:
            tracked_objects = self.tracker.update()
        else:
            norfair_dets = []
            for box, conf in zip(detections.xyxy, detections.confidence):
                norfair_dets.append(
                    NorfairDetection(
                        points=box.reshape(1, 4),
                        scores=np.array([float(conf)], dtype=np.float32)
                    )
                )
            tracked_objects = self.tracker.update(detections=norfair_dets)

        if not tracked_objects:
            return TrackedDetection.empty()

        boxes = []
        tids = []
        confs = []

        for obj in tracked_objects:
            if obj.estimate is not None and len(obj.estimate) > 0:
                box = obj.estimate[0]
                boxes.append(box)
                tids.append(int(obj.id))
                conf = float(obj.last_detection.scores[0]) if obj.last_detection is not None and obj.last_detection.scores is not None else 0.8
                confs.append(conf)

        if not boxes:
            return TrackedDetection.empty()

        return TrackedDetection(
            xyxy=np.asarray(boxes, dtype=np.float32),
            tracker_id=np.asarray(tids, dtype=int),
            confidence=np.asarray(confs, dtype=np.float32),
        )

