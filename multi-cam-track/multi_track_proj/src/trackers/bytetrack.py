"""ByteTrack tracker implementation using supervision."""
import numpy as np
import supervision as sv
from .base import BaseTracker, TrackedDetection
from ..detectors.base import Detection


class ByteTrackTracker(BaseTracker):
    """Wraps supervision ByteTrack implementation."""

    def __init__(self, cfg: dict, fps: float):
        super().__init__(cfg, fps)
        t_cfg = cfg['tracker']
        self.tracker = sv.ByteTrack(
            track_activation_threshold=t_cfg.get('track_high_thresh', 0.5),
            lost_track_buffer=int(t_cfg.get('track_buffer', 30)),
            frame_rate=int(round(fps)) if fps > 0 else 25,
        )

    def update(self, detections: Detection, frame: np.ndarray = None) -> TrackedDetection:
        if len(detections) == 0:
            sv_dets = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=int),
            )
        else:
            sv_dets = sv.Detections(
                xyxy=detections.xyxy,
                confidence=detections.confidence,
                class_id=detections.class_id,
            )

        tracked = self.tracker.update_with_detections(sv_dets)

        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            return TrackedDetection.empty()

        return TrackedDetection(
            xyxy=tracked.xyxy,
            tracker_id=tracked.tracker_id.astype(int),
            confidence=tracked.confidence if tracked.confidence is not None else np.ones(len(tracked.tracker_id), dtype=np.float32),
        )

