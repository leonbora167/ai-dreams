"""StrongSORT tracker implementation with camera motion compensation and appearance ReID."""
import numpy as np
from boxmot import StrongSort
from .base import BaseTracker, TrackedDetection
from ..detectors.base import Detection
from ..device import resolve_device


class StrongSortTracker(BaseTracker):
    """Wraps BoxMOT StrongSORT tracker."""

    def __init__(self, cfg: dict, fps: float):
        super().__init__(cfg, fps)
        t_cfg = cfg.get('tracker', {})
        reid_weights = t_cfg.get('reid_weights', 'osnet_x0_25_msmt17.pt')
        device = resolve_device(cfg.get('detector', {}).get('device', 'auto'))
        dev_str = str(device)
        self.tracker = StrongSort(
            reid_weights=reid_weights,
            device=dev_str,
            max_age=int(t_cfg.get('track_buffer', 30)),
            min_conf=float(t_cfg.get('track_low_thresh', 0.1)),
            n_init=int(t_cfg.get('n_init', 1)),
            max_iou_dist=float(t_cfg.get('match_thresh', 0.7)),
        )

    def update(self, detections: Detection, frame: np.ndarray = None) -> TrackedDetection:
        if len(detections) == 0:
            return TrackedDetection.empty()

        xyxy = detections.xyxy
        confs = detections.confidence[:, None]
        classes = detections.class_id[:, None]
        dets = np.hstack([xyxy, confs, classes]).astype(np.float32)

        dummy_frame = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = self.tracker.update(dets, dummy_frame)

        if tracks is None or len(tracks) == 0:
            return TrackedDetection.empty()

        tracks_arr = np.asarray(tracks, dtype=np.float32)
        out_xyxy = tracks_arr[:, 0:4]
        track_ids = tracks_arr[:, 4].astype(int)
        confidences = tracks_arr[:, 5]

        return TrackedDetection(
            xyxy=out_xyxy,
            tracker_id=track_ids,
            confidence=confidences,
        )

