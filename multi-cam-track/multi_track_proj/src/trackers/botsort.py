"""BoT-SORT tracker implementation using Ultralytics BOTSORT."""
from types import SimpleNamespace
import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from .base import BaseTracker, TrackedDetection
from ..detectors.base import Detection


class BoTSORTTracker(BaseTracker):
    """Wraps Ultralytics BoT-SORT multi-object tracker."""

    def __init__(self, cfg: dict, fps: float):
        super().__init__(cfg, fps)
        t_cfg = cfg.get('tracker', {})
        args = SimpleNamespace(
            track_high_thresh=float(t_cfg.get('track_high_thresh', 0.5)),
            track_low_thresh=float(t_cfg.get('track_low_thresh', 0.1)),
            new_track_thresh=float(t_cfg.get('new_track_thresh', 0.6)),
            track_buffer=int(t_cfg.get('track_buffer', 30)),
            match_thresh=float(t_cfg.get('match_thresh', 0.8)),
            gmc_method=str(t_cfg.get('gmc_method', 'sparseOptFlow')),
            proximity_thresh=float(t_cfg.get('proximity_thresh', 0.5)),
            appearance_thresh=float(t_cfg.get('appearance_thresh', 0.25)),
            with_reid=False,
            fuse_score=True,
            model=None,
            device='cpu',
            fps=float(fps) if fps > 0 else 25.0,
        )
        self.tracker = BOTSORT(args)

    def update(self, detections: Detection) -> TrackedDetection:
        if len(detections) == 0:
            return TrackedDetection.empty()

        # Construct (N, 6) tensor: [x1, y1, x2, y2, conf, cls]
        xyxy = detections.xyxy
        confs = detections.confidence[:, None]
        classes = detections.class_id[:, None]
        arr = np.hstack([xyxy, confs, classes]).astype(np.float32)

        tensor = torch.from_numpy(arr)
        boxes = Boxes(tensor, orig_shape=(1080, 1920))
        tracks = self.tracker.update(boxes)

        if tracks is None or len(tracks) == 0:
            return TrackedDetection.empty()

        # tracks format: [x1, y1, x2, y2, track_id, conf, cls, idx]
        tracks_arr = np.asarray(tracks, dtype=np.float32)
        out_xyxy = tracks_arr[:, 0:4]
        track_ids = tracks_arr[:, 4].astype(int)
        confidences = tracks_arr[:, 5]

        return TrackedDetection(
            xyxy=out_xyxy,
            tracker_id=track_ids,
            confidence=confidences,
        )

