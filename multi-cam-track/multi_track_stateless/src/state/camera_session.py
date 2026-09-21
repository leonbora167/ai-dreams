"""Stateful Camera Tracking Session.

Maintains per-camera single-target tracker state, Kalman filter states,
and reservoir crop collection across the tracklet's lifetime.
"""
from typing import Dict, List, Optional
import numpy as np
import random
from ..trackers import get_tracker, BaseTracker
from ..tracklets import Tracklet
from ..services.detector_service import Detection


class _WorkingTrack:
    """Internal accumulator for a live single-camera trajectory."""

    def __init__(self, camera_id: str, track_id: int, fps: float, video_path: str):
        self.camera_id = camera_id
        self.track_id = track_id
        self.fps = fps
        self.video_path = video_path

        self.frames: List[int] = []
        self.bboxes: List[List[float]] = []
        self.confidences: List[float] = []
        self.crops: List[np.ndarray] = []
        self.max_crops = 24
        self.total_seen = 0

    def add(self, frame_id: int, bbox: list, conf: float, frame: np.ndarray):
        self.frames.append(frame_id)
        self.bboxes.append(bbox)
        self.confidences.append(conf)
        self.total_seen += 1

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if (x2 - x1) > 10 and (y2 - y1) > 10:
            crop = frame[y1:y2, x1:x2].copy()
            # Reservoir sampling across track lifetime
            if len(self.crops) < self.max_crops:
                self.crops.append(crop)
            else:
                idx = random.randint(0, self.total_seen - 1)
                if idx < self.max_crops:
                    self.crops[idx] = crop

    @property
    def last_frame(self) -> int:
        return self.frames[-1] if self.frames else 0

    @property
    def start_frame(self) -> int:
        return self.frames[0] if self.frames else 0

    def finalize(self) -> Tracklet:
        t = Tracklet(
            camera_id=self.camera_id,
            local_id=self.track_id,
            frames=self.frames,
            boxes=self.bboxes,
            confidences=self.confidences,
            fps=self.fps,
            video_path=self.video_path,
        )
        t.crops = self.crops
        return t


class CameraTrackerSession:
    """Stateful tracking container for a single camera video stream."""

    def __init__(self, camera_id: str, cfg: dict, fps: float, video_path: str):
        self.camera_id = camera_id
        self.cfg = cfg
        self.fps = fps
        self.video_path = video_path
        self.timeout = int(cfg.get("tracker", {}).get("track_buffer", 30))

        # Instantiate dedicated stateful tracker instance for this camera
        self.tracker: BaseTracker = get_tracker(cfg, fps)
        self.active_tracks: Dict[int, _WorkingTrack] = {}
        self.finalized_tracklets: List[Tracklet] = []

    def process_frame(self, frame_no: int, frame: np.ndarray, detections: Detection) -> List[dict]:
        """Update tracker with detections and frame, returning active observation records."""
        tracked = self.tracker.update(detections, frame=frame)
        seen_tids = set()
        observations = []

        if len(tracked) > 0:
            for box, tid, conf in zip(tracked.xyxy.tolist(), tracked.tracker_id.tolist(), tracked.confidence.tolist()):
                tid = int(tid)
                seen_tids.add(tid)
                if tid not in self.active_tracks:
                    self.active_tracks[tid] = _WorkingTrack(self.camera_id, tid, self.fps, self.video_path)
                item = self.active_tracks[tid]
                item.add(frame_no, box, float(conf), frame)

                observations.append({
                    "camera_id": self.camera_id,
                    "frame_id": frame_no,
                    "timestamp_sec": round(frame_no / self.fps, 4),
                    "local_track_id": tid,
                    "bbox_xyxy": [round(float(x), 2) for x in box],
                    "confidence": round(float(conf), 5),
                    "global_id": None
                })

        # Evict stale tracks that crossed the timeout threshold
        for tid, item in list(self.active_tracks.items()):
            if (frame_no - item.last_frame) > self.timeout and tid not in seen_tids:
                if len(item.frames) >= 3:
                    self.finalized_tracklets.append(item.finalize())
                del self.active_tracks[tid]

        return observations

    def flush(self) -> List[Tracklet]:
        """Flush all remaining active tracks when camera stream ends."""
        for item in self.active_tracks.values():
            if len(item.frames) >= 3:
                self.finalized_tracklets.append(item.finalize())
        self.active_tracks.clear()
        return self.finalized_tracklets
