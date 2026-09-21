"""Abstract Base Tracker interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from ..services.detector_service import Detection


@dataclass
class TrackedDetection:
    """Standardized output of single-camera multi-target tracker."""
    xyxy: np.ndarray        # (M, 4) float bounding boxes
    tracker_id: np.ndarray  # (M,) int local tracker IDs
    confidence: np.ndarray  # (M,) float confidences

    @classmethod
    def empty(cls) -> "TrackedDetection":
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            tracker_id=np.empty((0,), dtype=int),
            confidence=np.empty((0,), dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.tracker_id)


class BaseTracker(ABC):
    """Abstract interface for single-camera multi-object trackers."""

    def __init__(self, cfg: dict, fps: float):
        self.cfg = cfg
        self.fps = fps

    @abstractmethod
    def update(self, detections: Detection, frame: np.ndarray = None) -> TrackedDetection:
        """Update tracker state with current frame's detections.

        Args:
            detections: Standardized Detection object from detector.
            frame: Optional BGR video frame array (H, W, 3) for visual appearance ReID.

        Returns:
            TrackedDetection: Active confirmed tracks with consistent local IDs.
        """
        pass

