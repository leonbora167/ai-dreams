"""Abstract Base Detector interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    """Standardized detection container."""
    xyxy: np.ndarray        # (N, 4) float bounding boxes [x1, y1, x2, y2]
    confidence: np.ndarray  # (N,) float confidence scores [0.0, 1.0]
    class_id: np.ndarray    # (N,) int class labels

    @classmethod
    def empty(cls) -> "Detection":
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=int),
        )

    def __len__(self) -> int:
        return len(self.xyxy)


class BaseDetector(ABC):
    """Abstract interface for object detectors."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Detection:
        """Execute detection on a single BGR video frame.

        Args:
            frame: OpenCV BGR image array (H, W, 3).

        Returns:
            Detection: Normalized detection results containing boxes, scores, and class IDs.
        """
        pass

