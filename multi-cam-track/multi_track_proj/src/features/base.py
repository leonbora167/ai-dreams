"""Abstract Base Feature Extractor interface."""
from abc import ABC, abstractmethod
import numpy as np


class BaseFeatureExtractor(ABC):
    """Abstract interface for feature extractors (appearance, color, pose, motion, etc.)."""

    def __init__(self, spec: dict, cfg: dict):
        self.spec = spec
        self.cfg = cfg
        self.weight = float(spec.get('weight', 1.0))
        self.enabled = bool(spec.get('enabled', True))

    @abstractmethod
    def extract(self, crops: list, tracklet) -> np.ndarray:
        """Extract a 1D feature representation for a finalized tracklet.

        Args:
            crops: List of representative BGR image crops (np.ndarray).
            tracklet: Tracklet instance containing trajectory and bounding boxes.

        Returns:
            np.ndarray: 1D feature vector.
        """
        pass

    @abstractmethod
    def similarity(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        """Compute similarity score between two feature vectors in range [-1.0, 1.0] or [0.0, 1.0].

        Args:
            feat_a: Feature array from tracklet A.
            feat_b: Feature array from tracklet B.

        Returns:
            float: Similarity metric (higher = more similar).
        """
        pass

