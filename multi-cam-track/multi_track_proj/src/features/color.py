"""HSV spatial color distribution feature extractor."""
import cv2
import numpy as np
from .base import BaseFeatureExtractor


class ColorFeatureExtractor(BaseFeatureExtractor):
    """HSV color histogram appearance extractor."""

    def __init__(self, spec: dict, cfg: dict):
        super().__init__(spec, cfg)
        self.h_bins = int(spec.get('h_bins', 6))
        self.s_bins = int(spec.get('s_bins', 4))

    def extract(self, crops: list, tracklet) -> np.ndarray:
        if not crops:
            return np.zeros(self.h_bins * self.s_bins, dtype=np.float32)

        hist_list = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [self.h_bins, self.s_bins], [0, 180, 0, 256])
            norm_hist = cv2.normalize(hist, None).flatten()
            hist_list.append(norm_hist)

        if not hist_list:
            return np.zeros(self.h_bins * self.s_bins, dtype=np.float32)

        pooled = np.mean(hist_list, axis=0)
        norm = np.linalg.norm(pooled) + 1e-8
        return (pooled / norm).astype(np.float32)

    def similarity(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        dot = float(np.dot(feat_a, feat_b))
        norm = float((np.linalg.norm(feat_a) * np.linalg.norm(feat_b)) + 1e-8)
        return float(dot / norm)

