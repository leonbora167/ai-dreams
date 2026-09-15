"""Modular feature extraction subsystem and FeatureManager orchestrator."""
from typing import Dict, Type
import numpy as np
from .base import BaseFeatureExtractor
from .reid import ReIDFeatureExtractor
from .color import ColorFeatureExtractor
from .pose import PoseFeatureExtractor

_FEATURE_REGISTRY: Dict[str, Type[BaseFeatureExtractor]] = {
    'reid': ReIDFeatureExtractor,
    'color': ColorFeatureExtractor,
    'pose': PoseFeatureExtractor,
}


def register_feature(name: str, extractor_cls: Type[BaseFeatureExtractor]):
    """Register a custom feature extractor in the registry."""
    _FEATURE_REGISTRY[name.lower()] = extractor_cls


class FeatureManager:
    """Orchestrates feature extraction and multi-modal similarity computation."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.features_cfg = cfg.get('features', {})
        self.extractors: Dict[str, BaseFeatureExtractor] = {}

        for name, spec in self.features_cfg.items():
            if not isinstance(spec, dict) or not spec.get('enabled', False):
                continue
            name_lower = name.lower()
            if name_lower in _FEATURE_REGISTRY:
                extractor_cls = _FEATURE_REGISTRY[name_lower]
                self.extractors[name] = extractor_cls(spec, cfg)

    def extract_crops(self, crops: list, tracklet) -> dict:
        """Extract multi-modal descriptors for a finalized tracklet."""
        result = {}

        # Quality gating check
        quality_cfg = self.features_cfg.get('quality', {})
        min_conf = quality_cfg.get('min_confidence', 0.0)
        avg_conf = float(np.mean(tracklet.confidences)) if tracklet.confidences else 0.0
        result['quality'] = avg_conf

        if quality_cfg.get('enabled', False) and avg_conf < min_conf:
            result['_skip'] = True
            return result

        for name, extractor in self.extractors.items():
            try:
                result[name] = extractor.extract(crops, tracklet)
            except Exception as e:
                print(f"Warning: Feature extractor '{name}' failed: {e}")

        return result

    def compute_similarity(self, desc_a: dict, desc_b: dict) -> float:
        """Compute weighted similarity score between two multi-modal descriptors."""
        scores = []
        weights = []

        for name, extractor in self.extractors.items():
            if name in desc_a and name in desc_b:
                feat_a = desc_a[name]
                feat_b = desc_b[name]
                if isinstance(feat_a, np.ndarray) and isinstance(feat_b, np.ndarray):
                    sim = extractor.similarity(feat_a, feat_b)
                    scores.append(sim)
                    weights.append(extractor.weight)

        if not scores or sum(weights) == 0:
            return -1.0

        return float(np.average(scores, weights=weights))


__all__ = [
    'BaseFeatureExtractor',
    'ReIDFeatureExtractor',
    'ColorFeatureExtractor',
    'PoseFeatureExtractor',
    'FeatureManager',
    'register_feature'
]

