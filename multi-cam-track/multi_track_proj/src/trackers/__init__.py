"""Modular tracker subsystem and factory registry."""
from typing import Dict, Type
from .base import BaseTracker, TrackedDetection
from .bytetrack import ByteTrackTracker
from .botsort import BoTSORTTracker

_TRACKER_REGISTRY: Dict[str, Type[BaseTracker]] = {
    'bytetrack': ByteTrackTracker,
    'botsort': BoTSORTTracker,
    'bot_sort': BoTSORTTracker,
}


def register_tracker(name: str, tracker_cls: Type[BaseTracker]):
    """Register a custom tracker class in the registry."""
    _TRACKER_REGISTRY[name.lower()] = tracker_cls


def get_tracker(cfg: dict, fps: float) -> BaseTracker:
    """Factory function: instantiates the configured tracker."""
    tracker_cfg = cfg.get('tracker', {})
    tracker_type = tracker_cfg.get('name') or tracker_cfg.get('type', 'bytetrack')
    tracker_cls = _TRACKER_REGISTRY.get(str(tracker_type).lower())
    if tracker_cls is None:
        raise ValueError(
            f"Unsupported tracker type '{tracker_type}'. Registered types: {list(_TRACKER_REGISTRY.keys())}"
        )

    return tracker_cls(cfg, fps)


__all__ = ['BaseTracker', 'TrackedDetection', 'ByteTrackTracker', 'BoTSORTTracker', 'get_tracker', 'register_tracker']

