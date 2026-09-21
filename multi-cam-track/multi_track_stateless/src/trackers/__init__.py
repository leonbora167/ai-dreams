"""Stateful single-camera tracker registry (OC-SORT and BoT-SORT)."""
from typing import Dict, Type
from .base import BaseTracker, TrackedDetection
from .ocsort import OCSortTracker
from .botsort import BoTSORTTracker

_TRACKER_REGISTRY: Dict[str, Type[BaseTracker]] = {
    'ocsort': OCSortTracker,
    'oc-sort': OCSortTracker,
    'oc_sort': OCSortTracker,
    'botsort': BoTSORTTracker,
    'bot-sort': BoTSORTTracker,
    'bot_sort': BoTSORTTracker,
}


def register_tracker(name: str, tracker_cls: Type[BaseTracker]):
    """Register a tracker implementation."""
    _TRACKER_REGISTRY[name.lower()] = tracker_cls


def get_tracker(cfg: dict, fps: float) -> BaseTracker:
    """Instantiate the stateful tracker for a camera stream."""
    tracker_cfg = cfg.get('tracker', {})
    tracker_name = tracker_cfg.get('name') or tracker_cfg.get('type', 'ocsort')
    tracker_cls = _TRACKER_REGISTRY.get(str(tracker_name).lower())
    if tracker_cls is None:
        raise ValueError(
            f"Unsupported tracker '{tracker_name}'. Available production trackers: {list(_TRACKER_REGISTRY.keys())}"
        )
    return tracker_cls(cfg, fps)


__all__ = [
    'BaseTracker',
    'TrackedDetection',
    'OCSortTracker',
    'BoTSORTTracker',
    'get_tracker',
    'register_tracker'
]

