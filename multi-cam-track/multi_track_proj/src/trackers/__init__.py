"""Modular tracker subsystem and factory registry."""
from typing import Dict, Type
from .base import BaseTracker, TrackedDetection
from .bytetrack import ByteTrackTracker
from .botsort import BoTSORTTracker
from .ocsort import OCSortTracker
from .deepocsort import DeepOCSortTracker
from .strongsort import StrongSortTracker
from .deepsort import DeepSortTracker
from .norfair_tracker import NorfairTrackerWrapper

_TRACKER_REGISTRY: Dict[str, Type[BaseTracker]] = {
    'bytetrack': ByteTrackTracker,
    'byte_track': ByteTrackTracker,
    'botsort': BoTSORTTracker,
    'bot_sort': BoTSORTTracker,
    'ocsort': OCSortTracker,
    'oc-sort': OCSortTracker,
    'oc_sort': OCSortTracker,
    'deepocsort': DeepOCSortTracker,
    'deep-oc-sort': DeepOCSortTracker,
    'deep_oc_sort': DeepOCSortTracker,
    'strongsort': StrongSortTracker,
    'strong-sort': StrongSortTracker,
    'strong_sort': StrongSortTracker,
    'deepsort': DeepSortTracker,
    'deep-sort': DeepSortTracker,
    'deep_sort': DeepSortTracker,
    'norfair': NorfairTrackerWrapper,
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


__all__ = [
    'BaseTracker',
    'TrackedDetection',
    'ByteTrackTracker',
    'BoTSORTTracker',
    'OCSortTracker',
    'DeepOCSortTracker',
    'StrongSortTracker',
    'DeepSortTracker',
    'NorfairTrackerWrapper',
    'get_tracker',
    'register_tracker'
]

