"""Modular detector subsystem and factory registry."""
from typing import Dict, Type
from .base import BaseDetector, Detection
from .yolo import YOLODetector
from .rfdetr import RFDETRDetector

_DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {
    'yolo': YOLODetector,
    'yolov8': YOLODetector,
    'yolov9': YOLODetector,
    'yolov10': YOLODetector,
    'yolo11': YOLODetector,
    'rtdetr': YOLODetector,
    'rfdetr': RFDETRDetector,
    'rf-detr': RFDETRDetector,
}


def register_detector(name: str, detector_cls: Type[BaseDetector]):
    """Register a custom detector class in the registry."""
    _DETECTOR_REGISTRY[name.lower()] = detector_cls


def get_detector(cfg: dict) -> BaseDetector:
    """Factory function: instantiates the configured detector."""
    det_cfg = cfg.get('detector', {})
    det_type = det_cfg.get('name') or det_cfg.get('type')
    model_name = det_cfg.get('model', 'yolo11n.pt')

    # If 'name' or 'type' is not explicitly specified, infer from model path
    if not det_type:
        model_lower = str(model_name).lower()
        if 'rtdetr' in model_lower:
            det_type = 'rtdetr'
        elif 'rfdetr' in model_lower or 'rf-detr' in model_lower:
            det_type = 'rfdetr'
        else:
            det_type = 'yolo'

    det_cls = _DETECTOR_REGISTRY.get(det_type.lower())
    if det_cls is None:
        raise ValueError(
            f"Unsupported detector type '{det_type}'. Registered types: {list(_DETECTOR_REGISTRY.keys())}"
        )

    return det_cls(cfg)


__all__ = ['BaseDetector', 'Detection', 'YOLODetector', 'RFDETRDetector', 'get_detector', 'register_detector']

