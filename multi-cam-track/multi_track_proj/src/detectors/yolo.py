"""Ultralytics YOLO and RT-DETR detector implementation."""
import numpy as np
from ultralytics import YOLO
from .base import BaseDetector, Detection
from ..device import resolve_device


class YOLODetector(BaseDetector):
    """Wraps Ultralytics YOLO/RT-DETR models for human detection."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        det_cfg = cfg['detector']
        model_path = det_cfg['model']
        self.conf = det_cfg.get('confidence', 0.35)
        self.classes = det_cfg.get('classes', [0])
        self.device = resolve_device(det_cfg.get('device', 'auto'))
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> Detection:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            classes=self.classes,
            device=self.device,
            verbose=False
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return Detection.empty()

        return Detection(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.int().cpu().numpy(),
        )
