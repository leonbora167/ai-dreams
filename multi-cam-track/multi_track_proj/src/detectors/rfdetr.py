"""Roboflow Detection Transformer (RF-DETR) detector implementation.

Apache 2.0 licensed transformer detector loaded from Hugging Face / Roboflow.
"""
import cv2
import numpy as np
from .base import BaseDetector, Detection


class RFDETRDetector(BaseDetector):
    """Wraps Roboflow RF-DETR models for human detection."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        det_cfg = cfg.get('detector', {})
        self.conf = float(det_cfg.get('confidence', 0.35))
        self.classes = det_cfg.get('classes', [0])
        model_name = str(det_cfg.get('model', 'rf-detr-nano')).lower()

        import rfdetr
        if 'small' in model_name:
            self.model = rfdetr.RFDETRSmall(trust_checkpoint=True)
        elif 'medium' in model_name:
            self.model = rfdetr.RFDETRMedium(trust_checkpoint=True)
        elif 'large' in model_name:
            self.model = rfdetr.RFDETRLarge(trust_checkpoint=True)
        elif 'base' in model_name:
            self.model = rfdetr.RFDETRBase(trust_checkpoint=True)
        else:
            self.model = rfdetr.RFDETRNano(trust_checkpoint=True)

    def detect(self, frame: np.ndarray) -> Detection:
        if frame is None or frame.size == 0:
            return Detection.empty()

        # Convert OpenCV BGR to contiguous RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            detections = self.model.predict(rgb, threshold=self.conf)
        except Exception:
            return Detection.empty()

        if len(detections.xyxy) == 0:
            return Detection.empty()

        boxes = detections.xyxy
        confs = detections.confidence
        classes = detections.class_id
        class_names = detections.data.get('class_name')

        # Filter for person: class_name == 'person' or class_id == 1 (COCO 91) or class_id in self.classes
        if class_names is not None and len(class_names) == len(boxes):
            mask = (class_names == 'person')
        else:
            # Fallback to class_id checking
            mask = np.isin(classes, [0, 1])

        if not np.any(mask):
            return Detection.empty()

        filtered_boxes = boxes[mask].astype(np.float32)
        filtered_confs = confs[mask].astype(np.float32)
        # Standardize class_id to 0 for downstream person tracking
        filtered_classes = np.zeros(len(filtered_boxes), dtype=int)

        return Detection(
            xyxy=filtered_boxes,
            confidence=filtered_confs,
            class_id=filtered_classes,
        )

