"""
Open-vocabulary object detection module powered by YOLO-World v2.
Runs independently from PersonDetector so the existing person-detection pipeline
remains completely unchanged.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLOWorld


class ObjectDetector:
    """
    Open-vocabulary object detector wrapping YOLOWorld (yolov8s-worldv2.pt).
    Configured once per pipeline run with user-specified target class prompts
    (e.g., ['bag', 'backpack', 'handbag', 'suitcase', 'water bottle']).
    """

    def __init__(
        self,
        target_classes: List[str],
        model_path: str = "yolov8s-worldv2.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.50,
        device: Optional[str] = None,
    ):
        cleaned = [c.strip() for c in target_classes if c and c.strip()]
        if not cleaned:
            raise ValueError("ObjectDetector requires at least one non-empty target class.")

        self.target_classes = cleaned
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = YOLOWorld(model_path)
        self.model.set_classes(self.target_classes)
        print(
            f"[ObjectDetector] Initialized '{model_path}' on {self.device} "
            f"for classes: {self.target_classes} (conf>={self.conf_threshold})"
        )

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Run open-vocabulary detection on a BGR frame.

        Returns:
            detections: np.ndarray of shape (N, 5) with [x1, y1, x2, y2, conf].
            labels: List of N class name strings matching each row in detections.
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )

        detections = []
        labels = []
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_idx in zip(xyxy, confs, clss):
                label = (
                    self.target_classes[cls_idx]
                    if 0 <= cls_idx < len(self.target_classes)
                    else str(cls_idx)
                )
                detections.append([
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    float(conf),
                ])
                labels.append(label)

        if not detections:
            return np.empty((0, 5), dtype=np.float32), []

        return np.array(detections, dtype=np.float32), labels
