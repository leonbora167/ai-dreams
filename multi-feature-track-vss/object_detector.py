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

    def detect_roi(
        self,
        frame: np.ndarray,
        roi_bbox: List[float],
        pad_ratio: float = 2.0,
        min_crop_size: int = 520,
        conf_override: float = 0.16,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Runs zoomed-in open-vocabulary detection around a recently lost object's
        bounding box so tiny stationary items (e.g. floor bags in 4K auditoriums)
        are recovered at full resolution without slowing down the whole frame.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = roi_bbox
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        bw = max(float(x2 - x1) * pad_ratio, float(min_crop_size))
        bh = max(float(y2 - y1) * pad_ratio, float(min_crop_size))

        rx1 = max(0, int(cx - 0.5 * bw))
        ry1 = max(0, int(cy - 0.5 * bh))
        rx2 = min(w, int(cx + 0.5 * bw))
        ry2 = min(h, int(cy + 0.5 * bh))
        if rx2 <= rx1 or ry2 <= ry1:
            return np.empty((0, 5), dtype=np.float32), []

        roi_crop = frame[ry1:ry2, rx1:rx2]
        results = self.model(
            roi_crop,
            conf=conf_override,
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
                    float(box[0] + rx1),
                    float(box[1] + ry1),
                    float(box[2] + rx1),
                    float(box[3] + ry1),
                    float(conf),
                ])
                labels.append(label)

        if not detections:
            return np.empty((0, 5), dtype=np.float32), []
        return np.array(detections, dtype=np.float32), labels

    def detect_tiled(self, frame: np.ndarray, overlap: float = 0.15) -> Tuple[np.ndarray, List[str]]:
        """
        Runs 2x2 overlapping spatial tiled detection for high-resolution 4K frames,
        merging cross-tile boxes with class-agnostic NMS.
        """
        h, w = frame.shape[:2]
        mid_x, mid_y = w * 0.5, h * 0.5
        ox, oy = w * (overlap * 0.5), h * (overlap * 0.5)
        tiles = [
            (0, 0, int(mid_x + ox), int(mid_y + oy)),
            (int(mid_x - ox), 0, w, int(mid_y + oy)),
            (0, int(mid_y - oy), int(mid_x + ox), h),
            (int(mid_x - ox), int(mid_y - oy), w, h),
        ]

        all_boxes = []
        all_labels = []
        for rx1, ry1, rx2, ry2 in tiles:
            crop = frame[ry1:ry2, rx1:rx2]
            dets, lbls = self.detect(crop)
            for row, lbl in zip(dets, lbls):
                all_boxes.append([row[0] + rx1, row[1] + ry1, row[2] + rx1, row[3] + ry1, row[4]])
                all_labels.append(lbl)

        if not all_boxes:
            return np.empty((0, 5), dtype=np.float32), []

        # Simple greedy NMS across overlapping tile borders
        arr = np.array(all_boxes, dtype=np.float32)
        order = np.argsort(-arr[:, 4])
        keep = []
        while len(order) > 0:
            idx = int(order[0])
            keep.append(idx)
            if len(order) == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(arr[idx, 0], arr[rest, 0])
            yy1 = np.maximum(arr[idx, 1], arr[rest, 1])
            xx2 = np.minimum(arr[idx, 2], arr[rest, 2])
            yy2 = np.minimum(arr[idx, 3], arr[rest, 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            area_i = (arr[idx, 2] - arr[idx, 0]) * (arr[idx, 3] - arr[idx, 1])
            area_r = (arr[rest, 2] - arr[rest, 0]) * (arr[rest, 3] - arr[rest, 1])
            iou = inter / np.maximum(1.0, area_i + area_r - inter)
            order = rest[iou <= 0.45]

        return arr[keep], [all_labels[i] for i in keep]
