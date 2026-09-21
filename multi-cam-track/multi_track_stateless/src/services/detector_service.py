"""Stateless Detector Service.

Pure functional detector interface (input: frame -> output: detections).
Holds no state, no tracking memory, and is directly convertible to a remote Triton Inference Server client.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import cv2
import numpy as np


@dataclass
class Detection:
    """Standardized detection container."""
    xyxy: np.ndarray        # (N, 4) float bounding boxes [x1, y1, x2, y2]
    confidence: np.ndarray  # (N,) float confidence scores [0.0, 1.0]
    class_id: np.ndarray    # (N,) int class labels

    @classmethod
    def empty(cls) -> "Detection":
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=int),
        )

    def __len__(self) -> int:
        return len(self.xyxy)


class BaseDetectorService(ABC):
    """Abstract interface for stateless detector services."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Detection:
        """Run inference on a single image frame."""
        pass


class LocalRFDETRService(BaseDetectorService):
    """Local RF-DETR transformer detector service (Apache 2.0)."""

    def __init__(self, model_name: str = "rf-detr-nano", confidence: float = 0.35, classes: List[int] = None):
        import rfdetr
        self.conf = float(confidence)
        self.classes = classes or [0]
        model_str = str(model_name).lower()

        if "small" in model_str:
            self.model = rfdetr.RFDETRSmall(trust_checkpoint=True)
        elif "medium" in model_str:
            self.model = rfdetr.RFDETRMedium(trust_checkpoint=True)
        elif "large" in model_str:
            self.model = rfdetr.RFDETRLarge(trust_checkpoint=True)
        elif "base" in model_str:
            self.model = rfdetr.RFDETRBase(trust_checkpoint=True)
        else:
            self.model = rfdetr.RFDETRNano(trust_checkpoint=True)

    def detect(self, frame: np.ndarray) -> Detection:
        if frame is None or frame.size == 0:
            return Detection.empty()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            detections = self.model.predict(rgb, threshold=self.conf)
        except Exception:
            return Detection.empty()

        if len(detections.xyxy) == 0:
            return Detection.empty()

        boxes = detections.xyxy
        confs = detections.confidence
        class_names = detections.data.get("class_name")

        if class_names is not None and len(class_names) == len(boxes):
            mask = (class_names == "person")
        else:
            mask = np.isin(detections.class_id, [0, 1])

        if not np.any(mask):
            return Detection.empty()

        filtered_boxes = boxes[mask].astype(np.float32)
        filtered_confs = confs[mask].astype(np.float32)
        filtered_classes = np.zeros(len(filtered_boxes), dtype=int)

        return Detection(
            xyxy=filtered_boxes,
            confidence=filtered_confs,
            class_id=filtered_classes,
        )


class LocalYOLOService(BaseDetectorService):
    """Local Ultralytics YOLO detector service."""

    def __init__(self, model_path: str = "yolo11n.pt", confidence: float = 0.35, classes: List[int] = None, device: str = "auto"):
        from ultralytics import YOLO
        from ..device import resolve_device
        self.conf = float(confidence)
        self.classes = classes or [0]
        self.device = resolve_device(device)
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> Detection:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return Detection.empty()

        return Detection(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.int().cpu().numpy(),
        )


class TritonDetectorService(BaseDetectorService):
    """Stateless Triton Inference Server client for remote object detection.

    Ready for production microservice deployment:
      client = TritonDetectorService(url="localhost:8001", model_name="rfdetr_nano")
    """

    def __init__(self, url: str = "localhost:8001", model_name: str = "detector", confidence: float = 0.35):
        self.url = url
        self.model_name = model_name
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> Detection:
        # Example Triton gRPC / HTTP payload transmission
        # When deploying on Triton, uncomment tritonclient logic:
        # import tritonclient.grpc as grpcclient
        # client = grpcclient.InferenceServerClient(url=self.url)
        # ...
        raise NotImplementedError("Connect to live Triton server with tritonclient")


def create_detector_service(cfg: dict) -> BaseDetectorService:
    """Factory creating the stateless detector service."""
    det_cfg = cfg.get("detector", {})
    name = str(det_cfg.get("name") or det_cfg.get("type", "rfdetr")).lower()
    model = det_cfg.get("model", "rf-detr-nano")
    conf = float(det_cfg.get("confidence", 0.35))
    classes = det_cfg.get("classes", [0])
    device = det_cfg.get("device", "auto")

    if "triton" in name:
        url = det_cfg.get("triton_url", "localhost:8001")
        return TritonDetectorService(url=url, model_name=model, confidence=conf)
    elif "rfdetr" in name or "rf-detr" in name:
        return LocalRFDETRService(model_name=model, confidence=conf, classes=classes)
    else:
        return LocalYOLOService(model_path=model, confidence=conf, classes=classes, device=device)

