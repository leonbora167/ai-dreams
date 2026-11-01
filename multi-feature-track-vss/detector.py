import numpy as np
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.40):
        self.conf_thresh = conf_thresh
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Runs person detection on a single BGR video frame.
        Returns:
            bboxes: np.ndarray of shape (N, 5) with [x1, y1, x2, y2, confidence]
            crops: list of np.ndarray image crops corresponding to each detected person
        """
        if frame is None or frame.size == 0:
            return np.empty((0, 5), dtype=np.float32), []

        # Run inference filtering for class 0 (person)
        results = self.model(frame, classes=[0], conf=self.conf_thresh, verbose=False)
        
        bboxes_list = []
        crops = []
        h, w = frame.shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                x1 = max(0, min(int(round(xyxy[0])), w - 1))
                y1 = max(0, min(int(round(xyxy[1])), h - 1))
                x2 = max(0, min(int(round(xyxy[2])), w))
                y2 = max(0, min(int(round(xyxy[3])), h))
                
                # Check valid crop dimensions
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2].copy()
                    bboxes_list.append([x1, y1, x2, y2, conf])
                    crops.append(crop)

        if len(bboxes_list) == 0:
            return np.empty((0, 5), dtype=np.float32), []

        bboxes = np.array(bboxes_list, dtype=np.float32)
        return bboxes, crops

