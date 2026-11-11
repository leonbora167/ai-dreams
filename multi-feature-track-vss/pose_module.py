import math
import numpy as np
import torch
from ultralytics import YOLO


class PoseAnalyzer:
    """
    Classifies person posture ('standing', 'sitting', or 'unknown') using
    YOLOv8n-Pose 17-keypoint skeletal geometry on both person crops and full
    auditorium scene keyframes.
    """

    # COCO keypoint indices
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    def __init__(self, model_path: str = "yolov8n-pose.pt", kpt_conf: float = 0.30):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.kpt_conf = kpt_conf

    @staticmethod
    def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Computes interior angle at vertex b (in degrees) for points a-b-c."""
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 180.0
        cos_val = float(np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0))
        return math.degrees(math.acos(cos_val))

    def _classify_keypoints(self, xy: np.ndarray, conf: np.ndarray) -> str:
        """
        Determines 'standing', 'sitting', or 'unknown' from (17, 2) xy and (17,) conf.
        """
        def _avg_pt(i1: int, i2: int):
            pts = []
            if conf[i1] >= self.kpt_conf:
                pts.append(xy[i1])
            if conf[i2] >= self.kpt_conf:
                pts.append(xy[i2])
            return np.mean(pts, axis=0) if pts else None

        shoulder = _avg_pt(self.L_SHOULDER, self.R_SHOULDER)
        hip = _avg_pt(self.L_HIP, self.R_HIP)
        knee = _avg_pt(self.L_KNEE, self.R_KNEE)
        ankle = _avg_pt(self.L_ANKLE, self.R_ANKLE)

        if hip is None or knee is None:
            return "unknown"

        dy_thigh = float(knee[1] - hip[1])
        torso_h = float(hip[1] - shoulder[1]) if shoulder is not None else 0.0

        # Check left/right knee bend angle if ankle is also visible
        knee_angles = []
        for h_i, k_i, a_i in [(self.L_HIP, self.L_KNEE, self.L_ANKLE), (self.R_HIP, self.R_KNEE, self.R_ANKLE)]:
            if conf[h_i] >= self.kpt_conf and conf[k_i] >= self.kpt_conf and conf[a_i] >= self.kpt_conf:
                knee_angles.append(self._angle_deg(xy[h_i], xy[k_i], xy[a_i]))

        if knee_angles:
            min_angle = min(knee_angles)
            if min_angle < 135.0:
                return "sitting"

        if torso_h > 8.0:
            thigh_ratio = dy_thigh / torso_h
            if thigh_ratio < 0.48:
                return "sitting"
            if ankle is not None:
                leg_span_ratio = float(ankle[1] - hip[1]) / torso_h
                if leg_span_ratio < 0.95:
                    return "sitting"
            return "standing"

        # Fallback if shoulders aren't visible cleanly
        thigh_len = float(np.linalg.norm(knee - hip))
        if thigh_len > 8.0 and (dy_thigh / thigh_len) < 0.55:
            return "sitting"
        return "standing"

    def classify_crop(self, crop: np.ndarray) -> str:
        """
        Runs pose estimation on a single person crop and returns 'standing', 'sitting', or 'unknown'.
        """
        if crop is None or crop.size == 0:
            return "unknown"

        results = self.model.predict(crop, conf=0.30, device=self.device, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return "unknown"

        # Pick largest/most centered person detection in crop
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else None
        best_idx = 0
        if boxes is not None and len(boxes) > 1:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = int(np.argmax(areas))

        xy = results[0].keypoints.xy[best_idx].cpu().numpy()
        conf = results[0].keypoints.conf[best_idx].cpu().numpy()
        return self._classify_keypoints(xy, conf)

    def analyze_scene(self, frame: np.ndarray) -> dict:
        """
        Runs pose estimation on a full scene keyframe to aggregate standing/sitting counts.
        """
        if frame is None or frame.size == 0:
            return {"person_count": 0, "standing_count": 0, "sitting_count": 0, "unknown_count": 0}

        results = self.model.predict(frame, conf=0.35, device=self.device, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return {"person_count": 0, "standing_count": 0, "sitting_count": 0, "unknown_count": 0}

        total = len(results[0].boxes)
        standing = 0
        sitting = 0
        unknown = 0

        if results[0].keypoints is not None and results[0].keypoints.conf is not None:
            kpts_xy = results[0].keypoints.xy.cpu().numpy()
            kpts_conf = results[0].keypoints.conf.cpu().numpy()
            for idx in range(len(kpts_xy)):
                pose = self._classify_keypoints(kpts_xy[idx], kpts_conf[idx])
                if pose == "standing":
                    standing += 1
                elif pose == "sitting":
                    sitting += 1
                else:
                    unknown += 1
        else:
            unknown = total

        return {
            "person_count": int(total),
            "standing_count": int(standing),
            "sitting_count": int(sitting),
            "unknown_count": int(unknown)
        }
