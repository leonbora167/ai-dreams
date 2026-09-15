"""Modular Pose Feature Extractor supporting YOLO11-Pose and RTMPose.

Extracts viewpoint orientation distributions and scale-invariant body proportions
from pedestrian tracklet crops.
"""
from typing import List, Optional
import cv2
import numpy as np
import torch
from .base import BaseFeatureExtractor
from ..device import resolve_device


class PoseFeatureExtractor(BaseFeatureExtractor):
    """Pose-based feature extractor for viewpoint orientation & geometric biometrics."""

    def __init__(self, spec: dict, cfg: dict):
        super().__init__(spec, cfg)
        self.model_type = str(spec.get('model_type', 'yolo11n-pose')).lower()
        self.device = resolve_device(cfg.get('detector', {}).get('device', 'auto'))
        self.min_kpt_conf = float(spec.get('min_kpt_conf', 0.25))

        self.yolo_model = None
        self.rtm_body = None

        if 'yolo' in self.model_type:
            from ultralytics import YOLO
            model_name = spec.get('model_name', 'yolo11n-pose.pt')
            self.yolo_model = YOLO(model_name)
        elif 'rtm' in self.model_type:
            from rtmlib import Body
            mode = spec.get('mode', 'lightweight')
            self.rtm_body = Body(mode=mode, to_openpose=False)
        else:
            raise ValueError(
                f"Unsupported pose model_type: '{self.model_type}'. "
                "Supported options: 'yolo11n-pose', 'rtmpose'."
            )

    def _extract_crop_keypoints(self, crop: np.ndarray):
        """Extract 17 COCO keypoints (x, y, conf) for a single pedestrian crop."""
        if crop is None or crop.size == 0:
            return None, None

        h, w = crop.shape[:2]
        if h < 20 or w < 10:
            return None, None

        if self.yolo_model is not None:
            results = self.yolo_model(crop, verbose=False, device=self.device)
            if not results or len(results) == 0 or results[0].keypoints is None:
                return None, None
            kpts_obj = results[0].keypoints
            if len(kpts_obj.xy) == 0:
                return None, None
            xy = kpts_obj.xy[0].cpu().numpy()  # (17, 2)
            if kpts_obj.conf is not None and len(kpts_obj.conf) > 0:
                conf = kpts_obj.conf[0].cpu().numpy()  # (17,)
            else:
                conf = np.ones(17, dtype=np.float32)
            return xy, conf

        if self.rtm_body is not None:
            try:
                kpts, scores = self.rtm_body.pose_model(crop)
                if kpts is None or len(kpts) == 0:
                    return None, None
                return kpts[0], scores[0]  # (17, 2), (17,)
            except Exception:
                return None, None

        return None, None

    def extract(self, crops: list, tracklet) -> np.ndarray:
        """Extract an 8-dimensional pose representation for a finalized tracklet.

        Descriptor Layout:
            [0:4] Viewpoint orientation distribution: [Front, Right_Profile, Back, Left_Profile]
            [4:7] Invariant biometric ratios: [shoulder/torso, hip/torso, leg/torso]
            [7]   Mean keypoint confidence
        """
        if not crops:
            return np.zeros(8, dtype=np.float32)

        view_counts = np.zeros(4, dtype=np.float32)  # 0: Front, 1: Right, 2: Back, 3: Left
        geom_ratios = []
        conf_values = []

        for crop in crops:
            kpts, conf = self._extract_crop_keypoints(crop)
            if kpts is None or conf is None:
                continue

            avg_conf = float(np.mean(conf))
            if avg_conf < self.min_kpt_conf:
                continue

            conf_values.append(avg_conf)
            h, w = crop.shape[:2]

            # COCO indices:
            # 0: nose, 1: l_eye, 2: r_eye, 3: l_ear, 4: r_ear
            # 5: l_shoulder, 6: r_shoulder, 11: l_hip, 12: r_hip
            # 15: l_ankle, 16: r_ankle
            l_sh, r_sh = kpts[5], kpts[6]
            l_hp, r_hp = kpts[11], kpts[12]
            l_ak, r_ak = kpts[15], kpts[16]

            sh_dx = float(l_sh[0] - r_sh[0])
            face_conf = float(np.mean(conf[0:3])) if len(conf) >= 3 else 0.0

            # Viewpoint Classification:
            # When facing camera (FRONT): Left shoulder is at greater x than right shoulder (sh_dx > 0)
            # When facing away (BACK): Left shoulder is at smaller x than right shoulder (sh_dx < 0)
            if face_conf > 0.35 and sh_dx > 0.10 * w:
                view_idx = 0  # Front
            elif sh_dx < -0.10 * w or (face_conf < 0.20 and sh_dx < 0):
                view_idx = 2  # Back
            else:
                # Profile view: person moving left or right
                nose_x = float(kpts[0][0])
                mid_sh_x = float((l_sh[0] + r_sh[0]) / 2.0)
                if nose_x > mid_sh_x:
                    view_idx = 1  # Right profile
                else:
                    view_idx = 3  # Left profile

            view_counts[view_idx] += 1.0

            # Scale-Invariant Biometrics:
            sh_mid = (l_sh + r_sh) / 2.0
            hp_mid = (l_hp + r_hp) / 2.0
            ak_mid = (l_ak + r_ak) / 2.0

            torso_len = float(np.linalg.norm(sh_mid - hp_mid))
            sh_width = float(np.linalg.norm(l_sh - r_sh))
            hp_width = float(np.linalg.norm(l_hp - r_hp))
            leg_len = float(np.linalg.norm(hp_mid - ak_mid))

            if torso_len > 12.0:
                r_sh = np.clip(sh_width / torso_len, 0.2, 1.5)
                r_hp = np.clip(hp_width / torso_len, 0.1, 1.2)
                r_leg = np.clip(leg_len / torso_len, 0.5, 2.5)
                geom_ratios.append([r_sh, r_hp, r_leg])

        total_views = np.sum(view_counts)
        if total_views > 0:
            view_dist = (view_counts / total_views).astype(np.float32)
        else:
            view_dist = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

        if geom_ratios:
            avg_geom = np.mean(geom_ratios, axis=0).astype(np.float32)
        else:
            avg_geom = np.array([0.7, 0.5, 1.3], dtype=np.float32)

        mean_c = float(np.mean(conf_values)) if conf_values else 0.0

        feat = np.concatenate([view_dist, avg_geom, [mean_c]]).astype(np.float32)
        return feat

    def similarity(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        """Compute pose similarity score between two tracklets in range [0.0, 1.0]."""
        if feat_a is None or feat_b is None or len(feat_a) < 8 or len(feat_b) < 8:
            return 0.5

        # Check for uninitialized descriptors
        if np.all(feat_a == 0) or np.all(feat_b == 0):
            return 0.5

        # 1. Viewpoint Compatibility Score
        view_a = feat_a[0:4]
        view_b = feat_b[0:4]
        hist_inter = float(np.sum(np.minimum(view_a, view_b)))
        cos_sim = float(np.dot(view_a, view_b) / (np.linalg.norm(view_a) * np.linalg.norm(view_b) + 1e-6))
        view_overlap = max(hist_inter, cos_sim)
        # Bounded between 0.4 (orthogonal/opposite viewpoints) and 1.0 (matching views)
        s_view = 0.4 + 0.6 * np.clip(view_overlap, 0.0, 1.0)

        # 2. Geometric Biometric Consistency Score
        geom_a = feat_a[4:7]
        geom_b = feat_b[4:7]
        diff = np.linalg.norm(geom_a - geom_b)
        s_geom = float(np.exp(- (diff ** 2) / (2.0 * (0.35 ** 2))))

        # Combined pose similarity
        s_pose = float(0.5 * s_view + 0.5 * s_geom)
        return float(np.clip(s_pose, 0.0, 1.0))

