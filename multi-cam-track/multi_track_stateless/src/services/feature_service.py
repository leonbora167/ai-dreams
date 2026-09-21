"""Stateless Feature Service.

Pure functional feature extraction interface (input: list of crops -> output: feature descriptors).
Holds no track state, no tracklets, and no gallery memory.
Ready for production hosting on Triton Inference Server.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import torch
from ..device import resolve_device


class BaseReIDExtractor(ABC):
    """Abstract interface for stateless ReID extractors."""
    @abstractmethod
    def extract_embedding(self, crops: List[np.ndarray]) -> np.ndarray:
        pass


class LocalOSNetExtractor(BaseReIDExtractor):
    """Local OSNet deep metric learning appearance extractor."""

    def __init__(self, weight_path: str, model_name: str = "osnet_ain_x1_0", input_size: tuple = (256, 128), device: str = "auto"):
        import torchreid
        self.device = resolve_device(device)
        self.input_size = input_size
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

        p = Path(weight_path)
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,
            pretrained=not p.exists()
        )
        if p.exists():
            torchreid.utils.load_pretrained_weights(self.model, str(p))

        self.model.to(self.device)
        self.model.eval()

    def extract_embedding(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros(512, dtype=np.float32)

        target_h, target_w = int(self.input_size[0]), int(self.input_size[1])
        batch = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            resized = cv2.resize(crop, (target_w, target_h))[:, :, ::-1]
            img = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
            normalized = (img - self.mean) / self.std
            batch.append(normalized)

        if not batch:
            return np.zeros(512, dtype=np.float32)

        with torch.no_grad():
            batch_tensor = torch.tensor(np.asarray(batch), dtype=torch.float32, device=self.device)
            embeddings = self.model(batch_tensor).cpu().numpy()

        pooled = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(pooled) + 1e-8
        return (pooled / norm).astype(np.float32)


class TritonReIDExtractor(BaseReIDExtractor):
    """Stateless Triton Inference Server client for remote ReID feature extraction.

    Drop-in replacement for production environments:
      reid_client = TritonReIDExtractor(url="localhost:8001", model_name="osnet_ain")
    """

    def __init__(self, url: str = "localhost:8001", model_name: str = "osnet_ain"):
        self.url = url
        self.model_name = model_name

    def extract_embedding(self, crops: List[np.ndarray]) -> np.ndarray:
        # Example Triton gRPC / HTTP payload transmission
        # When deploying on Triton, uncomment tritonclient logic:
        # import tritonclient.grpc as grpcclient
        # client = grpcclient.InferenceServerClient(url=self.url)
        # ...
        raise NotImplementedError("Connect to live Triton server with tritonclient")


class StatelessColorExtractor:
    """HSV color histogram spatial distribution extractor."""

    def __init__(self, h_bins: int = 6, s_bins: int = 4):
        self.h_bins = h_bins
        self.s_bins = s_bins

    def extract(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros(self.h_bins * self.s_bins, dtype=np.float32)

        hist_list = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [self.h_bins, self.s_bins], [0, 180, 0, 256])
            norm_hist = cv2.normalize(hist, None).flatten()
            hist_list.append(norm_hist)

        if not hist_list:
            return np.zeros(self.h_bins * self.s_bins, dtype=np.float32)

        pooled = np.mean(hist_list, axis=0)
        norm = np.linalg.norm(pooled) + 1e-8
        return (pooled / norm).astype(np.float32)


class StatelessPoseExtractor:
    """Pose viewpoint heading & scale-invariant body ratio extractor (Strategy 1)."""

    def __init__(self, model_type: str = "yolo11n-pose", device: str = "auto", min_kpt_conf: float = 0.25):
        self.model_type = str(model_type).lower()
        self.device = resolve_device(device)
        self.min_kpt_conf = float(min_kpt_conf)

        self.yolo_model = None
        self.rtm_body = None

        if "yolo" in self.model_type:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolo11n-pose.pt")
        elif "rtm" in self.model_type:
            from rtmlib import Body
            self.rtm_body = Body(mode="lightweight", to_openpose=False)

    def _extract_crop_keypoints(self, crop: np.ndarray):
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
            xy = kpts_obj.xy[0].cpu().numpy()
            conf = kpts_obj.conf[0].cpu().numpy() if kpts_obj.conf is not None else np.ones(17, dtype=np.float32)
            return xy, conf

        if self.rtm_body is not None:
            try:
                kpts, scores = self.rtm_body.pose_model(crop)
                if kpts is None or len(kpts) == 0:
                    return None, None
                return kpts[0], scores[0]
            except Exception:
                return None, None

        return None, None

    def extract(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros(8, dtype=np.float32)

        view_counts = np.zeros(4, dtype=np.float32)
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

            l_sh, r_sh = kpts[5], kpts[6]
            l_hp, r_hp = kpts[11], kpts[12]
            l_ak, r_ak = kpts[15], kpts[16]

            sh_dx = float(l_sh[0] - r_sh[0])
            face_conf = float(np.mean(conf[0:3])) if len(conf) >= 3 else 0.0

            if face_conf > 0.35 and sh_dx > 0.10 * w:
                view_idx = 0  # Front
            elif sh_dx < -0.10 * w or (face_conf < 0.20 and sh_dx < 0):
                view_idx = 2  # Back
            else:
                nose_x = float(kpts[0][0])
                mid_sh_x = float((l_sh[0] + r_sh[0]) / 2.0)
                view_idx = 1 if nose_x > mid_sh_x else 3

            view_counts[view_idx] += 1.0

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
        view_dist = (view_counts / total_views).astype(np.float32) if total_views > 0 else np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        avg_geom = np.mean(geom_ratios, axis=0).astype(np.float32) if geom_ratios else np.array([0.7, 0.5, 1.3], dtype=np.float32)
        mean_c = float(np.mean(conf_values)) if conf_values else 0.0

        return np.concatenate([view_dist, avg_geom, [mean_c]]).astype(np.float32)


class StatelessFeatureService:
    """Orchestrates stateless multi-modal feature extraction.

    Dispatches crops to ReID, Color, and Pose models without holding any tracking state.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.f_cfg = cfg.get("features", {})
        dev = cfg.get("detector", {}).get("device", "auto")

        # 1. ReID Service
        reid_spec = self.f_cfg.get("reid", {})
        self.reid_enabled = bool(reid_spec.get("enabled", True))
        self.reid_weight = float(reid_spec.get("weight", 0.8))
        if self.reid_enabled:
            reid_type = str(reid_spec.get("type", "local")).lower()
            if "triton" in reid_type:
                self.reid_extractor = TritonReIDExtractor(
                    url=reid_spec.get("triton_url", "localhost:8001"),
                    model_name=reid_spec.get("model", "osnet_ain")
                )
            else:
                self.reid_extractor = LocalOSNetExtractor(
                    weight_path=reid_spec.get("weight_path", "./models/osnet_ain_x1_0_msmt17_256x128.pth"),
                    model_name=reid_spec.get("model", "osnet_ain_x1_0"),
                    input_size=reid_spec.get("input_size", [256, 128]),
                    device=dev
                )
        else:
            self.reid_extractor = None

        # 2. Color Service
        color_spec = self.f_cfg.get("color", {})
        self.color_enabled = bool(color_spec.get("enabled", True))
        self.color_weight = float(color_spec.get("weight", 0.1))
        self.color_extractor = StatelessColorExtractor(
            h_bins=int(color_spec.get("h_bins", 6)),
            s_bins=int(color_spec.get("s_bins", 4))
        ) if self.color_enabled else None

        # 3. Pose Service
        pose_spec = self.f_cfg.get("pose", {})
        self.pose_enabled = bool(pose_spec.get("enabled", False))
        self.pose_weight = float(pose_spec.get("weight", 0.1))
        self.pose_extractor = StatelessPoseExtractor(
            model_type=pose_spec.get("model_type", "yolo11n-pose"),
            device=dev,
            min_kpt_conf=float(pose_spec.get("min_kpt_conf", 0.25))
        ) if self.pose_enabled else None

    def extract_descriptors(self, crops: List[np.ndarray], avg_conf: float = 1.0) -> dict:
        """Pure functional extraction: crops in -> multi-modal descriptors out."""
        result = {}

        # Quality gating check
        q_cfg = self.f_cfg.get("quality", {})
        if q_cfg.get("enabled", False):
            if avg_conf < float(q_cfg.get("min_confidence", 0.0)):
                result["_skip"] = True
                return result

        if self.reid_enabled and self.reid_extractor is not None:
            try:
                result["reid"] = self.reid_extractor.extract_embedding(crops)
            except Exception as e:
                print(f"Warning: ReID extraction failed: {e}")

        if self.color_enabled and self.color_extractor is not None:
            try:
                result["color"] = self.color_extractor.extract(crops)
            except Exception as e:
                print(f"Warning: Color extraction failed: {e}")

        if self.pose_enabled and self.pose_extractor is not None:
            try:
                result["pose"] = self.pose_extractor.extract(crops)
            except Exception as e:
                print(f"Warning: Pose extraction failed: {e}")

        return result

    def compute_similarity(self, desc_a: dict, desc_b: dict) -> float:
        """Compute weighted multi-modal similarity score in range [0.0, 1.0]."""
        scores = []
        weights = []

        # 1. ReID appearance cosine similarity
        if "reid" in desc_a and "reid" in desc_b and self.reid_enabled:
            fa, fb = desc_a["reid"], desc_b["reid"]
            sim = float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-8))
            scores.append(sim)
            weights.append(self.reid_weight)

        # 2. Color histogram cosine similarity
        if "color" in desc_a and "color" in desc_b and self.color_enabled:
            fa, fb = desc_a["color"], desc_b["color"]
            sim = float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-8))
            scores.append(sim)
            weights.append(self.color_weight)

        # 3. Pose viewpoint + geometry similarity
        if "pose" in desc_a and "pose" in desc_b and self.pose_enabled:
            fa, fb = desc_a["pose"], desc_b["pose"]
            if len(fa) >= 8 and len(fb) >= 8:
                # Viewpoint compatibility
                va, vb = fa[0:4], fb[0:4]
                overlap = max(float(np.sum(np.minimum(va, vb))), float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-6)))
                s_view = 0.4 + 0.6 * np.clip(overlap, 0.0, 1.0)

                # Biometric ratio consistency
                ga, gb = fa[4:7], fb[4:7]
                diff = np.linalg.norm(ga - gb)
                s_geom = float(np.exp(- (diff ** 2) / (2.0 * (0.35 ** 2))))

                s_pose = float(0.5 * s_view + 0.5 * s_geom)
                scores.append(s_pose)
                weights.append(self.pose_weight)

        if not scores or sum(weights) == 0:
            return -1.0

        return float(np.average(scores, weights=weights))

