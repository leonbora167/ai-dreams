import numpy as np
import cv2
import torch
import onnxruntime
from insightface.app import FaceAnalysis


class FaceModule:
    """
    Extracts 512-d ArcFace embeddings and face crops using InsightFace (buffalo_s).
    Includes border-padding and direct recognition fallback so tight face crops
    also resolve reliably during reference face search.
    """

    def __init__(self, model_name: str = "buffalo_s", min_conf: float = 0.60):
        self.min_conf = min_conf

        avail = onnxruntime.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in avail and torch.cuda.is_available():
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in avail:
            providers.append("CPUExecutionProvider")
        if not providers:
            providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(name=model_name, providers=providers)
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    @staticmethod
    def _l2_normalize(emb: np.ndarray) -> list[float]:
        emb = np.array(emb, dtype=np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.tolist()

    def _detect_with_padding(self, image: np.ndarray, pad_ratio: float = 0.25, min_side: int = 160):
        """
        Runs SCRFD face detection after optional border padding and upscaling.
        Returns (faces_list, pad_x, pad_y, scale).
        """
        h, w = image.shape[:2]
        scale = 1.0
        work = image
        if min(h, w) < min_side and min(h, w) > 0:
            scale = float(min_side) / float(min(h, w))
            work = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)

        wh, ww = work.shape[:2]
        pad_y = max(16, int(round(wh * pad_ratio)))
        pad_x = max(16, int(round(ww * pad_ratio)))
        padded = cv2.copyMakeBorder(
            work, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REPLICATE
        )
        try:
            faces = self.app.get(padded)
        except Exception:
            faces = []
        return faces, pad_x, pad_y, scale, padded

    def extract_face(self, crop: np.ndarray) -> tuple[list[float] | None, np.ndarray | None, float]:
        """
        Extracts ArcFace 512-d embedding and padded face crop from upper 45% of body crop.
        Returns:
            (face_vector_512d, face_crop_image, det_score) or (None, None, 0.0)
        """
        if crop is None or crop.size == 0:
            return None, None, 0.0

        h, w = crop.shape[:2]
        upper_h = max(16, int(round(h * 0.45)))
        upper_crop = crop[:upper_h, :].copy()

        # Mild padding helps SCRFD detect faces near top/side bbox edges
        faces, pad_x, pad_y, scale, _ = self._detect_with_padding(upper_crop, pad_ratio=0.20, min_side=128)
        if not faces:
            return None, None, 0.0

        valid_faces = [f for f in faces if hasattr(f, "det_score") and f.det_score >= self.min_conf]
        if not valid_faces:
            return None, None, 0.0

        best_face = max(valid_faces, key=lambda f: f.det_score)
        if not hasattr(best_face, "embedding") or best_face.embedding is None:
            return None, None, 0.0

        face_vector = self._l2_normalize(best_face.embedding)

        # Map detected bbox back from padded/scaled coordinates to original upper_crop with 20% margin
        bbox = best_face.bbox
        x1 = (bbox[0] - pad_x) / scale
        y1 = (bbox[1] - pad_y) / scale
        x2 = (bbox[2] - pad_x) / scale
        y2 = (bbox[3] - pad_y) / scale
        bw, bh = x2 - x1, y2 - y1
        margin_x = bw * 0.25
        margin_y = bh * 0.25

        fx1 = max(0, min(int(round(x1 - margin_x)), w - 1))
        fy1 = max(0, min(int(round(y1 - margin_y)), upper_h - 1))
        fx2 = max(0, min(int(round(x2 + margin_x)), w))
        fy2 = max(0, min(int(round(y2 + margin_y)), upper_h))

        face_crop = upper_crop[fy1:fy2, fx1:fx2].copy() if (fx2 > fx1 and fy2 > fy1) else None
        return face_vector, face_crop, float(best_face.det_score)

    def extract_best_face_from_crops(self, crops: list[np.ndarray]) -> tuple[list[float] | None, np.ndarray | None]:
        """
        Scans multiple sharpest crops from a tracklet window to find the clearest
        frontal face embedding rather than relying solely on one body crop.
        """
        best_vec = None
        best_crop = None
        best_score = -1.0

        for crop in crops:
            vec, f_crop, score = self.extract_face(crop)
            if vec is not None and score > best_score:
                best_score = score
                best_vec = vec
                best_crop = f_crop

        return best_vec, best_crop

    def extract_face_from_crop(self, face_img: np.ndarray) -> list[float] | None:
        """
        Extracts a 512-d ArcFace embedding from a standalone reference face photo
        OR a tight self-generated face crop (`face_local_*.jpg`).
        """
        if face_img is None or face_img.size == 0:
            return None

        # 1. Try normal detection on full image (with generous padding for tight crops)
        for pad_ratio, conf_floor in [(0.50, self.min_conf), (0.80, 0.35)]:
            faces, _, _, _, _ = self._detect_with_padding(face_img, pad_ratio=pad_ratio, min_side=160)
            valid = [f for f in faces if hasattr(f, "det_score") and f.det_score >= conf_floor and f.embedding is not None]
            if valid:
                best = max(valid, key=lambda f: f.det_score)
                return self._l2_normalize(best.embedding)

        # 2. Direct ArcFace recognition fallback if image is already a tightly cropped face
        rec_model = self.app.models.get("recognition")
        if rec_model is not None:
            aligned_112 = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_CUBIC)
            emb = rec_model.get_feat([aligned_112])
            if emb is not None and len(emb) > 0:
                return self._l2_normalize(emb[0])

        return None
