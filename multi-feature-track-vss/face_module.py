import numpy as np
import cv2
import torch
import onnxruntime
from insightface.app import FaceAnalysis

class FaceModule:
    def __init__(self, model_name: str = "buffalo_s", min_conf: float = 0.60):
        self.min_conf = min_conf
        
        # Check available ONNX runtime providers
        avail = onnxruntime.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in avail and torch.cuda.is_available():
            providers.append('CUDAExecutionProvider')
        if 'CPUExecutionProvider' in avail:
            providers.append('CPUExecutionProvider')
        if not providers:
            providers = ['CPUExecutionProvider']

        self.app = FaceAnalysis(name=model_name, providers=providers)
        ctx_id = 0 if 'CUDAExecutionProvider' in providers else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def extract_face(self, crop: np.ndarray) -> tuple[list[float] | None, np.ndarray | None]:
        """
        Extracts ArcFace 512-d embedding and face crop from upper 40% of body crop.
        Returns:
            (face_vector_512d, face_crop_image) or (None, None)
        """
        if crop is None or crop.size == 0:
            return None, None

        h, w = crop.shape[:2]
        # Inspect the upper 40% of the body crop
        upper_h = max(10, int(round(h * 0.40)))
        upper_crop = crop[:upper_h, :].copy()

        try:
            faces = self.app.get(upper_crop)
        except Exception as e:
            return None, None

        if not faces:
            return None, None

        # Filter faces by confidence threshold >= min_conf
        valid_faces = [f for f in faces if hasattr(f, 'det_score') and f.det_score >= self.min_conf]
        if not valid_faces:
            return None, None

        # Pick the most confident face
        best_face = max(valid_faces, key=lambda f: f.det_score)

        if not hasattr(best_face, 'embedding') or best_face.embedding is None:
            return None, None

        # L2-normalize face embedding
        emb = np.array(best_face.embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        face_vector = emb.tolist()

        # Extract face crop
        bbox = best_face.bbox.astype(int)
        fx1 = max(0, min(bbox[0], w - 1))
        fy1 = max(0, min(bbox[1], upper_h - 1))
        fx2 = max(0, min(bbox[2], w))
        fy2 = max(0, min(bbox[3], upper_h))

        if fx2 > fx1 and fy2 > fy1:
            face_crop = upper_crop[fy1:fy2, fx1:fx2].copy()
        else:
            face_crop = None

        return face_vector, face_crop

