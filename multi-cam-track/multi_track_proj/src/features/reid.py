"""OSNet / Torchreid deep appearance feature extractor."""
from pathlib import Path
import cv2
import numpy as np
import torch
from .base import BaseFeatureExtractor
from ..device import resolve_device


class ReIDFeatureExtractor(BaseFeatureExtractor):
    """Deep metric learning visual appearance extractor using OSNet."""

    def __init__(self, spec: dict, cfg: dict):
        super().__init__(spec, cfg)
        import torchreid

        self.device = resolve_device(cfg.get('detector', {}).get('device', 'auto'))
        self.input_size = spec.get('input_size', [256, 128])
        weight_path = spec.get('weight_path')

        if weight_path and not Path(weight_path).exists():
            raise FileNotFoundError(
                f"ReID checkpoint not found at: {weight_path}. "
                "Please run: python scripts/download_reid_weights.py"
            )

        model_name = spec.get('model', 'osnet_ain_x1_0')
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,
            pretrained=not bool(weight_path)
        )
        if weight_path:
            torchreid.utils.load_pretrained_weights(self.model, weight_path)

        self.model.to(self.device)
        self.model.eval()

        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def extract(self, crops: list, tracklet) -> np.ndarray:
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

    def similarity(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        dot = float(np.dot(feat_a, feat_b))
        norm = float((np.linalg.norm(feat_a) * np.linalg.norm(feat_b)) + 1e-8)
        return float(dot / norm)

