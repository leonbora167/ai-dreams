import cv2
import numpy as np
from pathlib import Path


def _crop(frame, box):
    h, w = frame.shape[:2]; x1, y1, x2, y2 = map(int, box)
    return frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]


class FeatureManager:
    def __init__(self, cfg):
        self.cfg = cfg; self.enabled = {k: v for k, v in cfg['features'].items() if v.get('enabled', False)}
        self.reid_model = None
        if 'reid' in self.enabled:
            import torchreid
            reid_cfg = self.enabled['reid']
            weight_path = reid_cfg.get('weight_path')
            if weight_path and not Path(weight_path).exists():
                raise FileNotFoundError(
                    f'ReID checkpoint not found: {weight_path}. '
                    'Run: python scripts/download_reid_weights.py')
            self.reid_model = torchreid.models.build_model(
                name=reid_cfg.get('model', 'osnet_x1_0'), num_classes=1,
                pretrained=not bool(weight_path))
            if weight_path:
                torchreid.utils.load_pretrained_weights(self.reid_model, weight_path)
            self.reid_model.eval()

    def extract(self, track):
        d = {}
        quality = self.enabled.get('quality', {})
        if quality and track.confidences and float(np.mean(track.confidences)) < quality.get('min_confidence', 0):
            return {'quality': float(np.mean(track.confidences)), '_skip': True}
        if 'color' in self.enabled or 'reid' in self.enabled:
            cap = cv2.VideoCapture(track.video_path); crops=[]
            for frame_no, box in zip(track.frames, track.boxes):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no); ok, frame = cap.read()
                if ok:
                    crop = _crop(frame, box)
                    if crop.size: crops.append(crop)
            cap.release()
            if 'color' in self.enabled: d['color'] = self._color(crops)
            if 'reid' in self.enabled: d['reid'] = self._reid(crops)
        if 'motion' in self.enabled: d['motion'] = self._motion(track)
        d['quality'] = float(np.mean(track.confidences)) if track.confidences else 0.
        return d

    def extract_crops(self, crops, track):
        """Extract descriptors from crops collected while a live stream ran."""
        d = {}
        quality = self.enabled.get('quality', {})
        if quality and track.confidences and float(np.mean(track.confidences)) < quality.get('min_confidence', 0):
            return {'quality': float(np.mean(track.confidences)), '_skip': True}
        if 'color' in self.enabled: d['color'] = self._color(crops)
        if 'reid' in self.enabled: d['reid'] = self._reid(crops)
        if 'motion' in self.enabled: d['motion'] = self._motion(track)
        d['quality'] = float(np.mean(track.confidences)) if track.confidences else 0.
        return d

    @staticmethod
    def _color(crops):
        if not crops: return np.zeros(24, dtype=np.float32)
        hist = [cv2.normalize(cv2.calcHist([c], [0,1], None, [6,4], [0,180,0,256]), None).flatten() for c in crops]
        v = np.mean(hist, axis=0); return v / (np.linalg.norm(v) + 1e-8)

    def _reid(self, crops):
        if not crops: return np.zeros(512, dtype=np.float32)
        import torch
        # Torchreid models expect RGB ImageNet-normalized tensors.
        size = self.enabled.get('reid', {}).get('input_size', [256, 128])
        mean = np.asarray([.485, .456, .406], dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray([.229, .224, .225], dtype=np.float32).reshape(3, 1, 1)
        batch = []
        for crop in crops:
            image = cv2.resize(crop, (int(size[1]), int(size[0])))[:,:,::-1].transpose(2,0,1).astype(np.float32) / 255.
            batch.append((image - mean) / std)
        with torch.no_grad(): v = self.reid_model(torch.tensor(np.asarray(batch), dtype=torch.float32)).cpu().numpy()
        v = np.mean(v, axis=0); return v / (np.linalg.norm(v) + 1e-8)

    @staticmethod
    def _motion(t):
        if len(t.boxes) < 2: return np.zeros(2, dtype=np.float32)
        a, b = np.asarray(t.boxes[0]), np.asarray(t.boxes[-1])
        return np.asarray([(b[0]+b[2]-a[0]-a[2])/2, (b[1]+b[3]-a[1]-a[3])/2], dtype=np.float32)
