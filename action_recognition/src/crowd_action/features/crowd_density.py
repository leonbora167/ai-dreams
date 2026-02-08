import math

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)


class CrowdDensityExtractor:
    def __init__(self, device: torch.device, person_thresh: float = 0.6, sigma: float = 12.0) -> None:
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, progress=True).to(device).eval()
        self.device = device
        self.person_thresh = person_thresh
        self.sigma = sigma

    def _gaussian_2d(self, h: int, w: int, cx: float, cy: float, sigma: float) -> torch.Tensor:
        ys = torch.arange(h).float()
        xs = torch.arange(w).float()
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        return torch.exp(-d2 / (2 * sigma**2))

    @torch.no_grad()
    def compute(self, clip: torch.Tensor) -> torch.Tensor:
        t, _, h, w = clip.shape
        maps = []
        for i in range(t):
            image = clip[i].to(self.device)
            preds = self.model([image])[0]
            density = torch.zeros((h, w), dtype=torch.float32)

            boxes = preds["boxes"].cpu()
            labels = preds["labels"].cpu()
            scores = preds["scores"].cpu()
            for box, label, score in zip(boxes, labels, scores):
                if int(label) != 1 or float(score) < self.person_thresh:
                    continue
                x1, y1, x2, y2 = box.tolist()
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                scale_sigma = self.sigma * max(1.0, math.sqrt((x2 - x1) * (y2 - y1)) / 40.0)
                density += self._gaussian_2d(h, w, cx, cy, scale_sigma) * float(score)

            if density.max() > 0:
                density = density / density.max()
            maps.append(density.unsqueeze(0))
        return torch.stack(maps, dim=0)
