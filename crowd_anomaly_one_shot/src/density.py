"""Density estimation module for Phase 1B using CSRNet.

Output data contracts:
- predict_density returns density map (H, W) float32. Sum across pixels equals the estimated total people count.
- density_to_grid returns (grid_h, grid_w) float32 where each cell contains the integrated count for that cell.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class CSRNet(nn.Module):
    """CSRNet architecture with VGG-16 frontend and dilated convolutional backend."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)
        # First 10 conv layers of VGG-16 (up to conv4_3)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.backend(self.frontend(x)))


def load_density_model(name: str = "csrnet", weights_path: str = "weights/csrnet_shanghaitech.pth", device: str = None) -> nn.Module:
    """Load pretrained density model onto target device and set to eval mode."""
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if name.lower() == "csrnet":
        assert os.path.exists(weights_path), f"CSRNet weights not found at {weights_path}"
        model = CSRNet().to(dev).eval()
        state = torch.load(weights_path, map_location=dev)
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        return model
    else:
        raise ValueError(f"Unsupported density model: {name}")


def predict_density(model: nn.Module, frame: np.ndarray, tile_size: int = None) -> np.ndarray:
    """Predict continuous crowd density map (H, W) float32 from a BGR frame (H, W, 3) uint8.
    
    The array values represent density per pixel, and np.sum(density) equals total estimated people count.
    If tile_size is specified and smaller than frame, processes with 25% overlap and Hann window blending.
    """
    assert isinstance(frame, np.ndarray), "frame must be a numpy ndarray"
    assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H, W, 3), got {frame.shape}"
    assert frame.dtype == np.uint8, "frame must be uint8"
    h, w = frame.shape[:2]

    device = next(model.parameters()).device
    # ImageNet standard normalization: (x - mean) / std
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = (rgb - mean) / std

    # Direct inference for standard analysis resolutions
    if tile_size is None or (h <= tile_size and w <= tile_size):
        tensor = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
        raw_density = out.squeeze().cpu().numpy().astype(np.float32)
        raw_density = np.maximum(raw_density, 0.0)
    else:
        # Tiled inference with overlap and cosine window stitching
        stride = int(tile_size * 0.75)
        raw_density = np.zeros((h // 8, w // 8), dtype=np.float32)
        weight_mask = np.zeros((h // 8, w // 8), dtype=np.float32)
        for y0 in range(0, h, stride):
            for x0 in range(0, w, stride):
                y1, x1 = min(y0 + tile_size, h), min(x0 + tile_size, w)
                patch = norm_img[y0:y1, x0:x1]
                tensor_p = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    out_p = model(tensor_p).squeeze().cpu().numpy()
                out_p = np.maximum(out_p, 0.0)
                ph, pw = out_p.shape
                # 2D Hann window for overlap smoothing
                win_y = np.hanning(ph)
                win_x = np.hanning(pw)
                win = np.outer(win_y, win_x).astype(np.float32)
                raw_density[y0 // 8 : (y0 // 8) + ph, x0 // 8 : (x0 // 8) + pw] += (out_p * win)
                weight_mask[y0 // 8 : (y0 // 8) + ph, x0 // 8 : (x0 // 8) + pw] += win
        raw_density = raw_density / np.maximum(weight_mask, 1e-6)

    # Upsample density map to full frame resolution (H, W) preserving total integral sum
    density_upsampled = cv2.resize(raw_density, (w, h), interpolation=cv2.INTER_LINEAR)
    density_upsampled = np.maximum(density_upsampled, 0.0)
    raw_sum = float(np.sum(raw_density))
    up_sum = float(np.sum(density_upsampled))
    if up_sum > 0:
        density_upsampled = density_upsampled * (raw_sum / up_sum)

    return density_upsampled.astype(np.float32)


def density_to_grid(density: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Sum continuous pixel density map (H, W) into (grid_h, grid_w) float32 per-cell counts."""
    assert density.ndim == 2, f"Expected 2D density map, got {density.shape}"
    h, w = density.shape
    y_edges = np.linspace(0, h, grid_h + 1, dtype=int)
    x_edges = np.linspace(0, w, grid_w + 1, dtype=int)

    grid_counts = np.zeros((grid_h, grid_w), dtype=np.float32)
    for r in range(grid_h):
        for c in range(grid_w):
            cell = density[y_edges[r] : y_edges[r + 1], x_edges[c] : x_edges[c + 1]]
            grid_counts[r, c] = float(np.sum(cell))

    return grid_counts
