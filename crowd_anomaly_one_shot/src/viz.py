"""Visualization helper functions for images, contact sheets, heatmaps, grids, and plots."""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_image(img: np.ndarray, path: str):
    """Save an image (H, W, 3 BGR or H, W uint8) to path."""
    assert isinstance(img, np.ndarray), "img must be a numpy ndarray"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    success = cv2.imwrite(path, img)
    assert success, f"Failed to save image to {path}"


def contact_sheet(images: list, cols: int, path: str, labels: list = None):
    """Assemble a list of images (H, W, 3 BGR) into a grid and save to path."""
    assert len(images) > 0, "Image list cannot be empty"
    assert cols > 0, "cols must be positive"
    h, w = images[0].shape[:2]
    n = len(images)
    rows = (n + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        assert img.shape[:2] == (h, w), f"Image size mismatch: {img.shape[:2]} vs {(h, w)}"
        r, c = idx // cols, idx % cols
        cell = img.copy()
        if labels is not None and idx < len(labels) and labels[idx]:
            cv2.putText(cell, str(labels[idx]), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(cell, str(labels[idx]), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = cell

    save_image(sheet, path)


def overlay_heatmap(frame: np.ndarray, heat: np.ndarray, alpha: float = 0.5, vmax: float = None) -> np.ndarray:
    """Blend a 2D heatmap (grid_h, grid_w) or (H, W) float array onto a BGR frame (H, W, 3) uint8."""
    assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H, W, 3), got {frame.shape}"
    h, w = frame.shape[:2]
    heat_f = heat.astype(np.float32)

    v_max = float(vmax) if vmax is not None else float(heat_f.max()) if heat_f.max() > 0 else 1.0
    v_min = float(heat_f.min())
    denom = max(v_max - v_min, 1e-6)
    heat_norm = np.clip((heat_f - v_min) / denom, 0.0, 1.0)
    heat_u8 = (heat_norm * 255).astype(np.uint8)

    heat_resized = cv2.resize(heat_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_color = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(frame, 1.0 - alpha, heat_color, alpha, 0)
    return blended


def draw_boxes(frame: np.ndarray, boxes: list, labels: list = None, color: tuple = (0, 255, 0)) -> np.ndarray:
    """Draw bounding boxes [x1, y1, x2, y2] and optional labels on a copy of frame (H, W, 3 uint8 BGR)."""
    assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H, W, 3), got {frame.shape}"
    out = frame.copy()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if labels is not None and idx < len(labels) and labels[idx]:
            text = str(labels[idx])
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, text, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def draw_grid(frame: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Draw grid_h rows and grid_w columns over a copy of frame (H, W, 3 uint8 BGR)."""
    assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H, W, 3), got {frame.shape}"
    out = frame.copy()
    h, w = frame.shape[:2]
    # Draw horizontal lines
    for r in range(1, grid_h):
        y = int(round(r * h / grid_h))
        cv2.line(out, (0, y), (w, y), (255, 255, 255), 1)
    # Draw vertical lines
    for c in range(1, grid_w):
        x = int(round(c * w / grid_w))
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1)
    return out


def plot_timeline(values_dict: dict, path: str, title: str):
    """Save a line plot of named numeric series in values_dict vs time index to path."""
    assert len(values_dict) > 0, "values_dict must not be empty"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.figure(figsize=(10, 4))
    for name, series in values_dict.items():
        plt.plot(series, label=str(name))
    plt.title(title)
    plt.xlabel("Analysis Frame Index")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def draw_flow_arrows(frame: np.ndarray, flow: np.ndarray, step: int = 24, scale: float = 2.0, max_len: float = 30.0, color: tuple = (0, 255, 0)) -> np.ndarray:
    """Draw sparse flow displacement arrows on a copy of frame (H, W, 3 uint8 BGR)."""
    assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H, W, 3), got {frame.shape}"
    assert flow.ndim == 3 and flow.shape[2] == 2, f"Expected (H, W, 2), got {flow.shape}"
    out = frame.copy()
    h, w = frame.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = float(flow[y, x, 0]), float(flow[y, x, 1])
            length = np.hypot(dx, dy)
            if length > 0.5:
                s = scale if (length * scale) <= max_len else (max_len / length)
                x2 = int(round(x + dx * s))
                y2 = int(round(y + dy * s))
                cv2.arrowedLine(out, (x, y), (x2, y2), color, 1, tipLength=0.3)
                cv2.circle(out, (x, y), 1, (0, 0, 255), -1)
    return out


def draw_grid_counts(frame: np.ndarray, grid_counts: np.ndarray, color: tuple = (0, 255, 255)) -> np.ndarray:
    """Draw grid lines and per-cell count numbers on a copy of frame (H, W, 3 uint8 BGR)."""
    assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected (H, W, 3), got {frame.shape}"
    grid_h, grid_w = grid_counts.shape
    out = draw_grid(frame, grid_h, grid_w)
    h, w = frame.shape[:2]
    cell_h, cell_w = h / grid_h, w / grid_w
    for r in range(grid_h):
        for c in range(grid_w):
            cnt = float(grid_counts[r, c])
            if cnt >= 0.05:
                text = f"{cnt:.1f}" if cnt < 10 else f"{int(round(cnt))}"
                cx = int(c * cell_w + cell_w * 0.5 - 12)
                cy = int(r * cell_h + cell_h * 0.5 + 5)
                cv2.putText(out, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
                cv2.putText(out, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out



