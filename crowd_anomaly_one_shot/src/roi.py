"""ROI crop, temporal buffering, square resizing, and tiling module for Phase 2A.

Extracts fixed-size spatial-temporal clips from detected events for Stage 2 classifiers,
with context expansion, square aspect-ratio preservation, shift-in bounds handling,
and multi-tile decomposition for large crowd regions.
"""

import cv2
import numpy as np


def expand_box(box: list, margin: float = 0.25, frame_shape: tuple = None) -> list:
    """Expand bounding box [x1, y1, x2, y2] by margin percentage on each side."""
    x1, y1, x2, y2 = box
    w, h = max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))
    dx, dy = w * margin, h * margin
    nx1, ny1, nx2, ny2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy
    if frame_shape is not None:
        h_f, w_f = frame_shape[:2]
        nx1, ny1 = max(0.0, nx1), max(0.0, ny1)
        nx2, ny2 = min(float(w_f), nx2), min(float(h_f), ny2)
    return [round(nx1, 1), round(ny1, 1), round(nx2, 1), round(ny2, 1)]


def square_box(box: list, frame_shape: tuple) -> list:
    """Make box square by expanding shorter side, shifting inside frame boundaries when possible."""
    h_f, w_f = frame_shape[:2]
    x1, y1, x2, y2 = box
    w, h = max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))
    side = max(w, h)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx1, ny1 = cx - side / 2.0, cy - side / 2.0
    nx2, ny2 = nx1 + side, ny1 + side

    # Shift back inside frame boundaries horizontally
    if nx1 < 0.0:
        shift_x = -nx1
        nx1, nx2 = nx1 + shift_x, nx2 + shift_x
    elif nx2 > w_f:
        shift_x = nx2 - w_f
        nx1, nx2 = nx1 - shift_x, nx2 - shift_x

    # Shift back inside frame boundaries vertically
    if ny1 < 0.0:
        shift_y = -ny1
        ny1, ny2 = ny1 + shift_y, ny2 + shift_y
    elif ny2 > h_f:
        shift_y = ny2 - h_f
        ny1, ny2 = ny1 - shift_y, ny2 - shift_y

    return [int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))]


def crop_and_resize(frame: np.ndarray, box: list, size: int = 224) -> np.ndarray:
    """Crop square region with zero-padding only if outside frame, then resize to (size, size, 3).
    
    Uses cv2.INTER_AREA for downsampling (prevents aliasing) and cv2.INTER_LINEAR for upsampling.
    """
    assert isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h_f, w_f = frame.shape[:2]
    crop_w, crop_h = max(1, x2 - x1), max(1, y2 - y1)
    crop_buf = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

    src_x1, src_y1 = max(0, x1), max(0, y1)
    src_x2, src_y2 = min(w_f, x2), min(h_f, y2)
    dst_x1, dst_y1 = max(0, src_x1 - x1), max(0, src_y1 - y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        crop_buf[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

    if crop_buf.shape[0] == size and crop_buf.shape[1] == size:
        return crop_buf

    interp = cv2.INTER_AREA if crop_buf.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(crop_buf, (size, size), interpolation=interp)


def needs_tiling(box: list, size: int = 224, ratio_limit: float = 2.0) -> bool:
    """Check if region size exceeds ratio_limit * size (448px for size=224)."""
    w = max(1.0, float(box[2] - box[0]))
    h = max(1.0, float(box[3] - box[1]))
    return max(w, h) >= (ratio_limit * size)


def tile_boxes(box: list, frame_shape: tuple, tile_size_px: int = 224, overlap: float = 0.5, max_tiles: int = 4) -> list:
    """Produce overlapping square tiles covering large box, capped at max_tiles."""
    x1, y1, x2, y2 = box
    w, h = float(x2 - x1), float(y2 - y1)
    tile_side = float(tile_size_px)
    step = tile_side * (1.0 - overlap)

    tiles = []
    # Determine tile layout (1x2, 2x1, or 2x2 grid)
    cols = 2 if w >= tile_side * 1.3 else 1
    rows = 2 if h >= tile_side * 1.3 else 1
    if cols * rows > max_tiles:
        cols, rows = 2, 2

    x_starts = [x1, x2 - tile_side] if cols == 2 else [x1 + (w - tile_side) / 2.0]
    y_starts = [y1, y2 - tile_side] if rows == 2 else [y1 + (h - tile_side) / 2.0]

    for ys in y_starts:
        for xs in x_starts:
            raw_tile = [xs, ys, xs + tile_side, ys + tile_side]
            sq = square_box(raw_tile, frame_shape)
            tiles.append(sq)
            if len(tiles) >= max_tiles:
                return tiles
    return tiles


def make_clip(frames: list, box: list, size: int = 224, num_frames: int = 16) -> np.ndarray:
    """Extract temporal clip (T, size, size, 3) uint8 using a single stationary crop box across all frames."""
    assert len(frames) > 0, "frames list cannot be empty"
    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    clip = np.empty((num_frames, size, size, 3), dtype=np.uint8)
    for idx, frame_idx in enumerate(indices):
        clip[idx] = crop_and_resize(frames[frame_idx], box, size=size)
    assert clip.shape == (num_frames, size, size, 3), f"Shape mismatch: {clip.shape}"
    assert clip.dtype == np.uint8, f"Dtype mismatch: {clip.dtype}"
    return clip
