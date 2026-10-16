"""Multi-scale per-camera baseline and anomaly scoring module for Phase 1C.

Computes grid-level features (speed_mean, speed_p90, dir_coherence, divergence, density),
fits robust baseline statistics (median, MAD) on normal footage, and scores test frames.
"""

import os
import cv2
import numpy as np


def compute_grid_stats(flow: np.ndarray, density: np.ndarray, grid_h: int, grid_w: int) -> dict:
    """Compute per-cell motion and density statistics of shape (grid_h, grid_w) float32."""
    assert flow.ndim == 3 and flow.shape[2] == 2, f"Expected (H, W, 2) flow, got {flow.shape}"
    h, w = flow.shape[:2]

    # 1. Flow speed and mean speed per cell
    speed = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)
    speed_mean = cv2.resize(speed, (grid_w, grid_h), interpolation=cv2.INTER_AREA)

    # 2. 90th percentile of speed per cell (subsampled for speed)
    speed_sub = speed[::2, ::2]
    hs, ws = speed_sub.shape
    y_edges = np.linspace(0, hs, grid_h + 1, dtype=int)
    x_edges = np.linspace(0, ws, grid_w + 1, dtype=int)
    speed_p90 = np.zeros((grid_h, grid_w), dtype=np.float32)
    for r in range(grid_h):
        for c in range(grid_w):
            cell = speed_sub[y_edges[r] : y_edges[r + 1], x_edges[c] : x_edges[c + 1]]
            if cell.size > 0:
                k = int(0.9 * (cell.size - 1))
                speed_p90[r, c] = np.partition(cell.ravel(), k)[k]

    # 3. Direction coherence: ||mean(v)|| / (mean(||v||) + eps)
    # 1.0 means all flow vectors point in identical direction (laminar flow);
    # ~0.0 indicates chaotic opposing movement (turbulent / violent struggle).
    u_mean = cv2.resize(flow[..., 0], (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    v_mean = cv2.resize(flow[..., 1], (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    vec_mag = np.sqrt(u_mean ** 2 + v_mean ** 2)
    dir_coherence = np.where(speed_mean < 0.1, 1.0, vec_mag / (speed_mean + 1e-6))
    dir_coherence = np.clip(dir_coherence, 0.0, 1.0).astype(np.float32)

    # 4. Divergence: div = du/dx + dv/dy (Sobel derivatives)
    # Positive indicates spreading / fleeing outward; negative indicates convergence / crushing inward.
    du_dx = cv2.Sobel(flow[..., 0], cv2.CV_32F, 1, 0, ksize=3)
    dv_dy = cv2.Sobel(flow[..., 1], cv2.CV_32F, 0, 1, ksize=3)
    divergence = cv2.resize(du_dx + dv_dy, (grid_w, grid_h), interpolation=cv2.INTER_AREA)

    # 5. Density from Phase 1B (integrated people count per cell)
    if density is not None:
        from src.density import density_to_grid
        cell_density = density_to_grid(density, grid_h, grid_w)
    else:
        cell_density = np.zeros((grid_h, grid_w), dtype=np.float32)

    return {
        "speed_mean": speed_mean.astype(np.float32),
        "speed_p90": speed_p90.astype(np.float32),
        "dir_coherence": dir_coherence.astype(np.float32),
        "divergence": divergence.astype(np.float32),
        "density": cell_density.astype(np.float32),
    }


def fit_baseline(stats_list: list, min_mad_floors: dict = None) -> dict:
    """Fit robust per-cell baseline statistics (median, scale=1.4826*MAD) across normal footage."""
    assert len(stats_list) > 0, "stats_list cannot be empty"
    # Physically calibrated scale floors matching pedestrian surveillance at 960 width:
    # prevents division-by-zero or extreme sensitivity in empty/still cells.
    floors = {
        "speed_mean": 4.0, "speed_p90": 6.0, "dir_coherence": 0.20,
        "divergence": 0.8, "density": 1.0
    }
    if min_mad_floors is not None:
        floors.update(min_mad_floors)

    keys = list(stats_list[0].keys())
    baseline = {"num_frames": len(stats_list)}
    for k in keys:
        arr = np.stack([s[k] for s in stats_list], axis=0)  # (N, gh, gw)
        med = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - med), axis=0)
        # Floor on scale prevents zero-MAD divisions in stationary or empty seat cells
        scale = np.maximum(1.4826 * mad, floors.get(k, 0.1))
        baseline[f"{k}_median"] = med.astype(np.float32)
        baseline[f"{k}_scale"] = scale.astype(np.float32)
        baseline[f"{k}_p10"] = np.percentile(arr, 10, axis=0).astype(np.float32)
        baseline[f"{k}_p90"] = np.percentile(arr, 90, axis=0).astype(np.float32)
        baseline[f"{k}_p99"] = np.percentile(arr, 99, axis=0).astype(np.float32)
    return baseline


def save_baseline(baseline: dict, path: str):
    """Save baseline dictionary to an npz file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **baseline)


def load_baseline(path: str) -> dict:
    """Load baseline dictionary from an npz file."""
    assert os.path.exists(path), f"Baseline file not found: {path}"
    data = np.load(path)
    return {k: data[k] for k in data.files}


def score_frame(stats: dict, baseline: dict) -> tuple:
    """Compute robust z-scores per cell and combined score map using np.maximum across features.
    
    Returns:
        score_map: (grid_h, grid_w) float32 combined anomaly score
        z_dict: dict of individual feature z-score maps
    """
    z_dict = {}
    # Speed increases (e.g. running, rushing)
    z_dict["speed_mean"] = np.maximum(0.0, (stats["speed_mean"] - baseline["speed_mean_median"]) / baseline["speed_mean_scale"])
    z_dict["speed_p90"] = np.maximum(0.0, (stats["speed_p90"] - baseline["speed_p90_median"]) / baseline["speed_p90_scale"])
    # Density spike (e.g. bottleneck surge, crowding)
    z_dict["density"] = np.maximum(0.0, (stats["density"] - baseline["density_median"]) / baseline["density_scale"])
    # Divergence deviation (both sudden dispersal > 0 and sudden crush < 0)
    z_dict["divergence"] = np.maximum(0.0, np.abs(stats["divergence"] - baseline["divergence_median"]) / baseline["divergence_scale"])
    # Coherence drop: abnormal when coherence falls below normal, provided motion is present
    coherence_dev = (baseline["dir_coherence_median"] - stats["dir_coherence"]) / baseline["dir_coherence_scale"]
    motion_mask = (stats["speed_mean"] > 0.3).astype(np.float32)
    z_dict["dir_coherence"] = np.maximum(0.0, coherence_dev * motion_mask)

    # Max combination: anomalies are heterogeneous; max allows any single distinct physical abnormality to trigger
    score_map = np.maximum.reduce(list(z_dict.values())).astype(np.float32)
    return score_map, z_dict


def smooth_scores(score_maps: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply temporal moving average over score_maps (T, grid_h, grid_w)."""
    assert score_maps.ndim == 3, f"Expected (T, H, W), got {score_maps.shape}"
    t, h, w = score_maps.shape
    if t < window or window <= 1:
        return score_maps.copy()
    smoothed = np.zeros_like(score_maps)
    for i in range(t):
        start_idx = max(0, i - window + 1)
        smoothed[i] = np.mean(score_maps[start_idx : i + 1], axis=0)
    return smoothed.astype(np.float32)


def merge_scales(score_maps_by_scale: dict, target_grid: tuple) -> np.ndarray:
    """Map multi-scale score maps to target_grid (gh, gw) via nearest-neighbor up/downsampling and take maximum.
    
    Nearest-neighbor preserves distinct block anomaly values without smoothing away sharp cell boundaries.
    """
    tgt_h, tgt_w = target_grid
    rescaled = []
    for (gh, gw), smap in score_maps_by_scale.items():
        if (gh, gw) == (tgt_h, tgt_w):
            rescaled.append(smap)
        else:
            resized = cv2.resize(smap, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)
            rescaled.append(resized)
    merged = np.maximum.reduce(rescaled)
    return merged.astype(np.float32)
