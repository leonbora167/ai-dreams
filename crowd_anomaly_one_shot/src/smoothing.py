"""Temporal smoothing and alert trigger module for Phase 3-A.

Implements exponential moving average (EMA) of class probability series,
k-of-n window persistence verification, and event alert trigger time determination.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


def ema(values: List[float], alpha: float = 0.35) -> List[float]:
    """Compute exponential moving average of a 1D sequence of float values."""
    if not values:
        return []
    smoothed = [float(values[0])]
    for v in values[1:]:
        s = alpha * float(v) + (1.0 - alpha) * smoothed[-1]
        smoothed.append(float(s))
    return smoothed


def k_of_n(flags: List[bool], k: int = 3, n: int = 4) -> bool:
    """Return True if at least k of the most recent n binary flags are True."""
    if not flags:
        return False
    window = flags[-n:] if len(flags) >= n else flags
    req_k = min(k, len(flags)) if len(flags) >= 2 else k
    return sum(1 for f in window if f) >= req_k


def smooth_probabilities(window_preds: List[Dict], alpha: float = 0.35, classes: Optional[List[str]] = None) -> List[Dict]:
    """Smooth per-class probability trajectories across temporal windows using EMA."""
    if not window_preds:
        return []
    if classes is None:
        classes = list(window_preds[0].get("probs", {}).keys())

    # Extract per-class time series
    raw_series = {c: [p["probs"].get(c, 0.0) for p in window_preds] for c in classes}
    smoothed_series = {c: ema(raw_series[c], alpha=alpha) for c in classes}

    num_windows = len(window_preds)
    smoothed_preds = []
    for i in range(num_windows):
        raw_dist = np.array([smoothed_series[c][i] for c in classes], dtype=np.float64)
        sum_p = np.sum(raw_dist)
        norm_dist = raw_dist / max(sum_p, 1e-12)
        top_idx = int(np.argmax(norm_dist))
        top_cls = classes[top_idx]
        top_p = float(norm_dist[top_idx])
        probs = {c: float(norm_dist[j]) for j, c in enumerate(classes)}
        smoothed_preds.append({
            "probs": probs,
            "top": top_cls,
            "top_prob": top_p,
            "classifier": "smoothed"
        })
    return smoothed_preds


def detect_alert_trigger(smoothed_preds: List[Dict], window_times: List[float],
                         class_thrs: Dict[str, float], k: int = 3, n: int = 4,
                         class_kn: Optional[Dict[str, Tuple[int, int]]] = None) -> Dict:
    """Evaluate k-of-n persistence and class thresholds to find exact alert fire time."""
    classes = [c for c in class_thrs.keys() if c != "none"]
    class_flags = {c: [] for c in classes}

    for idx, pred in enumerate(smoothed_preds):
        top_cls = pred["top"]
        top_prob = pred["top_prob"]

        for c in classes:
            pos = (top_cls == c and top_prob >= class_thrs.get(c, 0.50))
            class_flags[c].append(pos)
            k_c, n_c = class_kn.get(c, (k, n)) if class_kn else (k, n)
            if k_of_n(class_flags[c], k_c, n_c):
                fire_time = window_times[idx] if idx < len(window_times) else 0.0
                return {
                    "alert_fired": True,
                    "alert_class": c,
                    "fire_index": idx,
                    "fire_time": fire_time,
                    "fire_prob": top_prob,
                    "reason": f"Sustained {k_c}-of-{n_c} agreement at p={top_prob:.2f} >= {class_thrs[c]:.2f}"
                }

    # Final decision if no alert triggered
    last_top = smoothed_preds[-1]["top"] if smoothed_preds else "none"
    last_prob = smoothed_preds[-1]["top_prob"] if smoothed_preds else 0.0
    return {
        "alert_fired": False,
        "alert_class": "none",
        "fire_index": -1,
        "fire_time": None,
        "fire_prob": last_prob,
        "reason": f"Suppressed: top is '{last_top}' (p={last_prob:.2f}) or failed persistence"
    }


def smooth_event_predictions(window_preds: List[Dict], window_times: List[float],
                            event_start_sec: float, alpha: float = 0.35,
                            k: int = 3, n: int = 4,
                            class_thrs: Optional[Dict[str, float]] = None,
                            class_kn: Optional[Dict[str, Tuple[int, int]]] = None) -> Dict:
    """Full procedural smoothing pipeline for a single event over time."""
    if class_thrs is None:
        class_thrs = {"none": 0.50, "fight": 0.55, "running_surge": 0.50, "fire_smoke": 0.35}

    smoothed = smooth_probabilities(window_preds, alpha=alpha)
    trigger = detect_alert_trigger(smoothed, window_times, class_thrs=class_thrs, k=k, n=n, class_kn=class_kn)

    delay_sec = None
    if trigger["alert_fired"] and trigger["fire_time"] is not None:
        delay_sec = max(0.0, float(trigger["fire_time"]) - float(event_start_sec))

    return {
        "smoothed_preds": smoothed,
        "alert_fired": trigger["alert_fired"],
        "alert_class": trigger["alert_class"],
        "fire_index": trigger["fire_index"],
        "fire_time": trigger["fire_time"],
        "fire_prob": trigger["fire_prob"],
        "delay_sec": delay_sec,
        "reason": trigger["reason"]
    }
