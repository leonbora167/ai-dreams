"""Random audit sampler module for Phase 1E.

Samples random clips that bypass the Stage 1 gate to measure what the gate drops,
logs audit review items, and computes the estimated miss rate with Wilson confidence intervals.
"""

import json
import os
import random
import cv2
import numpy as np


def sample_audit_clips(duration_sec: float, frame_shape: tuple, rate_per_hour: float = 8.0,
                       seed: int = 42, avoid_intervals: list = None,
                       clip_dur_sec: float = 3.0, box_size: tuple = (240, 240)) -> list:
    """Sample random audit time windows and spatial boxes avoiding intervals covered by gate events.
    
    Returns list of dicts: {"id": int, "start_sec": float, "end_sec": float, "box": [x1, y1, x2, y2], "origin": "audit"}
    """
    if avoid_intervals is None:
        avoid_intervals = []

    rng = random.Random(seed)
    h_frame, w_frame = frame_shape[:2]
    bw, bh = box_size

    # Calculate number of clips based on video duration
    dur_hours = duration_sec / 3600.0
    num_samples = max(1, int(round(dur_hours * rate_per_hour)))

    audit_clips = []
    max_start = max(0.0, duration_sec - clip_dur_sec)

    attempts = 0
    while len(audit_clips) < num_samples and attempts < 100:
        attempts += 1
        st = rng.uniform(0.0, max_start)
        en = min(duration_sec, st + clip_dur_sec)

        # Check overlap with avoid_intervals (e.g. existing gate event times)
        overlap = False
        for (ast, aen) in avoid_intervals:
            if not (en < ast or st > aen):
                overlap = True
                break

        if overlap and attempts < 50:
            continue

        # Random spatial box placement
        x1 = rng.randint(0, max(0, w_frame - bw))
        y1 = rng.randint(0, max(0, h_frame - bh))
        x2 = min(w_frame, x1 + bw)
        y2 = min(h_frame, y1 + bh)

        audit_clips.append({
            "id": 1000 + len(audit_clips) + 1,
            "start_sec": round(st, 2),
            "end_sec": round(en, 2),
            "box": [x1, y1, x2, y2],
            "origin": "audit",
            "source": "motion",
            "level": None
        })

    return audit_clips


def log_audit_result(clip_info: dict, prediction: dict, manifest_path: str):
    """Append audit clip entry to audit_manifest.json with empty user_label field."""
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    manifest = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except Exception:
            manifest = []

    entry = {
        "clip_id": clip_info.get("id"),
        "video_name": clip_info.get("video_name", "unknown"),
        "clip_path": clip_info.get("clip_path", ""),
        "start_sec": clip_info.get("start_sec"),
        "end_sec": clip_info.get("end_sec"),
        "box": clip_info.get("box"),
        "prediction": prediction if prediction else {"probs": {"none": 1.0}, "top": "none", "top_prob": 1.0},
        "user_label": clip_info.get("user_label", "")  # For human review: "normal" | "abnormal"
    }
    manifest.append(entry)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def estimate_miss_rate(audit_rows: list, z_score: float = 1.96) -> dict:
    """Calculate the estimated gate miss rate and Wilson score 95% confidence interval.
    
    Wilson score formula provides valid coverage even for small sample sizes or zero hits.
    """
    labelled = [r for r in audit_rows if r.get("user_label") and r.get("user_label").strip()]
    n = len(labelled)
    if n == 0:
        return {
            "status": "pending_labels",
            "total_audited": len(audit_rows),
            "total_labelled": 0,
            "misses_found": 0,
            "miss_rate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "message": "Audit clips generated in review queue. Awaiting user labels."
        }

    k = sum(1 for r in labelled if r["user_label"].strip().lower() in ["abnormal", "fight", "running", "surge", "fire"])
    p_hat = k / n

    # Wilson score interval calculation
    z = z_score
    z2 = z ** 2
    n_tilde = n + z2
    p_tilde = (k + 0.5 * z2) / n_tilde
    margin = z * np.sqrt((p_tilde * (1.0 - p_tilde)) / n_tilde)

    ci_lower = max(0.0, float(p_tilde - margin))
    ci_upper = min(1.0, float(p_tilde + margin))

    is_wide = (ci_upper - ci_lower) > 0.25
    msg = "Wilson interval is wide due to small sample size; add more audit clips." if is_wide else "Sample size sufficient."

    return {
        "status": "computed",
        "total_audited": len(audit_rows),
        "total_labelled": n,
        "misses_found": k,
        "miss_rate": float(p_hat),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "message": msg
    }
