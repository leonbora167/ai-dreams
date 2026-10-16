"""Visual overlay engine for rendering annotated evaluation videos with top-left status HUD."""

import os
from typing import Dict, List
import cv2
import numpy as np

import eval.config_eval as config_eval


def draw_top_left_hud(frame: np.ndarray, state_info: Dict):
    """Render the 3-state classification status banner in the top-left corner."""
    state = state_info.get("state", "normal")
    z_score = state_info.get("z_score", 0.0)
    pred_class = state_info.get("class", "Normal")
    conf = state_info.get("confidence", 0.0)
    model_name = state_info.get("model", "SigLIP")
    lat_ms = state_info.get("latency_ms", 0.0)

    h, w = frame.shape[:2]
    x1, y1 = 12, 12
    max_hud_w = int(w * 0.58)

    if state == "normal":
        # Green quiet banner
        w_box, h_box = min(360, max_hud_w), 50
        sub = frame[y1:y1+h_box, x1:x1+w_box]
        bg = np.full_like(sub, (20, 50, 20), dtype=np.uint8)
        frame[y1:y1+h_box, x1:x1+w_box] = cv2.addWeighted(sub, 0.25, bg, 0.75, 0)
        cv2.rectangle(frame, (x1, y1), (x1+w_box, y1+h_box), (0, 220, 0), 2)
        cv2.putText(frame, "[STATUS: NORMAL]", (x1 + 8, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, "Stage 1 Gate Quiet | Scene Normal", (x1 + 8, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 255, 200), 1)

    elif state == "stage1_abnormal":
        # Amber/orange Stage 1 gate trigger banner
        w_box, h_box = min(360, max_hud_w), 50
        sub = frame[y1:y1+h_box, x1:x1+w_box]
        bg = np.full_like(sub, (15, 60, 130), dtype=np.uint8)
        frame[y1:y1+h_box, x1:x1+w_box] = cv2.addWeighted(sub, 0.25, bg, 0.75, 0)
        cv2.rectangle(frame, (x1, y1), (x1+w_box, y1+h_box), (0, 165, 255), 2)
        cv2.putText(frame, "[STATUS: ABNORMAL (S1)]", (x1 + 8, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        cv2.putText(frame, f"Gate Triggered (Z={z_score:.1f}) | Awaiting S2", (x1 + 8, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 235, 255), 1)

    elif state == "multimodal_abnormal":
        # Crimson Red Multimodal Confirmed Anomaly Banner
        w_box, h_box = min(360, max_hud_w), 68
        sub = frame[y1:y1+h_box, x1:x1+w_box]
        bg = np.full_like(sub, (20, 20, 150), dtype=np.uint8)
        frame[y1:y1+h_box, x1:x1+w_box] = cv2.addWeighted(sub, 0.20, bg, 0.80, 0)
        cv2.rectangle(frame, (x1, y1), (x1+w_box, y1+h_box), (0, 0, 255), 2)
        cv2.putText(frame, "[STATUS: ABNORMAL]", (x1 + 8, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        cv2.putText(frame, f"Classifier: {pred_class.upper()} ({conf*100:.0f}%)", (x1 + 8, y1 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2)
        cv2.putText(frame, f"Model: {model_name[:14]} | Lat: {lat_ms:.0f}ms", (x1 + 8, y1 + 59),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 255), 1)

    elif state == "multimodal_suppressed":
        # Gray / Cyan Suppressed as Normal Banner
        w_box, h_box = min(360, max_hud_w), 50
        sub = frame[y1:y1+h_box, x1:x1+w_box]
        bg = np.full_like(sub, (40, 40, 40), dtype=np.uint8)
        frame[y1:y1+h_box, x1:x1+w_box] = cv2.addWeighted(sub, 0.25, bg, 0.75, 0)
        cv2.rectangle(frame, (x1, y1), (x1+w_box, y1+h_box), (200, 200, 0), 2)
        cv2.putText(frame, "[STATUS: NORMAL]", (x1 + 8, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"Suppressed by Classifier ({conf*100:.0f}%)", (x1 + 8, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 255, 255), 1)


def get_hardware_label(dev_str: str = None) -> str:
    """Detect human-readable device name dynamically (e.g. RTX 4060, A100, Apple MPS, CPU)."""
    if dev_str and dev_str.lower() == "cpu":
        return "CPU"
    try:
        import torch
        if torch.cuda.is_available():
            raw = torch.cuda.get_device_name(0)
            clean = raw.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").replace("Laptop GPU", "").strip()
            return clean if clean else "CUDA GPU"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS"
        return "CPU"
    except Exception:
        return "EDGE"


def draw_telemetry_hud(frame: np.ndarray, cur_sec: float, tot_sec: float,
                       s1_ms: float, s2_ms: float, fps: float,
                       device_name: str = None):
    """Render operational telemetry box in the top-right corner."""
    h, w = frame.shape[:2]
    hud_w, hud_h = min(220, int(w * 0.36)), 68
    x1 = w - hud_w - 12
    y1 = 12

    if device_name and device_name.lower() not in ["cuda", "gpu", "none"]:
        dev_label = device_name
    else:
        dev_label = get_hardware_label(device_name)

    sub = frame[y1:y1+hud_h, x1:x1+hud_w]
    bg = np.full_like(sub, (10, 10, 10), dtype=np.uint8)
    frame[y1:y1+hud_h, x1:x1+hud_w] = cv2.addWeighted(sub, 0.30, bg, 0.70, 0)
    cv2.rectangle(frame, (x1, y1), (x1+hud_w, y1+hud_h), (80, 80, 80), 1)

    cv2.putText(frame, f"EDGE TELEMETRY ({dev_label[:12]})", (x1 + 6, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 215, 255), 1)
    cv2.putText(frame, f"Time: {cur_sec:.1f}s / {tot_sec:.1f}s", (x1 + 6, y1 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
    cv2.putText(frame, f"S1: {s1_ms:.1f}ms | S2: {s2_ms:.0f}ms", (x1 + 6, y1 + 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 255, 200), 1)
    cv2.putText(frame, f"Throughput: {fps:.1f} FPS", (x1 + 6, y1 + 61),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1)


def render_annotated_video(frames: List[np.ndarray], frame_states: List[Dict],
                           boxes_per_frame: List[List], out_path: str,
                           fps: float = 5.0, s1_ms: float = 35.0, s2_ms: float = 25.0,
                           device_name: str = None):
    """Write video with top-left status banner, bounding boxes, and HUD telemetry."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    tot_sec = len(frames) / fps
    eff_fps = 1000.0 / max(s1_ms + s2_ms, 1e-3)

    for i, f in enumerate(frames):
        draw_f = f.copy()
        cur_sec = float(i) / fps
        st_info = frame_states[i] if i < len(frame_states) else {"state": "normal"}

        # Draw any event bounding boxes
        if i < len(boxes_per_frame):
            for b_info in boxes_per_frame[i]:
                b = b_info["box"]
                col = b_info.get("color", (0, 0, 255))
                cv2.rectangle(draw_f, (b[0], b[1]), (b[2], b[3]), col, 2)
                lbl = b_info.get("label", "")
                if lbl:
                    cv2.putText(draw_f, lbl, (b[0], max(b[1] - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)

        # 1. Render Left-Hand Top Status Box
        draw_top_left_hud(draw_f, st_info)

        # 2. Render Top-Right Telemetry
        draw_telemetry_hud(draw_f, cur_sec, tot_sec, s1_ms, s2_ms, eff_fps, device_name=device_name)

        writer.write(draw_f)

    writer.release()

