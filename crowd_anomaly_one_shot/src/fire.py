"""Fire and smoke detection module for Phase 1F.

Runs YOLO fire/smoke detector, tracks detections across time for temporal persistence
to eliminate flicker false alarms, and produces bypass events with source='fire'.
"""

import os
import cv2
import numpy as np
from src.gate import box_iou


def load_fire_model(weights_path: str = "weights/yolov8n_fire_smoke.pt", device: str = None):
    """Load pretrained YOLO fire/smoke detector on target device."""
    assert os.path.exists(weights_path), f"Weights file not found: {weights_path}"
    from ultralytics import YOLO
    import torch
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(weights_path)
    model.to(dev)
    return model


def detect_fire_smoke(model, frame: np.ndarray, conf_thr: float = 0.25) -> list:
    """Run fire and smoke detector on a single frame (H, W, 3 BGR uint8).
    
    Returns list of {"box": [x1, y1, x2, y2], "cls": "fire"|"smoke", "conf": float}.
    """
    assert isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3
    results = model.predict(frame, conf=conf_thr, verbose=False)[0]
    detections = []
    names = model.names
    for box in results.boxes:
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        cls_name = names.get(cls_id, str(cls_id)).lower()
        xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        detections.append({"box": xyxy, "cls": cls_name, "conf": round(conf, 4)})
    return detections


def color_flicker_prefilter(frame_prev: np.ndarray, frame: np.ndarray, sat_thr: int = 100,
                            val_thr: int = 120, min_changed_pixels: int = 50) -> bool:
    """Classical rule-based check for reddish/orange saturated pixels with temporal change."""
    if frame_prev is None:
        return True
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, sat_thr, val_thr]), np.array([25, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, sat_thr, val_thr]), np.array([180, 255, 255]))
    fire_mask = cv2.bitwise_or(mask1, mask2)
    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_curr, gray_prev)
    changed = cv2.bitwise_and(fire_mask, ((diff > 15).astype(np.uint8) * 255))
    return int(np.count_nonzero(changed)) >= min_changed_pixels


def _track_update_step(active_tracks: list, valid_dets: list, frame_idx: int, max_gap: int) -> tuple:
    """Match detections to existing tracks and return updated active tracks and finalized tracks."""
    still_active, finished = [], []
    matched = [False] * len(valid_dets)
    for trk in active_tracks:
        best_iou, best_idx = 0.0, -1
        for idx, d in enumerate(valid_dets):
            if not matched[idx]:
                iou = box_iou(trk["box"], d["box"])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
        if best_iou >= 0.15:
            d = valid_dets[best_idx]
            matched[best_idx] = True
            trk.update({"end_frame": frame_idx, "box": d["box"], "gap": 0, "count": trk["count"] + 1})
            trk["max_conf"] = max(trk["max_conf"], d["conf"])
            trk["classes"].append(d["cls"])
            still_active.append(trk)
        else:
            trk["gap"] += 1
            (still_active if trk["gap"] <= max_gap else finished).append(trk)
    return still_active, finished, matched


def fire_events_from_detections(detections_over_time: list, min_frames: int = 3,
                               min_conf: float = 0.25, max_gap: int = 2) -> list:
    """Group fire/smoke detections across time into persistent events.
    
    detections_over_time: list of (frame_idx, detections_list)
    Requires persistence over >= min_frames detections to eliminate flicker false alarms.
    Emits events with source='fire', level=None, origin='gate' (bypassing classifier stage).
    """
    events, active_tracks, next_id = [], [], 1
    for frame_idx, dets in detections_over_time:
        valid_dets = [d for d in dets if d["conf"] >= min_conf]
        active_tracks, finished, matched = _track_update_step(active_tracks, valid_dets, frame_idx, max_gap)
        for trk in finished:
            if trk["count"] >= min_frames:
                dom_cls = max(set(trk["classes"]), key=trk["classes"].count)
                events.append({"id": trk["track_id"], "start_frame": trk["start_frame"], "end_frame": trk["end_frame"],
                               "box": trk["box"], "score": trk["max_conf"], "source": "fire", "cls": dom_cls,
                               "level": None, "origin": "gate"})
        for idx, d in enumerate(valid_dets):
            if not matched[idx]:
                active_tracks.append({"track_id": next_id, "start_frame": frame_idx, "end_frame": frame_idx,
                                      "box": d["box"], "max_conf": d["conf"], "count": 1, "gap": 0, "classes": [d["cls"]]})
                next_id += 1
    for trk in active_tracks:
        if trk["count"] >= min_frames:
            dom_cls = max(set(trk["classes"]), key=trk["classes"].count)
            events.append({"id": trk["track_id"], "start_frame": trk["start_frame"], "end_frame": trk["end_frame"],
                           "box": trk["box"], "score": trk["max_conf"], "source": "fire", "cls": dom_cls,
                           "level": None, "origin": "gate"})
    return events
