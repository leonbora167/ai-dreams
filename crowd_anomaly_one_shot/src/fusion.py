"""Multi-signal fusion and alert generation module for Phase 3-B.

Combines Stage 1 gate events, Stage 2 zero-shot predictions, and Phase 1F fire detections
into structured alerts and review queue items following deterministic operational rules.
"""

import os
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def fuse(gate_events: List[Dict], predictions: Dict[int, Dict], fire_events: List[Dict],
         fps: float = 5.0, extreme_z_thr: float = 15.0) -> Tuple[List[Dict], List[Dict]]:
    """Fuse motion gate events, classifier predictions, and fire detections into alerts and review items.
    
    All operational fusion rules are centralized here:
    Rule 1 (Fire immediate): fire events bypass the classifier and alert immediately.
    Rule 2 (Audit queue): audit clips (origin="audit") never raise alerts; log to review queue.
    Rule 3 (Suppressed normal): gate events where classifier says 'none' -> review queue as suppressed.
    Rule 4 (Confirmed anomaly): classifier says fight/running with threshold satisfied -> alert.
    Rule 5 (Extreme motion uncertain): extreme Z-score >= 15.0 but classifier uncertain -> review queue.
    """
    alerts = []
    review_items = []
    alert_counter = 1
    review_counter = 1

    # Rule 1: Fire events -> Immediate active alert
    for fe in fire_events:
        st = float(fe["start_frame"]) / fps
        et = float(fe["end_frame"]) / fps
        alerts.append({
            "id": f"ALERT_FIRE_{alert_counter:03d}",
            "type": "fire_smoke",
            "start_time": round(st, 2),
            "end_time": round(et, 2),
            "box": fe["box"],
            "confidence": round(float(fe.get("score", 0.90)), 3),
            "sources": ["fire_detector_yolov8n"],
            "classifier": "yolov8n_fire_smoke",
            "clip_path": None
        })
        alert_counter += 1

    # Evaluate each gate event with its Stage 2 prediction
    for ev in gate_events:
        ev_id = ev.get("id", 0)
        st = float(ev["start_frame"]) / fps
        et = float(ev["end_frame"]) / fps
        pred = predictions.get(ev_id, {"top": "none", "top_prob": 0.0, "classifier": "none", "alert_fired": False})

        # Rule 2: Random audit samples never alert
        if ev.get("origin") == "audit":
            review_items.append({
                "id": f"REV_AUDIT_{review_counter:03d}",
                "event_id": ev_id,
                "reason": "Audit sampler queue item",
                "start_time": round(st, 2),
                "end_time": round(et, 2),
                "box": ev["box"],
                "prediction": pred,
                "event_source": ev.get("source", "motion"),
                "score": round(float(ev.get("score", 0.0)), 2),
                "clip_path": None
            })
            review_counter += 1
            continue

        # Rule 3 & Rule 5: Classifier says none or uncertain
        top_cls = pred.get("top", "none")
        top_prob = float(pred.get("top_prob", 0.0))
        alert_fired = pred.get("alert_fired", False)

        if top_cls == "none" or not alert_fired:
            # Rule 5: Extreme motion score with unconfirmed classifier
            if float(ev.get("score", 0.0)) >= extreme_z_thr:
                reason = f"Uncertain Extreme Anomaly (Z={ev['score']:.1f} >= {extreme_z_thr:.1f}, top='{top_cls}', p={top_prob:.2f})"
            else:
                reason = f"Suppressed Normal Crowd (top='{top_cls}', p={top_prob:.2f})"

            review_items.append({
                "id": f"REV_SUPP_{review_counter:03d}",
                "event_id": ev_id,
                "reason": reason,
                "start_time": round(st, 2),
                "end_time": round(et, 2),
                "box": ev["box"],
                "prediction": pred,
                "event_source": ev.get("source", "motion"),
                "score": round(float(ev.get("score", 0.0)), 2),
                "clip_path": None
            })
            review_counter += 1
            continue

        # Rule 4: Confirmed anomaly -> Active alert
        alerts.append({
            "id": f"ALERT_{top_cls.upper()}_{alert_counter:03d}",
            "type": top_cls,
            "start_time": round(st, 2),
            "end_time": round(et, 2),
            "box": ev["box"],
            "confidence": round(top_prob, 3),
            "sources": ["motion_gate", pred.get("classifier", "cascade")],
            "classifier": pred.get("classifier", "cascade"),
            "clip_path": None
        })
        alert_counter += 1

    return alerts, review_items


def save_alert(alert: Dict, frames: List[np.ndarray], out_dir: str, fps: float = 5.0) -> str:
    """Save alert metadata JSON and annotated MP4 snippet with bounding box and label overlay."""
    os.makedirs(out_dir, exist_ok=True)
    aid = alert["id"]
    json_path = os.path.join(out_dir, f"{aid}.json")
    clip_path = os.path.join(out_dir, f"{aid}.mp4")

    # Slice frames corresponding to the alert timeline
    sf = int(round(alert["start_time"] * fps))
    ef = int(round(alert["end_time"] * fps))
    snippet = frames[sf:ef + 1] if ef < len(frames) else frames[sf:]
    if not snippet:
        snippet = [frames[min(sf, len(frames) - 1)]]

    h, w = snippet[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))

    bx = [int(v) for v in alert["box"]]
    for f in snippet:
        draw_f = f.copy()
        # Draw bold alert box
        cv2.rectangle(draw_f, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)
        # Draw header banner
        banner_text = f"ALERT: {alert['type'].upper()} ({alert['confidence']:.2f}) | ID: {aid}"
        cv2.rectangle(draw_f, (0, 0), (w, 28), (0, 0, 180), -1)
        cv2.putText(draw_f, banner_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        writer.write(draw_f)
    writer.release()

    alert["clip_path"] = clip_path
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(alert, jf, indent=2)
    return clip_path


def save_review_item(item: Dict, frames: List[np.ndarray], out_dir: str, fps: float = 5.0) -> str:
    """Save review item metadata JSON and annotated snippet to review queue directory."""
    os.makedirs(out_dir, exist_ok=True)
    rid = item["id"]
    json_path = os.path.join(out_dir, f"{rid}.json")
    clip_path = os.path.join(out_dir, f"{rid}.mp4")

    sf = int(round(item["start_time"] * fps))
    ef = int(round(item["end_time"] * fps))
    snippet = frames[sf:ef + 1] if ef < len(frames) else frames[sf:]
    if not snippet:
        snippet = [frames[min(sf, len(frames) - 1)]]

    h, w = snippet[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))

    bx = [int(v) for v in item["box"]]
    for f in snippet:
        draw_f = f.copy()
        # Yellow border for review/suppressed
        cv2.rectangle(draw_f, (bx[0], bx[1]), (bx[2], bx[3]), (0, 215, 255), 2)
        banner_text = f"REVIEW: {item['reason'][:40]} | ID: {rid}"
        cv2.rectangle(draw_f, (0, 0), (w, 26), (40, 40, 40), -1)
        cv2.putText(draw_f, banner_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 1)
        writer.write(draw_f)
    writer.release()

    item["clip_path"] = clip_path
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(item, jf, indent=2)
    return clip_path
