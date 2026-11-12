"""
Open-vocabulary object tracker, person-object ownership binder,
and deterministic abandoned-object state machine.
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import supervision as sv

try:
    import trackers
    HAS_TRACKERS_PKG = True
except ImportError:
    HAS_TRACKERS_PKG = False


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    return (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))


def _intersection_over_object(obj_bbox: List[float], person_bbox: List[float]) -> float:
    ix1 = max(obj_bbox[0], person_bbox[0])
    iy1 = max(obj_bbox[1], person_bbox[1])
    ix2 = min(obj_bbox[2], person_bbox[2])
    iy2 = min(obj_bbox[3], person_bbox[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    obj_area = max(1.0, (obj_bbox[2] - obj_bbox[0]) * (obj_bbox[3] - obj_bbox[1]))
    return inter / obj_area


def _normalized_proximity_score(
    obj_bbox: List[float],
    person_bbox: List[float],
    margin_ratio: float = 0.35,
) -> float:
    """
    Returns an association score > 0 if the object overlaps or sits within an
    expanded proximity margin of the person bbox. Higher = closer / stronger carrier match.
    """
    pw = max(1.0, person_bbox[2] - person_bbox[0])
    ph = max(1.0, person_bbox[3] - person_bbox[1])
    margin_x = pw * margin_ratio
    margin_y = ph * (margin_ratio * 0.6)

    ex1 = person_bbox[0] - margin_x
    ey1 = person_bbox[1] - margin_y
    ex2 = person_bbox[2] + margin_x
    ey2 = person_bbox[3] + margin_y

    ocx, ocy = _bbox_center(obj_bbox)
    inside_expanded = (ex1 <= ocx <= ex2) and (ey1 <= ocy <= ey2)
    ioo = _intersection_over_object(obj_bbox, person_bbox)

    if not inside_expanded and ioo < 0.10:
        return 0.0

    # Distance to person lower-torso/hand zone (around 60% height of person box)
    pcx = 0.5 * (person_bbox[0] + person_bbox[2])
    pcy_hand = person_bbox[1] + 0.60 * ph
    diag = math.hypot(pw, ph)
    dist_norm = math.hypot(ocx - pcx, ocy - pcy_hand) / max(1.0, diag)

    # Combine overlap + inverse center distance
    return float(max(0.05, 1.5 * ioo + max(0.0, 1.0 - dist_norm)))


class ObjectTracker:
    """
    Tracks open-vocab objects across frames, binds them to their most likely
    human carrier/owner (Global Person ID), and monitors stationary unattended
    durations against `abandon_seconds`.
    """

    def __init__(
        self,
        track_activation_threshold: int = 6,
        abandon_seconds: float = 5.0,
        movement_threshold_ratio: float = 0.025,
        carrier_proximity_ratio: float = 0.40,
    ):
        self.track_activation_threshold = track_activation_threshold
        self.abandon_seconds = float(abandon_seconds)
        self.movement_threshold_ratio = float(movement_threshold_ratio)
        self.carrier_proximity_ratio = float(carrier_proximity_ratio)

        if HAS_TRACKERS_PKG:
            self.tracker = trackers.ByteTrackTracker(
                lost_track_buffer=60,
                track_activation_threshold=0.25,
                high_conf_det_threshold=0.30,
                minimum_consecutive_frames=2,
            )
            self.is_standalone_tracker = True
        else:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=60,
                minimum_consecutive_frames=2,
            )
            self.is_standalone_tracker = False

        self.active_buffers: Dict[int, dict] = {}

    def _compute_quality(self, crop: np.ndarray, bbox: List[float], confidence: float) -> float:
        if crop is None or crop.size == 0:
            return 0.0
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            sharpness = 0.0
        area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
        return sharpness * max(0.01, float(confidence)) * math.sqrt(area)

    def _match_label_for_box(
        self,
        tracked_bbox: List[float],
        detections: np.ndarray,
        labels: List[str],
    ) -> str:
        if len(detections) == 0 or not labels:
            return "object"
        best_iou = -1.0
        best_label = labels[0]
        tx1, ty1, tx2, ty2 = tracked_bbox
        t_area = max(1.0, (tx2 - tx1) * (ty2 - ty1))
        for row, lbl in zip(detections, labels):
            dx1, dy1, dx2, dy2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            ix1 = max(tx1, dx1)
            iy1 = max(ty1, dy1)
            ix2 = min(tx2, dx2)
            iy2 = min(ty2, dy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            d_area = max(1.0, (dx2 - dx1) * (dy2 - dy1))
            iou = inter / max(1.0, t_area + d_area - inter)
            if iou > best_iou:
                best_iou = iou
                best_label = lbl
        return best_label

    def update(
        self,
        detections: np.ndarray,
        labels: List[str],
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
        person_candidates: Optional[List[dict]] = None,
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        Update object tracks, ownership scores, and abandonment state.

        Args:
            detections: (N, 5) float array [x1, y1, x2, y2, conf].
            labels: List of N class labels corresponding to rows in detections.
            frame: Current BGR video frame.
            frame_id: Sequential frame index.
            timestamp: Video elapsed time in seconds.
            person_candidates: Optional list of dicts from active person tracking:
                [{'person_id': str, 'bbox': [x1, y1, x2, y2], 'crop_path': str}]

        Returns:
            active_objects: Tracked objects currently visible in this frame.
            newly_confirmed: Objects crossing `track_activation_threshold` in this frame.
            newly_abandoned: Alert dicts emitted when an object crosses `abandon_seconds`.
        """
        h, w = frame.shape[:2]
        frame_diag = max(1.0, math.hypot(w, h))
        person_candidates = person_candidates or []

        if detections is None or len(detections) == 0:
            sv_dets = sv.Detections.empty()
        else:
            # Assign uniform class_id=0 for tracking continuity so label jitter
            # (e.g. 'bag' <-> 'backpack') doesn't fragment an object's ID.
            sv_dets = sv.Detections(
                xyxy=detections[:, :4],
                confidence=detections[:, 4],
                class_id=np.zeros(len(detections), dtype=int),
            )

        if self.is_standalone_tracker:
            tracked = self.tracker.update(sv_dets)
        else:
            tracked = self.tracker.update_with_detections(sv_dets)

        active_objects: List[dict] = []
        newly_confirmed: List[dict] = []
        newly_abandoned: List[dict] = []

        if tracked.tracker_id is None or len(tracked) == 0:
            return active_objects, newly_confirmed, newly_abandoned

        for i in range(len(tracked)):
            obj_id = int(tracked.tracker_id[i])
            if obj_id < 0:
                continue
            x1, y1, x2, y2 = tracked.xyxy[i]
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
            label = self._match_label_for_box(bbox, detections, labels)

            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(w, int(x2)), min(h, int(y2))
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            crop = frame[iy1:iy2, ix1:ix2]
            quality = self._compute_quality(crop, bbox, conf)

            if obj_id not in self.active_buffers:
                self.active_buffers[obj_id] = {
                    "indexed": False,
                    "start_frame": frame_id,
                    "start_sec": timestamp,
                    "end_frame": frame_id,
                    "end_sec": timestamp,
                    "label_counts": Counter(),
                    "best_crop": crop.copy(),
                    "best_quality": quality,
                    "best_bbox": bbox,
                    "owner_scores": {},       # person_id -> float cumulative score
                    "owner_crop_paths": {},   # person_id -> crop_path
                    "owner_track_id": None,
                    "owner_crop_path": "",
                    "stationary_anchor": bbox,
                    "stationary_since_sec": timestamp,
                    "last_attended_sec": timestamp,
                    "status": "attended",
                    "abandoned_since_sec": -1.0,
                    "abandoned_alerted": False,
                    "abandoned_frame": None,
                    "frame_count": 0,
                }

            buf = self.active_buffers[obj_id]
            buf["end_frame"] = frame_id
            buf["end_sec"] = timestamp
            buf["frame_count"] += 1
            buf["label_counts"][label] += 1
            dominant_label = buf["label_counts"].most_common(1)[0][0]

            if quality > buf["best_quality"]:
                buf["best_quality"] = quality
                buf["best_crop"] = crop.copy()
                buf["best_bbox"] = bbox

            # 1. Person-Object Proximity / Carrier Binding
            any_person_near = False
            for pc in person_candidates:
                pid = str(pc.get("person_id", ""))
                pbbox = pc.get("bbox")
                if not pid or not pbbox:
                    continue
                score = _normalized_proximity_score(
                    bbox, pbbox, margin_ratio=self.carrier_proximity_ratio
                )
                if score > 0.0:
                    any_person_near = True
                    buf["owner_scores"][pid] = buf["owner_scores"].get(pid, 0.0) + score
                    if pc.get("crop_path"):
                        buf["owner_crop_paths"][pid] = pc["crop_path"]

            if buf["owner_scores"]:
                best_owner = max(buf["owner_scores"].items(), key=lambda kv: kv[1])[0]
                buf["owner_track_id"] = best_owner
                buf["owner_crop_path"] = buf["owner_crop_paths"].get(best_owner, "")

            # 2. Stationary + Unattended Abandonment State Machine
            acx, acy = _bbox_center(buf["stationary_anchor"])
            ccx, ccy = _bbox_center(bbox)
            disp_ratio = math.hypot(ccx - acx, ccy - acy) / frame_diag

            move_th = 0.08 if buf["status"] == "abandoned" else self.movement_threshold_ratio
            if disp_ratio > move_th:
                # Object moved noticeably -> reset stationary timer
                buf["stationary_anchor"] = bbox
                buf["stationary_since_sec"] = timestamp
                if any_person_near:
                    buf["status"] = "attended"
                elif buf["status"] != "abandoned":
                    buf["status"] = "moving"

            if any_person_near:
                buf["last_attended_sec"] = timestamp
                # Do not downgrade 'abandoned' just because a passerby walks past;
                # only real displacement (>8%) with a carrier recovers it above.
                if buf["status"] != "abandoned":
                    buf["status"] = "attended"
            else:
                unattended_start = max(buf["stationary_since_sec"], buf["last_attended_sec"])
                unattended_dur = max(0.0, timestamp - unattended_start)
                if unattended_dur >= self.abandon_seconds:
                    if buf["status"] != "abandoned":
                        buf["status"] = "abandoned"
                        buf["abandoned_since_sec"] = unattended_start + self.abandon_seconds
                        buf["abandoned_frame"] = frame.copy()
                        if not buf["abandoned_alerted"]:
                            buf["abandoned_alerted"] = True
                            newly_abandoned.append({
                                "object_id": str(obj_id),
                                "label": dominant_label,
                                "bbox": bbox,
                                "owner_track_id": buf["owner_track_id"] or "Unknown",
                                "owner_crop_path": buf["owner_crop_path"],
                                "abandoned_since_sec": buf["abandoned_since_sec"],
                                "timestamp": timestamp,
                                "frame_id": frame_id,
                                "abandoned_frame": buf["abandoned_frame"],
                                "best_crop": buf["best_crop"],
                            })
                else:
                    if buf["status"] != "abandoned":
                        buf["status"] = "stationary"

            obj_state = {
                "object_id": str(obj_id),
                "label": dominant_label,
                "bbox": bbox,
                "confidence": conf,
                "owner_track_id": buf["owner_track_id"] or "Unknown",
                "owner_crop_path": buf["owner_crop_path"],
                "status": buf["status"],
                "abandoned_since_sec": buf["abandoned_since_sec"],
                "start_sec": buf["start_sec"],
                "end_sec": buf["end_sec"],
                "start_frame": buf["start_frame"],
                "end_frame": buf["end_frame"],
            }
            active_objects.append(obj_state)

            # Activation check
            if not buf["indexed"] and buf["frame_count"] >= self.track_activation_threshold:
                buf["indexed"] = True
                newly_confirmed.append({
                    **obj_state,
                    "best_crop": buf["best_crop"].copy(),
                })

        return active_objects, newly_confirmed, newly_abandoned

    def get_all_indexed_objects(self) -> List[dict]:
        """Returns final state snapshot for every indexed object track at end of video."""
        results = []
        for obj_id, buf in self.active_buffers.items():
            if not buf["indexed"]:
                continue
            dominant_label = (
                buf["label_counts"].most_common(1)[0][0]
                if buf["label_counts"]
                else "object"
            )
            results.append({
                "object_id": str(obj_id),
                "label": dominant_label,
                "owner_track_id": buf["owner_track_id"] or "Unknown",
                "owner_crop_path": buf["owner_crop_path"],
                "status": buf["status"],
                "abandoned_since_sec": float(buf["abandoned_since_sec"]),
                "start_sec": float(buf["start_sec"]),
                "end_sec": float(buf["end_sec"]),
                "start_frame": int(buf["start_frame"]),
                "end_frame": int(buf["end_frame"]),
                "best_crop": buf["best_crop"],
                "abandoned_frame": buf["abandoned_frame"],
            })
        return results
