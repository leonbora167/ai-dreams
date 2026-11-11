import cv2
import math
import numpy as np
import supervision as sv

try:
    import trackers
    HAS_TRACKERS_PKG = True
except ImportError:
    HAS_TRACKERS_PKG = False

class AuditoriumTracker:
    def __init__(self, track_activation_threshold: int = 24):
        self.track_activation_threshold = track_activation_threshold
        if HAS_TRACKERS_PKG:
            # Future-proof standalone trackers package from supervision
            self.tracker = trackers.ByteTrackTracker()
            self.is_standalone_tracker = True
        else:
            # Legacy supervision built-in ByteTrack
            self.tracker = sv.ByteTrack()
            self.is_standalone_tracker = False

        # Buffer structure: track_id -> dict
        # {
        #   'indexed': bool,
        #   'start_frame': int,
        #   'start_sec': float,
        #   'end_frame': int,
        #   'end_sec': float,
        #   'trajectory': list,
        #   'tentative_entries': list
        # }
        self.active_buffers: dict[int, dict] = {}

    def _compute_quality(self, crop: np.ndarray, bbox: list[float], confidence: float) -> float:
        """
        Calculates image quality score: Sharpness * Confidence * sqrt(Crop Area)
        Sharpness = Var(Laplacian(gray))
        """
        if crop is None or crop.size == 0:
            return 0.0

        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            sharpness = 0.0

        area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
        quality = sharpness * max(0.01, float(confidence)) * math.sqrt(area)
        return quality

    def update(self, detections: np.ndarray, frame: np.ndarray, frame_id: int, timestamp: float) -> tuple[list[dict], list[dict]]:
        """
        Associates detections across frames, maintains 24-frame RAM buffer,
        and selects the highest quality crop once confirmed.

        Args:
            detections: np.ndarray of shape (N, 5) with [x1, y1, x2, y2, confidence]
            frame: current BGR video frame
            frame_id: sequential integer frame index
            timestamp: elapsed video time in seconds

        Returns:
            active_tracks: list of dicts [{'track_id': int, 'bbox': [x1, y1, x2, y2]}]
            newly_confirmed_tracks: list of dicts [{'track_id': int, 'best_crop': np.ndarray, 'start_frame': int, 'start_sec': float, 'trajectory': list}]
        """
        active_tracks: list[dict] = []
        newly_confirmed_tracks: list[dict] = []

        if detections is None or len(detections) == 0:
            sv_detections = sv.Detections.empty()
        else:
            xyxy = detections[:, :4]
            confidence = detections[:, 4]
            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=np.zeros(len(detections), dtype=int)
            )

        if self.is_standalone_tracker:
            tracked_detections = self.tracker.update(sv_detections, timestamp=timestamp)
        else:
            tracked_detections = self.tracker.update_with_detections(sv_detections)

        if tracked_detections is None or len(tracked_detections) == 0 or tracked_detections.tracker_id is None:
            return active_tracks, newly_confirmed_tracks

        h, w = frame.shape[:2]

        for i in range(len(tracked_detections)):
            track_id = int(tracked_detections.tracker_id[i])
            if track_id < 0:
                continue
            box = tracked_detections.xyxy[i]
            x1 = max(0, min(int(round(box[0])), w - 1))
            y1 = max(0, min(int(round(box[1])), h - 1))
            x2 = max(0, min(int(round(box[2])), w))
            y2 = max(0, min(int(round(box[3])), h))
            bbox = [x1, y1, x2, y2]
            
            conf = float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else 1.0

            active_tracks.append({
                'track_id': track_id,
                'bbox': bbox
            })

            # Crop for quality evaluation
            crop = frame[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None

            if track_id not in self.active_buffers:
                self.active_buffers[track_id] = {
                    'indexed': False,
                    'start_frame': frame_id,
                    'start_sec': timestamp,
                    'end_frame': frame_id,
                    'end_sec': timestamp,
                    'trajectory': [],
                    'tentative_entries': []
                }

            track_info = self.active_buffers[track_id]
            track_info['end_frame'] = frame_id
            track_info['end_sec'] = timestamp
            track_info['trajectory'].append({
                'frame_id': frame_id,
                'timestamp': timestamp,
                'bbox': bbox
            })

            if not track_info['indexed']:
                if crop is not None:
                    track_info['tentative_entries'].append({
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'bbox': bbox,
                        'confidence': conf,
                        'crop': crop
                    })

                # Check if reached activation threshold (24 frames)
                if len(track_info['tentative_entries']) >= self.track_activation_threshold:
                    scored_entries = []
                    for idx, entry in enumerate(track_info['tentative_entries']):
                        score = self._compute_quality(entry['crop'], entry['bbox'], entry['confidence'])
                        scored_entries.append((score, idx, entry['crop']))

                    scored_entries.sort(key=lambda x: x[0], reverse=True)
                    best_crop = scored_entries[0][2] if scored_entries else None
                    # Keep up to 5 top-quality crops spread across the buffer for face detection
                    top_crops = [item[2] for item in scored_entries[:5]]

                    track_info['indexed'] = True
                    # Release crops from RAM to prevent unbounded memory growth
                    track_info['tentative_entries'] = []

                    newly_confirmed_tracks.append({
                        'track_id': track_id,
                        'best_crop': best_crop,
                        'top_crops': top_crops,
                        'start_frame': track_info['start_frame'],
                        'start_sec': track_info['start_sec'],
                        'trajectory': list(track_info['trajectory'])
                    })

        return active_tracks, newly_confirmed_tracks

    def get_track_summary(self, track_id: int) -> dict | None:
        """Returns final timeline summary for a given tracklet."""
        return self.active_buffers.get(track_id)

