"""Cumulative cross-camera gallery state and sequential identity manager."""
from typing import Dict, List, Tuple
from ..tracklets import Tracklet
from ..services.feature_service import StatelessFeatureService


class GlobalGallery:
    """State store managing cross-camera identity assignments."""

    def __init__(self, cfg: dict, feature_service: StatelessFeatureService):
        self.cfg = cfg
        self.feature_service = feature_service
        self.sim_threshold = float(cfg.get("association", {}).get("similarity_threshold", 0.70))
        topo = cfg.get("cameras", {}).get("topology", {})
        self.max_transition_sec = float(topo.get("max_transition_seconds", 30))
        self.min_transition_sec = float(topo.get("min_transition_seconds", 0))

        # Stores tuples of (Tracklet, descriptor_dict, global_id)
        self.entries: List[Tuple[Tracklet, dict, int]] = []
        self._next_id = 1

    def assign_identity(self, tracklet: Tracklet, descriptor: dict) -> int:
        """Sequential single-pass identity matching against active cumulative gallery."""
        if not self.entries or descriptor.get("_skip"):
            gid = self._next_id
            self._next_id += 1
            self.entries.append((tracklet, descriptor, gid))
            return gid

        best_gid = None
        best_score = -1.0

        for old_track, old_desc, gid in self.entries:
            # Physical conflict check: cannot be in the same camera at overlapping times
            if old_track.camera_id == tracklet.camera_id:
                if not (tracklet.end_time < old_track.start_time or tracklet.start_time > old_track.end_time):
                    continue

            # Temporal travel window constraint across different cameras
            if old_track.camera_id != tracklet.camera_id:
                gap = max(0.0, tracklet.start_time - old_track.end_time, old_track.start_time - tracklet.end_time)
                if gap < self.min_transition_sec or gap > self.max_transition_sec:
                    continue

            # Compute multi-modal similarity via stateless feature service
            score = self.feature_service.compute_similarity(descriptor, old_desc)
            if score > best_score:
                best_gid = gid
                best_score = score

        if best_gid is not None and best_score >= self.sim_threshold:
            assigned_gid = best_gid
        else:
            assigned_gid = self._next_id
            self._next_id += 1

        self.entries.append((tracklet, descriptor, assigned_gid))
        return assigned_gid

    @property
    def total_identities(self) -> int:
        return self._next_id - 1

