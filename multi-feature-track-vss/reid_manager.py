import json
from pathlib import Path
import numpy as np

from vector_store import AuditoriumVectorDB

class OnlineReIDManager:
    """
    Manages online person re-identification and global identity assignment.
    Ensures that re-appearing persons across temporal breaks/occlusions
    re-adopt their existing Global Track ID from the vector gallery rather
    than being assigned redundant, fragmented IDs.
    """
    def __init__(
        self,
        vector_db: AuditoriumVectorDB,
        reid_sim_threshold: float = 0.75,
        face_sim_threshold: float = 0.60,
        min_time_gap_sec: float = 0.5
    ):
        self.vector_db = vector_db
        self.reid_sim_threshold = float(reid_sim_threshold)
        self.face_sim_threshold = float(face_sim_threshold)
        self.min_time_gap_sec = float(min_time_gap_sec)

        # Mapping: local_track_id (int) -> global_id (str)
        self.local_to_global: dict[int, str] = {}

        # Tracking active tracks on screen at current frame
        self.active_local_ids: set[int] = set()
        self.active_global_ids: set[str] = set()

        # In-memory synchronized gallery for instantaneous candidate matching
        # Key: global_id (str) -> dict of features, intervals, timestamps
        self.gallery: dict[str, dict] = {}
        self._init_gallery_from_db()

    def _init_gallery_from_db(self) -> None:
        """Loads any existing pre-indexed track records from LanceDB into gallery."""
        records = self.vector_db.get_all_records()
        for r in records:
            gid = str(r["track_id"])
            intervals = []
            if "intervals_json" in r and r["intervals_json"]:
                try:
                    intervals = json.loads(r["intervals_json"])
                except Exception:
                    intervals = []
            if not intervals:
                intervals = [{
                    "start_sec": float(r.get("start_sec", 0.0)),
                    "end_sec": float(r.get("end_sec", 0.0)),
                    "start_frame": int(r.get("start_frame", 0)),
                    "end_frame": int(r.get("end_frame", 0))
                }]

            self.gallery[gid] = {
                "track_id": gid,
                "body_vector": np.array(r["body_vector"], dtype=np.float32),
                "face_vector": np.array(r["face_vector"], dtype=np.float32) if r.get("face_vector") is not None else None,
                "siglip_vector": np.array(r["siglip_vector"], dtype=np.float32),
                "has_face": bool(r.get("has_face", False)),
                "best_crop_path": str(r.get("best_crop_path", "")),
                "face_crop_path": str(r.get("face_crop_path", "")),
                "video_source": str(r.get("video_source", "")),
                "intervals": intervals,
                "last_seen_sec": float(r.get("end_sec", 0.0)),
                "last_seen_frame": int(r.get("end_frame", 0))
            }

    def update_frame_state(self, active_tracks: list[dict], timestamp: float, frame_id: int) -> None:
        """
        Updates the set of active tracks present on-screen in the current video frame
        and extends the active duration for confirmed global tracks.
        """
        current_local_ids = {t['track_id'] for t in active_tracks}
        self.active_local_ids = current_local_ids
        
        # Determine currently visible global tracks
        self.active_global_ids = {
            self.local_to_global[lid] for lid in current_local_ids if lid in self.local_to_global
        }

        # Extend ongoing interval for active global tracks
        for gid in self.active_global_ids:
            if gid in self.gallery and self.gallery[gid]["intervals"]:
                self.gallery[gid]["intervals"][-1]["end_sec"] = timestamp
                self.gallery[gid]["intervals"][-1]["end_frame"] = frame_id
                self.gallery[gid]["last_seen_sec"] = timestamp
                self.gallery[gid]["last_seen_frame"] = frame_id

    def get_global_id(self, local_id: int) -> str | None:
        """Returns the assigned global ID for a local tracker ID, if confirmed."""
        return self.local_to_global.get(local_id, None)

    def get_display_label(self, local_id: int) -> str:
        """Returns a formatted label for visual bounding box display."""
        if local_id in self.local_to_global:
            return f"ID: {self.local_to_global[local_id]}"
        return f"~{local_id}"

    def resolve_track_id(
        self,
        local_id: int,
        body_vec: list[float],
        siglip_vec: list[float],
        face_vec: list[float] | None,
        has_face: bool,
        start_sec: float,
        start_frame: int,
        current_sec: float,
        current_frame: int,
        best_crop_path: str,
        face_crop_path: str,
        video_source: str
    ) -> tuple[str, bool, dict | None]:
        """
        Resolves a newly confirmed local tracklet to either an existing Global ID
        (if matching appearance embeddings are found in the vector gallery) or
        creates and registers a brand new Global ID.

        Returns:
            global_id (str): The assigned global identity ID.
            is_reappearance (bool): True if matched to an existing person; False if new.
            match_details (dict | None): Diagnostics if matched (similarity, matched_id).
        """
        # If already resolved earlier, return immediately
        if local_id in self.local_to_global:
            gid = self.local_to_global[local_id]
            return gid, False, None

        b_vec = np.array(body_vec, dtype=np.float32)
        b_norm = np.linalg.norm(b_vec)
        if b_norm > 0:
            b_vec = b_vec / b_norm

        f_vec = None
        if has_face and face_vec is not None:
            f_vec = np.array(face_vec, dtype=np.float32)
            f_norm = np.linalg.norm(f_vec)
            if f_norm > 0:
                f_vec = f_vec / f_norm

        best_match_id = None
        best_match_score = -1.0
        match_type = ""

        # Search candidates in Gallery
        for gid, entry in self.gallery.items():
            # Constraint 1: Spatiotemporal Mutual Exclusion
            # A person cannot be in two different bounding boxes at the exact same frame!
            if gid in self.active_global_ids:
                continue

            # Constraint 2: Min time gap
            # Person must have disappeared prior to this candidate's start time
            time_gap = start_sec - entry["last_seen_sec"]
            if time_gap < self.min_time_gap_sec and time_gap >= -0.5:
                # If appeared almost simultaneously without prior gap, check if valid
                pass

            cand_score = 0.0
            cand_type = ""

            # Tier 1: Face matching (highest confidence)
            if has_face and f_vec is not None and entry["has_face"] and entry["face_vector"] is not None:
                face_sim = float(np.dot(f_vec, entry["face_vector"]))
                if face_sim >= self.face_sim_threshold:
                    cand_score = face_sim
                    cand_type = f"FaceMatch (Sim={face_sim:.3f})"

            # Tier 2: Body Re-ID matching (OSNet)
            body_sim = float(np.dot(b_vec, entry["body_vector"]))
            if body_sim >= self.reid_sim_threshold:
                if body_sim > cand_score:
                    cand_score = body_sim
                    cand_type = f"BodyReID (Sim={body_sim:.3f})"

            if cand_score > best_match_score and cand_score > 0.0:
                best_match_score = cand_score
                best_match_id = gid
                match_type = cand_type

        # ----------------------------------------------------------------------
        # Case A: Match Found -> Re-assign to Existing Global Track ID
        # ----------------------------------------------------------------------
        if best_match_id is not None:
            gid = best_match_id
            print(f"  [Global Re-ID] Matched Local Track #{local_id} -> Global Person #{gid} via {match_type}")

            entry = self.gallery[gid]
            new_interval = {
                "start_sec": float(start_sec),
                "end_sec": float(current_sec),
                "start_frame": int(start_frame),
                "end_frame": int(current_frame)
            }
            entry["intervals"].append(new_interval)
            entry["last_seen_sec"] = float(current_sec)
            entry["last_seen_frame"] = int(current_frame)

            # Smooth body embedding with running average (70% prior, 30% new view)
            smoothed_body = 0.70 * entry["body_vector"] + 0.30 * b_vec
            s_norm = np.linalg.norm(smoothed_body)
            if s_norm > 0:
                smoothed_body = smoothed_body / s_norm
            entry["body_vector"] = smoothed_body

            # Update LanceDB record
            self.vector_db.update_track_reappearance(
                track_id=gid,
                end_sec=float(current_sec),
                end_frame=int(current_frame),
                intervals_json=json.dumps(entry["intervals"]),
                body_vector=smoothed_body.tolist()
            )

            self.local_to_global[local_id] = gid
            self.active_global_ids.add(gid)
            return gid, True, {"matched_id": gid, "score": best_match_score, "type": match_type}

        # ----------------------------------------------------------------------
        # Case B: No Match -> Mint New Global Track ID
        # ----------------------------------------------------------------------
        new_gid = str(local_id)
        # Ensure ID doesn't collide with existing gallery IDs
        if new_gid in self.gallery:
            # Pick next available integer
            existing_ints = [int(k) for k in self.gallery.keys() if k.isdigit()]
            next_int = (max(existing_ints) + 1) if existing_ints else 1
            new_gid = str(next_int)

        print(f"  [Global Re-ID] Local Track #{local_id} registered as new Global Person #{new_gid}")

        initial_interval = [{
            "start_sec": float(start_sec),
            "end_sec": float(current_sec),
            "start_frame": int(start_frame),
            "end_frame": int(current_frame)
        }]

        new_entry = {
            "track_id": new_gid,
            "body_vector": b_vec,
            "face_vector": f_vec,
            "siglip_vector": np.array(siglip_vec, dtype=np.float32),
            "has_face": has_face,
            "best_crop_path": best_crop_path,
            "face_crop_path": face_crop_path,
            "video_source": video_source,
            "intervals": initial_interval,
            "last_seen_sec": float(current_sec),
            "last_seen_frame": int(current_frame)
        }
        self.gallery[new_gid] = new_entry

        record = {
            "track_id": new_gid,
            "siglip_vector": siglip_vec,
            "body_vector": b_vec.tolist(),
            "face_vector": f_vec.tolist() if f_vec is not None else None,
            "has_face": has_face,
            "start_sec": float(start_sec),
            "end_sec": float(current_sec),
            "start_frame": int(start_frame),
            "end_frame": int(current_frame),
            "best_crop_path": best_crop_path,
            "face_crop_path": face_crop_path,
            "video_source": video_source,
            "intervals": initial_interval
        }
        self.vector_db.insert_track(record)

        self.local_to_global[local_id] = new_gid
        self.active_global_ids.add(new_gid)
        return new_gid, False, None

    def finalize(self) -> None:
        """
        Final sync of all intervals and end boundaries to LanceDB upon video completion.
        """
        for gid, entry in self.gallery.items():
            if entry["intervals"]:
                last_interval = entry["intervals"][-1]
                self.vector_db.update_track_duration(
                    track_id=gid,
                    end_sec=last_interval["end_sec"],
                    end_frame=last_interval["end_frame"],
                    intervals_json=json.dumps(entry["intervals"])
                )

