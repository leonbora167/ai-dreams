import json
import numpy as np

from vector_store import AuditoriumVectorDB


class OnlineReIDManager:
    """
    Manages online person re-identification and global identity assignment.
    Uses MSMT17-trained OSNet-AIN body embeddings + ArcFace facial identity
    with spatiotemporal mutual exclusion so people visible simultaneously
    are never collapsed into the same identity.
    """

    def __init__(
        self,
        vector_db: AuditoriumVectorDB,
        reid_sim_threshold: float = 0.72,
        face_sim_threshold: float = 0.55,
        min_time_gap_sec: float = 0.5,
        multi_angle_reid: bool = False,
        secondary_reid_threshold: float = 0.76,
    ):
        self.vector_db = vector_db
        self.reid_sim_threshold = float(reid_sim_threshold)
        self.face_sim_threshold = float(face_sim_threshold)
        self.min_time_gap_sec = float(min_time_gap_sec)
        self.multi_angle_reid = bool(multi_angle_reid)
        self.secondary_reid_threshold = float(secondary_reid_threshold)

        self.local_to_global: dict[int, str] = {}
        self.active_local_ids: set[int] = set()
        self.active_global_ids: set[str] = set()

        self.gallery: dict[str, dict] = {}
        self._init_gallery_from_db()

    def _init_gallery_from_db(self) -> None:
        records = self.vector_db.get_all_records()
        for r in records:
            gid = str(r["track_id"])
            intervals = []
            if r.get("intervals_json"):
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

            crops = []
            if r.get("appearance_crops_json"):
                try:
                    crops = json.loads(r["appearance_crops_json"])
                except Exception:
                    crops = []
            if not crops and r.get("best_crop_path"):
                crops = [str(r["best_crop_path"])]

            self.gallery[gid] = {
                "track_id": gid,
                "body_vector": np.array(r["body_vector"], dtype=np.float32),
                "body_gallery": [np.array(r["body_vector"], dtype=np.float32)],
                "face_vector": np.array(r["face_vector"], dtype=np.float32) if r.get("face_vector") is not None else None,
                "siglip_vector": np.array(r["siglip_vector"], dtype=np.float32),
                "has_face": bool(r.get("has_face", False)),
                "pose_label": str(r.get("pose_label", "unknown")),
                "best_crop_path": str(r.get("best_crop_path", "")),
                "face_crop_path": str(r.get("face_crop_path", "")),
                "video_source": str(r.get("video_source", "")),
                "intervals": intervals,
                "appearance_crops": crops,
                "last_seen_sec": float(r.get("end_sec", 0.0)),
                "last_seen_frame": int(r.get("end_frame", 0))
            }

    def update_frame_state(self, active_tracks: list[dict], timestamp: float, frame_id: int) -> None:
        current_local_ids = {t["track_id"] for t in active_tracks}
        self.active_local_ids = current_local_ids

        self.active_global_ids = {
            self.local_to_global[lid] for lid in current_local_ids if lid in self.local_to_global
        }

        for gid in self.active_global_ids:
            if gid in self.gallery and self.gallery[gid]["intervals"]:
                self.gallery[gid]["intervals"][-1]["end_sec"] = timestamp
                self.gallery[gid]["intervals"][-1]["end_frame"] = frame_id
                self.gallery[gid]["last_seen_sec"] = timestamp
                self.gallery[gid]["last_seen_frame"] = frame_id

    def maybe_add_angle_vector(
        self,
        local_id: int,
        body_vec: list[float],
        is_isolated: bool,
        max_gallery: int = 5,
    ) -> bool:
        """
        Opt-in (`multi_angle_reid=True`) helper: stores an additional angle OSNet
        embedding only when the track is isolated (no overlapping person IoU > 0.15)
        and within a safe similarity band (0.58..0.88) to the primary anchor.
        """
        if not self.multi_angle_reid or not is_isolated:
            return False
        gid = self.local_to_global.get(local_id)
        if not gid or gid not in self.gallery:
            return False
        entry = self.gallery[gid]
        gal = entry.setdefault("body_gallery", [entry["body_vector"]])
        if len(gal) >= max_gallery:
            return False

        vec = np.array(body_vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm <= 0:
            return False
        vec = vec / norm

        sim_primary = float(np.dot(vec, entry["body_vector"]))
        if 0.58 <= sim_primary <= 0.88:
            gal.append(vec)
            return True
        return False

    def get_global_id(self, local_id: int) -> str | None:
        return self.local_to_global.get(local_id, None)

    def get_display_label(self, local_id: int) -> str:
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
        pose_label: str,
        start_sec: float,
        start_frame: int,
        current_sec: float,
        current_frame: int,
        best_crop_path: str,
        face_crop_path: str,
        video_source: str
    ) -> tuple[str, bool, dict | None]:
        if local_id in self.local_to_global:
            return self.local_to_global[local_id], False, None

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

        for gid, entry in self.gallery.items():
            # Mutual exclusion: cannot match someone currently visible on screen
            if gid in self.active_global_ids:
                continue

            time_gap = start_sec - entry["last_seen_sec"]
            if time_gap < self.min_time_gap_sec:
                continue

            cand_score = 0.0
            cand_type = ""

            # Tier 1: Face match
            if has_face and f_vec is not None and entry["has_face"] and entry["face_vector"] is not None:
                face_sim = float(np.dot(f_vec, entry["face_vector"]))
                if face_sim >= self.face_sim_threshold:
                    cand_score = face_sim + 0.10  # prioritize strong face confirmation
                    cand_type = f"FaceMatch (Sim={face_sim:.3f})"

            # Tier 2: MSMT17 Body Re-ID match
            body_sim = float(np.dot(b_vec, entry["body_vector"]))
            matched_body_sim = body_sim if body_sim >= self.reid_sim_threshold else -1.0
            matched_type = f"BodyReID (Sim={body_sim:.3f})"

            if self.multi_angle_reid and matched_body_sim < 0.0:
                sec_vecs = entry.get("body_gallery", [])[1:]
                if sec_vecs:
                    sec_sims = [float(np.dot(b_vec, sv)) for sv in sec_vecs]
                    best_sec = max(sec_sims)
                    if best_sec >= self.secondary_reid_threshold:
                        matched_body_sim = best_sec
                        matched_type = f"BodyReID-MultiAngle (Sim={best_sec:.3f})"

            if matched_body_sim > cand_score:
                cand_score = matched_body_sim
                cand_type = matched_type

            if cand_score > best_match_score and cand_score > 0.0:
                best_match_score = cand_score
                best_match_id = gid
                match_type = cand_type

        if best_match_id is not None:
            gid = best_match_id
            print(f"  [Global Re-ID] Matched Local Track #{local_id} -> Global Person #{gid} via {match_type}")

            entry = self.gallery[gid]
            entry["intervals"].append({
                "start_sec": float(start_sec),
                "end_sec": float(current_sec),
                "start_frame": int(start_frame),
                "end_frame": int(current_frame)
            })
            if best_crop_path and best_crop_path not in entry["appearance_crops"]:
                entry["appearance_crops"].append(best_crop_path)

            if entry["pose_label"] == "unknown" and pose_label != "unknown":
                entry["pose_label"] = pose_label

            # Promote face if previously missing
            updated_face = None
            updated_has_face = None
            updated_face_crop = None
            if not entry["has_face"] and has_face and f_vec is not None:
                entry["has_face"] = True
                entry["face_vector"] = f_vec
                entry["face_crop_path"] = face_crop_path
                updated_face = f_vec.tolist()
                updated_has_face = True
                updated_face_crop = face_crop_path

            entry["last_seen_sec"] = float(current_sec)
            entry["last_seen_frame"] = int(current_frame)

            smoothed_body = 0.70 * entry["body_vector"] + 0.30 * b_vec
            s_norm = np.linalg.norm(smoothed_body)
            if s_norm > 0:
                smoothed_body = smoothed_body / s_norm
            entry["body_vector"] = smoothed_body

            self.vector_db.update_track_reappearance(
                track_id=gid,
                end_sec=float(current_sec),
                end_frame=int(current_frame),
                intervals_json=json.dumps(entry["intervals"]),
                body_vector=smoothed_body.tolist(),
                pose_label=entry["pose_label"],
                appearance_crops_json=json.dumps(entry["appearance_crops"]),
                face_vector=updated_face,
                has_face=updated_has_face,
                face_crop_path=updated_face_crop
            )

            self.local_to_global[local_id] = gid
            self.active_global_ids.add(gid)
            return gid, True, {"matched_id": gid, "score": best_match_score, "type": match_type}

        # Case B: New person
        new_gid = str(local_id)
        if new_gid in self.gallery:
            existing_ints = [int(k) for k in self.gallery.keys() if k.isdigit()]
            next_int = (max(existing_ints) + 1) if existing_ints else 1
            new_gid = str(next_int)

        print(f"  [Global Re-ID] Local Track #{local_id} registered as new Global Person #{new_gid} (Pose: {pose_label})")

        initial_interval = [{
            "start_sec": float(start_sec),
            "end_sec": float(current_sec),
            "start_frame": int(start_frame),
            "end_frame": int(current_frame)
        }]
        initial_crops = [best_crop_path] if best_crop_path else []

        self.gallery[new_gid] = {
            "track_id": new_gid,
            "body_vector": b_vec,
            "body_gallery": [b_vec],
            "face_vector": f_vec,
            "siglip_vector": np.array(siglip_vec, dtype=np.float32),
            "has_face": has_face,
            "pose_label": pose_label,
            "best_crop_path": best_crop_path,
            "face_crop_path": face_crop_path,
            "video_source": video_source,
            "intervals": initial_interval,
            "appearance_crops": initial_crops,
            "last_seen_sec": float(current_sec),
            "last_seen_frame": int(current_frame)
        }

        self.vector_db.insert_track({
            "track_id": new_gid,
            "siglip_vector": siglip_vec,
            "body_vector": b_vec.tolist(),
            "face_vector": f_vec.tolist() if f_vec is not None else None,
            "has_face": has_face,
            "pose_label": pose_label,
            "start_sec": float(start_sec),
            "end_sec": float(current_sec),
            "start_frame": int(start_frame),
            "end_frame": int(current_frame),
            "best_crop_path": best_crop_path,
            "face_crop_path": face_crop_path,
            "video_source": video_source,
            "intervals": initial_interval,
            "appearance_crops": initial_crops
        })

        self.local_to_global[local_id] = new_gid
        self.active_global_ids.add(new_gid)
        return new_gid, False, None

    def finalize(self) -> None:
        for gid, entry in self.gallery.items():
            if entry["intervals"]:
                last_interval = entry["intervals"][-1]
                self.vector_db.update_track_duration(
                    track_id=gid,
                    end_sec=last_interval["end_sec"],
                    end_frame=last_interval["end_frame"],
                    intervals_json=json.dumps(entry["intervals"])
                )
