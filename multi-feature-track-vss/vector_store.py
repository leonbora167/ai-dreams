import json
from pathlib import Path
import lancedb
import pyarrow as pa

TABLE_NAME = "person_tracks"
SCENE_TABLE_NAME = "scene_frames"
OBJECT_TABLE_NAME = "object_tracks"
VIDEO_EVENT_TABLE_NAME = "video_events"


def build_track_schema(siglip_dim: int = 768) -> pa.Schema:
    return pa.schema([
        pa.field("track_id", pa.string()),
        pa.field("siglip_vector", pa.list_(pa.float32(), siglip_dim)),
        pa.field("body_vector", pa.list_(pa.float32(), 512)),
        pa.field("face_vector", pa.list_(pa.float32(), 512), nullable=True),
        pa.field("has_face", pa.bool_()),
        pa.field("pose_label", pa.string()),
        pa.field("start_sec", pa.float64()),
        pa.field("end_sec", pa.float64()),
        pa.field("start_frame", pa.int64()),
        pa.field("end_frame", pa.int64()),
        pa.field("best_crop_path", pa.string()),
        pa.field("face_crop_path", pa.string()),
        pa.field("video_source", pa.string()),
        pa.field("intervals_json", pa.string()),
        pa.field("appearance_crops_json", pa.string()),
    ])


def build_scene_schema(siglip_dim: int = 768) -> pa.Schema:
    return pa.schema([
        pa.field("frame_id", pa.int64()),
        pa.field("timestamp_sec", pa.float64()),
        pa.field("frame_path", pa.string()),
        pa.field("person_count", pa.int64()),
        pa.field("standing_count", pa.int64()),
        pa.field("sitting_count", pa.int64()),
        pa.field("unknown_count", pa.int64()),
        pa.field("siglip_vector", pa.list_(pa.float32(), siglip_dim)),
        pa.field("video_source", pa.string()),
    ])


def build_object_schema(siglip_dim: int = 768) -> pa.Schema:
    return pa.schema([
        pa.field("object_id", pa.string()),
        pa.field("label", pa.string()),
        pa.field("siglip_vector", pa.list_(pa.float32(), siglip_dim)),
        pa.field("owner_track_id", pa.string()),
        pa.field("owner_crop_path", pa.string()),
        pa.field("status", pa.string()),
        pa.field("abandoned_since_sec", pa.float64()),
        pa.field("abandoned_frame_path", pa.string()),
        pa.field("start_sec", pa.float64()),
        pa.field("end_sec", pa.float64()),
        pa.field("start_frame", pa.int64()),
        pa.field("end_frame", pa.int64()),
        pa.field("best_crop_path", pa.string()),
        pa.field("intervals_json", pa.string()),
        pa.field("video_source", pa.string()),
    ])


def build_video_event_schema(siglip_dim: int = 768) -> pa.Schema:
    return pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("start_sec", pa.float64()),
        pa.field("end_sec", pa.float64()),
        pa.field("trigger_reason", pa.string()),
        pa.field("summary_text", pa.string()),
        pa.field("tile_notes_json", pa.string()),
        pa.field("referenced_track_ids_json", pa.string()),
        pa.field("referenced_object_ids_json", pa.string()),
        pa.field("keyframe_path", pa.string()),
        pa.field("siglip_vector", pa.list_(pa.float32(), siglip_dim)),
        pa.field("video_source", pa.string()),
    ])


class AuditoriumVectorDB:
    def __init__(self, db_path: str = "data/lancedb", recreate: bool = False, siglip_dim: int = 768):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self.siglip_dim = siglip_dim

        track_schema = build_track_schema(siglip_dim)
        scene_schema = build_scene_schema(siglip_dim)
        object_schema = build_object_schema(siglip_dim)
        event_schema = build_video_event_schema(siglip_dim)

        self.table = self._open_or_create_table(
            TABLE_NAME, track_schema, required_cols={"intervals_json", "pose_label", "appearance_crops_json"}, recreate=recreate
        )
        self.scene_table = self._open_or_create_table(
            SCENE_TABLE_NAME, scene_schema, required_cols={"frame_path", "standing_count"}, recreate=recreate
        )
        self.object_table = self._open_or_create_table(
            OBJECT_TABLE_NAME, object_schema, required_cols={"object_id", "owner_track_id", "status", "abandoned_since_sec"}, recreate=recreate
        )
        self.event_table = self._open_or_create_table(
            VIDEO_EVENT_TABLE_NAME, event_schema, required_cols={"chunk_id", "summary_text", "referenced_track_ids_json"}, recreate=recreate
        )

    def _open_or_create_table(self, name: str, schema: pa.Schema, required_cols: set[str], recreate: bool):
        table_names = self.db.table_names()
        if name in table_names:
            if recreate:
                self.db.drop_table(name)
                return self.db.create_table(name, schema=schema)
            existing = self.db.open_table(name)
            existing_names = set(existing.schema.names)
            # Check both required columns and matching siglip_vector dimension
            sig_field = existing.schema.field("siglip_vector") if "siglip_vector" in existing_names else None
            dim_mismatch = sig_field is not None and getattr(sig_field.type, "list_size", self.siglip_dim) != self.siglip_dim
            if not required_cols.issubset(existing_names) or dim_mismatch:
                print(f"Updating LanceDB table schema for '{name}'...")
                self.db.drop_table(name)
                return self.db.create_table(name, schema=schema)
            return existing
        return self.db.create_table(name, schema=schema)

    def insert_track(self, record: dict) -> None:
        intervals = record.get("intervals", None)
        if intervals is None:
            intervals_json = record.get("intervals_json", "")
            if not intervals_json:
                intervals_json = json.dumps([{
                    "start_sec": float(record.get("start_sec", 0.0)),
                    "end_sec": float(record.get("end_sec", 0.0)),
                    "start_frame": int(record.get("start_frame", 0)),
                    "end_frame": int(record.get("end_frame", 0))
                }])
        else:
            intervals_json = json.dumps(intervals) if isinstance(intervals, list) else str(intervals)

        crops = record.get("appearance_crops", None)
        if crops is None:
            crops_json = record.get("appearance_crops_json", "")
            if not crops_json:
                best_c = str(record.get("best_crop_path", ""))
                crops_json = json.dumps([best_c] if best_c else [])
        else:
            crops_json = json.dumps(crops) if isinstance(crops, list) else str(crops)

        clean_record = {
            "track_id": str(record.get("track_id", "")),
            "siglip_vector": [float(x) for x in record.get("siglip_vector", [])],
            "body_vector": [float(x) for x in record.get("body_vector", [])],
            "face_vector": [float(x) for x in record["face_vector"]] if record.get("face_vector") is not None else None,
            "has_face": bool(record.get("has_face", False)),
            "pose_label": str(record.get("pose_label", "unknown")),
            "start_sec": float(record.get("start_sec", 0.0)),
            "end_sec": float(record.get("end_sec", 0.0)),
            "start_frame": int(record.get("start_frame", 0)),
            "end_frame": int(record.get("end_frame", 0)),
            "best_crop_path": str(record.get("best_crop_path", "")),
            "face_crop_path": str(record.get("face_crop_path", "")),
            "video_source": str(record.get("video_source", "")),
            "intervals_json": intervals_json,
            "appearance_crops_json": crops_json,
        }
        self.table.add([clean_record])

    def insert_scene_frame(self, record: dict) -> None:
        clean_record = {
            "frame_id": int(record.get("frame_id", 0)),
            "timestamp_sec": float(record.get("timestamp_sec", 0.0)),
            "frame_path": str(record.get("frame_path", "")),
            "person_count": int(record.get("person_count", 0)),
            "standing_count": int(record.get("standing_count", 0)),
            "sitting_count": int(record.get("sitting_count", 0)),
            "unknown_count": int(record.get("unknown_count", 0)),
            "siglip_vector": [float(x) for x in record.get("siglip_vector", [])],
            "video_source": str(record.get("video_source", "")),
        }
        self.scene_table.add([clean_record])

    def update_track_duration(
        self,
        track_id: str,
        end_sec: float,
        end_frame: int,
        intervals_json: str | None = None
    ) -> None:
        values = {
            "end_sec": float(end_sec),
            "end_frame": int(end_frame)
        }
        if intervals_json is not None:
            values["intervals_json"] = str(intervals_json)

        self.table.update(where=f"track_id = '{track_id}'", values=values)

    def update_track_reappearance(
        self,
        track_id: str,
        end_sec: float,
        end_frame: int,
        intervals_json: str,
        body_vector: list[float] | None = None,
        siglip_vector: list[float] | None = None,
        pose_label: str | None = None,
        appearance_crops_json: str | None = None,
        face_vector: list[float] | None = None,
        has_face: bool | None = None,
        face_crop_path: str | None = None
    ) -> None:
        values = {
            "end_sec": float(end_sec),
            "end_frame": int(end_frame),
            "intervals_json": str(intervals_json)
        }
        if body_vector is not None:
            values["body_vector"] = [float(x) for x in body_vector]
        if siglip_vector is not None:
            values["siglip_vector"] = [float(x) for x in siglip_vector]
        if pose_label is not None:
            values["pose_label"] = str(pose_label)
        if appearance_crops_json is not None:
            values["appearance_crops_json"] = str(appearance_crops_json)
        if face_vector is not None and has_face:
            values["face_vector"] = [float(x) for x in face_vector]
            values["has_face"] = True
            if face_crop_path:
                values["face_crop_path"] = str(face_crop_path)

        self.table.update(where=f"track_id = '{track_id}'", values=values)

    def search_by_text_vector(self, vector: list[float], top_k: int = 5, pose_filter: str | None = None) -> list[dict]:
        vec = [float(x) for x in vector]
        query = self.table.search(vec, vector_column_name="siglip_vector").metric("cosine")
        if pose_filter:
            query = query.where(f"pose_label = '{pose_filter}'")
        return query.limit(top_k).to_list()

    def search_by_body_vector(self, vector: list[float], top_k: int = 5) -> list[dict]:
        vec = [float(x) for x in vector]
        return (
            self.table.search(vec, vector_column_name="body_vector")
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )

    def search_by_face_vector(self, vector: list[float], top_k: int = 5) -> list[dict]:
        vec = [float(x) for x in vector]
        return (
            self.table.search(vec, vector_column_name="face_vector")
            .metric("cosine")
            .where("has_face = true")
            .limit(top_k)
            .to_list()
        )

    def search_scene_frames(self, vector: list[float], top_k: int = 5) -> list[dict]:
        vec = [float(x) for x in vector]
        return (
            self.scene_table.search(vec, vector_column_name="siglip_vector")
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )

    def get_all_records(self) -> list[dict]:
        return self.table.to_pandas().to_dict(orient="records")

    def get_all_scene_frames(self) -> list[dict]:
        return self.scene_table.to_pandas().to_dict(orient="records")

    def get_structured_stats(self) -> dict:
        """
        Aggregates exact DB-level counts across unique global person tracks and
        sampled full scene frames for accurate counting Q&A.
        """
        tracks = self.get_all_records()
        scenes = self.get_all_scene_frames()

        standing_tracks = sum(1 for t in tracks if t.get("pose_label") == "standing")
        sitting_tracks = sum(1 for t in tracks if t.get("pose_label") == "sitting")
        unknown_tracks = sum(1 for t in tracks if t.get("pose_label") not in ("standing", "sitting"))
        face_tracks = sum(1 for t in tracks if t.get("has_face"))

        max_people_frame = max((s.get("person_count", 0) for s in scenes), default=0)
        max_standing_frame = max((s.get("standing_count", 0) for s in scenes), default=0)
        max_sitting_frame = max((s.get("sitting_count", 0) for s in scenes), default=0)

        return {
            "total_unique_tracks": len(tracks),
            "standing_tracks": standing_tracks,
            "sitting_tracks": sitting_tracks,
            "unknown_pose_tracks": unknown_tracks,
            "tracks_with_faces": face_tracks,
            "total_scene_keyframes": len(scenes),
            "peak_simultaneous_people": max_people_frame,
            "peak_simultaneous_standing": max_standing_frame,
            "peak_simultaneous_sitting": max_sitting_frame,
        }

    # --- Phase 1: Object Tracking & Ownership CRUD / Search ---

    def insert_object(self, record: dict) -> None:
        intervals = record.get("intervals", None)
        if intervals is None:
            intervals_json = record.get("intervals_json", "")
            if not intervals_json:
                intervals_json = json.dumps([{
                    "start_sec": float(record.get("start_sec", 0.0)),
                    "end_sec": float(record.get("end_sec", 0.0)),
                    "start_frame": int(record.get("start_frame", 0)),
                    "end_frame": int(record.get("end_frame", 0)),
                }])
        else:
            intervals_json = json.dumps(intervals) if isinstance(intervals, list) else str(intervals)

        row = {
            "object_id": str(record["object_id"]),
            "label": str(record.get("label", "object")),
            "siglip_vector": [float(x) for x in record["siglip_vector"]],
            "owner_track_id": str(record.get("owner_track_id", "Unknown")),
            "owner_crop_path": str(record.get("owner_crop_path", "")),
            "status": str(record.get("status", "attended")),
            "abandoned_since_sec": float(record.get("abandoned_since_sec", -1.0)),
            "abandoned_frame_path": str(record.get("abandoned_frame_path", "")),
            "start_sec": float(record.get("start_sec", 0.0)),
            "end_sec": float(record.get("end_sec", 0.0)),
            "start_frame": int(record.get("start_frame", 0)),
            "end_frame": int(record.get("end_frame", 0)),
            "best_crop_path": str(record.get("best_crop_path", "")),
            "intervals_json": intervals_json,
            "video_source": str(record.get("video_source", "")),
        }
        self.object_table.add([row])

    def update_object(
        self,
        object_id: str,
        owner_track_id: str | None = None,
        owner_crop_path: str | None = None,
        status: str | None = None,
        abandoned_since_sec: float | None = None,
        abandoned_frame_path: str | None = None,
        end_sec: float | None = None,
        end_frame: int | None = None,
        intervals_json: str | None = None,
        siglip_vector: list[float] | None = None,
    ) -> None:
        values = {}
        if owner_track_id is not None:
            values["owner_track_id"] = str(owner_track_id)
        if owner_crop_path is not None:
            values["owner_crop_path"] = str(owner_crop_path)
        if status is not None:
            values["status"] = str(status)
        if abandoned_since_sec is not None:
            values["abandoned_since_sec"] = float(abandoned_since_sec)
        if abandoned_frame_path is not None:
            values["abandoned_frame_path"] = str(abandoned_frame_path)
        if end_sec is not None:
            values["end_sec"] = float(end_sec)
        if end_frame is not None:
            values["end_frame"] = int(end_frame)
        if intervals_json is not None:
            values["intervals_json"] = str(intervals_json)
        if siglip_vector is not None:
            values["siglip_vector"] = [float(x) for x in siglip_vector]

        if values:
            self.object_table.update(where=f"object_id = '{object_id}'", values=values)

    def search_objects_by_vector(
        self,
        vector: list[float],
        top_k: int = 5,
        status_filter: str | None = None,
    ) -> list[dict]:
        vec = [float(x) for x in vector]
        query = self.object_table.search(vec, vector_column_name="siglip_vector").metric("cosine")
        if status_filter:
            query = query.where(f"status = '{status_filter}'")
        return query.limit(top_k).to_list()

    def get_all_objects(self) -> list[dict]:
        return self.object_table.to_pandas().to_dict(orient="records")

    def get_objects_by_owner(self, owner_track_id: str) -> list[dict]:
        return [
            r for r in self.get_all_objects()
            if str(r.get("owner_track_id", "")) == str(owner_track_id)
        ]

    # --- Phase 2: Video Event Chunk Summaries CRUD / Search ---

    def insert_video_event(self, record: dict) -> None:
        tile_notes = record.get("tile_notes", [])
        tile_notes_json = (
            json.dumps(tile_notes) if not isinstance(tile_notes, str) else tile_notes
        )
        ref_tracks = record.get("referenced_track_ids", [])
        ref_tracks_json = (
            json.dumps([str(x) for x in ref_tracks])
            if not isinstance(ref_tracks, str)
            else ref_tracks
        )
        ref_objs = record.get("referenced_object_ids", [])
        ref_objs_json = (
            json.dumps([str(x) for x in ref_objs])
            if not isinstance(ref_objs, str)
            else ref_objs
        )

        row = {
            "chunk_id": str(record["chunk_id"]),
            "start_sec": float(record.get("start_sec", 0.0)),
            "end_sec": float(record.get("end_sec", 0.0)),
            "trigger_reason": str(record.get("trigger_reason", "periodic")),
            "summary_text": str(record.get("summary_text", "")),
            "tile_notes_json": tile_notes_json,
            "referenced_track_ids_json": ref_tracks_json,
            "referenced_object_ids_json": ref_objs_json,
            "keyframe_path": str(record.get("keyframe_path", "")),
            "siglip_vector": [float(x) for x in record["siglip_vector"]],
            "video_source": str(record.get("video_source", "")),
        }
        self.event_table.add([row])

    def search_video_events(self, vector: list[float], top_k: int = 5) -> list[dict]:
        vec = [float(x) for x in vector]
        return (
            self.event_table.search(vec, vector_column_name="siglip_vector")
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )

    def get_all_video_events(self) -> list[dict]:
        rows = self.event_table.to_pandas().to_dict(orient="records")
        return sorted(rows, key=lambda r: float(r.get("start_sec", 0.0)))
