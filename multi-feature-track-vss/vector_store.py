import json
from pathlib import Path
import lancedb
import pyarrow as pa
import numpy as np

TABLE_NAME = "person_tracks"

SCHEMA = pa.schema([
    pa.field("track_id", pa.string()),
    pa.field("siglip_vector", pa.list_(pa.float32(), 768)),
    pa.field("body_vector", pa.list_(pa.float32(), 512)),
    pa.field("face_vector", pa.list_(pa.float32(), 512), nullable=True),
    pa.field("has_face", pa.bool_()),
    pa.field("start_sec", pa.float64()),
    pa.field("end_sec", pa.float64()),
    pa.field("start_frame", pa.int64()),
    pa.field("end_frame", pa.int64()),
    pa.field("best_crop_path", pa.string()),
    pa.field("face_crop_path", pa.string()),
    pa.field("video_source", pa.string()),
    pa.field("intervals_json", pa.string())
])

class AuditoriumVectorDB:
    def __init__(self, db_path: str = "data/lancedb", recreate: bool = False):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        
        table_names = self.db.table_names()
        if TABLE_NAME in table_names:
            if recreate:
                print(f"Recreating LanceDB table '{TABLE_NAME}'...")
                self.db.drop_table(TABLE_NAME)
                self.table = self.db.create_table(TABLE_NAME, schema=SCHEMA)
            else:
                existing_table = self.db.open_table(TABLE_NAME)
                # Check schema compatibility (ensure intervals_json exists)
                if "intervals_json" not in existing_table.schema.names:
                    print(f"Updating table schema for '{TABLE_NAME}' to include 'intervals_json'...")
                    self.db.drop_table(TABLE_NAME)
                    self.table = self.db.create_table(TABLE_NAME, schema=SCHEMA)
                else:
                    self.table = existing_table
        else:
            self.table = self.db.create_table(TABLE_NAME, schema=SCHEMA)

    def insert_track(self, record: dict) -> None:
        """
        Inserts a single global track record into LanceDB.
        """
        # Format default intervals_json if not provided
        intervals = record.get("intervals", None)
        if intervals is None:
            intervals_json = record.get("intervals_json", "")
            if not intervals_json:
                intervals = [{
                    "start_sec": float(record.get("start_sec", 0.0)),
                    "end_sec": float(record.get("end_sec", 0.0)),
                    "start_frame": int(record.get("start_frame", 0)),
                    "end_frame": int(record.get("end_frame", 0))
                }]
                intervals_json = json.dumps(intervals)
        else:
            intervals_json = json.dumps(intervals) if isinstance(intervals, list) else str(intervals)

        clean_record = {
            "track_id": str(record.get("track_id", "")),
            "siglip_vector": [float(x) for x in record.get("siglip_vector", [])],
            "body_vector": [float(x) for x in record.get("body_vector", [])],
            "face_vector": [float(x) for x in record["face_vector"]] if record.get("face_vector") is not None else None,
            "has_face": bool(record.get("has_face", False)),
            "start_sec": float(record.get("start_sec", 0.0)),
            "end_sec": float(record.get("end_sec", 0.0)),
            "start_frame": int(record.get("start_frame", 0)),
            "end_frame": int(record.get("end_frame", 0)),
            "best_crop_path": str(record.get("best_crop_path", "")),
            "face_crop_path": str(record.get("face_crop_path", "")),
            "video_source": str(record.get("video_source", "")),
            "intervals_json": intervals_json
        }
        self.table.add([clean_record])

    def update_track_duration(
        self,
        track_id: str,
        end_sec: float,
        end_frame: int,
        intervals_json: str | None = None
    ) -> None:
        """
        Updates the end_sec, end_frame, and optionally intervals_json for a global track.
        """
        values = {
            "end_sec": float(end_sec),
            "end_frame": int(end_frame)
        }
        if intervals_json is not None:
            values["intervals_json"] = str(intervals_json)

        self.table.update(
            where=f"track_id = '{track_id}'",
            values=values
        )

    def update_track_reappearance(
        self,
        track_id: str,
        end_sec: float,
        end_frame: int,
        intervals_json: str,
        body_vector: list[float] | None = None,
        siglip_vector: list[float] | None = None
    ) -> None:
        """
        Updates an existing global track record when the person re-appears in a new interval.
        """
        values = {
            "end_sec": float(end_sec),
            "end_frame": int(end_frame),
            "intervals_json": str(intervals_json)
        }
        if body_vector is not None:
            values["body_vector"] = [float(x) for x in body_vector]
        if siglip_vector is not None:
            values["siglip_vector"] = [float(x) for x in siglip_vector]

        self.table.update(
            where=f"track_id = '{track_id}'",
            values=values
        )

    def search_by_text_vector(self, vector: list[float], top_k: int = 5) -> list[dict]:
        """
        Searches using a 768-d SigLIP text/image embedding against siglip_vector.
        """
        vec = [float(x) for x in vector]
        results = (
            self.table.search(vec, vector_column_name="siglip_vector")
            .limit(top_k)
            .to_list()
        )
        return results

    def search_by_body_vector(self, vector: list[float], top_k: int = 5) -> list[dict]:
        """
        Searches using a 512-d OSNet body embedding against body_vector.
        """
        vec = [float(x) for x in vector]
        results = (
            self.table.search(vec, vector_column_name="body_vector")
            .limit(top_k)
            .to_list()
        )
        return results

    def search_by_face_vector(self, vector: list[float], top_k: int = 5) -> list[dict]:
        """
        Searches using a 512-d ArcFace embedding against face_vector for tracks with faces.
        """
        vec = [float(x) for x in vector]
        results = (
            self.table.search(vec, vector_column_name="face_vector")
            .where("has_face = true")
            .limit(top_k)
            .to_list()
        )
        return results

    def get_all_records(self) -> list[dict]:
        """
        Returns all track records stored in the database.
        """
        return self.table.to_pandas().to_dict(orient="records")
