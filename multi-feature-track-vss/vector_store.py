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
    pa.field("video_source", pa.string())
])

class AuditoriumVectorDB:
    def __init__(self, db_path: str = "data/lancedb"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        
        table_names = self.db.table_names()
        if TABLE_NAME in table_names:
            self.table = self.db.open_table(TABLE_NAME)
        else:
            self.table = self.db.create_table(TABLE_NAME, schema=SCHEMA)

    def insert_track(self, record: dict) -> None:
        """
        Inserts a single track record into LanceDB.
        """
        # Ensure correct vector formatting and nullable face_vector
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
            "video_source": str(record.get("video_source", ""))
        }
        self.table.add([clean_record])

    def update_track_duration(self, track_id: str, end_sec: float, end_frame: int) -> None:
        """
        Updates the final end_sec and end_frame for a completed or ongoing tracklet.
        """
        self.table.update(
            where=f"track_id = '{track_id}'",
            values={"end_sec": float(end_sec), "end_frame": int(end_frame)}
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

