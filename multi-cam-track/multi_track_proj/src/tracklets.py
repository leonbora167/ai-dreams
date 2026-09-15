from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np


@dataclass
class Tracklet:
    camera_id: str
    local_id: int
    frames: list
    boxes: list
    confidences: list
    fps: float
    video_path: str
    global_id: int = None

    @property
    def start_time(self): return self.frames[0] / self.fps if self.frames else 0.
    @property
    def end_time(self): return self.frames[-1] / self.fps if self.frames else 0.
    @property
    def last_box(self): return np.asarray(self.boxes[-1], dtype=float)
    @property
    def first_box(self): return np.asarray(self.boxes[0], dtype=float)


def save_tracklets(path, tracks):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(t) for t in tracks], indent=2), encoding='utf-8')


def load_tracklets(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return [Tracklet(**x) for x in data]
