# Technical Guide & Architecture Manual (`multi_track_proj`)

This document is the comprehensive technical reference for the **Sequential Multi-Target Multi-Camera (MTMC) Tracker**. It explains the internal data flow, state management, physical constraint rules, and instructions on how to plug in new detectors, trackers, and feature extractors.

---

## 1. End-to-End Sequential Pipeline Flow

The sequential pipeline processes camera videos one-by-one, carrying forward a **continuous cumulative gallery** in memory.

```text
[Video 1: cam01]
  ├── YOLO Detector ──> Detections [xyxy, conf, cls]
  ├── ByteTrack Tracker ──> TrackedDetections [xyxy, local_id, conf]
  ├── Stream frame detections to observations/cam01.jsonl
  │
  ├── When video ends:
  │     ├── Order tracklets chronologically by arrival time (start_frame)
  │     ├── First person appearing gets G1 [L1], second gets G2 [L2]...
  │     ├── Extract ReID & Color descriptors via FeatureManager
  │     ├── Initialize Cumulative Gallery in RAM with cam01 descriptors
  │     ├── Save tracks/cam01.json and features/cam01_*.npz
  │     └── Render individual cam01_tracked.mp4 immediately
  │
  ▼
[Video 2: cam02]  <── (Carries forward Cumulative Gallery from cam01!)
  ├── YOLO Detector + ByteTrack Tracker run on cam02
  │
  ├── When video ends:
  │     ├── Order tracklets chronologically by arrival time (start_frame)
  │     ├── Match each tracklet against Cumulative Gallery:
  │     │     ├── Check intra-camera overlap (cannot match concurrent track)
  │     │     ├── Check temporal transition window (gap <= max_transition_seconds)
  │     │     ├── Compute multi-modal similarity score
  │     │     ├── If similarity >= threshold ──> Re-use existing Global ID (G1, G2...)
  │     │     └── If no match ──> Mint next available Global ID (G4)
  │     ├── Update Cumulative Gallery with cam02 descriptors
  │     ├── Save tracks/cam02.json and features/cam02_*.npz
  │     └── Render individual cam02_tracked.mp4 immediately
  │
  ▼
[Subsequent Videos: cam03, cam04...]
  └── Continue accumulating gallery memory and assigning consistent IDs
  │
  ▼
[Stitching Stage]
  └── Combine all per-camera videos into synchronized stitched_review.mp4
```

---

## 2. Developer Extension Guide: How to Swap Models

The system is decoupled into three modular subsystems with factory registries:

### A. How to Add a New Detector

1. Create a new file in `src/detectors/` (e.g. `src/detectors/my_detector.py`).
2. Subclass `BaseDetector` and implement `detect(frame)`:

```python
from src.detectors.base import BaseDetector, Detection
import numpy as np

class MyDetector(BaseDetector):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # Initialize your custom detector model here
        self.model = ...

    def detect(self, frame: np.ndarray) -> Detection:
        # Run inference on BGR frame (H, W, 3)
        boxes, confs, class_ids = self.model.predict(frame)
        return Detection(
            xyxy=np.asarray(boxes, dtype=np.float32),        # (N, 4)
            confidence=np.asarray(confs, dtype=np.float32),  # (N,)
            class_id=np.asarray(class_ids, dtype=int)       # (N,)
        )
```

### A. Built-in Detectors (Switch via `detector.name`)

The pipeline includes built-in detector backends (`src/detectors/`):
1. **`rfdetr` (`RFDETRDetector`)**: Roboflow Detection Transformer from Hugging Face (**Apache 2.0 open-source license**, resolves AGPL licensing concerns). Supports `rf-detr-nano`, `rf-detr-small`, etc.
2. **`yolo` (`YOLODetector`)**: Ultralytics YOLO11, YOLOv8, YOLOv9, YOLOv10.
3. **`rtdetr` (`YOLODetector`)**: Real-Time DETR.

#### Configuration Example in YAML:
```yaml
detector:
  name: rfdetr          # 'rfdetr', 'yolo', or 'rtdetr'
  model: rf-detr-nano   # model weight name or path
  confidence: 0.35
  classes: [0]
  device: auto          # 'auto', 'mps', 'cuda', or 'cpu'
```

---

### B. Built-in Trackers (Switch via `tracker.name`)

The pipeline includes built-in multi-object tracking backends (`src/trackers/`):
1. **`bytetrack` (`ByteTrackTracker`)**: ByteTrack two-stage association using `supervision`.
2. **`botsort` (`BoTSORTTracker`)**: BoT-SORT tracker with camera motion compensation.

#### Configuration Example in YAML:
```yaml
tracker:
  name: bytetrack       # 'bytetrack' or 'botsort'
  track_high_thresh: 0.5
  track_low_thresh: 0.1
  new_track_thresh: 0.6
  track_buffer: 30
```

---

### C. Built-in Multi-Modal Feature Extractors

The pipeline features a modular multi-modal extraction engine (`src/features/`):
1. **`reid` (`ReIDFeatureExtractor`)**: OSNet AIN deep appearance metric learning (default weight: `0.8`).
2. **`color` (`ColorFeatureExtractor`)**: 2D HSV color histogram spatial representation (default weight: `0.1`).
3. **`pose` (`PoseFeatureExtractor`)**: Strategy 1 Viewpoint & Body Geometry extractor (default weight: `0.1`):
   - **Viewpoint Orientation**: Extracts 17 COCO keypoints to determine heading angle (`Front`, `Right Profile`, `Back`, `Left Profile`) across tracklet crops.
   - **Scale-Invariant Biometrics**: Computes body proportions (shoulder/torso, hip/torso, leg/torso).
   - **Dual Model Support**:
     - `model_type: yolo11n-pose`: Ultralytics YOLO11 Nano Pose (PyTorch/MPS accelerated).
     - `model_type: rtmpose`: OpenMMLab RTMPose-S via `rtmlib` (ONNX Runtime).

#### Configuring Pose in YAML:
```yaml
features:
  reid:
    enabled: true
    weight: 0.8
  color:
    enabled: true
    weight: 0.1
  pose:
    enabled: true
    model_type: yolo11n-pose  # or 'rtmpose'
    weight: 0.1
```

### D. How to Add an Additional Feature Extractor (e.g., Gait, Face, or Attribute)

1. Create a new file in `src/features/` (e.g. `src/features/gait.py`).
2. Subclass `BaseFeatureExtractor` and implement `extract(crops, tracklet)` and `similarity(feat_a, feat_b)`:

```python
from src.features.base import BaseFeatureExtractor
import numpy as np

class GaitFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, spec: dict, cfg: dict):
        super().__init__(spec, cfg)
        self.gait_model = ...

    def extract(self, crops: list, tracklet) -> np.ndarray:
        # Extract gait embedding from sequential tracklet crops
        return ...

    def similarity(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        # Return similarity in range [0.0, 1.0]
        return float(...)
```

3. Register it in `src/features/__init__.py`:
```python
from .gait import GaitFeatureExtractor
register_feature('gait', GaitFeatureExtractor)
```

4. Enable it in your YAML config with your chosen weight:
```yaml
features:
  gait:
    enabled: true
    weight: 0.1
```
*`FeatureManager` will automatically instantiate the extractor, extract its embeddings, and include its score in the weighted multi-modal similarity computation!*

---

## 3. Physical Constraints & Memory Rules

The tracker enforces strict real-world physical rules:

### 1. Inactivity Timeout (`tracker.track_buffer: 30`)
- ByteTrack maintains track state for each person.
- If a person is lost (e.g. occlusion) for up to **30 frames**, the tracker waits. If 30 frames pass with no re-detection, the tracklet closes and moves to the gallery.

### 2. Intra-Camera Co-Location Constraint (Zero Overlap Rule)
- **A single person cannot be in two different bounding boxes at the exact same millisecond in the same camera.**
- If tracklet $A$ and tracklet $B$ co-occur at overlapping frames in the same camera, the system strictly forbids matching them to the same ID, ensuring **0 Overlap Conflicts**.

### 3. Temporal Travel Window (`cameras.topology.max_transition_seconds: 60`)
- When matching a tracklet against previous gallery tracklets, the time gap is checked:
  $$\Delta t = \max(0.0,\, t_{\text{start}} - t_{\text{last\_seen}})$$
- If $\Delta t \le \text{max\_transition\_seconds}$, the person is eligible for re-identification.
- If $\Delta t > \text{max\_transition\_seconds}$, the old entry is considered expired, preventing false associations with new people arriving minutes later.

### 4. Intra-Camera Re-Identification
- If a person walks behind a pillar or exits and re-enters the *same camera* at a non-overlapping time within the temporal window, ReID can re-link them to their original identity.

---

## 4. Evaluation Metrics Reference

| Metric | Full Name | Target | Interpretation |
| :--- | :--- | :---: | :--- |
| **MOTA** | Multi-Object Tracking Accuracy | $> 65\%$ | Combines false alarms, misses, and ID flips: $1 - \frac{\text{FN} + \text{FP} + \text{IDSW}}{\text{GT}}$ |
| **MOTP** | Multi-Object Tracking Precision | $> 75\%$ | Average spatial bounding box overlap ($\text{IoU}$) on correctly tracked targets |
| **Precision** | Detection Precision | $> 90\%$ | Ratio of true pedestrian boxes over all predicted boxes: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ |
| **Recall** | Detection Recall | $> 75\%$ | Ratio of true pedestrian boxes over all ground truth boxes: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ |
| **IDF1** | Identification F1-Score | $> 50\%$ | Measures global trajectory-level identity preservation across occlusions and camera handoffs |
| **IDSW** | Identity Switches | Low | Total count of identity flips within tracking trajectories |
| **Overlap Conflicts** | Spatial Co-location Violations | **0** | Must always be 0; verifies zero identities appear twice concurrently in any camera |
| **Consensus Coverage** | Cross-Camera Consensus | High | Number of unique identities tracked across $\ge 2$, $\ge 3$, or all cameras |

