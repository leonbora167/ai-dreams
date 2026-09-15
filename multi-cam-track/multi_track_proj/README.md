# Sequential Multi-Target Multi-Camera (MTMC) Tracker

A clean, production-ready, modular computer vision pipeline for multi-camera pedestrian tracking using **sequential camera-by-camera inference with continuous cumulative gallery memory**.

- **Detailed Technical Guide & Extension Manual**: See [`TRACKER_GUIDE.md`](TRACKER_GUIDE.md).

---

## 1. Key Architectural Highlights

1. **Sequential Processing with Cumulative Memory**:
   - Cameras are processed one-by-one (`cam01`, `cam02`, `cam03`...).
   - **Video 1 (`cam01`)**: Processed first. Identifies pedestrians in chronological arrival order (`start_frame`), minting IDs `G1 [L1]`, `G2 [L2]`, `G3 [L3]`...
   - **Video 2 (`cam02`)**: Starts with the cumulative gallery and ID counter preserved from Video 1. Returning individuals keep their IDs (e.g. `G1`, `G2`), while new individuals receive the next sequential ID (e.g. `G4`).
   - **Zero Retroactive Renumbering**: What happens in Video 1 remains fixed. No future cameras alter earlier cameras' IDs.
2. **Fully Pluggable Architecture (Switch by `name:` in config)**:
   - **Detectors** (`src/detectors/`): Switchable by `name:`:
     - `rfdetr`: Roboflow Detection Transformer Nano/Small/Medium/Large from Hugging Face (**Apache 2.0 open-source license**, resolves AGPL licensing concerns).
     - `yolo`: Ultralytics YOLO11, YOLOv8, YOLOv9, YOLOv10.
     - `rtdetr`: Real-Time DETR.
   - **Trackers** (`src/trackers/`): Switchable by `name:`:
     - `bytetrack`: ByteTrack two-stage association.
     - `botsort`: BoT-SORT motion-compensated tracker.
   - **Features** (`src/features/`): Multi-modal appearance & geometry engine:
     - `reid`: OSNet AIN deep metric learning.
     - `color`: 2D HSV spatial color histogram.
     - `pose`: Viewpoint heading angle + biometric body ratios (supports `yolo11n-pose` and `rtmpose`).
3. **Dual Video Output**:
   - Renders individual high-resolution review videos for each camera (`cam01_tracked.mp4`, etc.).
   - Renders a synchronized multi-camera stitched grid review video (`stitched_review.mp4`).

---

## 2. Quickstart & Hardware Setup

### Environment Activation
```bash
conda activate cv
```

### Hardware Acceleration (MPS / CUDA / CPU)
The system automatically chooses the best available hardware via `src/device.py`:
- **Apple Silicon (M1/M2/M3/M4)**: Uses Metal Performance Shaders (`mps`).
- **NVIDIA GPU**: Uses `cuda`.
- **CPU**: Fallback when no accelerator is detected.

### Pretrained Weights
Ensure OSNet ReID weights are present in `models/`:
```bash
python scripts/download_reid_weights.py
```

---

## 3. Supported Datasets & Automated Downloaders

All datasets download and convert into synchronized MP4 files automatically:

| Dataset | Download Command | Config File | Cameras | Frames | Ground Truth |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **CAVIAR (Corridor)** | `python scripts/download_caviar.py` | `configs/caviar.yaml` | 2 | 1,645 | CVML Ground Truth (`cam01_gt.json`) |
| **PETS 2009 (S2.L1)** | `python scripts/download_pets2009.py` | `configs/pets2009.yaml` | 4 | 795 | Benchmark GT (`cam01_gt.json`) |
| **EPFL Passageway 1** | `python scripts/download_epfl_passageway.py` | `configs/epfl_passageway.yaml`| 4 | 2,500 | Multi-camera tunnel sequence |
| **EPFL Terrace 1** | `python scripts/download_epfl_terrace.py` | `configs/epfl_terrace1.yaml` | 4 | 5,010 | Outdoor crossing terrace |

---

## 4. Running Sequential Tracking

To track across cameras for any dataset, run `run.py`:

```bash
# CAVIAR (Corridor)
python run.py --config configs/caviar.yaml --output-dir outputs/caviar_sequential_v1

# PETS 2009 (S2.L1)
python run.py --config configs/pets2009.yaml --output-dir outputs/pets2009_sequential_v1

# EPFL Passageway 1
python run.py --config configs/epfl_passageway.yaml --output-dir outputs/epfl_passageway_sequential_v1

# EPFL Terrace 1
python run.py --config configs/epfl_terrace1.yaml --output-dir outputs/epfl_terrace1_sequential_v1

# Custom Videos (template in configs/custom.yaml)
python run.py --config configs/custom.yaml

# Or override paths on the fly:
python run.py --config configs/custom.yaml --videos /path/to/my_videos --output-dir outputs/my_custom_run
```


---

## 5. Running Evaluations

Run `scripts/evaluate.py` to calculate MOT metrics (MOTA, MOTP, Precision, Recall, IDF1, ID switches) and multi-camera structural consistency:

```bash
# Benchmark Evaluation with Ground Truth (e.g. CAVIAR):
python scripts/evaluate.py \
  --output-dir outputs/caviar_sequential_v1 \
  --video-dir data/caviar/videos \
  --gt-path data/caviar/ground_truth/cam01_gt.json \
  --gt-camera cam01 \
  --save-video --max-video-frames 500

# Multi-Camera Consistency Evaluation (e.g. EPFL Passageway):
python scripts/evaluate.py \
  --output-dir outputs/epfl_passageway_sequential_v1 \
  --video-dir data/epfl_passageway/videos \
  --save-video --max-video-frames 500
```

To compile a consolidated CSV report across all evaluated datasets:
```bash
python scripts/consolidate_sequential_benchmarks.py
```

---

## 6. Modular Configuration Guide

You can customize the detector, tracker, or features directly inside any YAML config file:

```yaml
# Detector Configuration
detector:
  type: yolo                     # 'yolo', 'rtdetr', or custom registered detector
  model: yolo11n.pt              # Model weights or checkpoint path
  device: auto                   # 'auto', 'mps', 'cuda', or 'cpu'
  confidence: 0.35               # Detection confidence threshold
  classes: [0]                   # 0 = person in COCO

# Tracker Configuration
tracker:
  type: bytetrack                # 'bytetrack' or custom registered tracker
  track_high_thresh: 0.5         # High detection threshold for association
  track_low_thresh: 0.1          # Low detection threshold for recovery
  track_buffer: 30               # Lost track buffer (frames)

# Feature Extraction Configuration
features:
  reid:
    enabled: true
    model: osnet_ain_x1_0
    weight_path: ./models/osnet_ain_x1_0_msmt17_256x128.pth
    weight: 0.9                  # Weight in multi-modal similarity score
  color:
    enabled: true
    weight: 0.1                  # Weight in multi-modal similarity score
  quality:
    enabled: true
    min_confidence: 0.5          # Minimum average tracklet confidence to extract ReID
```

---

## 7. Output Directory Structure

Each run generates standard artifacts:
```text
outputs/<run_name>/
├── tracks/
│   ├── cam01.json                 # Tracklet histories (frames, boxes, confidences, IDs)
│   └── ...
├── features/
│   ├── cam01_1.npz                # 512-D ReID embedding + color descriptor + metadata
│   └── ...
├── observations/
│   ├── cam01.jsonl                # Frame-by-frame JSON log of every detection & ID
│   └── ...
├── visualization/
│   ├── cam01_tracked.mp4          # Individual tracked review video for Camera 1
│   ├── cam02_tracked.mp4          # Individual tracked review video for Camera 2
│   └── stitched_review.mp4        # Synchronized multi-camera grid review video
├── evaluation/
│   ├── summary_metrics.csv        # MOTA, MOTP, precision, recall, overlap conflicts
│   ├── global_identities.csv      # Per-identity multi-camera lifespan and camera coverage
│   ├── identity_timeline.png      # Gantt chart of identities across cameras over time
│   └── sample_review_cam01.mp4    # Side-by-side tracker vs ground-truth visual review
└── global_id_map.json             # Lookup dictionary mapping (camera, local_id) -> global_id
```

