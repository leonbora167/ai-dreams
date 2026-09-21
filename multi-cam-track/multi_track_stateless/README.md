# Decoupled Stateless-Service Multi-Target Multi-Camera (MTMC) Tracker

An enterprise-ready, modular Multi-Target Multi-Camera (MTMC) tracking system designed specifically for **decoupled microservice architectures** and **Triton Inference Server** deployment.

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │            Stateless Model Services (Triton-Ready)          │
                    │  ┌──────────────────────┐    ┌───────────────────────────┐  │
                    │  │   Detector Service   │    │      Feature Service      │  │
                    │  │ (RF-DETR Nano / YOLO)│    │  (OSNet ReID, Pose, HSV)  │  │
                    │  └──────────┬───────────┘    └─────────────▲─────────────┘  │
                    └─────────────┼──────────────────────────────┼────────────────┘
                        Frames In │ Detections      Crops In     │ Embeddings Out
                                  ▼                              │
                    ┌────────────────────────────────────────────┴────────────────┐
                    │               Stateful Tracking Runtime                     │
                    │                                                             │
                    │  ┌───────────────────────────────────────────────────────┐  │
                    │  │        Camera Tracker Session (per camera stream)     │  │
                    │  │  - Single-Camera Tracker Instance (OC-SORT / BoT-SORT)│  │
                    │  │  - Kalman Filter State Vectors                        │  │
                    │  │  - 24-Crop Dynamic Reservoir Sampling                 │  │
                    │  └──────────────────────────┬────────────────────────────┘  │
                    │                             │ Finished Tracklets            │
                    │                             ▼                               │
                    │  ┌───────────────────────────────────────────────────────┐  │
                    │  │        Global Cross-Camera Association Gallery        │  │
                    │  │  - Temporal Travel Windows (Δt ≤ max_transition)     │  │
                    │  │  - Physical Camera Spatio-Temporal Mutual Exclusion   │  │
                    │  │  - Multi-Modal Cosine Metric Fusion                   │  │
                    │  └───────────────────────────────────────────────────────┘  │
                    └─────────────────────────────────────────────────────────────┘
```

---

## Key Architectural Principles

### 1. Pure Stateless Inference Services (`src/services/`)
- **Zero Tracker State**: Detectors and Feature Extractors take raw NumPy arrays (`frame` or `crops`) and return bounding boxes or feature vectors. They do **not** maintain track IDs, frame histories, or identity state.
- **Triton Inference Server Ready**: Any model service can be switched from local in-process inference to a remote Triton gRPC/HTTP endpoint (`TritonDetectorService`, `TritonReIDExtractor`) simply by changing the config file. No tracker logic needs to be modified.
- **Licensing Compliance**: Default person detection uses **RF-DETR** (`rf-detr-nano`), licensed under permissive **Apache 2.0**.

### 2. Stateful Tracking Sessions (`src/state/`)
- **`CameraTrackerSession`**: Each camera video stream is processed with its own isolated tracking session. Holds the single-camera tracker instance (`OCSortTracker` or `BoTSORTTracker`), Kalman filters, and a 24-crop reservoir buffer per active track.
- **`GlobalGallery`**: Cross-camera identity matching state store. Evaluates physical constraints (same target cannot appear simultaneously in two views) and maximum inter-camera transit time before performing multi-modal feature similarity matching.

### 3. Production Trackers (`src/trackers/`)
Only high-performance, industry-standard trackers are included:
1. **OC-SORT (Observation-Centric SORT)** *(Default)*:
   - Momentum-aware Kalman filtering with Observation-Centric Recovery (OCR) and Observation-Centric Online Smoothing (OCOS).
   - Minimizes identity switches during sharp direction reversals and non-linear movement.
2. **BoT-SORT**:
   - Advanced Kalman filter formulation with camera motion compensation (GMC) and two-stage high/low confidence association.

---

## Directory Structure

```
multi_track_stateless/
├── configs/
│   ├── pets2009.yaml            # Benchmark configuration on PETS 2009 dataset
│   └── custom.yaml              # Ready-to-use template for your custom video streams
├── data -> ../data              # Symlinked dataset root
├── models -> ../models          # Symlinked deep learning checkpoints
├── outputs/                     # Track outputs, evaluation metrics, and review videos
├── scripts/
│   └── evaluate.py              # Benchmark evaluation against ground truth annotations
├── src/
│   ├── config.py                # Configuration loading & video discovery
│   ├── device.py                # Auto-device resolution (MPS / CUDA / CPU)
│   ├── pipeline.py              # Decoupled sequential MTMC orchestration engine
│   ├── tracklets.py             # Tracklet data containers & serialization
│   ├── visualization.py         # Per-camera video renderer & grid stitcher
│   ├── services/                # Stateless inference services
│   │   ├── detector_service.py  # RF-DETR, YOLO, and Triton detector client stubs
│   │   └── feature_service.py   # OSNet ReID, Color, Pose, and Triton client stubs
│   ├── state/                   # Stateful tracking layer
│   │   ├── camera_session.py    # Per-camera tracker session & reservoir buffer
│   │   └── global_gallery.py    # Cross-camera identity gallery & travel windows
│   └── trackers/                # Stateful single-camera tracker implementations
│       ├── base.py              # Abstract BaseTracker interface
│       ├── ocsort.py            # OC-SORT tracker implementation
│       └── botsort.py           # BoT-SORT tracker implementation
├── run.py                       # CLI execution entry point
├── README.md                    # System documentation and usage guide
└── STATELESS_TRITON_GUIDE.md    # Production deployment & Triton Server integration guide
```

---

## Quickstart

### 1. Run on PETS 2009 Benchmark

To run sequential tracking using RF-DETR + OC-SORT + Pose Estimation:

```bash
/opt/miniconda3/envs/cv/bin/python multi_track_stateless/run.py \
  --config multi_track_stateless/configs/pets2009.yaml
```

Outputs produced in `multi_track_stateless/outputs/pets2009_stateless_ocsort/`:
- `tracks/cam01.json` ... `cam04.json`: Complete tracklet coordinate trajectories and assigned global IDs.
- `observations/cam01.jsonl` ... `cam04.jsonl`: Frame-by-frame detection records with local and global IDs.
- `features/cam01_features.npz`: Extracted multi-modal embeddings for each tracklet.
- `visualization/cam01_tracked.mp4` ... `cam04_tracked.mp4`: Individual annotated camera review videos.
- `visualization/stitched_review.mp4`: Synchronized multi-camera grid video with consistent target colors.

### 2. Evaluate Benchmark Performance

To evaluate tracking accuracy against ground truth (`cam01_gt.json`):

```bash
/opt/miniconda3/envs/cv/bin/python multi_track_stateless/scripts/evaluate.py \
  --output-dir multi_track_stateless/outputs/pets2009_stateless_ocsort \
  --gt-path data/pets2009/ground_truth/cam01_gt.json \
  --gt-camera cam01
```

### 3. Run on Custom Videos

1. Place your video files in a folder, e.g. `data/my_cameras/cam01.mp4`, `cam02.mp4`.
2. Edit `multi_track_stateless/configs/custom.yaml` or provide CLI arguments:

```bash
/opt/miniconda3/envs/cv/bin/python multi_track_stateless/run.py \
  --config multi_track_stateless/configs/custom.yaml \
  --videos /path/to/my_cameras \
  --output-dir multi_track_stateless/outputs/my_custom_run
```

---

## Switching Trackers and Detectors

In your YAML configuration file:

### Switching Tracker:
```yaml
tracker:
  name: ocsort   # or 'botsort'
  track_high_thresh: 0.5
  track_low_thresh: 0.1
  new_track_thresh: 0.6
  track_buffer: 30
```

### Switching Detector:
```yaml
# Option A: RF-DETR (Apache 2.0 license)
detector:
  name: rfdetr
  model: rf-detr-nano   # or 'rf-detr-small'
  confidence: 0.35
  classes: [0]

# Option B: YOLO
detector:
  name: yolo
  model: yolo11n.pt
  confidence: 0.35
  classes: [0]
```

### Toggling Pose Estimation Feature:
```yaml
features:
  pose:
    enabled: true        # set to false to disable
    model_type: yolo11n-pose # or 'rtmpose'
    weight: 0.1
```

