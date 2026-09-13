# Multi-Target Multi-Camera (MTMC) Tracker

A modular, configuration-driven multi-camera pedestrian tracking system supporting local tracking (ByteTrack + YOLO), appearance re-identification (OSNet ReID), spatial/temporal constraints, and multi-camera association.

- **ELI5 Intuition & Technical Blueprint**: See [`INTUITION_GUIDE.md`](INTUITION_GUIDE.md) for a plain-language walkthrough and architectural deep-dive.
- **Tracker Operations Guide**: See [`TRACKER_GUIDE.md`](TRACKER_GUIDE.md) for operational details, data contracts, and pipeline workflows.
- **Consolidated Offline Benchmark Results**: See [`benchmark_evaluation_summary.csv`](benchmark_evaluation_summary.csv).
- **Consolidated Sequential Benchmark Results**: See [`benchmark_sequential_evaluation_summary.csv`](benchmark_sequential_evaluation_summary.csv).

---

## 1. Quickstart & Hardware Setup

### Environment Activation
The project runs with Python 3.10+ and standard computer vision packages (`torch`, `ultralytics`, `supervision`, `opencv-python`, `scipy`, `tqdm`).
```bash
conda activate cv
```

### Dynamic Compute Hardware Acceleration
The tracker includes automatic device selection (`src/device.py`):
- **Apple Silicon (M1/M2/M3/M4)**: Automatically selects Metal Performance Shaders (`mps`).
- **NVIDIA GPU**: Automatically selects `cuda`.
- **Fallback**: Falls back to `cpu` if no GPU accelerator is present.

### Pretrained ReID Model Weights
Download the recommended Torchreid OSNet model (`osnet_ain_x1_0_msmt17_256x128.pth`) into `models/`:
```bash
python scripts/download_reid_weights.py
```

---

## 2. Downloadable Benchmark Datasets

The repository includes dedicated automated downloaders for standard public multi-camera benchmarks. All scripts automatically download raw video streams, convert them to synchronized MP4s, and parse ground truth annotations where available:

| Dataset | Downloader Script | Config File | Cameras | Frames / Length | Annotations / Ground Truth |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **EPFL Lab (4-Person)** | `scripts/download_poc_dataset.py` | `configs/epfl_4p.yaml` | 4 | ~2.5 mins (25 fps) | Multi-camera indoor lab walkthrough |
| **CAVIAR (Corridor)** | `scripts/download_caviar.py` | `configs/caviar.yaml` | 2 | 1,645 frames | Official CVML XML ground truth (`cam01_gt.json`) |
| **PETS 2009 (S2.L1)** | `scripts/download_pets2009.py` | `configs/pets2009.yaml` | 4 | 795 frames | Official XML/JSON benchmark GT (`cam01_gt.json`) |
| **EPFL Passageway 1** | `scripts/download_epfl_passageway.py` | `configs/epfl_passageway.yaml`| 4 | 2,500 frames | Indoor-to-underground pedestrian tunnel |
| **EPFL Terrace 1** | `scripts/download_epfl_terrace.py` | `configs/epfl_terrace1.yaml` | 4 | 5,010 frames | Outdoor terrace walking and crossing |

### How to Download Each Dataset

```bash
# 1. EPFL 4-Person Lab Sequence (~150 MB)
python scripts/download_poc_dataset.py

# 2. CAVIAR Corridor Sequence + Ground Truth (~50 MB)
python scripts/download_caviar.py

# 3. PETS 2009 Benchmark Dataset + Ground Truth (~200 MB)
python scripts/download_pets2009.py

# 4. EPFL Passageway 1 Sequence (~100 MB)
python scripts/download_epfl_passageway.py

# 5. EPFL Terrace 1 Sequence (~120 MB)
python scripts/download_epfl_terrace.py
```

*Note: All video streams are placed in `data/<dataset_name>/videos/` as `cam01.mp4`, `cam02.mp4`, etc.*

---

## 3. Execution Modes & Comparison

The codebase supports three distinct tracking flows depending on whether you need retrospective forensic accuracy, real-time live simulation, or sequential room-by-room inspection:

| Mode | Entry Command | Video Ingestion | Association Timing | ID Numbering Progression | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Offline Batch** | `python run.py --config <cfg>` | Camera-by-camera, batching tracks to disk | **Post-hoc global pass**: Global hierarchical agglomerative clustering across all cameras | Facility-wide global clustering (`G1`, `G2`... based on cluster order across all streams) | Maximum global accuracy, forensic analysis, benchmark replication |
| **2. Incremental / Streaming** | `python run.py --config <cfg> --incremental` | Interleaved frame-by-frame (`c1_f1 → c2_f1 → c3_f1...`) | **Online + Post-hoc**: Provisional online gallery matching as tracks close + final Union-Find pass | Provisional live IDs updated during stream, final IDs settled at video end | Live RTSP streaming, real-time multi-camera edge deployment |
| **3. Sequential (Camera-by-Camera)** | `python run_sequential.py --config <cfg>` | Camera-by-camera start to finish (`cam01` then `cam02`...) | **Online cumulative**: Matches against growing cumulative gallery in arrival order | `cam01` starts strictly at `G1 [L1]`, `G2 [L2]`... `cam02` preserves gallery (`G1`, `G2` + new `G4`) | Step-by-step camera inspection, per-camera video review, zero retroactive ID flipping |

> For complete flowcharts, internal lifecycle details, and scenario walkthroughs, see [`TRACKER_GUIDE.md#2-the-three-execution-modes-compared`](TRACKER_GUIDE.md#2-the-three-execution-modes-compared).

### Mode A: Offline Batch Tracking (`run.py`)
Processes all camera videos, extracts appearance descriptors, and performs global hierarchical clustering across the entire facility:
```bash
python run.py --config configs/epfl_passageway.yaml --videos data/epfl_passageway/videos --output-dir outputs/epfl_passageway_v1
```

### Mode B: Incremental / Streaming Simulation (`run.py --incremental`)
Simulates live RTSP streams by interleaving frames from all cameras concurrently (`cam1_f1 -> cam2_f1 -> cam3_f1 -> cam1_f2 -> ...`), assigning provisional IDs and performing online gallery matching:
```bash
python run.py --config configs/epfl_passageway.yaml --videos data/epfl_passageway/videos --output-dir outputs/epfl_passageway_incremental_v1 --incremental
```

### Mode C: Sequential Camera-by-Camera Tracking (`run_sequential.py`)
Processes each camera stream completely one-by-one with continuous cumulative gallery memory:
- **Video 1 (`cam01`)**: Starts with a clean slate; tracklets receive IDs `G1 [L1]`, `G2 [L2]`, `G3 [L3]...` strictly in order of appearance (`start_frame`).
- **Video 2 (`cam02`)**: Starts with the cumulative gallery preserved; returning persons keep their IDs (e.g. `G1`, `G2`), while new people get the next available IDs (e.g. `G4`).
- **Subsequent Videos**: Continue this cumulative memory progression.
- **Renders**: Produces both individual per-camera videos (`cam01_tracked.mp4`, etc.) and a synchronized review video (`stitched_review.mp4`).

```bash
# Run sequential tracking on any dataset:
python run_sequential.py --config configs/caviar.yaml --output-dir outputs/caviar_sequential_v1
python run_sequential.py --config configs/pets2009.yaml --output-dir outputs/pets2009_sequential_v1
python run_sequential.py --config configs/epfl_passageway.yaml --output-dir outputs/epfl_passageway_sequential_v1
python run_sequential.py --config configs/epfl_terrace1.yaml --output-dir outputs/epfl_terrace1_sequential_v1
```


---

## 4. How to Run Evaluations

The evaluation module (`scripts/evaluate.py`) computes standard multi-object tracking metrics (MOTA, MOTP, Precision, Recall, IDF1, ID switches) when ground truth is provided, along with multi-camera structural metrics (overlap conflicts, cross-camera consensus, tracklet durations), diagnostic Gantt plots, and annotated review videos.

### Evaluating Datasets with Ground Truth
```bash
# CAVIAR (Evaluates cam01 against official CVML Ground Truth):
python scripts/evaluate.py \
  --output-dir outputs/caviar_sequential_v1 \
  --video-dir data/caviar/videos \
  --gt-path data/caviar/ground_truth/cam01_gt.json \
  --gt-camera cam01 \
  --save-video --max-video-frames 500

# PETS 2009 (Evaluates cam01 against official benchmark Ground Truth):
python scripts/evaluate.py \
  --output-dir outputs/pets2009_sequential_v1 \
  --video-dir data/pets2009/videos \
  --gt-path data/pets2009/ground_truth/cam01_gt.json \
  --gt-camera cam01 \
  --save-video --max-video-frames 500
```

### Evaluating Multi-Camera Structural Consistency (Unannotated Datasets)
```bash
# EPFL Passageway 1:
python scripts/evaluate.py \
  --output-dir outputs/epfl_passageway_sequential_v1 \
  --video-dir data/epfl_passageway/videos \
  --save-video --max-video-frames 500

# EPFL Terrace 1:
python scripts/evaluate.py \
  --output-dir outputs/epfl_terrace1_sequential_v1 \
  --video-dir data/epfl_terrace/videos \
  --save-video --max-video-frames 500
```

### Consolidating Benchmark Summaries
Run either consolidation script to generate clean, consolidated CSV summaries:
```bash
# Generate benchmark_evaluation_summary.csv (for Offline runs):
python scripts/consolidate_benchmarks.py

# Generate benchmark_sequential_evaluation_summary.csv (for Sequential runs):
python scripts/consolidate_sequential_benchmarks.py
```

---

## 5. Output Directory Structure

Each run creates an isolated output folder with structured artifacts:
```text
outputs/<run_name>/
├── tracks/
│   ├── cam01.json                 # Finalized tracklets per camera (frames, boxes, IDs)
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

