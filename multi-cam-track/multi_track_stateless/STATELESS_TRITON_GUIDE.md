# Triton Inference Server Decoupling & Deployment Guide

This guide details how the **`multi_track_stateless`** architecture separates compute-intensive neural network inference from multi-camera tracking state, enabling zero-friction hosting on **NVIDIA Triton Inference Server**.

---

## 1. The Monolithic Tracker Problem

Traditional multi-camera trackers tightly couple tracking logic with neural networks:

```
Monolithic Design (Difficult to Scale):
Frame ──> [ Class Tracker { Kalman Filter + PyTorch Net + ID History } ] ──> Output
```

**Why this breaks in production**:
- **Cannot Scale Dynamically**: Running 20 camera streams on a single GPU runs out of VRAM; moving models to a shared Triton cluster fails because the tracker cannot be serialized across network boundaries without losing its Kalman state.
- **Resource Inefficiency**: If camera 1 sees 10 people and camera 2 sees 0 people, GPU resources are locked to fixed processes rather than batched dynamically.
- **Deployment Coupling**: Upgrading a detector model requires restarting all camera tracking instances, dropping active trajectories and Kalman filters.

---

## 2. Decoupled Architecture

`multi_track_stateless` enforces a strict architectural boundary:

```
                       ┌─────────────────────────────────────────────────────────────┐
                       │           Stateless Model Cluster (Triton Server)           │
                       │                                                             │
                       │   Input Tensors (NCHW)  ──>  Output Tensors (Scores/Embed)  │
                       │   • RF-DETR Detection        • OSNet ReID Metric            │
                       │   • Dynamic Server Batching  • Multi-GPU Load Balancing     │
                       └───────────────▲─────────────────────────────┬───────────────┘
                                       │ gRPC / HTTP                 │
                   Raw Frames / Crops  │                             │ BBoxes / Embeddings
                                       │                             ▼
┌──────────────────────────────────────┴───────────────────────────────────────────────────────┐
│                           Stateful Edge Tracking Runtime                                     │
│                                                                                              │
│   Camera 1 ──> [ CameraTrackerSession (OC-SORT/BoT-SORT #1) ] ──> Tracklets 1 ──┐             │
│   Camera 2 ──> [ CameraTrackerSession (OC-SORT/BoT-SORT #2) ] ──> Tracklets 2 ──┼──> Global   │
│   Camera N ──> [ CameraTrackerSession (OC-SORT/BoT-SORT #N) ] ──> Tracklets N ──┘   Gallery  │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Division of Responsibilities

| Component | Stateful / Stateless | Role & Memory |
| :--- | :--- | :--- |
| **`StatelessDetectorService`** | **Pure Stateless** | Input: `np.ndarray` (H, W, 3). Output: `Detection(xyxy, conf, cls)`. Zero memory of previous frames or IDs. Hosted on Triton. |
| **`StatelessFeatureService`** | **Pure Stateless** | Input: `List[np.ndarray]` (crops). Output: pooled embedding vector (512-dim). Pure functional mapping. Hosted on Triton. |
| **`CameraTrackerSession`** | **Stateful (Per-Camera)** | Maintains dedicated tracker instance (`OCSortTracker` or `BoTSORTTracker`), Kalman state covariance matrices $\mathbf{x}, \mathbf{P}$, observation velocity, and 24-crop reservoir buffer. |
| **`GlobalGallery`** | **Stateful (System-Wide)** | Cumulative cross-camera identity dictionary. Enforces physical non-overlap and transit time windows before cosine matching. |

---

## 3. Triton Model Repository Setup

To host the models on Triton, organize your Triton model repository as follows:

```
triton_model_repo/
├── rfdetr_nano/
│   ├── 1/
│   │   └── model.onnx (or model.plan for TensorRT)
│   └── config.pbtxt
└── osnet_ain/
    ├── 1/
    │   └── model.onnx (or model.plan)
    └── config.pbtxt
```

### Example `config.pbtxt` for OSNet ReID:

```protobuf
name: "osnet_ain"
platform: "onnxruntime_onnx"
max_batch_size: 64

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 256, 128 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 512 ]
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 5000
}
```

---

## 4. Connecting to Triton in Python

In `multi_track_stateless/src/services/feature_service.py`, `TritonReIDExtractor` is already pre-structured. To enable live gRPC calls:

```python
import tritonclient.grpc as grpcclient
import numpy as np

class TritonReIDExtractor(BaseReIDExtractor):
    def __init__(self, url: str = "localhost:8001", model_name: str = "osnet_ain"):
        self.client = grpcclient.InferenceServerClient(url=url)
        self.model_name = model_name

    def extract_embedding(self, crops: list) -> np.ndarray:
        if not crops:
            return np.zeros(512, dtype=np.float32)

        # 1. Preprocess crops into (B, 3, 256, 128) float32 batch
        batch = preprocess_crops(crops)  # (B, 3, 256, 128)

        # 2. Build Triton inference inputs
        inputs = [grpcclient.InferInput("input", batch.shape, "FP32")]
        inputs[0].set_data_from_numpy(batch)

        outputs = [grpcclient.InferRequestedOutput("output")]

        # 3. Synchronous / Asynchronous remote RPC
        response = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        embeddings = response.as_numpy("output")  # (B, 512)

        # 4. Average pooling & L2 normalization
        pooled = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(pooled) + 1e-8
        return (pooled / norm).astype(np.float32)
```

### Switching in Configuration

Once deployed on Triton, update `configs/custom.yaml`:

```yaml
features:
  reid:
    enabled: true
    type: triton
    triton_url: "triton-cluster.internal:8001"
    model: osnet_ain
    weight: 0.8
```

Notice that:
- `CameraTrackerSession` does **not** change.
- `OCSortTracker` / `BoTSORTTracker` Kalman filters do **not** change.
- `GlobalGallery` does **not** change.
- Output formats (`.json`, `.jsonl`, `.mp4`) remain **100% identical**.

---

## 5. Benefits of this Architecture

1. **Massive Throughput via Server Batching**:
   Crops from all cameras (Cam 1, Cam 2, Cam 3, Cam 4) can be pooled together into a single dynamic inference batch sent to Triton, maximizing GPU compute utilization.
2. **Zero Tracker State Drift**:
   Because models are strictly functional ($f(\mathbf{x}) \to \mathbf{y}$), network dropped packets or server restarts never corrupt Kalman filter covariance matrices or active track IDs.
3. **Seamless Hot-Swapping**:
   Upgrade the ReID model checkpoint on Triton without restarting the video ingestion services or losing active tracking state.

