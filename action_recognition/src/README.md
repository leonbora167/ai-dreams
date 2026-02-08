# `src/` Technical Overview

This document explains the modeling and training internals implemented under `src/crowd_action/`.

## 1) Model Components Used

### Video Swin Transformer (`swin3d_t`)

File: `src/crowd_action/models/multimodal_swin.py`

- Backbone: `torchvision.models.video.swin3d_t` with pretrained weights.
- Role: learns spatio-temporal appearance features from RGB video clips.
- In code:
  - The Swin classification head is replaced with `Identity`.
  - Swin outputs a feature vector `rgb_feat` per clip.

### RAFT Optical Flow

File: `src/crowd_action/features/raft_extractor.py`

- Model: `torchvision.models.optical_flow.raft_small` with pretrained weights.
- Role: estimates motion between consecutive frames.
- Output: for a clip of `T` frames, flow tensor of shape `(T-1, 2, H, W)`.
  - `2` channels are horizontal and vertical flow components.
  - Flow is normalized/clamped to `[-1, 1]`.

### Crowd Density Extractor

File: `src/crowd_action/features/crowd_density.py`

- Detector: `fasterrcnn_mobilenet_v3_large_320_fpn` pretrained on COCO.
- Role: finds persons in each frame and builds a soft density map.
- Process:
  - Keep detections with label `person` and score above threshold.
  - Place Gaussian kernels at person box centers (sigma scales with box size).
  - Sum kernels and normalize map per frame.
- Output: `(T, 1, H, W)` density tensor.

### Auxiliary 3D Encoders + Fusion

File: `src/crowd_action/models/multimodal_swin.py`

- Two lightweight 3D CNN encoders are used:
  - `flow_encoder` for RAFT flow maps
  - `crowd_encoder` for density maps
- Final fusion:
  - `rgb_feat`, `flow_feat`, `crowd_feat` are concatenated.
  - Concatenated vector is passed through an MLP classifier for logits.

## 2) End-to-End Data Flow (Video -> Frames -> Features -> Classification)

### Clip sampling and transforms

Files:
- `src/crowd_action/data/dataset.py`
- `scripts/precompute_aux.py`

For each video:

1. Video is decoded into frames (`TCHW` format in torchvision I/O).
2. `frames_per_clip` indices are sampled using linear spacing (`np.linspace`).
3. RGB frames are resized and normalized for Swin input.

### Auxiliary feature generation

In preprocessing (`scripts/precompute_aux.py`):

1. Same sampled clip is resized (without RGB normalization).
2. RAFT computes flow between consecutive frames.
3. Crowd density extractor computes one map per frame.
4. Both are stored in compressed `.npz` (`flow`, `crowd`) in `data_training/aux/`.

### Training-time tensor assembly

In dataset loader (`src/crowd_action/data/dataset.py`):

1. RGB clip is loaded from video each sample.
2. Matching aux file is loaded by `video_id` when available.
3. If aux is missing, zero tensors are used as fallback.
4. Batch shapes before model:
  - RGB: `B x T x 3 x H x W`
  - Flow: `B x (T-1) x 2 x H x W`
  - Crowd: `B x T x 1 x H x W`

### Model input ordering and fusion

In `MultiModalSwinClassifier.forward`:

1. Tensors are permuted to `B x C x T x H x W` for 3D models.
2. Branch outputs:
  - `rgb_feat = Swin(rgb)`
  - `flow_feat = flow_encoder(flow)`
  - `crowd_feat = crowd_encoder(crowd)`
3. Feature concatenation:
  - `fused = cat([rgb_feat, flow_feat, crowd_feat], dim=1)`
4. MLP produces final class logits.

So yes: modalities are **encoded separately first**, then **concatenated at feature level** (late fusion), not concatenated as raw input frames/channels.

## 3) Loss Computation and Optimization

File: `src/crowd_action/train.py`

### Loss

- Criterion: `torch.nn.CrossEntropyLoss()`
- Inputs:
  - `logits`: model output of shape `B x num_classes`
  - `labels`: integer class indices of shape `B`
- Objective:
  - Minimize negative log-likelihood of the true class under softmax logits.

### Gradient accumulation

- Config field: `train.grad_accum_steps`
- Effective behavior:
  - Backprop uses `loss / grad_accum_steps`
  - Optimizer step occurs every `grad_accum_steps` mini-batches
- This reduces per-step memory pressure while keeping a larger effective batch.

### Mixed precision

- Enabled when CUDA is available and `train.mixed_precision=true`.
- Uses `torch.autocast` + `GradScaler`.

### Gradient clipping

- If `grad_clip_norm > 0`, gradients are norm-clipped before optimizer step.

### OOM handling (optional)

- If `skip_oom_batches=true`, CUDA OOM batches are skipped and training continues.

### Metrics logged

Per epoch:
- `train_loss`, `train_acc`, `val_loss`, `val_acc`

Saved under run directory:
- `best.pt`, `last.pt`, `metrics.csv`, `summary.json`
