# Comprehensive Drift Analysis Reference Guide

This document explains every model drift methodology implemented across the pipeline, broken down into **Inference Drift** and **Training Drift (Pre-Training and During/Post-Training)**, covering which mathematical and physical metrics are evaluated, why they exist, and how they apply to **Object Detection** vs. **Image Classification**.

---

## 1. Architectural Overview & Workflow Separation

The system separates drift analysis into two independent lifecycles:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           1. INFERENCE DRIFT ASSESSMENT                         │
│                    (Fixed Production Model on Live Client Data)                 │
│                                                                                 │
│  [ Golden Reference Dataset ]  vs  [ New Unlabelled/Labelled Client Data ]      │
│                                ↓                                                │
│         5 Drift Dimensions: Quality, Input, Embeddings, Predictions, mAP        │
│                                ↓                                                │
│          Outputs: Executive HTML Report + Evidently AI HTML Report              │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           2. TRAINING DRIFT PIPELINE                            │
│                  (Guarding Data Integrity & Model Convergence)                  │
│                                                                                 │
│  STAGE A: PRE-TRAINING DATA AUDIT                                               │
│  • Compares raw Golden Train Data vs. New Candidate Training Data               │
│  • Go/No-Go Decision Gate BEFORE burning GPU/Compute hours                      │
│  • Output: Pre-Training Readiness HTML Audit                                    │
│                                ↓                                                │
│  STAGE B: DURING & POST-TRAINING DYNAMICS DRIFT                                 │
│  • Epoch-by-epoch loss convergence gap (Δ Loss = Loss_dirty - Loss_golden)      │
│  • Real-time accuracy divergence curve                                          │
│  • Generalization gap / Overfitting penalty                                     │
│  • Outputs: TensorBoard Event Logs (:6006) + Dynamics HTML Report               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Inference Drift Analysis (Ad-Hoc Assessment)

Inference drift answers: *"Has the live data in the field changed significantly compared to the baseline on which the model was validated, and is model inference at risk?"*

Executed via:
```bash
python -m src.drift.runner --model rf_detr --golden data/golden/voc_golden --new data/new/voc_camera_degraded
```

### The 5 Evaluated Drift Dimensions

#### A. Data Quality Drift (Physical Image Level)
Measures the physical integrity of incoming images before they pass into neural network layers:
- **Sharpness / Blur ($Var(\nabla^2 I)$)**: Variance of the Laplacian kernel across grayscale pixels. Low values flag smudged lenses, defocus, or motion blur.
- **Contrast ($\sigma_I$)**: Standard deviation of pixel luminance. Low contrast indicates washed-out images or fog.
- **Brightness ($\mu_I$)**: Mean pixel intensity. Detects overexposure or underexposed night conditions.
- **Sensor Noise**: Median filter residual deviation. Detects high-ISO sensor grain or dirty hardware sensors.
- **Dimensions / Aspect Ratio ($W/H$)**: Detects camera resolution mismatches or distorted pre-processing.

#### B. Data / Input Drift (Statistical Distribution Level)
Compares the 1D continuous distributions of quality features between Golden and New datasets:
- **Two-Sample Kolmogorov-Smirnov Test (KS Test)**: Non-parametric test comparing cumulative empirical distribution functions ($F_1$ and $F_2$). Returns a $p$-value. If $p < 0.05$, the distributions are statistically distinct.
- **Population Stability Index (PSI)**: Quantifies shift into quantile buckets ($PSI = \sum (Actual\% - Expected\%) \times \ln(Actual\% / Expected\%)$). Values $> 0.25$ indicate significant population drift.
- **Wasserstein Distance (Earth Mover's Distance)**: Minimum cost of turning distribution $A$ into distribution $B$.

#### C. Feature / Latent Embedding Drift (Model Backbone Level)
Measures changes in high-dimensional feature representations inside the neural network:
- **Classification (Inception-v3)**: Extracts 2048-dimensional vectors from the final average pooling layer.
- **Detection (RF-DETR / Faster R-CNN)**: Extracts spatial feature map representations from the ResNet-50 / FPN backbone.
- **Metrics Evaluated**:
  - *Centroid Shift Distance*: Euclidean distance $\|\mu_{feat, golden} - \mu_{feat, new}\|$ between feature centers in latent space.
  - *Embedding Norm Drift*: Two-sample KS test on vector norms ($\|v\|_2$).

#### D. Prediction Drift (Output Probability & Geometry Level)
Monitors changes in the model's output decisions:
- **Classification**:
  - *Confidence Distribution*: Two-sample KS test on top predicted class probabilities.
  - *Prediction Entropy*: Measures model certainty ($H = -\sum p_i \ln p_i$). Higher entropy indicates confusion.
  - *Class Frequency Distribution*: Categorical shift across output classes.
- **Detection**:
  - *Detection Count per Image*: KS test on number of objects detected per image.
  - *Average Detection Confidence*: KS test on bounding-box score distributions.
  - *Bounding-Box Area Distribution*: Detects scale shifts (e.g., objects suddenly appearing much smaller or closer).

#### E. Performance Drift (Ground-Truth Level)
- **Only evaluated when labels are present in the new dataset**:
  - Classification: $\Delta \text{Accuracy}$, $\Delta \text{Precision}$, $\Delta \text{Recall}$, $\Delta \text{F1}$.
  - Detection: $\Delta \text{mAP@50}$ (Mean Average Precision at IoU 0.5), $\Delta \text{Precision}$, $\Delta \text{Recall}$.
- **If unlabelled**: Reports `Performance Drift: NOT AVAILABLE` with a risk advisory.

---

## 3. Training Drift Analysis (Pre-Training & During/Post-Training)

Training drift protects against corrupt training supervision, bad labels, and wasted compute.

---

### A. Pre-Training Drift & Data Quality Gate
Evaluates candidate training data **before** launching expensive GPU jobs:
```bash
python -m src.drift.pre_training_audit \
  --golden_dataset data/golden/voc_golden \
  --new_dataset data/new/voc_camera_degraded \
  --run_id pre_train_audit_01
```

- **Objective**: Prevent training on poor data.
- **Metrics Evaluated**:
  1. *Physical Image Quality Drift*: Checks KS $p$-values and PSI across sharpness, noise, brightness, and contrast.
  2. *Annotation Coverage Audit*: Validates percentage of labelled samples and flags missing/corrupt bounding boxes.
- **Decision Gate**:
  - `LOW RISK — SAFE TO TRAIN`: Candidate data matches golden training quality.
  - `MEDIUM RISK — PROCEED WITH CAUTION`: Minor lighting or blur shifts; recommends specific data augmentations.
  - `HIGH RISK — DO NOT TRAIN`: Rejects candidate data due to high noise or $>10\%$ missing labels.
- **Output Report**: `results/pre_training_drift/<run_id>/pre_training_audit_report.html`

---

### B. During & Post-Training Dynamics Drift
Measures how learning dynamics and convergence trajectories degrade when training on dirty vs. golden data:
```bash
python -m experiments.training_drift.run_training_drift \
  --model inception_v3 \
  --golden_dataset data/golden/cifar10_golden \
  --dirty_dataset data/new/cifar10_camera_degraded \
  --epochs 15 \
  --run_id dynamics_exp_01
```

- **Metrics Evaluated**:
  1. **Dynamic Loss Gap ($\Delta \text{Loss}(e)$)**:
     $$\Delta \text{Loss}(e) = \text{Loss}_{\text{dirty}}(e) - \text{Loss}_{\text{golden}}(e)$$
     Tracks how much higher the training loss plateaus due to contradictory labels or noisy imagery.
  2. **Dynamic Accuracy Gap ($\Delta \text{Acc}(e)$)**:
     $$\Delta \text{Acc}(e) = \text{Acc}_{\text{golden}}(e) - \text{Acc}_{\text{dirty}}(e)$$
     Measures real-time accuracy divergence across epochs.
  3. **Generalization Gap / Memorization Penalty**:
     $$\text{GenGap}(e) = \text{Loss}_{\text{val}}(e) - \text{Loss}_{\text{train}}(e)$$
     A widening generalization gap on dirty data flags that the model is overfitting to noise.
  4. **Overall Training Dynamics Verdict**:
     Computes average loss gap and final degradation drop to output a final rating:
     - `STABLE CONVERGENCE` ($\Delta \text{Loss} \le 0.2$)
     - `MODERATE DIVERGENCE` ($0.2 < \Delta \text{Loss} \le 0.4$)
     - `HIGH DIVERGENCE / SEVERE INSTABILITY` ($\Delta \text{Loss} > 0.4$)
- **Outputs**:
  - Real-time event logs streamed to **TensorBoard** (`results/training_drift/<run_id>/tensorboard_logs`).
  - Standalone HTML report: `results/training_drift/<run_id>/training_dynamics_report.html`.

---

## 4. Summary Matrix: Detection vs. Classification Across All Stages

| Lifecycle Stage | Metric Dimension | Image Classification (Inception-v3) | Object Detection (RF-DETR) |
| :--- | :--- | :--- | :--- |
| **Inference Drift** | **Data Quality** | Sharpness, Contrast, Brightness, Noise | Sharpness, Contrast, Brightness, Noise |
| | **Latent Embeddings** | 2048-d feature pooling vector distance | ResNet-50 / FPN multi-scale feature distance |
| | **Prediction Drift** | Output probability confidence & entropy | Detection counts per image & box area |
| | **Performance** | Accuracy, Precision, Recall, F1 | mAP@50 (IoU 0.5), Precision, Recall |
| **Pre-Training Drift** | **Quality Gate** | Image quality KS tests & missing label ratio | Image quality KS tests & invalid box coordinates |
| | **Verdict** | Safe to train / Caution / Do not train | Safe to train / Caution / Do not train |
| **Training Dynamics** | **Loss Dynamics** | Cross-Entropy Loss Gap ($\Delta \mathcal{L}$) | Combined BBox Regression + Classification Loss Gap |
| | **Accuracy Dynamics**| Top-1 Accuracy Divergence Curve | mAP Convergence Divergence Curve |
| | **Generalization** | Validation cross-entropy divergence | Validation detection loss divergence |
| | **Observability** | TensorBoard (:6006) + HTML Report | TensorBoard (:6006) + HTML Report |

