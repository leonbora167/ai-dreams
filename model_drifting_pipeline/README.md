# AI Model Drift Assessment & Observability Engine

Phase 1 Model Drift Assessment POC built for evaluating deployed Computer Vision models (**RF-DETR** object detection & **Inception-v3** image classification) against newly supplied client datasets relative to a reference golden dataset.

For a full conceptual explanation, metric breakdown, and architecture guide designed for all skill levels, please read **[flow.md](flow.md)**.

---

## Key Highlights
- **100% Free & Open Source**: Powered by **Evidently AI** and standalone self-contained visualization reports.
- **Zero Heavy Web Server Dependencies**: No Streamlit or continuous SaaS servers required. Generates professional, interactive HTML reports viewable in any web browser.
- **Strict Separation of Drift Dimensions**:
  1. Data Quality Drift (blur, contrast, brightness, noise)
  2. Data / Input Drift (Kolmogorov-Smirnov p-value, Wasserstein distance, PSI)
  3. Feature / Embedding Drift (latent feature centroid shift)
  4. Prediction Drift (confidence drops, entropy, detection counts)
  5. Performance Drift (mAP@50 or Accuracy degradation measured strictly when ground truth is provided)

---

## Quickstart

### 1. Environment Setup
```bash
conda activate temp-env
pip install -r requirements.txt
```

### 2. Download / Prepare Benchmark Datasets
```bash
# CIFAR-10 classification benchmark datasets
PYTHONPATH=. python src/utils/prepare_datasets.py

# PASCAL VOC object detection benchmark datasets
PYTHONPATH=. python src/utils/prepare_voc_dataset.py
```

### 3. Run Inference Drift Assessments
Run an ad-hoc assessment on an existing model:

**Image Classification (Inception-v3):**
```bash
python -m src.drift.runner \
  --model inception_v3 \
  --golden data/golden/cifar10_golden \
  --new data/new/cifar10_camera_degraded \
  --assessment_id custom_client_eval_01
```

**Object Detection (RF-DETR):**
```bash
python -m src.drift.runner \
  --model rf_detr \
  --golden data/golden/voc_golden \
  --new data/new/voc_camera_degraded \
  --assessment_id custom_detection_eval_01
```

Every run automatically outputs:
- **Executive Visual Report**: `results/assessments/<assessment_id>/executive_report.html` (open in your browser)
- **Evidently AI Interactive Report**: `results/assessments/<assessment_id>/evidently_drift_report.html`
- **Machine-readable JSON**: `results/assessments/<assessment_id>/summary.json`

### 4. Run Training Drift Experiments & Launch TensorBoard
```bash
# Synthetic noise sweep (0%, 5%, 10%, 20%)
python -m experiments.training_drift.run_training_drift --model inception_v3 --run_id sweep_01

# Or with your own custom Golden vs Dirty datasets
python -m experiments.training_drift.run_training_drift \
  --model rf_detr \
  --golden_dataset data/golden/voc_golden \
  --dirty_dataset data/new/voc_camera_degraded \
  --run_id custom_voc_train_01

# Launch TensorBoard:
tensorboard --logdir results/training_drift --port 6006
```
Report generated at: `results/training_drift/training_drift_report.html`.

### 5. Run Full Acceptance Validation Suite
```bash
PYTHONPATH=. python experiments/validate_all.py
```
Validates all 5 handover scenarios:
1. Golden vs Identical Copy -> `LOW RISK` (No drift)
2. Golden vs Degraded -> `HIGH RISK` (Data quality & feature drift)
3. Golden vs Distribution Shifted -> Prediction drift detected
4. Golden vs Unlabelled Dataset -> Performance marked `NOT AVAILABLE` (no false degradation claims)
5. RF-DETR Detection with Degraded Dataset -> Severe mAP drop detected
