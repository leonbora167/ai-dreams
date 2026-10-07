# Pipeline Architecture & Operational Flow Guide

This document is designed for both **non-technical stakeholders** and **engineers/domain specialists**. It explains how the Model Drift Pipeline works, what each file does, which metrics are computed, how Object Detection and Image Classification differ, and step-by-step instructions to set up and run this project on any server with custom datasets.

---

## 1. High-Level Concept: What is Model Drift?

When a Computer Vision (CV) model is deployed to production, real-world data constantly changes (new lighting, camera wear, weather, shifting environments). 

We must distinguish between:
1. **Ad-Hoc Inference Drift Assessment (Primary Platform Feature)**:
   A model is *already deployed* in production. A client sends a new batch of data. Without retraining, the system compares the new data against the original **"Golden" reference dataset** to detect if the data quality has dropped, if features have shifted, or if model performance is at risk.
2. **Controlled Training Drift (Experimental Benchmark)**:
   Proving how bad/corrupted training data (label noise, bad annotations) causes a trained model's accuracy to degrade over time.

```
┌───────────────────────────────────────────────────────────┐
│                       INPUT DATA                          │
│  Golden / Reference Dataset      New Client Dataset       │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                 DRIFT ASSESSMENT ENGINE                   │
│                                                           │
│  1. Image Quality Drift     (Blur, Noise, Brightness)     │
│  2. Data / Input Drift       (Statistical Distributions)  │
│  3. Feature Drift           (Latent Embedding Shift)      │
│  4. Prediction Drift        (Confidence & Class Shift)    │
│  5. Performance Drift*      (mAP or Accuracy Drop)        │
│     * Only when ground-truth labels are present           │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│                     VISUAL REPORTS                        │
│  • Executive Visual HTML Report (Self-Contained)          │
│  • Evidently AI Interactive Report                        │
```

---

## 1.1 Ground Truth vs. "New" Client Data: What Artificial Artifacts Were Added?

In real production deployments, you rarely get to know in advance *how* data will break. To rigorously test and prove this POC without waiting months for real camera hardware to degrade in the field, we created **controlled, synthetic drift experiments** matching realistic industrial failure modes.

### What is the "Golden" / Ground Truth Dataset?
- **Golden Dataset**: The pristine reference baseline. This represents the exact distribution of data on which the model was validated or initially deployed.
  - Contains clean, sharp, properly lit images.
  - Accompanied by **100% verified ground-truth labels** (class names for CIFAR-10, bounding boxes and labels for PASCAL VOC).
  - Used by the drift engine as the **Reference Anchor** to compute statistical baselines.

### What is the "New" Client Dataset, and what artifacts were injected?
The "New" datasets represent data captured weeks or months after deployment. In [`src/utils/perturbations.py`](src/utils/perturbations.py), we injected specific real-world physical and distribution phenomena:

| Test Dataset Scenario | Physical Failure Mode Simulated | Specific Artificial Artifacts Injected | What Drift It Triggers |
| :--- | :--- | :--- | :--- |
| **`cifar10_identical`** | Ideal production day. Camera, lighting, and subjects are identical. | **Zero artifacts added.** Bit-for-bit identical copy of golden images. | **No Drift (LOW RISK)**. Proves the engine has zero false alarms. |
| **`cifar10_camera_degraded`** & **`voc_camera_degraded`** | Dirty/fogged camera lens, low-light sensor ISO noise, and optical out-of-focus blur. | 1. **Gaussian Blur (`radius=2.5`)**: Blurs high-frequency edge gradients (simulates unfocused optics or smeared lens).<br>2. **Brightness Attenuation (`factor=0.6`)**: Diminishes illumination by 40% (simulates night/poor lighting).<br>3. **Gaussian Sensor Noise (`sigma=20.0`)**: Injects zero-mean Gaussian electronic noise across RGB channels (simulates cheap/underexposed sensors). | **Data Quality & Feature Drift (HIGH RISK)**. Causes Laplacian sharpness variance to drop and feature embeddings to migrate away from golden centroids. |
| **`cifar10_distribution_shifted`** | Seasonal/demographic shift in subject matter. | **Subject sampling re-weighting**: In the golden dataset, all 10 classes are evenly balanced (10% each). In the shifted dataset, we force **80% animals (dogs & cats) and only 10% vehicles (cars & planes)**. Images themselves remain clean. | **Prediction Drift (HIGH RISK)**. Model predictions tilt heavily toward animal classes; output entropy and class frequencies shift dramatically. |
| **`cifar10_unlabelled` & `voc_unlabelled`** | Client uploads raw imagery without spending money on human annotators. | The imagery is camera-degraded, but **all annotation files (`labels.json` or XML/JSON boxes) are completely omitted**. | **Performance = NOT AVAILABLE**. Proves the system does *not* hallucinate accuracy drops without labels, but flags Data Quality & Prediction Drift with an advisory. |
| **Training Quality Drift (`label_noise`)** | Sloppy crowdsourced data labelling or human annotator error during training. | We systematically invert ground-truth labels at **5%, 10%, and 20% random corruption rates** prior to model training. | **Training Performance Drift**. Directly demonstrates how dirty training data degrades model accuracy from 88.5% down to 61.3%. |

---

## 2. File-by-File Breakdown: How Each Component Works

| Directory / File | What it does & Why it exists |
| :--- | :--- |
| `config.yaml` | System configuration (model parameters, target classes, statistical drift thresholds, output paths). |
| `requirements.txt` | Minimal Python dependencies required to run the pipeline on any Linux/Mac server. |
| **`src/adapters/models/`** | **Model Abstraction Layer** |
| ├── `base.py` | Defines `BaseModelAdapter`. Allows connecting *any* model to the drift engine using standard methods: `predict()`, `extract_features()`, and `evaluate()`. |
| ├── `inception_adapter.py` | Implementation for **Image Classification** (Inception-v3). Extracts average-pooled feature vectors (2048-d), outputs predicted classes & entropy. |
| ├── `rf_detr_adapter.py` | Implementation for **Object Detection** (RF-DETR / Faster R-CNN architecture). Computes bounding box areas, detection counts, confidence scores, and mAP@50. |
| └── `__init__.py` | Model factory (`get_model_adapter(name)`). |
| **`src/adapters/datasets/`** | **Dataset Abstraction Layer** |
| ├── `base.py` | Defines `BaseDatasetAdapter` to load and validate images, format annotations, and handle metadata uniformly. |
| ├── `cifar10_adapter.py` | Loads Image Classification datasets (supports folder-per-class or flat image directories with `labels.json`). |
| ├── `pascal_voc_adapter.py`| Loads Object Detection datasets (supports PASCAL VOC format with XML/JSON bounding boxes). |
| └── `__init__.py` | Dataset factory (`get_dataset_adapter(name)`). |
| **`src/quality/`** | **Image Quality Analyzer** |
| └── `quality_analyzer.py` | Measures raw image characteristics before the model even sees them: sharpness (Laplacian variance), contrast, brightness, noise estimation, and dimensions. |
| **`src/drift/`** | **Statistical Engine & Assessment Execution** |
| ├── `drift_stats.py` | Core mathematical tests: Kolmogorov-Smirnov test (p-value), Population Stability Index (PSI), Wasserstein Distance, and multidimensional centroid shifts. |
| ├── `engine.py` | The main `DriftEngine`. Executes all 5 drift dimensions, runs Evidently AI report generation, and computes overall risk (`LOW`, `MEDIUM`, `HIGH`). |
| └── `runner.py` | Command-Line Interface (CLI) entrypoint for running ad-hoc assessments and generating reports. |
| **`src/utils/`** | **Utilities & Report Generation** |
| ├── `perturbations.py` | Generates controlled distortions: Gaussian blur, low-light sensor noise, camera degradation, and JPEG compression. |
| ├── `generate_visual_report.py`| Converts assessment JSON results into standalone visual HTML dashboards with Chart.js charts and executive summaries. |
| ├── `prepare_datasets.py` | Downloads CIFAR-10 and builds the reference golden, degraded, and shifted subsets. |
| └── `prepare_voc_dataset.py` | Downloads PASCAL VOC 2007 and builds reference golden and camera-degraded detection sets. |
| **`experiments/`** | **Experiments & Automated Verification** |
| ├── `validate_all.py` | Complete test suite validating all 5 handover acceptance scenarios. |
| └── `training_drift/run_training_drift.py` | Controlled training experiment proving degradation under 0%, 5%, 10%, and 20% bad labels. |

---

## 3. Metrics Comparison: Training vs. Inferencing & Detection vs. Classification

### A. Inference Drift vs. Training Drift

| Scenario | Objective | When it Runs | Key Metrics Evaluated | Ground Truth Required? |
| :--- | :--- | :--- | :--- | :--- |
| **Inference Drift Assessment** | Check whether new real-world data differs from baseline operating conditions. | Ad-hoc in production when new batch arrives. | **Image Quality**: Sharpness, Noise, Contrast.<br>**Input Drift**: KS p-value, PSI, Wasserstein.<br>**Feature Drift**: Latent centroid distance.<br>**Prediction Drift**: Confidence shift, entropy. | **No.** If unlabelled, reports data/prediction drift and risk advisory. If labelled, also measures mAP/Accuracy. |
| **Training Quality Drift** | Demonstrate how noisy training supervision degrades model performance. | During model training / fine-tuning experiments. | **Supervision Quality**: % Corrupted / inverted labels.<br>**Degradation**: Accuracy Drop, Precision Drop, Recall Drop, F1 degradation. | **Yes.** Evaluated against the clean golden test benchmark. |

---

### B. Object Detection vs. Image Classification Metrics

| Dimension | Image Classification (e.g. Inception-v3) | Object Detection (e.g. RF-DETR) |
| :--- | :--- | :--- |
| **Data Quality Drift** | *Identical*: Brightness, contrast, blur/sharpness, sensor noise, aspect ratio. | *Identical*: Brightness, contrast, blur/sharpness, sensor noise, aspect ratio. |
| **Feature Drift** | Backbone pooled vector (2048-d) centroid distance & norm drift. | ResNet/FPN backbone spatial feature map pooled vector centroid distance. |
| **Prediction Drift** | • Confidence score distribution<br>• Prediction entropy (uncertainty)<br>• Predicted class distribution shifts | • Detections count per image<br>• Average detection confidence<br>• Bounding-box area & aspect ratio distributions |
| **Performance Drift** *(When labelled)* | • **Accuracy**<br>• **Precision & Recall**<br>• **F1-Score** | • **mAP@50** (Mean Average Precision at IoU 0.5)<br>• **Detection Precision & Recall**<br>• **True/False Positives count** |

---

## 4. Setup Guide: Deploying on a Fresh Server

Follow these simple steps when cloning the repository on any Ubuntu/Debian/RHEL/Mac machine:

### Step 1: Clone & Create Environment
```bash
# Clone the repository
git clone <YOUR_GIT_REPO_URL>
cd model_drifting_pipeline

# Option A: Conda
conda create -n drift-env python=3.10 -y
conda activate drift-env

# Option B: Python Virtual Environment
# python3 -m venv venv && source venv/bin/activate
```

### Step 2: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Download & Prepare Benchmark Datasets
Run the automated preparation scripts (only needed once to populate reference benchmarks):
```bash
# Prepare CIFAR-10 classification reference and drift datasets
PYTHONPATH=. python src/utils/prepare_datasets.py

# Prepare PASCAL VOC detection reference and drift datasets
PYTHONPATH=. python src/utils/prepare_voc_dataset.py
```

### Step 4: Run Training Drift Experiment & View TensorBoard
You can evaluate training drift either via synthetic noise sweep (0%, 5%, 10%, 20%) or using your own custom clean vs. dirty training datasets:

```bash
# Option A: Synthetic noise corruption sweep (benchmarks 0%, 5%, 10%, 20% label noise)
python -m experiments.training_drift.run_training_drift \
  --model inception_v3 \
  --run_id sweep_run_01

# Option B: Using Custom Datasets (compares your clean golden training data vs dirty training data)
python -m experiments.training_drift.run_training_drift \
  --model rf_detr \
  --golden_dataset data/golden/voc_golden \
  --dirty_dataset data/new/voc_camera_degraded \
  --run_id custom_voc_train_01
```

#### Launching the TensorBoard Dashboard:
TensorBoard logs are saved per run in `results/training_drift/<run_id>/tensorboard_logs`. To view all runs:
```bash
tensorboard --logdir results/training_drift --port 6006
```
Open **[http://localhost:6006](http://localhost:6006)** in your browser.

> **How does TensorBoard display the two datasets (Golden vs. Dirty) in a single run?**
> TensorBoard groups metrics using `add_scalars`:
> - Under **`DataQuality/`**, each chart (e.g. `DataQuality/Sharpness`) plots **two side-by-side curves/bars**: a blue line for `Golden` and an orange line for `Dirty`.
> - Under **`Performance/`**, it plots `GoldenModel` vs `DirtyDataModel` together on the same graph, along with a scalar for `Performance/Degradation_Drop_pp`.
> - You can check/uncheck runs in the left sidebar to overlay and compare multiple experiment runs simultaneously.

Generate the standalone HTML report for the experiment:
```bash
python -m src.utils.generate_visual_report --training_drift
```

### Step 5: Run Inference Drift Assessment
Run an ad-hoc drift assessment on any model and specify a custom run folder name (`--assessment_id`):

**For Classification:**
```bash
python -m src.drift.runner \
  --model inception_v3 \
  --golden data/golden/cifar10_golden \
  --new data/new/cifar10_camera_degraded \
  --assessment_id client_cifar_eval_01
```

**For Object Detection:**
```bash
python -m src.drift.runner \
  --model rf_detr \
  --golden data/golden/voc_golden \
  --new data/new/voc_camera_degraded \
  --assessment_id client_voc_eval_01
```

### Step 6: View the Visual Reports
Every run automatically generates:
1. **Executive Visualization Report**:
   Located at: `results/assessments/<assessment_id>/executive_report.html`
   Double-click to open in Chrome/Safari/Firefox. Shows overall risk cards, image quality charts, and metrics.
2. **Evidently AI Interactive Report**:
   Located at: `results/assessments/<assessment_id>/evidently_drift_report.html`
   Interactive distributions, Kolmogorov-Smirnov cumulative graphs, and quantile views.

---

## 5. How to Use Your Own Custom Data

You do **not** need to modify the core pipeline code to evaluate your own datasets.

### Use Case 1: Custom Image Classification
Place your images in any folder using either structure:

**Structure A: Subdirectories by class name**
```
my_custom_dataset/
├── dog/
│   ├── img001.jpg
│   └── img002.jpg
└── cat/
    ├── img003.jpg
    └── img004.jpg
```

**Structure B: Flat image folder with `labels.json` (or unlabelled without `labels.json`)**
```
my_custom_dataset/
├── images/
│   ├── sample1.jpg
│   └── sample2.jpg
└── labels.json  (optional: {"sample1.jpg": "dog", "sample2.jpg": "cat"})
```

**Run Assessment:**
```bash
PYTHONPATH=. python src/drift/runner.py \
  --model inception_v3 \
  --golden path/to/my_golden_dataset \
  --new path/to/my_new_dataset
```

---

### Use Case 2: Custom Object Detection
Place your detection images and annotations in the standard layout:

```
my_detection_dataset/
├── images/
│   ├── frame_001.jpg
│   └── frame_002.jpg
└── annotations/
    ├── frame_001.json (or .xml)
    └── frame_002.json
```

**JSON Annotation Format:**
```json
{
  "boxes": [[xmin, ymin, xmax, ymax]],
  "labels": ["person"]
}
```
*(If the new client dataset is unlabelled, simply omit the `annotations/` folder. The system will evaluate Data Quality, Feature Drift, and Prediction Drift, reporting Performance as NOT AVAILABLE.)*

**Run Assessment:**
```bash
PYTHONPATH=. python src/drift/runner.py \
  --model rf_detr \
  --golden path/to/my_detection_golden \
  --new path/to/my_detection_new
```

---

## 6. Interpreting the Output Decision Logic

The report outputs an overall risk status:
- **`LOW RISK`**: The new dataset matches the baseline operating distribution. Safe for production inference.
- **`MEDIUM RISK`**: Slight distribution or quality change observed (e.g. minor lighting shift). Recommend spot-checking low confidence inferences.
- **`HIGH RISK`**: Significant blur, noise, embedding deviation, or accuracy drop detected. Do not auto-promote inferences. Audit camera hardware or collect domain data for fine-tuning.

