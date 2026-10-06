# AI Model Drift POC — Agent Handover README

## 1. Objective

Build a **Phase 1 Model Drift Assessment POC** for the existing Computer Vision platform.

The platform already supports:

- Model training
- Model inference
- Computer Vision models
- Pipeline construction
- Dataset management
- Model deployment

The purpose of this POC is to add a **Model Drift Assessment capability** that can be invoked **ad hoc** for an already-trained/deployed model.

### Primary use case

A client already has a model deployed in production.

The client provides:

1. The existing production model
2. A **golden/reference dataset**
3. A **new dataset** that they want to evaluate

The system should determine whether the new dataset is significantly different from the golden dataset and whether there is evidence of potential model-performance degradation.

This is **not primarily a continuous monitoring system**.

The primary workflow is:

```text
Existing Production Model
          +
Golden / Reference Dataset
          +
New Dataset
          ↓
    Drift Assessment
          ↓
 ┌───────────────────────┐
 │ Data / Input Drift    │
 │ Feature Drift         │
 │ Prediction Drift     │
 │ Data Quality Drift    │
 │ Performance Drift*   │
 └───────────────────────┘
          ↓
      Drift Report
          ↓
       Dashboard
```

`* Performance drift requires ground-truth labels in the new dataset.`

---

# 2. Important Terminology

Do not use "model drift" as a catch-all term.

The system must distinguish between:

### Input / Data Drift

The distribution of incoming/new data differs from the reference data.

Examples:

- Different lighting
- Different camera
- Different image quality
- Different environments
- Different object distribution
- Different image resolution
- Different feature distribution

### Data Quality Drift

The quality of the new data has changed.

Examples:

- Blur
- Noise
- Poor exposure
- Compression
- Corrupt images
- Incorrect dimensions
- Missing data

### Prediction Drift

The model produces a different prediction distribution on the new dataset.

Examples:

- Confidence distribution changes
- Class distribution changes
- Detection count changes
- Bounding-box distribution changes

### Performance Drift

The model's actual predictive performance has changed.

This requires ground-truth labels.

Example:

```text
Golden dataset
    RF-DETR
    mAP = 91%

New labelled dataset
    RF-DETR
    mAP = 74%

→ Actual performance degradation detected
```

### Important

If the new dataset is **unlabelled**, the system must NOT claim that model accuracy has decreased.

Instead it should report:

> Significant data/prediction drift detected. This indicates potential model-performance risk, but actual performance degradation cannot be confirmed without ground-truth labels.

---

# 3. Phase 1 Models

Use two Computer Vision models.

## Object Detection

Model:

**RF-DETR**

Dataset:

**PASCAL VOC 2007**

Use a manageable subset of classes if required.

Suggested classes:

```text
person
car
bicycle
bus
motorbike
dog
cat
```

The exact subset can be adjusted based on RF-DETR compatibility and available compute.

---

## Image Classification

Model:

**Inception-v3**

Dataset:

**CIFAR-10**

Use appropriate resizing/preprocessing for Inception-v3.

---

# 4. Why These Datasets

The datasets are primarily for building and validating the POC.

They allow us to create controlled experiments where the expected drift is known.

### Detection

PASCAL VOC 2007 provides:

- Images
- Bounding boxes
- Class labels
- A manageable dataset size

### Classification

CIFAR-10 provides:

- Small images
- 10 classes
- Fast training
- Easy controlled corruption and distribution-shift experiments

The goal is not to achieve state-of-the-art accuracy.

The goal is to validate the **drift assessment architecture**.

---

# 5. Primary User Workflow

The most important workflow is:

```text
1. Select existing model
        ↓
2. Select golden dataset
        ↓
3. Upload/select new dataset
        ↓
4. Run Drift Assessment
        ↓
5. Execute existing model on datasets
        ↓
6. Extract metrics/features
        ↓
7. Compare golden vs new
        ↓
8. Generate drift report
        ↓
9. Display results in dashboard
```

This must work without retraining the production model.

---

# 6. Golden Dataset

The golden dataset represents the expected/reference operating distribution of the model.

It should contain, where available:

```text
images
ground-truth labels
annotations
metadata
```

The system should calculate and store baseline/reference statistics from this dataset.

Examples:

### Classification

```text
Class distribution
Feature distribution
Embedding distribution
Confidence distribution
Image quality statistics
```

### Detection

```text
Class distribution
Objects/image
Confidence distribution
Bounding-box size
Bounding-box position
Image quality
Feature/embedding distribution
```

The golden dataset should become the reference against which future ad-hoc datasets are evaluated.

---

# 7. New Dataset

The new dataset represents data that a client wants to evaluate.

It may be:

### Case A — Unlabelled

```text
new_dataset/
    image_001.jpg
    image_002.jpg
    ...
```

The system can assess:

- Data drift
- Feature drift
- Data quality drift
- Prediction drift

But it cannot calculate true accuracy/mAP/F1.

---

### Case B — Labelled

```text
new_dataset/
    images/
    annotations/
```

The system can additionally calculate:

- Accuracy
- Precision
- Recall
- F1
- mAP
- Per-class performance
- Confusion matrix

This is the preferred scenario for confirming actual performance degradation.

---

# 8. Drift Experiment Types

The agent must implement controlled drift experiments to validate the system.

## 8.1 Image Quality Drift

Generate variants of the dataset with:

```text
Gaussian blur
Motion blur
Noise
Brightness changes
Contrast changes
JPEG compression
Resolution reduction
Color shifts
```

---

## 8.2 Camera Degradation Simulation

Simulate:

```text
Dirty/blurred lens
Low-resolution camera
Sensor noise
Poor lighting
Compression artifacts
```

The objective is to reproduce a realistic production scenario:

```text
Original deployment
        ↓
Camera degradation
        ↓
New dataset
        ↓
Drift assessment
```

---

# 9. Distribution Drift

Create datasets where the composition of the data changes.

Example:

```text
Golden dataset

person     40%
car        30%
dog        10%
bicycle     5%
...


New dataset

person     70%
car        10%
dog         2%
bicycle     1%
...
```

The system should identify these distribution changes.

---

# 10. Prediction Drift

Run the existing model against:

```text
Golden dataset
New dataset
```

Compare:

### Detection

- Detection count
- Confidence distribution
- Class distribution
- Bounding-box sizes
- Bounding-box locations
- Objects/image

### Classification

- Predicted class distribution
- Confidence distribution
- Prediction entropy
- Per-class prediction frequency

---

# 11. Feature / Embedding Drift

Where practical, extract intermediate model features or embeddings.

Compare:

```text
Golden feature distribution
             vs
New feature distribution
```

Possible metrics:

- PSI
- KL divergence
- Jensen-Shannon divergence
- Wasserstein distance
- Statistical distribution tests

Do not assume one metric is universally best.

The system should allow multiple metrics where appropriate.

---

# 12. Data Quality Assessment

For both datasets calculate:

```text
Image dimensions
Brightness
Contrast
Sharpness / blur
Noise
Compression indicators
Corrupt image count
Missing annotations
Invalid annotations
```

The dashboard should clearly show:

```text
Golden Dataset       New Dataset
      │                    │
      └────────┬───────────┘
               ↓
       Quality Comparison
```

---

# 13. Performance Drift

Only calculate performance drift when ground truth exists.

## Detection

Compare:

```text
mAP
Precision
Recall
Per-class AP
```

Example:

```text
Golden mAP: 91%
New mAP:    74%

Change: -17 percentage points
```

## Classification

Compare:

```text
Accuracy
Precision
Recall
F1
Per-class metrics
Confusion matrix
```

---

# 14. Controlled Training/Data-Quality Drift

Phase 1 should also demonstrate that **bad training data can result in model degradation**.

Create corrupted training datasets.

## Wrong Labels

Introduce:

```text
5%
10%
20%
```

incorrect labels.

## Missing Labels

Remove labels from configurable percentages of training data.

## Detection Annotation Errors

For RF-DETR:

```text
Shift bounding boxes
Shrink boxes
Expand boxes
Remove boxes
Assign incorrect classes
```

## Classification Crop Errors

Create:

```text
Incorrect crops
Background-only crops
Partial object crops
Wrong image-class associations
```

## Class Imbalance

Create increasingly imbalanced training datasets.

---

# 15. Training Drift Experiment

Run:

```text
Experiment 0
Clean training data

Experiment 1
5% bad labels

Experiment 2
10% bad labels

Experiment 3
20% bad labels

Experiment 4
Incorrect crops

Experiment 5
Missing labels

Experiment 6
Class imbalance
```

For each experiment:

1. Train model.
2. Evaluate against the same clean test/golden dataset.
3. Compare with the baseline model.
4. Record performance degradation.
5. Record the associated training-data quality metrics.

The objective is to demonstrate:

```text
Training Data Quality
        ↓
Model Training
        ↓
Model Behaviour
        ↓
Performance Change
```

---

# 16. Open-Source Drift Library

Use:

**Evidently**

as the primary drift-analysis library where applicable.

Potential supporting libraries:

```text
PyTorch
TorchVision
RF-DETR
OpenCV
NumPy
Pandas
scikit-learn
SciPy
```

Do not introduce unnecessary external services.

---

# 17. Platform Integration Concept

The existing platform already handles model and pipeline construction.

Therefore, the drift component should **not rebuild the training/inference platform**.

Instead, implement a reusable **Drift Assessment Module**.

Conceptually:

```text
Existing Platform
│
├── Model Training
├── Model Registry
├── Dataset Management
├── Pipeline Builder
├── Inference
│
└── Drift Assessment
       │
       ├── Reference Dataset
       ├── New Dataset
       ├── Model Inference
       ├── Feature Extraction
       ├── Data Quality
       ├── Drift Engine
       └── Report
```

The drift module should consume existing models and datasets through clean interfaces.

---

# 18. Model Adapter Architecture

Do not hard-code the drift engine specifically for RF-DETR.

Create a model adapter interface.

Conceptually:

```python
class ModelAdapter:

    def predict(dataset):
        ...

    def extract_features(dataset):
        ...

    def evaluate(dataset, ground_truth):
        ...
```

Then implement:

```text
RFDETRAdapter
InceptionAdapter
```

This allows future CV models to be integrated without rewriting the drift engine.

---

# 19. Dataset Adapter Architecture

Similarly:

```python
class DatasetAdapter:

    def load(dataset):
        ...

    def validate(dataset):
        ...

    def get_labels(dataset):
        ...

    def get_metadata(dataset):
        ...
```

Implement adapters for:

```text
PASCAL VOC
CIFAR-10
```

The production platform should eventually be able to pass its own dataset format through the same interface.

---

# 20. Drift Assessment API / Service

The implementation should expose a reusable function or service conceptually equivalent to:

```text
assess_drift(
    model,
    golden_dataset,
    new_dataset,
    options
)
```

Return structured results:

```json
{
  "overall_status": "warning",
  "data_drift": {},
  "feature_drift": {},
  "prediction_drift": {},
  "data_quality": {},
  "performance": {},
  "recommendation": {}
}
```

Do not hard-code this exact schema if the existing platform has a preferred schema, but preserve the same conceptual separation.

---

# 21. Drift Decision Logic

Do not simply return:

```text
DRIFT = TRUE
```

Provide a structured assessment.

Example:

```text
Overall Assessment
------------------
Status: HIGH RISK

Data Drift:           HIGH
Feature Drift:        HIGH
Prediction Drift:     MEDIUM
Data Quality:         HIGH
Performance Drift:    NOT AVAILABLE

Reason:
The new dataset has significantly different image-quality
and feature distributions compared with the golden dataset.

Performance degradation cannot be confirmed because the
new dataset contains no ground-truth labels.
```

If labelled:

```text
Performance Drift: HIGH

Golden mAP: 91%
New mAP:    74%

Performance change: -17 pp
```

---

# 22. Dashboard

Build a local dashboard initially using:

**Streamlit**

The dashboard must represent an **ad-hoc assessment**, not a continuous monitoring console.

---

## Dashboard Flow

### Step 1 — Select Model

```text
Model
[ RF-DETR v1.2 ]
```

### Step 2 — Select Golden Dataset

```text
Reference Dataset
[ Client Golden Dataset ]
```

### Step 3 — Select New Dataset

```text
Dataset to Assess
[ New Client Dataset ]
```

### Step 4 — Run Assessment

```text
[ Run Drift Assessment ]
```

### Step 5 — Results

Display:

```text
Overall Risk
────────────
LOW / MEDIUM / HIGH
```

Then:

```text
Data Drift
Feature Drift
Prediction Drift
Data Quality
Performance Drift
```

---

# 23. Dashboard Visualizations

## Dataset Comparison

Show:

```text
Golden Dataset       New Dataset
--------------------------------
Samples              Samples
Classes              Classes
Image resolution     Image resolution
Brightness           Brightness
Sharpness            Sharpness
```

---

## Distribution Comparison

Show baseline vs new:

- Class distributions
- Confidence distributions
- Feature distributions
- Detection counts
- Bounding-box distributions

---

## Image Quality

Show:

```text
Brightness
Contrast
Sharpness
Noise
Resolution
Compression
```

---

## Model Performance

Only show this section when labels are available.

Display:

```text
Golden Performance
vs
New Dataset Performance
```

with:

- mAP
- Precision
- Recall
- Accuracy
- F1
- Confusion matrix

---

## Visual Examples

Display representative examples:

```text
Golden Dataset

[image] [image] [image]


New Dataset

[image] [image] [image]
```

For detection, overlay predictions and bounding boxes.

---

# 24. Results Storage

Do not hardcode dashboard results.

Create:

```text
results/
├── assessments/
│   ├── <assessment_id>/
│   │   ├── summary.json
│   │   ├── drift_metrics.json
│   │   ├── quality_metrics.json
│   │   ├── prediction_metrics.json
│   │   ├── performance_metrics.json
│   │   └── visualizations/
│
├── training_drift/
│
└── baseline/
```

Every assessment should have a unique ID.

Example:

```text
assessment_20261005_143500
```

---

# 25. Required Project Structure

```text
model-drift-poc/
│
├── README.md
├── requirements.txt
├── config.yaml
│
├── data/
│
├── models/
│
├── src/
│   ├── adapters/
│   │   ├── models/
│   │   └── datasets/
│   │
│   ├── drift/
│   │
│   ├── quality/
│   │
│   ├── evaluation/
│   │
│   ├── training/
│   │
│   └── utils/
│
├── experiments/
│   ├── inference_drift/
│   └── training_drift/
│
├── results/
│
└── dashboard/
    └── app.py
```

---

# 26. Agent Execution Plan

The agent reading this README should execute the following.

## Step 1

Inspect the existing environment and platform repository.

Do not overwrite existing platform functionality.

Identify:

- Python version
- Existing ML environment
- Existing model interfaces
- Existing dataset interfaces
- Existing dashboard technology
- Existing dependency versions

Reuse existing components wherever possible.

---

## Step 2

Install only missing dependencies.

Do not unnecessarily create duplicate environments or reinstall existing libraries.

---

## Step 3

Download and prepare:

```text
PASCAL VOC 2007
CIFAR-10
```

---

## Step 4

Train baseline models:

```text
RF-DETR
Inception-v3
```

---

## Step 5

Build model adapters.

---

## Step 6

Build dataset adapters.

---

## Step 7

Implement the Drift Assessment Engine.

---

## Step 8

Implement controlled inference-drift experiments.

---

## Step 9

Implement controlled training/data-quality drift experiments.

---

## Step 10

Implement the golden-vs-new-dataset assessment workflow.

This is the **highest-priority feature**.

---

## Step 11

Persist assessment results.

---

## Step 12

Integrate the results into the dashboard.

---

## Step 13

Run end-to-end validation.

Test at minimum:

```text
Test 1:
Golden dataset vs identical copy
→ No significant drift

Test 2:
Golden dataset vs blurred dataset
→ Data-quality/input drift

Test 3:
Golden dataset vs distribution-shifted dataset
→ Distribution drift

Test 4:
Golden dataset vs heavily corrupted dataset
→ Significant drift

Test 5:
Labelled new dataset with actual performance degradation
→ Performance drift
```

---

# 27. Acceptance Criteria

The POC is complete only when:

### Environment

- [ ] Dependencies install successfully
- [ ] Existing platform dependencies are reused where possible

### Models

- [ ] RF-DETR works
- [ ] Inception-v3 works
- [ ] Model adapters are implemented

### Datasets

- [ ] PASCAL VOC 2007 works
- [ ] CIFAR-10 works
- [ ] Dataset validation works
- [ ] Golden datasets can be registered

### Drift

- [ ] Data drift works
- [ ] Feature drift works where supported
- [ ] Prediction drift works
- [ ] Data-quality drift works
- [ ] Performance drift works when labels exist

### Ad-hoc Assessment

- [ ] Existing model can be selected
- [ ] Golden dataset can be selected
- [ ] New dataset can be supplied
- [ ] Assessment can be executed without retraining
- [ ] Results are generated
- [ ] Results are persisted
- [ ] Results can be displayed in dashboard

### Dashboard

- [ ] Golden vs new dataset comparison
- [ ] Drift summary
- [ ] Data quality comparison
- [ ] Prediction comparison
- [ ] Performance comparison when labels exist
- [ ] Visual examples
- [ ] No hardcoded metrics

---

# 28. Definition of Done

The most important demonstration should be:

```text
Existing RF-DETR Model
        +
Golden Dataset
        +
New Client Dataset
        ↓
     RUN TEST
        ↓
┌──────────────────────────┐
│ Drift Assessment         │
│                          │
│ Data Drift       HIGH    │
│ Feature Drift    HIGH    │
│ Prediction Drift MEDIUM  │
│ Data Quality     HIGH    │
│ Performance      N/A     │
└──────────────────────────┘
        ↓
   Dashboard Report
```

And when the new dataset contains ground truth:

```text
Existing RF-DETR Model
        +
Golden Dataset
        +
New Labelled Dataset
        ↓
     RUN TEST
        ↓
Golden mAP: 91%
New mAP:    74%
        ↓
Performance Drift: HIGH
        ↓
   Dashboard Report
```

The POC should prove that an already-deployed production model can be **evaluated ad hoc against newly supplied client data using a golden/reference dataset**, without requiring continuous monitoring or retraining.

That is the primary product capability this Phase 1 implementation must validate.