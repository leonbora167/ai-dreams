# Standalone Video Anomaly Evaluation Suite

This directory contains the self-contained evaluation suite for the two-stage surveillance anomaly detection pipeline using **strictly non-Chinese open-source models**:
- **Tier 1 (Fast-Pass Screening)**: Google SigLIP (`google/siglip-base-patch16-224`, 86M params, US)
- **Tier 2 (Multimodal Verification)**: Hugging Face SmolVLM2 (`HuggingFaceTB/SmolVLM2-2.2B-Instruct`, 2.2B params, US/France)

---

## 1. Directory Structure

```
eval/
├── config_eval.py             # Class taxonomy, models, and threshold configurations
├── prompts_eval.py            # Surveillance-tuned zero-shot captions and VLM prompts
├── models_eval.py             # Model loading and inference routines (SigLIP & SmolVLM2)
├── viz_eval.py                # Visual overlay engine with 3-state top-left status HUD
├── scripts/
│   ├── download_weights.py    # Downloads SigLIP & SmolVLM2 into 'weights/' for offline use
│   ├── download_and_sample.py # Ingestion tool: samples 100 clips & cleans up raw archives
│   └── run_evaluation.py      # Main benchmark script with verbose tqdm progress bar
├── data/
│   ├── manifest.csv           # Registry of evaluation video clips, labels, and metadata
│   ├── xd_violence/           # Sampled XD-Violence benchmark videos
│   └── ucf_crime/             # Sampled UCF-Crime benchmark videos
└── outputs/
    ├── videos/                # Generated surveillance videos with top-left status HUD
    ├── eval_results.csv       # Comprehensive per-video CSV results log
    ├── metrics_summary.md     # Classification report (Precision, Recall, F1 per class)
    └── confusion_matrix.png   # High-resolution confusion matrix plot
```

---

## 2. Target 7-Class Surveillance Taxonomy

1. **`Normal`**: Benign everyday crowd activity, walking, standing, sitting.
2. **`Violence`**: Broad aggression, riots, assaults, vandalism, civil unrest.
3. **`Fighting`**: Hand-to-hand physical brawls, punching, kicking, wrestling.
4. **`Arson`**: Deliberate fire-setting, burning objects, smoke plumes, flaming materials.
5. **`Car Accident`**: Vehicular collisions, traffic crashes, vehicle impact.
6. **`Explosion`**: Sudden violent blasts, detonations, fireball shockwaves.
7. **`Shooting`**: Firearms discharged, active armed assailants, gunfire.

---

## 3. Quick Start (Reproducing on Another Machine)

### Step 1: Install Dependencies
```bash
pip install torch torchvision transformers sentencepiece num2words opencv-python tqdm pandas matplotlib kagglehub huggingface_hub
```

### Step 2: Download Model Weights (Offline Support)
To download the Hugging Face model weights directly into `weights/` (useful when working on servers with restricted internet or SSL proxies):
```bash
python download_weights.py --models so400m smolvlm2
```
* The pipeline automatically checks `weights/` first (`weights/siglip-so400m-patch14-384`, `weights/SmolVLM2-2.2B-Instruct`).
* If found locally, it runs **100% offline with zero external network or SSL calls**.
* If not found, it automatically falls back to downloading directly from Hugging Face.

### Step 3: Download & Sample 100 Balanced Clips
To download and extract 100 balanced clips from XD-Violence and UCF-Crime (automatically pruning the remaining raw archive files to conserve disk space):
```bash
python eval/scripts/download_and_sample.py --dataset all --count 100
```

### Step 4: Run Full Evaluation with Verbose Progress Bar
```bash
python eval/scripts/run_evaluation.py --device cuda
```
For a quick test on the first 5 videos:
```bash
python eval/scripts/run_evaluation.py --device cuda --max_videos 5
```

---

## 4. Top-Left Video Status HUD States

Every generated video in `eval/outputs/videos/` displays an annotated status banner in the top-left corner:
- **Green Box**: `[STATUS: NORMAL] Stage 1 Gate Quiet | Motion & Density Normal`
- **Amber Box**: `[STATUS: ABNORMAL (STAGE 1)] Motion/Density Gate Triggered (Z=score) | Awaiting S2`
- **Red Box**: `[STATUS: ABNORMAL] Classifier: <CLASS> (<CONFIDENCE>%) | Alert Dispatched`
- **Slate Box**: `[STATUS: NORMAL] Suppressed by Classifier (Normal Crowd, <CONFIDENCE>%)`

