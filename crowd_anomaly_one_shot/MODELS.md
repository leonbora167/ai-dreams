# Model registry and recommendations

Status values: UNVERIFIED (default) | VERIFIED-BY-USER | REJECTED.
You (the agent) may NOT use a model until its status is VERIFIED-BY-USER for the current phase.

Names, repos and APIs change quickly, especially for VLMs. Treat every "likely source" as a lead to check, not a fact. Before proposing a VLM, search for what is current and loadable, and tell the user what you found, with sizes and licences.

## How to verify a model (per model, in the phase that needs it)

1. Tell the user which model you intend to try and where you believe it comes from (package, repo, weights URL).
2. Confirm network access, or ask the user to place weights in `weights/`.
3. State size and licence from the model card (if unsure, say "unsure, please check").
4. Load it and run a dummy forward pass. Report: expected input shape, output shape, parameter count, memory, and latency on the user's hardware (batch 1, warm-up excluded).
5. Run it on one real sample and save a visual result.
6. Add an entry to the "Verification log" at the bottom. The USER sets the status.

## Training legend

- DIRECT: pretrained weights work as-is, no training.
- FINETUNE: pretrained weights exist but you must fine-tune on the user's data.
- FIT: no labels needed, but must be fitted or calibrated per camera on normal footage.
- RULE: classical code or logic, nothing to train.

## Recommended starting choices (propose these first, the user decides)

| Step | First choice | Next to try | Why |
|---|---|---|---|
| Flow | OpenCV DIS (RULE) | FastFlowNet or RAFT-small (DIRECT) | DIS is cheap and CPU-only; learned flow only if DIS is too noisy |
| Density | Pretrained counting model (P2PNet or DM-Count), sanity-checked against manual counts | Head detector if heads are large; grid-flow only if heads are tiny | choice depends on measured head size |
| Baseline | Robust per-cell z-score at two grid scales (FIT) | GMM / Isolation Forest | simplest thing that can work |
| Fire/smoke | YOLOv8n or YOLO11n fine-tuned on D-Fire plus own hard negatives | YOLOv8s | small and fast |
| Classifier A | CLIP-style model on frames (SigLIP or CLIP via `open_clip`) | n/a | cheapest and edge-friendly; run first as the floor |
| Classifier B | SmolVLM2 video-instruct checkpoint (256M / 500M / 2.2B) | Qwen3-VL small variant, InternVL small variant | edge-sized VLMs with video support |
| Classifier C | Server or cloud VLM (only if the user allows) | n/a | strongest, flagged clips only |

## Candidates per step

### Phase 1A: motion estimation
| Candidate | Training | Likely source (verify) | Notes |
|---|---|---|---|
| Frame differencing / block matching | RULE | numpy / OpenCV | cheapest baseline, no direction |
| OpenCV DIS optical flow | RULE | OpenCV (`cv2.DISOpticalFlow_create`) | CPU, good default |
| Hardware optical flow engine | RULE | Jetson / NVIDIA optical flow SDK | only on supported hardware |
| FastFlowNet | DIRECT | github ltkong218/FastFlowNet | light GPU flow, check weights availability |
| RAFT-small | DIRECT | torchvision `torchvision.models.optical_flow.raft_small` | heavier, quality reference |

### Phase 1B: density / crowd estimation
| Candidate | Training | Likely source (verify) | Notes |
|---|---|---|---|
| P2PNet | DIRECT then maybe FINETUNE | github TencentYoutuResearch/CrowdCounting-P2PNet | point-based counting with locations |
| DM-Count | DIRECT then maybe FINETUNE | github cvlab-stonybrook/DM-Count | density map |
| CSRNet | DIRECT then maybe FINETUNE | various PyTorch ports (check) | light density map |
| Head detector (YOLO nano/small) | FINETUNE | `ultralytics` + head datasets | only if heads are large enough |
| Grid-flow only | RULE | n/a | fallback for tiny heads |

Pretrained counting weights usually come from public counting benchmarks; expect a domain gap to stadium CCTV and check against manual counts.

### Phase 1C and 1E: baseline and second gate
| Candidate | Training | Notes |
|---|---|---|
| Robust z-score per cell, two grid scales | FIT | start here |
| GMM / Isolation Forest (`scikit-learn`) | FIT | on flow and density features |
| Small autoencoder on flow patches | FIT | only if the simple ones fail |
| Speed-category bins (VelocityNet-style, arXiv 2510.18187) | FIT | calibrate per camera instead of using global thresholds |
| Small CLIP-style image encoder for the appearance check | DIRECT embedding, FIT normal set | optional; must earn its place in the with/without test |

### Phase 1F: fire and smoke
| Candidate | Training | Likely source (verify) | Notes |
|---|---|---|---|
| YOLOv8n / YOLOv8s / YOLO11n | FINETUNE | `ultralytics` + D-Fire (see DATASETS.md) | check dataset licence |
| Colour / flicker pre-filter | RULE | OpenCV | optional false-alarm reducer |

### Phase 2C: classifier candidates (no video network is trained)
| Candidate | Training | Likely source (verify) | Notes |
|---|---|---|---|
| CLIP-style frame classifier (CLIP, SigLIP family) | DIRECT | `open_clip` or Hugging Face `transformers`; check which variants are current | text prompts per class, aggregate over frames; sees frames more than motion |
| SmolVLM2 video-instruct (256M, 500M, 2.2B checkpoints) | DIRECT | Hugging Face (HuggingFaceTB) | designed for low memory and on-device use; a search this session found the SmolVLM paper reporting under 1 GB GPU RAM for the smallest model and competitive video-benchmark scores; verify the exact checkpoint names and sizes |
| Qwen3-VL (small variants) | DIRECT | Hugging Face (Qwen) | a 2026 article calls Qwen-VL the strongest all-round open family and reports Apache 2.0 licensing for Qwen3-VL; verify licence and which small sizes exist on the model card |
| InternVL (small variants) | DIRECT | Hugging Face (OpenGVLab) | described as the closest open rival to Qwen-VL; verify licence and sizes |
| Server or cloud VLM | DIRECT | provider chosen by the user | only if clips may leave the site; record latency and cost per call |
| Frozen CLIP features + light trained head | FINETUNE (head only) | reference implementation github nwpu-zxr/vadclip (confirm) | only if the decision memo calls for it; needs a small labelled set |
| LAVAD-style captioning + LLM scoring | DIRECT | paper arXiv 2404.01014 (CVPR 2024) | reference point only; captions many frames, probably too heavy for the edge |

Implementation rule for every VLM: present the labels as lettered options and read the probabilities of the option letters (or compare label log-likelihoods), then normalise. Do not parse free text.

Related papers worth skimming before prompt design (found via search this session; check each yourself): LAVAD (arXiv 2404.01014); a reality check on zero-shot multimodal LLMs for surveillance (arXiv 2603.04727), which reports a strong "normal" bias and low recall, improved by naming the classes in the prompt; benchmarking compact VLMs for surveillance anomaly detection (arXiv 2603.13306); a cascading architecture that uses VLMs selectively (arXiv 2601.06204). None evaluates stadium crowds, so measure on the user's own footage.

## Verification log

### OpenCV DIS Optical Flow (Phase 1A)
- Date: 2026-10-15
- Source / weights URL: OpenCV built-in (`cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)`)
- Licence: Apache 2.0
- Params / memory / input shape / output shape: 0 parameters (algorithmic), input: 2x (H, W) uint8 grayscale, output: (H, W, 2) float32
- Latency on user hardware (batch 1, 960x720, host CPU): 28.75 - 54.85 ms/frame (~18 - 35 FPS on CPU)
- Sample result saved at: `outputs/phase1a/normal_pets2009_dis_speed_overlay.mp4`, `outputs/phase1a/normal_pets2009_dis_arrows.png`
- Status: PROPOSED-DEFAULT (recommended for Phase 1C)

### Torchvision RAFT-small (Phase 1A)
- Date: 2026-10-15
- Source / weights URL: `torchvision.models.optical_flow.raft_small(weights=Raft_Small_Weights.DEFAULT)` (~21 MB)
- Licence: BSD 3-Clause
- Params / memory / input shape / output shape: 990,162 parameters, input: 2x (1, 3, H, W) float32, output: (1, 2, H, W) float32
- Latency on user hardware (batch 1, 960x720, RTX 4060 GPU): 109.65 - 137.82 ms/frame (~7.3 - 9.1 FPS on GPU)
- Sample result saved at: `outputs/phase1a/normal_pets2009_learned_speed_overlay.mp4`, `outputs/phase1a/normal_pets2009_compare.png`
- Status: UNVERIFIED (tested as learned comparison reference)

### FastFlowNet (Phase 1A)
- Date: 2026-10-15
- Source / weights URL: `github.com/ltkong218/FastFlowNet` checkpoint `weights/fastflownet_ft_mix.pth` (5.48 MB)
- Licence: Apache 2.0 / Academic Research
- Params / memory / input shape / output shape: 1,368,098 parameters (~5.5 MB weights), input: (1, 6, H_pad, W_pad) float32, output: (H, W, 2) float32
- Latency on user hardware (batch 1, 960x720, RTX 4060 GPU): 64.47 - 84.32 ms/frame (~12 - 15.5 FPS on GPU)
- Sample result saved at: `outputs/phase1a/normal_pets2009_fastflownet_speed_overlay.mp4`, `outputs/phase1a/normal_pets2009_fastflownet_arrows.png`
- Status: AVAILABLE-IN-CONFIG (can be selected via `MOTION_METHOD = "fastflownet"`)

### CSRNet (Phase 1B)
- Date: 2026-10-15
- Source / weights URL: Hugging Face `muasifk/CSRNet` checkpoint `weights/csrnet_shanghaitech.pth` (65.06 MB)
- Licence: Research / Academic (ShanghaiTech trained)
- Params / memory / input shape / output shape: 16,262,465 parameters (~65 MB weights), input: (1, 3, H, W) float32, output: (1, 1, H/8, W/8) continuous density map
- Latency on user hardware (batch 1, 960x720, RTX 4060 GPU): 59.41 - 92.92 ms/frame (~11 - 17 FPS on GPU)
- Sample result saved at: `outputs/phase1b/density_overlay_contact_sheet.png`, `outputs/phase1b/grid_counts_overlay.png`, `outputs/phase1b/count_vs_manual.png`
- Status: VERIFIED-BY-USER (approved approach for Phase 1B)

### MobileNetV3-Small (Phase 1E Second Gate)
- Date: 2026-10-15
- Source / weights URL: `torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)` (9.79 MB)
- Licence: BSD 3-Clause
- Params / memory / input shape / output shape: 933,568 parameters, input: (1, 3, 224, 224) float32, output: (576,) L2-normalized embedding
- Latency on user hardware (batch 1, RTX 4060 GPU): 3.2 ms/crop
- Sample result saved at: `outputs/phase1e/appearance_model_comparison.png`
- Status: VERIFIED-DEFAULT (preferred for low-level structural anomaly detection, 4x wider novelty margin)

### CLIP ViT-B/32 (Phase 1E Second Gate Alternative)
- Date: 2026-10-15
- Source / weights URL: Hugging Face `openai/clip-vit-base-patch32` (351 MB)
- Licence: MIT
- Params / memory / input shape / output shape: 87,849,216 vision parameters, input: (1, 3, 224, 224) float32, output: (512,) L2-normalized embedding
- Latency on user hardware (batch 1, RTX 4060 GPU): 8.8 ms/crop
- Sample result saved at: `outputs/phase1e/appearance_model_comparison.png`
- Status: AVAILABLE-IN-CONFIG (can be selected via `APPEARANCE_MODEL = "clip"`)

### YOLOv8n Fire and Smoke Detector (Phase 1F)
- Date: 2026-10-15
- Source / weights URL: Pretrained YOLOv8n checkpoint `weights/yolov8n_fire_smoke.pt` (5.94 MB, classes: `{0: 'smoke', 1: 'fire'}`)
- Licence: AGPL-3.0 (Ultralytics) / CC-BY-4.0 (D-Fire weights)
- Params / memory / input shape / output shape: 3,011,238 parameters, 8.2 GFLOPs, input: (1, 3, 720, 960) float32, output: detected bounding boxes [x1, y1, x2, y2], conf, cls
- Latency on user hardware (batch 1, 960x720, RTX 4060 GPU): 9.57 ms/frame (~104.4 FPS on GPU)
- Sample result saved at: `outputs/phase1f/fire_samples_contact_sheet.png`, `outputs/phase1f/false_positive_hunt.png`, `outputs/phase1f/confidence_histogram.png`
- Status: VERIFIED-PROPOSED (ready for review in Phase 1F)



