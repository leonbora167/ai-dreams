# Model registry

Status values: UNVERIFIED (default) | VERIFIED-BY-USER | REJECTED.
You (the agent) may NOT use a model until its status is VERIFIED-BY-USER for the current phase.

This list comes from earlier design discussion. Names, repos and APIs change. Treat every "likely source" as a lead to check, not a fact.

## How to verify a model (do this per model, in the phase that needs it)

1. Tell the user which model you intend to try and where you believe it comes from (package, repo, weights URL).
2. Confirm network access / ask the user to place weights in `weights/` if you cannot download.
3. State size and licence of the weights (check the repo; if unsure, say "unsure, please check").
4. Load it and run a dummy forward pass. Report: input shape expected, output shape, parameter count, memory used, latency on the user's hardware.
5. Run it on one real sample and save a visual result.
6. Add an entry to the "Verification log" at the bottom of this file. The USER changes the status to VERIFIED-BY-USER.

## Training legend

- DIRECT: pretrained weights work as-is for our use, no training.
- FINETUNE: pretrained weights exist but you must fine-tune on our data.
- FIT: no labels needed, but must be fitted/calibrated per camera on normal footage.
- RULE: classical code or logic, nothing to train.

## Candidates per stage

### Phase 1A: motion estimation
| Candidate | Training | Likely source (verify) | Notes |
|---|---|---|---|
| Frame differencing / block matching | RULE | numpy / OpenCV | cheapest baseline |
| OpenCV DIS optical flow | RULE | OpenCV (`cv2.DISOpticalFlow_create`) | CPU, good default first try |
| Hardware optical flow engine | RULE | Jetson / NVIDIA optical flow SDK | only on supported hardware |
| FastFlowNet | DIRECT | github ltkong218/FastFlowNet | light GPU flow, check weights availability |
| RAFT-small | DIRECT | torchvision `torchvision.models.optical_flow.raft_small` | heavier, good quality reference |

### Phase 1B: density / people localisation
| Candidate | Training | Likely source (verify) | Notes |
|---|---|---|---|
| Head detector (YOLO nano/small) | FINETUNE | `ultralytics` + head datasets (SCUT-HEAD, CrowdHuman head boxes, JHU-Crowd) | only if heads are large enough |
| P2PNet | FINETUNE | github TencentYoutuResearch/CrowdCounting-P2PNet | point-based counting, gives locations |
| DM-Count | FINETUNE | github cvlab-stonybrook/DM-Count | density map |
| CSRNet | FINETUNE | various PyTorch ports (check) | light density map |
| Grid-flow only (no density model) | RULE | n/a | use if heads are too tiny; density proxy from motion/texture |

Pretrained crowd counting weights exist (ShanghaiTech, UCF-QNRF, NWPU) but domain gap to stadium CCTV is likely. Plan to fine-tune or at least sanity-check against manual counts.

### Phase 1C: per-camera baseline
| Candidate | Training | Notes |
|---|---|---|
| Percentile / z-score per grid cell | FIT | start here |
| GMM or Isolation Forest on grid stats | FIT | `scikit-learn` |
| Small autoencoder on flow patches | FIT | only if the simple ones fail |
| VelocityNet-style speed bins | FIT | calibrate per camera; paper: arXiv 2510.18187 |

### Phase 1E: fire and smoke
| Candidate | Training | Likely source (verify) | Notes |
|---|---|---|---|
| YOLOv8n / YOLOv8s / YOLO11n | FINETUNE | `ultralytics` + D-Fire or similar | check dataset licence |
| Colour / flicker pre-filter | RULE | OpenCV | optional false-alarm reducer |

### Phase 2B: compact action model
| Candidate | Training | Likely source (verify) | Notes |
|---|---|---|---|
| X3D-XS / X3D-S | FINETUNE | PyTorchVideo (torch hub) | Kinetics-pretrained; good first choice |
| MoViNet-A0 / A1 | FINETUNE | third-party PyTorch ports or official TF | check which port is maintained |
| TSM + MobileNetV2 / ResNet-18 | FINETUNE | github mit-han-lab/temporal-shift-module | very cheap |
| MobileNetV3 / EfficientNet-Lite + GRU/TCN | FINETUNE (head trained, backbone pretrained) | torchvision / timm | simplest to understand and modify |
| RTMPose-tiny + ST-GCN | FINETUNE (GCN trained; pose net DIRECT) | rtmlib / mmpose; mmaction2 or pyskl | only if people are large in the crop |
| torchvision video models (r2plus1d_18, s3d, ...) | FINETUNE | torchvision.models.video | easy fallback baseline, heavier |

Start with ONE model (suggested: X3D-S, or the torchvision fallback if install is painful). Compare a second one only after the first is evaluated.

Datasets for action fine-tuning (check access and licence): RWF-2000, NTU CCTV-Fights, XD-Violence, UBI-Fights, plus the user's own site footage. UCF-Crime has weak video-level labels; use cautiously.

### Optional, after Phase 4 and only if asked: server-side verifier
| Candidate | Training | Notes |
|---|---|---|
| VLM (zero-shot) | DIRECT | server only, never on edge |
| VideoMAE-B | FINETUNE | server only (Hugging Face transformers) |

## Verification log

(Append entries here. Format:)
```
### <model name>  (phase <id>)
- Date:
- Source / weights URL:
- Licence:
- Params / memory / input shape / output shape:
- Latency on user hardware (batch 1):
- Sample result saved at:
- Status: UNVERIFIED  ->  user sets to VERIFIED-BY-USER or REJECTED
```
