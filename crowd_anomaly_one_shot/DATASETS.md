# Datasets guide: what to test on

No data is included. This file lists open datasets the user can test each part of the pipeline on, how to get them, and what to watch out for.

## Rules for the agent

1. Never download a dataset without telling the user: source URL, expected size, terms, and storage location. Wait for approval.
2. Some datasets need a request form or registration. Do NOT work around that. Tell the user what is needed; the user obtains access and places files in `data/datasets/<name>/`.
3. Each entry has a "Checked" field:
   - **CHECKED (this session)**: details were confirmed from the dataset's page or a paper at the time of writing. Re-confirm access and terms before downloading; pages move.
   - **NOT CHECKED**: written from general knowledge. Treat every detail (size, access, label format, terms) as a lead and verify with the user.
4. Terms and licences are deliberately NOT stated, because they were not verified. Many of these datasets were scraped from YouTube and released for research. If the goal is commercial deployment, the user must check the terms (and may want legal advice) early.
5. Before using any dataset, open 30-50 random samples with the viz helpers and report label noise, resolution, duplicates, overlays or watermarks, and how far the footage is from stadium CCTV.
6. Split by source video, never by clip, so near-duplicates do not leak between splits.
7. Record each dataset you actually obtain in the Verification log at the bottom.

## Important honesty note

No public dataset is stadium CCTV. Results on public data are sanity checks and debugging aids, not proof the system works in the stadium. The user's own footage (normal footage per camera, plus any real events) is the data that matters, and Phase 2B has the user label a small evaluation set from their own cameras.

## Local layout

```
data/datasets/<name>/
data/normal/<camera_name>/       # user's own normal footage
data/labelled_events/            # user's labelled events (Phase 2B)
```

---

## A. Test Stage 1 (flow, density, baseline, gate)

| Dataset | What to test with it | Checked | Notes |
|---|---|---|---|
| Violent-Flows (crowd violence) | Does the gate score differ between violent and non-violent crowd clips? Good crowd-specific test for the motion statistics | CHECKED (this session) | 246 real-world crowd videos from YouTube, half violent and half non-violent; clips are short (about 1 to 6.5 s, 3.6 s on average), so it cannot be used to fit a baseline, only to compare scores. Benchmark protocols (5-fold) are provided for violent/non-violent classification and violence outbreak detection. The authors' page asks you to fill a form to get FTP credentials, and requests citation of the Violent Flows paper |
| UMN crowd (escape / dispersal) | Sudden-running and dispersal events: does the speed-based score jump? | PARTIALLY (a crowd-analysis survey lists it as publicly available with normal and abnormal crowd videos) | Old and low-resolution; useful as a quick sanity check for motion-based scoring. Confirm access and format |
| UCF-Crime | False alarms per hour on long normal CCTV footage; broad anomaly smoke test | PARTIALLY (survey tables list 1,900 untrimmed CCTV videos, about 128 hours) | Weak video-level labels, very mixed scenes and quality. Temporal annotations for test videos may exist; confirm |
| ShanghaiTech Campus (anomaly) | Scoring sanity checks | NOT CHECKED | Campus scenes with a fixed camera each; useful for per-camera baseline logic |
| CHAD | Anomaly scoring, high-resolution scenes | NOT CHECKED | Named as a benchmark in a zero-shot surveillance paper; check what it contains before using |
| UBnormal | Scoring sanity checks | NOT CHECKED | Synthetic scenes; large domain gap |

For density (Phase 1B), counting benchmarks are the usual test sets and sources of pretrained weights (all NOT CHECKED): ShanghaiTech Part A/B, UCF-QNRF, NWPU-Crowd, JHU-Crowd++. Match them to the user's measured head size, and verify counts on the user's own frames against manual counts.

Do not tune gate thresholds on these datasets and assume they transfer to the stadium.

---

## B. Test Stage 2 (CLIP / VLM classifier)

| Dataset | What to test with it | Checked | Notes |
|---|---|---|---|
| Violent-Flows | Zero-shot violent vs non-violent in crowds, the closest public match to the use case | CHECKED (this session) | See above. Small, so treat results as indicative |
| RWF-2000 | Fight vs non-fight on surveillance-style clips | CHECKED (this session) | 2,000 clips extracted from surveillance videos, described as freely released for research, with a GitHub repo (`mchengny/RWF2000-Video-Database-for-Violence-Detection`, confirm URL), 80/20 train/test split by the authors, video-level labels, about 5 s clips. Its "violence" definition is broader than fighting (it lists robbery, explosion, shooting, blood and assault among others). Use the test part for zero-shot evaluation |
| NTU CCTV-Fights | Fights in real CCTV with temporal annotations | CHECKED (this session) | 1,000 videos: 280 CCTV (about 8.5 hours) and 720 non-CCTV; about 7.2 GB; long videos (5 s to 12 min) with annotated fight segments, so clips must be cut from the annotations. NTU states it is available on request (ROSE Lab dataset page) and requires a specific acknowledgment in publications. Ask the user to request access early |
| Real Life Violence Situations (RLVS) | Extra violence / non-violence clips | CHECKED (this session, partially) | 1,000 violence and 1,000 non-violence YouTube videos; non-violence is mostly everyday actions and sports, so results can reflect shortcuts like "sport vs street fight"; authors request citation; usual host is Kaggle (confirm) |
| XD-Violence | Violence types in varied footage | PARTIALLY (a paper's comparison table lists 4,754 videos, about 217 hours, from movies, sports, games, hand-held cameras and CCTV, 6 violence types) | Many clips are not CCTV; check access and label format |
| UCF-Crime | Mixed crime classes | PARTIALLY | Weak labels, low resolution; use only after inspecting samples |
| Hockey Fight | Sports-broadcast false positives (normal play vs fighting) | PARTIALLY (named in a violence-dataset review as 1,000 NHL clips, 500 violent and 500 not) | Broadcast video, not CCTV; useful mainly to see whether the classifier confuses normal sports contact with fights |

Class mapping suggestions (the user decides in Phase 2B):
- `fight` from RWF-2000 violent, CCTV-Fights fight segments, Violent-Flows violent, RLVS violence
- `none` from the non-violent counterparts and, above all, the user's own normal footage cut into clips
- `running_surge`, `fire_smoke`: almost no public clip-level data for surges. Ask the user for footage. UMN may help for dispersal. If no data exists, handle surges through the Stage 1 gate alone and say so.

Pitfalls to report:
- Non-violence classes can teach shortcuts (indoor vs outdoor, camera type).
- Re-uploads and shared sources create near-duplicates across splits.
- Clip length and fps differ across datasets; resample consistently to the pipeline's analysis fps and clip length.
- Public clips are fine for debugging prompts, but prompts must be frozen on a DEV set and judged on a TEST set made from the user's own events where possible.

---

## C. Test fire and smoke (Phase 1F)

| Dataset | Use | Checked | Notes |
|---|---|---|---|
| D-Fire | Fine-tune and evaluate the YOLO fire/smoke detector | CHECKED (this session) | Over 21,000 images; YOLO-format annotations (normalised coordinates); classes fire and smoke; pre-split train/val/test and some surveillance videos; authors ask for citation of their papers; repository `gaiasd/DFireDataset` on GitHub (confirm URL). Includes fire-like non-fire images, which helps with false positives |
| Wildfire smoke sets (e.g. on Kaggle) | Extra smoke examples | NOT CHECKED | Mostly forest scenes, not stadiums |
| Hard negatives from the user's cameras | Floodlights, flags, red/orange clothing, flares, sun glare, steam | n/a | Must come from the user. Decide with the user whether flares and pyrotechnics count as alarms |

---

## D. Normal footage for fitting baselines

Baselines must be fitted on the deployment camera's own normal footage (`data/normal/<camera>/`). No public dataset substitutes for this. Public long normal videos (for example UCF-Crime normal videos) are fine for testing the fitting code and for false-alarm-rate sanity checks only.

---

## Recommended order of acquisition

1. The user's own footage first (normal, plus any events).
2. RWF-2000 (open, simple labels): debug the classifier pipeline and prompts.
3. Violent-Flows (form needed, small): the crowd-specific test for both the gate and the classifier.
4. D-Fire when starting Phase 1F.
5. NTU CCTV-Fights (request needed): ask the user to request access early because approval can take time.
6. Counting datasets only if Phase 1B shows pretrained counting weights are inadequate.
7. Everything else only on request.

## Verification log

(Append one entry per dataset when it is actually obtained.)
```
### <dataset name>
- Date obtained:
- Source URL / how obtained (open download, request, purchase):
- Size on disk:
- Terms as stated by the source (copy the wording or link; if unclear write "unclear"):
- Number of videos/images, classes, label format:
- Resolution / fps / clip length observed:
- Sample inspection result (30-50 random samples):
- Used for: (phase, class mapping)
- Split strategy:
```
