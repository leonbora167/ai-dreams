# Datasets guide

This file tells the agent which datasets are candidates for each stage, what each is for, how to obtain it, and what to check first. It contains NO data. Nothing is downloaded until the user approves.

## Rules for the agent

1. Never download a dataset without telling the user: source URL, expected size, licence/terms, and where it will be stored. Wait for approval.
2. Some datasets require a request form or registration. In that case do NOT try to work around it. Tell the user what is needed; the user obtains access and places the files in `data/datasets/<name>/`.
3. Every entry below has a "Checked" field:
   - **CHECKED (this session)**: details confirmed from the dataset's own page or paper at the time this file was written. Re-confirm access and terms yourself before downloading; pages move.
   - **NOT CHECKED**: written from general knowledge only. Treat every detail (size, access, licence, label format) as a lead. Verify with the user before using.
4. Licences: I have deliberately NOT stated licences, because I did not verify them. Many of these datasets were scraped from YouTube and are released for research. If the end goal is a commercial deployment, the user must check terms and consider legal advice. Raise this early rather than at deployment.
5. Before training on any dataset, open 30-50 random samples (use the viz helpers) and look at them. Report label noise, resolution, duplicated content, watermarks/overlays, and whether footage looks like CCTV or like phone video.
6. Keep a record of what is used for what in the Verification log at the bottom.

## Local layout

```
data/datasets/
  rwf2000/
  cctv_fights/
  <other>/
data/normal/<camera_name>/      # user's own normal footage (most important data)
data/action_clips/              # prepared clips for Phase 2B
```

## The most important dataset is the user's own

Public datasets differ from stadium CCTV (camera height, crowd density, resolution, lighting, scale of people). Plan for these from the start:
- Normal footage from each deployment camera (needed for Phase 1C baselines, no labels required).
- A small set of labelled events from the user's cameras (needed for the Phase 2B domain-gap fix).
- Footage of the events the user actually cares about, even if only a handful. Ask the user what exists.

---

## Phase 1B: crowd counting / head detection

| Dataset | Used for | Checked | Notes |
|---|---|---|---|
| ShanghaiTech (Part A/B) | Pretraining/sanity check of density models (P2PNet, DM-Count, CSRNet) | NOT CHECKED | Standard crowd counting benchmark; most pretrained counting weights come from it or UCF-QNRF |
| UCF-QNRF | Dense crowd counting | NOT CHECKED | Very dense scenes; useful for domain gap checks |
| NWPU-Crowd | Counting, varied scenes | NOT CHECKED | Large benchmark; test labels may be withheld (evaluation server); check |
| JHU-Crowd++ | Counting, with head-level annotations and weather conditions | NOT CHECKED | Check if head boxes/points are provided in a usable format |
| SCUT-HEAD | Head detector fine-tuning (YOLO) | NOT CHECKED | Classroom-style scenes, probably larger heads than stadium |
| CrowdHuman (head boxes) | Head detector fine-tuning | NOT CHECKED | Dense pedestrians with head/visible/full-body boxes; check access terms |

Guidance:
- Match the dataset to the user's head size (measured in Phase 1B Step 0). Public datasets contain many different scales; none is guaranteed to look like a stadium stand.
- Counting networks can usually be run with provided pretrained weights first. Fine-tune only if the manual count check in Phase 1B shows a problem, and then the user's own annotated frames matter more than public data.
- Annotating even 20-50 of the user's own frames (head points) is often more valuable than any public dataset. Offer this as an option.

---

## Phase 1C: baseline fitting

No public dataset is appropriate for fitting. Baselines must come from the deployment camera's own normal footage. For evaluating the scoring logic (not fitting) the following may help; all NOT CHECKED:

| Dataset | Use | Notes |
|---|---|---|
| UBnormal | Sanity checks of anomaly scoring | Synthetic (virtual) scenes; domain gap is large |
| ShanghaiTech Campus (anomaly) | Scoring sanity checks | Campus scenes, not stadiums; different from ShanghaiTech crowd counting |
| UMN crowd escape | Sudden-running crowd events | Old, small, low-resolution, but directly about crowd dispersal; useful for a quick test of speed-based scoring |

Warning: do not tune thresholds on these and assume they transfer to the stadium.

---

## Phase 1E: fire and smoke

| Dataset | Used for | Checked | Notes |
|---|---|---|---|
| D-Fire | Fire/smoke detector fine-tuning and evaluation | CHECKED (this session) | Over 21,000 images; YOLO-format annotations (normalised coordinates); classes fire and smoke; pre-split train/val/test and some surveillance videos are provided; authors ask for citation of their papers; repository `gaiasd/DFireDataset` on GitHub (confirm URL). Includes non-fire images that look fire-like, which helps with false positives |
| Wildfire smoke datasets (e.g. on Kaggle) | Extra smoke examples | NOT CHECKED | Often forest/outdoor scenes, not stadiums |
| Hard-negative smoke/cloud sets | Reducing false positives | NOT CHECKED | Clouds and haze are common false positives |

Guidance:
- Public fire data is mostly wildfire/outdoor. Stadium-specific false positives (floodlights, flags, red/orange clothing, flares, sun glare, steam) must be collected from the user's cameras as hard negatives.
- Decide with the user whether flares/pyrotechnics are alarms. If yes, the user must supply footage; public sets rarely include them.
- Split by source/video, not by image, when images come from video frames.

---

## Phase 2B: action classification (fight/violence, crowd events)

| Dataset | Used for | Checked | Notes |
|---|---|---|---|
| RWF-2000 | Fight vs non-fight classification | CHECKED (this session) | 2,000 clips from surveillance cameras (collected from YouTube), 80/20 train/test split provided by the authors, video-level labels; described as open access with the database and code in a GitHub repo (`mchengny/RWF2000-Video-Database-for-Violence-Detection`, confirm URL). Violence includes more than fighting (authors list robbery, explosion, shooting, blood, assault), so labels are broader than "fight". Check clip lengths and resolution on download |
| NTU CCTV-Fights | Fight detection with temporal annotation | CHECKED (this session) | 1,000 videos: 280 CCTV (about 8.5 hours) plus 720 non-CCTV (mobile, dash-cam, drone); about 7.2 GB; videos are long (5 s to 12 min), with annotated fight instances, so you must cut clips from the annotations; NTU states it is available on request (see the dataset page at rose1.ntu.edu.sg) and requires a specific acknowledgment line in publications. The original paper reports that training on non-CCTV data and then fine-tuning on CCTV helps, so consider a two-stage fine-tune |
| Real Life Violence Situations (RLVS) | Extra violence/non-violence clips | CHECKED (this session, partially) | 1,000 violence and 1,000 non-violence videos from YouTube; violence is mostly street fights; non-violence is everyday actions like sports and walking, so a model can learn "sport vs fight" shortcuts; citation requested by authors; the usual host is Kaggle (confirm) |
| Crowd Violence (Hockey/Movies/Crowd) | Crowd-scene violence | PARTIALLY (named in the RWF-2000 paper) | Crowd Violence has 246 videos of crowded scenes (2012). Hockey Fight and Movies Fight are older, small, and not CCTV. Check access and whether worth the effort |
| XD-Violence | Violence classes with audio, weakly labelled | NOT CHECKED | Large, includes movies/games; labels are video-level; many clips are not CCTV |
| UBI-Fights | Fight vs normal with frame-level annotation | NOT CHECKED | Check size and annotation format |
| UCF-Crime | Several crime classes | NOT CHECKED | Weak video-level labels, many classes with few examples, mostly low-resolution; user already found generalisation poor. Use only for specific classes after inspecting samples |
| Kinetics (400/600/700) | Pretrained weights for X3D/MoViNet/TSM | NOT CHECKED | You normally do not download Kinetics itself; use the pretrained weights from the model library. Do not download the dataset |
| Commercially sold simulated-fight datasets | Extra fights | CHECKED (exists, not evaluated) | Some vendors sell staged-fight videos with sample downloads. These are staged, not real; evaluate samples before considering and check cost/terms |

Class mapping suggestions (the user decides in sub-step 2B-1):
- `fight` <- RWF-2000 violent, CCTV-Fights fight segments, RLVS violence, UBI-Fights fight
- `normal` <- RWF-2000 non-fight, RLVS non-violence, CCTV-Fights segments with no annotated fight, and above all the user's own normal footage cut into clips
- `running_surge` <- almost no public data of this exact event; ask the user for footage; UMN crowd escape may help for dispersal; otherwise this class may be left to Tier 1 motion only and not be a Tier 2 class. Discuss honestly with the user.

Known pitfalls to report to the user:
- RWF-2000 and RLVS non-violence classes can teach shortcuts (e.g. indoor vs outdoor, camera type).
- Many violence videos share sources or are re-uploads. Check for near-duplicates across splits (same source video, same scene). Split by source.
- Clip lengths and fps differ across datasets; resample to the pipeline's ANALYSIS_FPS and clip length consistently.
- The domain gap to stadium CCTV is large for all of the above. Plan on a small set of the user's own labelled events.

---

## Recommended order of acquisition

1. User's own footage first (normal, plus any events). Nothing else substitutes for it.
2. RWF-2000 (open, small, simple labels): first action training run and pipeline debugging.
3. D-Fire when starting Phase 1E.
4. CCTV-Fights (request needed): ask the user to request access early, since approval may take time.
5. Counting datasets, only if Phase 1B shows the pretrained counting weights are inadequate.
6. Everything else only on request.

## Verification log

(Append one entry per dataset when it is actually obtained.)
```
### <dataset name>
- Date obtained:
- Source URL / how obtained (open download, request, purchase):
- Size on disk:
- Terms/licence as stated by the source (copy the wording or link; if unclear write "unclear"):
- Number of videos/images, classes, label format:
- Resolution / fps / clip length observed:
- Sample inspection result (what you saw in 30-50 random samples):
- Used for: (phase, class mapping)
- Split strategy:
```
