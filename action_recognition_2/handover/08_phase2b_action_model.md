# Phase 2B: compact action model (data, fine-tune, evaluate)

Goal: classify a clip from Phase 2A into our action classes. This phase is the largest, so it is split into sub-steps with separate approvals: 2B-1 to 2B-5. Do NOT merge them.

Prerequisite: Phase 2A approved. Read MODELS.md section "Phase 2B".

Read DATASETS.md section "Phase 2B" before sub-step 2B-1. Follow its download rules and class mapping notes.

---

## Sub-step 2B-1: classes, data inventory, model choice (no code yet beyond data listing)

Ask the user:
1. Final class list. Suggest starting small, e.g. `normal`, `fight`, `running_surge`, plus anything else they can actually get data for. Do not include a class without data.
2. Which datasets/footage they have access to and where (RWF-2000, NTU CCTV-Fights, XD-Violence, UBI-Fights, own footage). Check licences before use. UCF-Crime only with caution (weak labels).
3. How many labelled clips per class, from how many distinct cameras/sources.
4. Compute available for training.

Then produce `outputs/phase2b/1_data_inventory.md` with per-class clip counts, sources, duration, resolution, and an honest statement of risks (class imbalance, one source per class, etc.).

Pick ONE model to start (suggested X3D-S, or a torchvision video model if installation is painful), verified per MODELS.md. Explain why, with the user's agreement.

**Gate: user approves classes, data and model before continuing.**

## Sub-step 2B-2: dataset preparation

Implement `src/action.py` (plain functions) and `scripts/p2b_prepare_data.py`:
- `list_clips(data_dir) -> list of (path, label, source_id)`
- `split_by_source(clips, val_frac, test_frac, seed) -> train, val, test`: split by SOURCE VIDEO or camera, NEVER by clip, otherwise near-duplicate clips leak between splits and the scores will be fake. Print counts per split per class and state how the split was done.
- `load_clip(path, num_frames, size, train) -> (T, S, S, 3)`: apply the SAME crop logic as Phase 2A where bounding boxes exist; for datasets without boxes, use a defined whole-frame or centre crop and say so. Training augmentation kept simple: horizontal flip, small scale/position jitter of the crop box (to mimic noisy tier 1 boxes), mild brightness/contrast. Put augmentations in one visible function.
- Normalisation constants matching the pretrained model.

Visual outputs (`outputs/phase2b/2_data/`):
- `train_batch_grid.png`: 16 clips as strips with labels (after augmentation).
- `class_counts_by_split.png`
- `augmentation_examples.png`: same clip with 6 augmentations.

**Gate: user checks the samples look right and the split is clean.**

## Sub-step 2B-3: pretrained sanity forward pass

Before training anything:
- Load the model, swap the classification head for our classes, run a forward pass on one batch, print shapes and parameter count, measure latency (batch 1, warm-up excluded) on the user's hardware, and GPU memory.
- Save `outputs/phase2b/3_sanity/latency_table.md`.

**Gate: user accepts latency/memory for the target hardware.** If too slow, propose the next smaller model and re-verify. Do not train a model that cannot meet the budget.

## Sub-step 2B-4: fine-tune (simple loop)

Write `scripts/p2b_train.py` with a plain training loop (no Lightning):
- Freeze the backbone for the first epochs, train the head, then unfreeze the last blocks with a lower learning rate. Explain these choices in comments.
- AdamW, cosine or step schedule, mixed precision only if it works without fuss, class-weighted loss or balanced sampling if classes are imbalanced (say which).
- Log train/val loss and accuracy per epoch to a CSV and print them. Save best checkpoint by validation macro-F1 to `weights/action_<model>_best.pt`.
- Keep epochs small for the first run (e.g. 5-10) so feedback is fast. Report time per epoch.

Visual outputs (`outputs/phase2b/4_train/`): `loss_curves.png`, `val_f1_curve.png`, `train_log.csv`.

**Gate: user reviews curves.** Overfitting (train up, val flat) should be called out honestly, with options (more data, stronger augmentation, fewer trainable layers).

## Sub-step 2B-5: evaluation

Write `scripts/p2b_eval.py` that evaluates on the held-out TEST split:
- Confusion matrix (counts and row-normalised), per-class precision/recall/F1, macro-F1.
- Grids of clips: true positives, false positives, false negatives per class (with predicted probabilities in the label).
- Per-source results (are some cameras/sources much worse?).
- Latency on the final model.

Visual outputs (`outputs/phase2b/5_eval/`): `confusion_matrix.png`, `per_class_metrics.png`, `fp_grid.png`, `fn_grid.png`, `tp_grid.png`, `per_source_metrics.png`.

Then the critical domain check, done with the user:
- Run Tier 1 (Phases 1A-1D) on the user's OWN stadium footage, take the events it proposes, crop with Phase 2A, and classify with this model. Save `outputs/phase2b/5_eval/tier1_events_predictions.png` (event keyframes + predicted class and probability).
- Public datasets differ from stadium CCTV. Expect a gap and report it honestly. Offer a plan: the user labels a small set of these tier 1 events, they are added to training, and we re-run 2B-4/2B-5. Do not do this without approval.

## What the user should check overall

- Accuracy on held-out sources, not just on random clips.
- Behaviour on crops from tier 1 events (the real deployment distribution).
- False positives: what normal activities look like "fight" (e.g. celebrations, crowd cheering)?
- Latency/memory on target hardware.

## Acceptance checklist

- [ ] 2B-1 approved (classes, data, model)
- [ ] 2B-2 approved (splits clean, samples look right)
- [ ] 2B-3 approved (latency/memory OK)
- [ ] 2B-4 approved (training behaves)
- [ ] 2B-5 approved (eval + tier 1 domain check reviewed)
- [ ] `config.py`: `ACTION_MODEL`, `ACTION_WEIGHTS`, `ACTION_CLASSES`, `ACTION_CONF_THR` set
- [ ] Optional second model comparison (TSM or other): only if the user requests it, as a repeat of 2B-3 to 2B-5

STOP after each sub-step with a phase report. Wait for "approved 2B-<n>".
