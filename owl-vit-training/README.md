# OWL-ViT Fine-Tuning On YOLO-Style Datasets

This repository is a compact but complete example of how to fine-tune `google/owlvit-base-patch32` on a tiny detection dataset first, then reuse the same pipeline for your own classes such as `hazmat suit`, `lego`, or anything else that can be annotated with bounding boxes.

It starts with `COCO8` because that dataset is extremely small and easy to inspect, but the code is now generalized to work with any dataset that follows the same folder layout and label format.

## What OWL-ViT Is

OWL-ViT is an open-vocabulary object detector. Instead of predicting only from a fixed classifier head like a traditional closed-set detector, it takes:

- an image
- a set of text queries such as `["a photo of a hazmat suit", "a photo of a person"]`

and returns:

- detection scores for those queries
- bounding boxes for regions that match them

Conceptually, it learns an alignment between image regions and text prompts.

## What Fine-Tuning Means Here

This pipeline performs supervised fine-tuning.

The supervision comes from:

- the image
- the ground-truth bounding boxes
- the class id per box
- the class-id-to-class-name mapping from your dataset YAML

During loading, the code turns each class name into a text prompt using a template like:

```text
a photo of a {}
```

So if your YAML says:

```yaml
names:
  0: hazmat suit
  1: person
```

the model is trained against prompts like:

- `a photo of a hazmat suit`
- `a photo of a person`

The model is therefore learning two things at once:

- localization: where the object is
- grounding: which text prompt that object should align with

## High-Level Architecture

The end-to-end flow is:

1. Load `dataset.yaml`.
2. Read all image files from `images/train` and `images/val`.
3. Read YOLO label files from `labels/train` and `labels/val`.
4. Convert each class id into a prompt string using the YAML class names.
5. Pass `text + image` into `OwlViTProcessor`.
6. Run `OwlViTForObjectDetection`.
7. Match predictions to targets with Hungarian matching.
8. Compute classification, L1 box, and GIoU losses.
9. Update the model weights.

## Repository Layout

- [download_coco8.py](C:/Projects/ai-dreams/owl-vit-training/download_coco8.py): downloads and prepares the tiny starter dataset
- [train_owlvit_yolo.py](C:/Projects/ai-dreams/owl-vit-training/train_owlvit_yolo.py): generic trainer for YOLO-style datasets
- [validate_yolo_dataset.py](C:/Projects/ai-dreams/owl-vit-training/validate_yolo_dataset.py): checks that a dataset is structurally compatible before training
- [infer_owlvit.py](C:/Projects/ai-dreams/owl-vit-training/infer_owlvit.py): runs inference on one image
- [visualize_val_predictions.py](C:/Projects/ai-dreams/owl-vit-training/visualize_val_predictions.py): saves validation images with predicted and ground-truth boxes
- [src/owlvit_dataset.py](C:/Projects/ai-dreams/owl-vit-training/src/owlvit_dataset.py): dataset config loader, prompt builder, YOLO reader, and custom loss
- [example-hazmat-dataset.yaml](C:/Projects/ai-dreams/owl-vit-training/example-hazmat-dataset.yaml): example config for a custom dataset

## Where Prompt Text Is Built And Sent To The Model

These are the most important code points:

- Prompt template definition: [src/owlvit_dataset.py](C:/Projects/ai-dreams/owl-vit-training/src/owlvit_dataset.py:16)
- Text query construction from class names: [src/owlvit_dataset.py](C:/Projects/ai-dreams/owl-vit-training/src/owlvit_dataset.py:112)
- Sample stores `text_queries`: [src/owlvit_dataset.py](C:/Projects/ai-dreams/owl-vit-training/src/owlvit_dataset.py:247)
- Processor receives `text=text_queries`: [src/owlvit_dataset.py](C:/Projects/ai-dreams/owl-vit-training/src/owlvit_dataset.py:257)
- Model receives the encoded tensors: [train_owlvit_yolo.py](C:/Projects/ai-dreams/owl-vit-training/train_owlvit_yolo.py:57)

That is the exact path by which a class id becomes a text prompt and then becomes model input.

## Expected Dataset Structure

Your custom dataset should look like this:

```text
your-dataset/
  dataset.yaml
  images/
    train/
      img_001.jpg
      img_002.jpg
    val/
      img_101.jpg
      img_102.jpg
  labels/
    train/
      img_001.txt
      img_002.txt
    val/
      img_101.txt
      img_102.txt
```

Rules:

- every image should have a matching label file with the same stem
- labels must be YOLO detection format
- class ids must start at `0` and be contiguous
- bounding boxes must be normalized to `[0, 1]`

## YOLO Label Format

Each label file contains one object per line:

```text
class_id center_x center_y width height
```

Example:

```text
0 0.52 0.48 0.31 0.72
1 0.67 0.61 0.14 0.20
```

This means:

- one object of class `0`
- one object of class `1`

with normalized `cxcywh` boxes.

## Dataset YAML Format

Your `dataset.yaml` should look like this:

```yaml
path: ./data/hazmat-demo
train: images/train
val: images/val
names:
  0: hazmat suit
  1: person
prompt_template: "a photo of a {}"
```

Fields:

- `path`: dataset root
- `train`: relative or absolute path to training images
- `val`: relative or absolute path to validation images
- `names`: class-id-to-class-name mapping
- `prompt_template`: optional text template used to build OWL-ViT prompts

## Why The YAML Matters

The label files do not contain text prompts directly. They only contain numeric class ids.

The YAML is what tells the pipeline:

- `0 -> hazmat suit`
- `1 -> person`

Then the code turns that into text prompts automatically.

So the minimal ingredients for a new dataset are:

- images
- bounding boxes
- class ids
- a YAML mapping from class ids to class names

## First-Time Setup With COCO8

Download and prepare the tiny starter dataset:

```powershell
conda run -n pyt-dl python download_coco8.py
```

Validate it:

```powershell
conda run -n pyt-dl python validate_yolo_dataset.py --dataset-config data/coco8/dataset.yaml
```

Run a short smoke-test training:

```powershell
conda run -n pyt-dl python train_owlvit_yolo.py --dataset-config data/coco8/dataset.yaml --freeze-text --epochs 5 --batch-size 1
```

## Training Your Own Dataset

Once your dataset follows the same format, train it with:

```powershell
conda run -n pyt-dl python validate_yolo_dataset.py --dataset-config path\to\your-dataset\dataset.yaml
conda run -n pyt-dl python train_owlvit_yolo.py --dataset-config path\to\your-dataset\dataset.yaml --freeze-text --epochs 20 --batch-size 1
```

Recommended for the first run:

- keep `--freeze-text`
- start with `--epochs 5`
- use `--batch-size 1`
- confirm that qualitative predictions improve before tuning aggressively

## Inference

Run inference on a single image:

```powershell
conda run -n pyt-dl python infer_owlvit.py `
  --checkpoint artifacts/owlvit-yolo/best.pt `
  --image path\to\image.jpg `
  --labels "hazmat suit" person
```

If you omit `--labels`, the script will use the class names stored inside the checkpoint.

## Visualization

Render predictions and ground truth on the validation set:

```powershell
conda run -n pyt-dl python visualize_val_predictions.py `
  --checkpoint artifacts/owlvit-yolo/best.pt `
  --dataset-config data/coco8/dataset.yaml `
  --output-dir artifacts/owlvit-yolo/val-viz
```

Ground-truth boxes are drawn in green and predictions in red.

## TensorBoard

Training logs are written to `runs/owlvit-yolo/<timestamp>/`.

Launch TensorBoard with:

```powershell
conda run -n pyt-dl tensorboard --logdir runs/owlvit-yolo
```

Useful charts to watch:

- `loss/train_total`
- `loss/val_total`
- `loss/train_bbox`
- `loss/val_bbox`
- `loss/train_giou`
- `loss/val_giou`
- `matching/train_objects`
- `matching/val_objects`

## Outputs

Training writes to `artifacts/owlvit-yolo/`:

- `best.pt`
- `last.pt`
- `history.json`
- `processor/`

Optional visualization output can go to:

- `artifacts/owlvit-yolo/val-viz/`

## Technical Notes

- The current `transformers` OWL-ViT API exposes predictions cleanly, but not a built-in end-to-end fine-tuning loss for this training setup.
- Because of that, this repository implements a compact custom loss on top of model outputs.
- Matching is done with Hungarian assignment.
- The loss combines:
  - classification loss
  - L1 box loss
  - generalized IoU loss

This is why the pipeline is conceptually close to DETR-style training, while still using OWL-ViT as the detector backbone and text-grounding model.

## Practical Expectations

For `COCO8`:

- the goal is understanding the pipeline
- the metrics are noisy
- overfitting is expected

For a real dataset like `hazmat suit`:

- the same pipeline becomes meaningful once you have enough clean annotations
- the model should improve at grounding that class name to the correct region
- the better the class-name wording and data consistency, the better the results usually are

## Suggested Progression

1. Run `COCO8` end to end.
2. Inspect the validation visualizations.
3. Validate your custom dataset with `validate_yolo_dataset.py`.
4. Train on your custom classes.
5. Compare predictions qualitatively before spending time on heavier metric work.

## Important Limitation

This is a strong learning and prototyping pipeline, not yet a production training framework.

If you later want to scale this up seriously, the next improvements would be:

- better evaluation metrics such as mAP
- checkpoint resume support
- data augmentation
- richer prompt strategies
- more robust experiment configuration
