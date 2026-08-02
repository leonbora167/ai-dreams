from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from src.owlvit_dataset import OwlViTYoloDataset, load_dataset_config


def draw_boxes(
    image: Image.Image,
    pred_boxes,
    pred_labels,
    pred_scores,
    gt_boxes_xyxy: torch.Tensor,
    gt_class_ids: torch.Tensor,
    class_names: list[str],
) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)

    for box, class_id in zip(gt_boxes_xyxy.tolist(), gt_class_ids.tolist()):
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
        draw.text((x1 + 4, y1 + 4), f"GT: {class_names[int(class_id)]}", fill="lime")

    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1 + 4, max(y1 - 14, 0)), f"PR: {label} {float(score):.2f}", fill="red")

    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize OWL-ViT predictions on COCO8 validation images.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint produced by training.")
    parser.add_argument(
        "--dataset-config",
        default="data/coco8/dataset.yaml",
        help="YAML config with dataset paths and class names.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/owlvit-yolo/val-viz",
        help="Directory where rendered validation images will be saved.",
    )
    parser.add_argument("--threshold", type=float, default=0.15, help="Confidence threshold.")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint["model_name"]

    processor_dir = Path(args.checkpoint).parent / "processor"
    if processor_dir.exists():
        processor = OwlViTProcessor.from_pretrained(processor_dir)
    else:
        processor = OwlViTProcessor.from_pretrained(model_name)

    model = OwlViTForObjectDetection.from_pretrained(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset_config = load_dataset_config(args.dataset_config)
    dataset = OwlViTYoloDataset(dataset_config, split="val")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = checkpoint.get("class_names", dataset_config.class_names)
    prompt_template = checkpoint.get("prompt_template", dataset_config.prompt_template)
    text_labels = [[prompt_template.format(label) for label in class_names]]

    for sample in dataset:
        inputs = processor(text=text_labels, images=sample.image, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([(sample.image.height, sample.image.width)])
        result = processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=args.threshold,
            text_labels=text_labels,
        )[0]

        gt_boxes_xyxy = sample.boxes_xyxy.clone()
        gt_boxes_xyxy[:, [0, 2]] *= sample.image.width
        gt_boxes_xyxy[:, [1, 3]] *= sample.image.height

        rendered = draw_boxes(
            image=sample.image,
            pred_boxes=result["boxes"],
            pred_labels=result["text_labels"],
            pred_scores=result["scores"],
            gt_boxes_xyxy=gt_boxes_xyxy,
            gt_class_ids=sample.class_labels,
            class_names=class_names,
        )
        output_path = output_dir / Path(sample.image_path).name
        rendered.save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
