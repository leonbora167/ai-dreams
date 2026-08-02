from __future__ import annotations

import argparse
from pathlib import Path

from src.owlvit_dataset import OwlViTYoloDataset, load_dataset_config, read_yolo_labels


def validate_split(dataset: OwlViTYoloDataset, split: str, num_classes: int) -> list[str]:
    issues: list[str] = []

    for image_path in dataset.image_paths:
        label_path = dataset.labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            issues.append(f"[{split}] Missing label file for image: {image_path.name}")
            continue

        class_ids, boxes = read_yolo_labels(label_path)
        if len(class_ids) != len(boxes):
            issues.append(f"[{split}] Class/box length mismatch in: {label_path.name}")
            continue

        for index, class_id in enumerate(class_ids.tolist()):
            if class_id < 0 or class_id >= num_classes:
                issues.append(
                    f"[{split}] Invalid class id {class_id} in {label_path.name}. "
                    f"Expected range [0, {num_classes - 1}]"
                )

            box = boxes[index].tolist()
            cx, cy, width, height = box
            if not all(0.0 <= value <= 1.0 for value in box):
                issues.append(f"[{split}] Non-normalized box {box} in {label_path.name}")
            if width <= 0 or height <= 0:
                issues.append(f"[{split}] Non-positive box size {box} in {label_path.name}")
            if cx - (width / 2) < 0 or cy - (height / 2) < 0 or cx + (width / 2) > 1 or cy + (height / 2) > 1:
                issues.append(f"[{split}] Box extends outside image bounds {box} in {label_path.name}")

    label_files = sorted(path for path in dataset.labels_dir.glob("*.txt") if path.is_file())
    image_stems = {path.stem for path in dataset.image_paths}
    extra_labels = [path.name for path in label_files if path.stem not in image_stems]
    for label_name in extra_labels:
        issues.append(f"[{split}] Label file has no matching image: {label_name}")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a YOLO-style dataset before OWL-ViT fine-tuning.")
    parser.add_argument(
        "--dataset-config",
        required=True,
        help="YAML config with dataset paths and class names.",
    )
    args = parser.parse_args()

    dataset_config = load_dataset_config(args.dataset_config)
    print(f"Dataset root: {dataset_config.root}")
    print(f"Classes ({len(dataset_config.class_names)}): {', '.join(dataset_config.class_names)}")
    print(f"Prompt template: {dataset_config.prompt_template}")

    train_dataset = OwlViTYoloDataset(dataset_config, split="train")
    val_dataset = OwlViTYoloDataset(dataset_config, split="val")

    issues = []
    issues.extend(validate_split(train_dataset, "train", len(dataset_config.class_names)))
    issues.extend(validate_split(val_dataset, "val", len(dataset_config.class_names)))

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")

    if issues:
        print("\nValidation failed with the following issues:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("\nValidation passed. Dataset format looks compatible with this pipeline.")


if __name__ == "__main__":
    main()
