from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers.loss.loss_for_object_detection import generalized_box_iou


DEFAULT_PROMPT_TEMPLATE = "a photo of a {}"

COCO80_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


@dataclass
class OwlViTDatasetConfig:
    root: Path
    train_images_dir: Path
    val_images_dir: Path
    train_labels_dir: Path
    val_labels_dir: Path
    class_names: list[str]
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE

    @property
    def text_queries(self) -> list[str]:
        return [self.prompt_template.format(name) for name in self.class_names]


@dataclass
class DetectionSample:
    image: Image.Image
    image_path: str
    class_labels: torch.Tensor
    boxes_cxcywh: torch.Tensor
    boxes_xyxy: torch.Tensor
    text_queries: list[str]


def normalize_class_names(names: dict | list) -> list[str]:
    if isinstance(names, list):
        return [str(name) for name in names]

    if isinstance(names, dict):
        normalized = {int(key): str(value) for key, value in names.items()}
        expected = list(range(len(normalized)))
        if sorted(normalized.keys()) != expected:
            raise ValueError(f"Class ids must be contiguous starting at 0, got keys {sorted(normalized.keys())}")
        return [normalized[index] for index in expected]

    raise TypeError("`names` must be a list or dict.")


def resolve_split_dir(root: Path, split_value: str | list[str], default_dir: str) -> Path:
    if isinstance(split_value, list):
        if len(split_value) != 1:
            raise ValueError("Only a single directory per split is supported in this minimal pipeline.")
        split_value = split_value[0]

    if not split_value:
        return root / default_dir

    candidate = Path(split_value)
    return candidate if candidate.is_absolute() else root / candidate


def load_dataset_config(config_path: str | Path) -> OwlViTDatasetConfig:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())
    root_value = raw.get("path", ".")
    root = Path(root_value)
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()

    class_names = normalize_class_names(raw["names"])
    prompt_template = raw.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)

    train_images_dir = resolve_split_dir(root, raw.get("train"), "images/train")
    val_images_dir = resolve_split_dir(root, raw.get("val"), "images/val")
    train_labels_dir = root / "labels" / "train"
    val_labels_dir = root / "labels" / "val"

    return OwlViTDatasetConfig(
        root=root,
        train_images_dir=train_images_dir,
        val_images_dir=val_images_dir,
        train_labels_dir=train_labels_dir,
        val_labels_dir=val_labels_dir,
        class_names=class_names,
        prompt_template=prompt_template,
    )


def default_coco8_config(root: str | Path = "data/coco8") -> OwlViTDatasetConfig:
    root = Path(root).resolve()
    return OwlViTDatasetConfig(
        root=root,
        train_images_dir=root / "images" / "train",
        val_images_dir=root / "images" / "val",
        train_labels_dir=root / "labels" / "train",
        val_labels_dir=root / "labels" / "val",
        class_names=[COCO80_NAMES[index] for index in range(len(COCO80_NAMES))],
    )


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - (w / 2), cy - (h / 2), cx + (w / 2), cy + (h / 2)), dim=-1)


def read_yolo_labels(label_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not label_path.exists():
        return torch.zeros((0,), dtype=torch.long), torch.zeros((0, 4), dtype=torch.float32)

    class_ids = []
    boxes = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        class_ids.append(int(parts[0]))
        boxes.append([float(value) for value in parts[1:5]])

    if not boxes:
        return torch.zeros((0,), dtype=torch.long), torch.zeros((0, 4), dtype=torch.float32)

    return torch.tensor(class_ids, dtype=torch.long), torch.tensor(boxes, dtype=torch.float32)


class OwlViTYoloDataset(Dataset[DetectionSample]):
    def __init__(self, dataset_config: OwlViTDatasetConfig, split: str):
        self.dataset_config = dataset_config
        self.split = split

        if split == "train":
            self.images_dir = dataset_config.train_images_dir
            self.labels_dir = dataset_config.train_labels_dir
        elif split == "val":
            self.images_dir = dataset_config.val_images_dir
            self.labels_dir = dataset_config.val_labels_dir
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.image_paths = sorted(path for path in self.images_dir.glob("*") if path.is_file())
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> DetectionSample:
        image_path = self.image_paths[index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        class_labels, boxes_cxcywh = read_yolo_labels(label_path)
        boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh) if len(boxes_cxcywh) else boxes_cxcywh.clone()

        return DetectionSample(
            image=image,
            image_path=str(image_path),
            class_labels=class_labels,
            boxes_cxcywh=boxes_cxcywh,
            boxes_xyxy=boxes_xyxy,
            text_queries=self.dataset_config.text_queries,
        )


def collate_fn(batch: list[DetectionSample], processor) -> dict[str, Any]:
    images = [sample.image for sample in batch]
    text = [sample.text_queries for sample in batch]
    encoded = processor(text=text, images=images, return_tensors="pt", truncation=True)
    encoded["targets"] = [
        {
            "class_labels": sample.class_labels,
            "boxes_cxcywh": sample.boxes_cxcywh,
            "boxes_xyxy": sample.boxes_xyxy,
            "image_path": sample.image_path,
        }
        for sample in batch
    ]
    return encoded


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    modulating = (1.0 - p_t) ** gamma
    loss = ce_loss * modulating

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    return loss.mean()


def hungarian_match(
    pred_logits: torch.Tensor,
    pred_boxes_cxcywh: torch.Tensor,
    target_classes: torch.Tensor,
    target_boxes_cxcywh: torch.Tensor,
    target_boxes_xyxy: torch.Tensor,
    class_cost: float,
    bbox_cost: float,
    giou_cost: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_classes.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.long, device=pred_logits.device)
        return empty, empty

    pred_probs = pred_logits.sigmoid()
    class_scores = pred_probs[:, target_classes]
    class_term = -class_scores

    bbox_term = torch.cdist(pred_boxes_cxcywh, target_boxes_cxcywh, p=1)
    pred_boxes_xyxy = cxcywh_to_xyxy(pred_boxes_cxcywh)
    giou = generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy)
    giou_term = -giou

    cost = (class_cost * class_term) + (bbox_cost * bbox_term) + (giou_cost * giou_term)
    pred_indices, target_indices = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(pred_indices, dtype=torch.long, device=pred_logits.device),
        torch.as_tensor(target_indices, dtype=torch.long, device=pred_logits.device),
    )


def owlvit_detection_loss(
    outputs,
    targets: list[dict[str, torch.Tensor]],
    class_cost: float = 1.0,
    bbox_cost: float = 5.0,
    giou_cost: float = 2.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    total_cls = outputs.logits.new_tensor(0.0)
    total_bbox = outputs.logits.new_tensor(0.0)
    total_giou = outputs.logits.new_tensor(0.0)
    total_objects = 0

    for batch_index, target in enumerate(targets):
        pred_logits = outputs.logits[batch_index]
        pred_boxes_cxcywh = outputs.pred_boxes[batch_index]

        target_classes = target["class_labels"].to(pred_logits.device)
        target_boxes_cxcywh = target["boxes_cxcywh"].to(pred_logits.device)
        target_boxes_xyxy = target["boxes_xyxy"].to(pred_logits.device)

        matched_pred, matched_target = hungarian_match(
            pred_logits=pred_logits,
            pred_boxes_cxcywh=pred_boxes_cxcywh,
            target_classes=target_classes,
            target_boxes_cxcywh=target_boxes_cxcywh,
            target_boxes_xyxy=target_boxes_xyxy,
            class_cost=class_cost,
            bbox_cost=bbox_cost,
            giou_cost=giou_cost,
        )

        cls_targets = torch.zeros_like(pred_logits)
        if matched_pred.numel():
            cls_targets[matched_pred, target_classes[matched_target]] = 1.0
        total_cls = total_cls + sigmoid_focal_loss(pred_logits, cls_targets, alpha=focal_alpha, gamma=focal_gamma)

        if matched_pred.numel():
            matched_pred_boxes = pred_boxes_cxcywh[matched_pred]
            matched_target_boxes = target_boxes_cxcywh[matched_target]
            total_bbox = total_bbox + F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction="mean")

            pred_xyxy = cxcywh_to_xyxy(matched_pred_boxes)
            target_xyxy = target_boxes_xyxy[matched_target]
            giou = generalized_box_iou(pred_xyxy, target_xyxy)
            diag_giou = torch.diag(giou)
            total_giou = total_giou + (1.0 - diag_giou).mean()
            total_objects += int(matched_pred.numel())

    batch_size = max(len(targets), 1)
    total_cls = total_cls / batch_size
    total_bbox = total_bbox / batch_size
    total_giou = total_giou / batch_size
    loss = total_cls + (bbox_cost * total_bbox) + (giou_cost * total_giou)

    metrics = {
        "loss": float(loss.detach().cpu()),
        "loss_cls": float(total_cls.detach().cpu()),
        "loss_bbox": float(total_bbox.detach().cpu()),
        "loss_giou": float(total_giou.detach().cpu()),
        "matched_objects": float(total_objects),
    }
    return loss, metrics
