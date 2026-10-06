import os
import json
from PIL import Image
from typing import List, Dict, Any, Optional
from .base import BaseDatasetAdapter

class CIFAR10Adapter(BaseDatasetAdapter):
    """
    Adapter for CIFAR-10 / Image Classification dataset directories.
    Can read from class-subfolders (e.g. dir/airplane/img.png) or flat image dir with labels.json.
    """
    CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def load(self, path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        items = []
        if not os.path.exists(path):
            return items

        # Check for subdirectories (class folder format)
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if subdirs and any(d in self.CLASSES for d in subdirs):
            for cls_name in subdirs:
                cls_folder = os.path.join(path, cls_name)
                files = sorted(os.listdir(cls_folder))
                for f in files:
                    if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    img_path = os.path.join(cls_folder, f)
                    try:
                        img = Image.open(img_path).convert("RGB")
                        items.append({
                            "image": img,
                            "image_path": img_path,
                            "metadata": {"width": img.width, "height": img.height, "class": cls_name},
                            "annotations": {"label": cls_name}
                        })
                    except Exception:
                        continue
                    if max_samples and len(items) >= max_samples:
                        return items
            return items

        # Check for labels.json
        labels_map = {}
        json_path = os.path.join(path, "labels.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as jf:
                    labels_map = json.load(jf)
            except Exception:
                pass

        # Flat directory scan
        img_dir = os.path.join(path, "images") if os.path.isdir(os.path.join(path, "images")) else path
        files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if max_samples:
            files = files[:max_samples]

        for f in files:
            img_path = os.path.join(img_dir, f)
            try:
                img = Image.open(img_path).convert("RGB")
                label_val = labels_map.get(f, labels_map.get(os.path.splitext(f)[0], None))
                items.append({
                    "image": img,
                    "image_path": img_path,
                    "metadata": {"width": img.width, "height": img.height},
                    "annotations": {"label": label_val} if label_val is not None else None
                })
            except Exception:
                continue

        return items

    def validate(self, path: str) -> Dict[str, Any]:
        items = self.load(path)
        valid_count = len(items)
        annotated_count = sum(1 for it in items if it["annotations"] is not None)
        return {
            "total_images": valid_count,
            "has_ground_truth": annotated_count > 0,
            "labelled_ratio": annotated_count / valid_count if valid_count > 0 else 0.0
        }

    def get_labels(self, items: List[Dict[str, Any]]) -> List[Any]:
        return [it["annotations"] for it in items if it.get("annotations") is not None]
