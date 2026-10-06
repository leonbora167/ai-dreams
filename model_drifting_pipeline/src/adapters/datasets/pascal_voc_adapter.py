import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from typing import List, Dict, Any, Optional
from .base import BaseDatasetAdapter

class PascalVOCAdapter(BaseDatasetAdapter):
    """
    Adapter for PASCAL VOC Detection format.
    Supports directory with images and XML/JSON annotations.
    """
    def __init__(self, target_classes: Optional[List[str]] = None):
        self.target_classes = target_classes or ["person", "car", "bicycle", "bus", "motorcycle", "dog", "cat"]

    def load(self, path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        items = []
        img_dir = path
        ann_dir = None

        if os.path.isdir(os.path.join(path, "images")):
            img_dir = os.path.join(path, "images")
        if os.path.isdir(os.path.join(path, "annotations")):
            ann_dir = os.path.join(path, "annotations")
        elif os.path.isdir(os.path.join(path, "Annotations")):
            ann_dir = os.path.join(path, "Annotations")

        if not os.path.exists(img_dir):
            return items

        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if max_samples:
            img_files = img_files[:max_samples]

        for fname in img_files:
            img_path = os.path.join(img_dir, fname)
            try:
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                
                # Try loading annotations
                boxes = []
                labels = []
                stem = os.path.splitext(fname)[0]

                if ann_dir:
                    xml_path = os.path.join(ann_dir, f"{stem}.xml")
                    json_path = os.path.join(ann_dir, f"{stem}.json")
                    if os.path.exists(xml_path):
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        for obj in root.findall("object"):
                            name = obj.find("name").text
                            bnd = obj.find("bndbox")
                            xmin = float(bnd.find("xmin").text)
                            ymin = float(bnd.find("ymin").text)
                            xmax = float(bnd.find("xmax").text)
                            ymax = float(bnd.find("ymax").text)
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(name)
                    elif os.path.exists(json_path):
                        with open(json_path, 'r') as jf:
                            data = json.load(jf)
                            boxes = data.get("boxes", [])
                            labels = data.get("labels", [])

                item = {
                    "image": img,
                    "image_path": img_path,
                    "metadata": {"width": w, "height": h, "filename": fname},
                    "annotations": {"boxes": boxes, "labels": labels} if boxes else None
                }
                items.append(item)
            except Exception as e:
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
