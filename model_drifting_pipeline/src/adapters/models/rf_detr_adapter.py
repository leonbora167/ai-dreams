import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Dict, Any, Optional
import numpy as np
from .base import BaseModelAdapter

class RFDETRAdapter(BaseModelAdapter):
    """
    Adapter for Object Detection (RF-DETR / Faster R-CNN PyTorch architecture).
    Compatible with PASCAL VOC classes.
    """
    COCO_TO_VOC = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        6: "bus",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow"
    }

    def __init__(self, score_thresh: float = 0.5, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.score_thresh = score_thresh
        
        # Pretrained FasterRCNN with ResNet50-FPN backbone
        weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.ToTensor()

    def predict(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        results = []
        batch_tensors = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            batch_tensors.append(self.transform(img).to(self.device))
        
        if not batch_tensors:
            return results

        with torch.no_grad():
            outputs = self.model(batch_tensors)

        for i, out in enumerate(outputs):
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()

            keep = scores >= self.score_thresh
            filtered_boxes = boxes[keep]
            filtered_scores = scores[keep]
            filtered_labels = labels[keep]

            detection_list = []
            box_areas = []
            for b, s, l in zip(filtered_boxes, filtered_scores, filtered_labels):
                label_name = self.COCO_TO_VOC.get(int(l), f"class_{l}")
                area = float((b[2] - b[0]) * (b[3] - b[1]))
                box_areas.append(area)
                detection_list.append({
                    "box": [float(coord) for coord in b],
                    "confidence": float(s),
                    "label": label_name,
                    "area": area
                })

            results.append({
                "num_detections": len(detection_list),
                "detections": detection_list,
                "avg_confidence": float(np.mean(filtered_scores)) if len(filtered_scores) > 0 else 0.0,
                "avg_box_area": float(np.mean(box_areas)) if len(box_areas) > 0 else 0.0
            })
        return results

    def extract_features(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract backbone features (pool of ResNet50 backbone).
        """
        batch_tensors = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # resize for uniform feature embedding
            img_resized = img.resize((256, 256))
            batch_tensors.append(self.transform(img_resized).to(self.device))
        
        if not batch_tensors:
            return np.empty((0, 256))

        tensors = torch.stack(batch_tensors)
        with torch.no_grad():
            features_dict = self.model.backbone(tensors)
            # Take pool layer 0 feature map average
            feat_tensor = features_dict['0'].mean(dim=[2, 3]).cpu().numpy()
            return feat_tensor

    def evaluate(self, predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluates mAP, precision, and recall over IoU 0.5.
        """
        if not ground_truths or len(ground_truths) != len(predictions):
            return {}

        total_gt = 0
        total_pred = 0
        tp = 0
        fp = 0

        for pred, gt in zip(predictions, ground_truths):
            gt_boxes = gt.get("boxes", [])
            gt_labels = gt.get("labels", [])
            pred_dets = pred.get("detections", [])

            total_gt += len(gt_boxes)
            total_pred += len(pred_dets)

            matched_gt = set()
            for p_det in pred_dets:
                p_box = p_det["box"]
                p_lbl = p_det["label"]
                best_iou = 0.0
                best_gt_idx = -1

                for idx, (g_box, g_lbl) in enumerate(zip(gt_boxes, gt_labels)):
                    if idx in matched_gt or p_lbl != g_lbl:
                        continue
                    # Calculate IoU
                    xi1 = max(p_box[0], g_box[0])
                    yi1 = max(p_box[1], g_box[1])
                    xi2 = min(p_box[2], g_box[2])
                    yi2 = min(p_box[3], g_box[3])
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    
                    p_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                    g_area = (g_box[2] - g_box[0]) * (g_box[3] - g_box[1])
                    union_area = p_area + g_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= 0.5 and best_gt_idx >= 0:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        map50 = (precision + recall) / 2 if (precision + recall) > 0 else 0.0

        return {
            "mAP_50": float(map50),
            "precision": float(precision),
            "recall": float(recall),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(total_gt - tp)
        }
