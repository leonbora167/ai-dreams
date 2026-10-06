import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .base import BaseModelAdapter

class InceptionAdapter(BaseModelAdapter):
    """
    Adapter for Inception-v3 Classification Model.
    Supports feature extraction, inference, and classification metrics evaluation.
    """
    DEFAULT_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    def __init__(self, device: str = "cpu", classes: Optional[List[str]] = None):
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.classes = classes or self.DEFAULT_CLASSES
        
        # Load pre-trained Inception-v3
        weights = models.Inception_V3_Weights.DEFAULT
        self.model = models.inception_v3(weights=weights)
        self.model.eval()
        self.model.to(self.device)

        # Standard Inception preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Feature hook to grab embeddings before final fc
        self._features = []
        def hook(module, input, output):
            # Output of AdaptiveAvgPool2d is (N, 2048, 1, 1)
            self._features.append(output.detach().cpu().squeeze())
        
        self.model.avgpool.register_forward_hook(hook)

    def predict(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        results = []
        batch_tensors = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            batch_tensors.append(self.transform(img))
        
        if not batch_tensors:
            return results

        tensors = torch.stack(batch_tensors).to(self.device)
        with torch.no_grad():
            self._features.clear()
            logits = self.model(tensors)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Map predictions to designated classes (using modulo/argmax over class space)
            # Or map top classes
            top_indices = np.argmax(probs, axis=-1)
            top_confs = np.max(probs, axis=-1)
            
            for i in range(len(images)):
                # Map ImageNet class index to one of our target classes deterministically
                target_cls_idx = top_indices[i] % len(self.classes)
                pred_label = self.classes[target_cls_idx]
                conf = float(top_confs[i])
                results.append({
                    "class_id": int(target_cls_idx),
                    "label": pred_label,
                    "confidence": conf,
                    "entropy": float(-np.sum(probs[i] * np.log(probs[i] + 1e-12)))
                })
        return results

    def extract_features(self, images: List[Image.Image]) -> np.ndarray:
        batch_tensors = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            batch_tensors.append(self.transform(img))

        if not batch_tensors:
            return np.empty((0, 2048))

        tensors = torch.stack(batch_tensors).to(self.device)
        with torch.no_grad():
            self._features.clear()
            _ = self.model(tensors)
            if self._features:
                feats = torch.cat([f.unsqueeze(0) if f.ndim == 1 else f for f in self._features], dim=0).numpy()
                return feats
            return np.empty((len(images), 2048))

    def evaluate(self, predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]) -> Dict[str, float]:
        if not ground_truths or len(ground_truths) != len(predictions):
            return {}
        
        y_true = [gt["label"] if isinstance(gt, dict) else gt for gt in ground_truths]
        y_pred = [p["label"] for p in predictions]

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1)
        }
