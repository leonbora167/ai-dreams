from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

class BaseModelAdapter(ABC):
    """
    Abstract Base Class for Model Adapters.
    Allows decoupling any CV model (Classification, Detection, Segmentation)
    from the drift assessment engine.
    """

    @abstractmethod
    def predict(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch or list of PIL images.
        Returns list of structured prediction dictionaries.
        """
        pass

    @abstractmethod
    def extract_features(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract latent representations/embeddings for feature drift analysis.
        Returns numpy array of shape (N, feature_dim).
        """
        pass

    @abstractmethod
    def evaluate(self, predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate predictions against ground-truth labels.
        Returns dictionary of metrics (accuracy, precision, recall, f1, mAP, etc.).
        """
        pass
