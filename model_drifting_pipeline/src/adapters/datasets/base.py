from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from PIL import Image

class BaseDatasetAdapter(ABC):
    """
    Abstract Base Class for Dataset Adapters.
    Encapsulates image loading, annotation extraction, and validation.
    """

    @abstractmethod
    def load(self, path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Loads dataset items. Each item is a dictionary containing:
        - "image": PIL.Image
        - "image_path": str
        - "annotations": Dict or List (optional, ground truth)
        - "metadata": Dict (size, format, etc.)
        """
        pass

    @abstractmethod
    def validate(self, path: str) -> Dict[str, Any]:
        """
        Validates the integrity of the dataset (corrupt images, missing labels, formats).
        """
        pass

    @abstractmethod
    def get_labels(self, items: List[Dict[str, Any]]) -> List[Any]:
        """
        Extract ground truth labels from items. Returns empty list if unlabelled.
        """
        pass
