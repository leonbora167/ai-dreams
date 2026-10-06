from .base import BaseDatasetAdapter
from .pascal_voc_adapter import PascalVOCAdapter
from .cifar10_adapter import CIFAR10Adapter

def get_dataset_adapter(dataset_type: str, **kwargs) -> BaseDatasetAdapter:
    name_clean = dataset_type.lower().replace("-", "_")
    if "cifar" in name_clean or "classification" in name_clean:
        return CIFAR10Adapter(**kwargs)
    elif "voc" in name_clean or "pascal" in name_clean or "detection" in name_clean:
        return PascalVOCAdapter(**kwargs)
    else:
        # Default fallback to classification or general
        return CIFAR10Adapter(**kwargs)
