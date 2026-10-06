from .base import BaseModelAdapter
from .inception_adapter import InceptionAdapter
from .rf_detr_adapter import RFDETRAdapter

def get_model_adapter(model_name: str, **kwargs) -> BaseModelAdapter:
    name_clean = model_name.lower().replace("-", "_")
    if "inception" in name_clean:
        return InceptionAdapter(**kwargs)
    elif "rf_detr" in name_clean or "detr" in name_clean or "detection" in name_clean:
        return RFDETRAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported: 'inception_v3', 'rf_detr'")
