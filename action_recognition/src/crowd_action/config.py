from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    manifest_path: str
    aux_dir: Optional[str] = None
    class_names: List[str] = field(default_factory=list)
    split_train: str = "train"
    split_val: str = "val"
    frames_per_clip: int = 20
    image_size: int = 224
    num_workers: int = 4


@dataclass
class TrainConfig:
    batch_size: int = 2
    grad_accum_steps: int = 1
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    mixed_precision: bool = True
    grad_clip_norm: float = 1.0
    gpu_memory_fraction: float = 0.95
    skip_oom_batches: bool = True
    seed: int = 42
    output_dir: str = "outputs"


@dataclass
class ModelConfig:
    num_classes: int = 0
    dropout: float = 0.2
    flow_weight: float = 1.0
    crowd_weight: float = 1.0


@dataclass
class AppConfig:
    data: DataConfig
    train: TrainConfig
    model: ModelConfig


def load_config(path: str) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(
        data=DataConfig(**raw["data"]),
        train=TrainConfig(**raw["train"]),
        model=ModelConfig(**raw["model"]),
    )
