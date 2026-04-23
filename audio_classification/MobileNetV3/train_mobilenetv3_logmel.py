from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

from training_common import train_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV3 on log-mel spectrograms.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=320)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--f-min", type=float, default=20.0)
    parser.add_argument("--f-max", type=float, default=8000.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument(
        "--label-mode",
        type=str,
        default="status",
        choices=["status", "machine_status", "machine_id_status"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def create_model(num_classes: int) -> nn.Module:
    model = mobilenet_v3_small(weights=None)
    classifier_in = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(classifier_in, num_classes)
    return model


def make_feature_adapter(image_size: int):
    def adapt_features(log_mel: torch.Tensor) -> torch.Tensor:
        image = log_mel.unsqueeze(0).unsqueeze(0)
        image = F.interpolate(
            image,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        return image.squeeze(0).repeat(3, 1, 1)

    return adapt_features


def main() -> None:
    args = parse_args()

    def save_model_artifact(model, artifact_dir: Path, label_names, args) -> None:
        torch.save(model.state_dict(), artifact_dir / "model.pt")

    train_model(
        model_name="mobilenetv3_logmel",
        model_builder=lambda num_classes, _label_names: create_model(num_classes=num_classes),
        feature_adapter=make_feature_adapter(args.image_size),
        model_saver=save_model_artifact,
        args=args,
    )


if __name__ == "__main__":
    main()
