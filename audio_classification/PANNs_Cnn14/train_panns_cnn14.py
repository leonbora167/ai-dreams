from __future__ import annotations

import argparse
from pathlib import Path

import torch

from training_common import train_model
from panns_models import Cnn14


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PANNs Cnn14 on log-mel spectrograms.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=320)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--f-min", type=float, default=50.0)
    parser.add_argument("--f-max", type=float, default=8000.0)
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


def adapt_features(log_mel: torch.Tensor) -> torch.Tensor:
    return log_mel.transpose(0, 1).unsqueeze(0)


def main() -> None:
    args = parse_args()

    def save_model_artifact(model, artifact_dir: Path, label_names, args) -> None:
        torch.save(model.state_dict(), artifact_dir / "model.pt")

    train_model(
        model_name="panns_cnn14",
        model_builder=lambda num_classes, _label_names: Cnn14(num_classes=num_classes, mel_bins=args.n_mels),
        feature_adapter=adapt_features,
        model_saver=save_model_artifact,
        args=args,
    )


if __name__ == "__main__":
    main()
