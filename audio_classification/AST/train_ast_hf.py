from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import ASTForAudioClassification

from training_common import train_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Hugging Face AST model on log-mel spectrograms.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--model-name", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--f-min", type=float, default=20.0)
    parser.add_argument("--f-max", type=float, default=8000.0)
    parser.add_argument("--ast-max-length", type=int, default=1024)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument(
        "--label-mode",
        type=str,
        default="machine_status",
        choices=["status", "machine_status", "machine_id_status"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_feature_adapter(ast_max_length: int):
    def adapt_features(log_mel: torch.Tensor) -> torch.Tensor:
        time_major = log_mel.transpose(0, 1)
        time_steps = time_major.shape[0]
        if time_steps < ast_max_length:
            time_major = F.pad(time_major, (0, 0, 0, ast_max_length - time_steps))
        else:
            time_major = time_major[:ast_max_length, :]
        return time_major

    return adapt_features


def make_model_builder(model_name: str):
    def build_model(num_classes: int, label_names: list[str]) -> ASTForAudioClassification:
        id2label = {idx: label for idx, label in enumerate(label_names)}
        label2id = {label: idx for idx, label in enumerate(label_names)}
        return ASTForAudioClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    return build_model


def main() -> None:
    args = parse_args()

    def save_model_artifact(model, artifact_dir: Path, label_names, args) -> None:
        model.save_pretrained(artifact_dir / "hf_model")

    train_model(
        model_name="ast_hf",
        model_builder=make_model_builder(args.model_name),
        feature_adapter=make_feature_adapter(args.ast_max_length),
        model_saver=save_model_artifact,
        args=args,
    )


if __name__ == "__main__":
    main()
