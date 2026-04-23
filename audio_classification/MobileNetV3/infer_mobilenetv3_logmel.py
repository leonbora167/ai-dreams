from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from torchvision.models import mobilenet_v3_small

from training_common import LogMelTransform


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def latest_artifact_dir(root: Path) -> Path:
    candidates = [p for p in root.glob("mobilenetv3_logmel_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No artifact directories found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def create_model(num_classes: int) -> torch.nn.Module:
    model = mobilenet_v3_small(weights=None)
    classifier_in = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(classifier_in, num_classes)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained MobileNetV3 log-mel model.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir or latest_artifact_dir(ARTIFACTS_DIR)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    transform = LogMelTransform(
        sample_rate=metadata["sample_rate"],
        duration_seconds=metadata["duration_seconds"],
        n_fft=metadata["n_fft"],
        hop_length=metadata["hop_length"],
        win_length=metadata["win_length"],
        n_mels=metadata["n_mels"],
        f_min=metadata["f_min"],
        f_max=metadata["f_max"],
    )

    sample_rate, data = wavfile.read(args.audio_path)
    if sample_rate != metadata["sample_rate"]:
        raise ValueError(f"Expected sample rate {metadata['sample_rate']}, got {sample_rate}")
    if data.ndim > 1:
        data = data[:, 0]
    waveform = data.astype(np.float32) / np.iinfo(np.int16).max
    log_mel = transform(waveform)
    image = log_mel.unsqueeze(0).unsqueeze(0)
    image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
    image = image.squeeze(0).repeat(3, 1, 1).unsqueeze(0)

    model = create_model(len(metadata["label_names"]))
    state = torch.load(artifact_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    print(f"artifact_dir={artifact_dir}")
    print(f"predicted_label={metadata['label_names'][pred_idx]}")
    print(f"predicted_index={pred_idx}")
    print("probabilities=")
    for idx, label in enumerate(metadata["label_names"]):
        print(f"  {label}: {float(probs[idx]):.6f}")


if __name__ == "__main__":
    main()
