from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from transformers import Wav2Vec2ForSequenceClassification


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def latest_artifact_dir(root: Path) -> Path:
    candidates = [p for p in root.glob("wav2vec2_seqcls_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No artifact directories found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained Wav2Vec2 sequence classifier.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir or latest_artifact_dir(ARTIFACTS_DIR)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

    sample_rate, data = wavfile.read(args.audio_path)
    if sample_rate != metadata["sample_rate"]:
        raise ValueError(f"Expected sample rate {metadata['sample_rate']}, got {sample_rate}")
    if data.ndim > 1:
        data = data[:, 0]
    waveform = data.astype(np.float32) / np.iinfo(np.int16).max
    target_samples = int(metadata["sample_rate"] * metadata["duration_seconds"])
    length = min(waveform.shape[0], target_samples)
    if waveform.shape[0] < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.shape[0]))
    else:
        waveform = waveform[:target_samples]

    input_values = torch.from_numpy(waveform).float().unsqueeze(0)
    attention_mask = torch.zeros((1, target_samples), dtype=torch.long)
    attention_mask[:, :length] = 1

    model = Wav2Vec2ForSequenceClassification.from_pretrained(artifact_dir / "hf_model")
    model.eval()
    with torch.no_grad():
        outputs = model(input_values=input_values, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    print(f"artifact_dir={artifact_dir}")
    print(f"predicted_label={metadata['label_names'][pred_idx]}")
    print(f"predicted_index={pred_idx}")
    print("probabilities=")
    for idx, label in enumerate(metadata["label_names"]):
        print(f"  {label}: {float(probs[idx]):.6f}")


if __name__ == "__main__":
    main()
