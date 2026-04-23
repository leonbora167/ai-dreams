from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

try:
    import tensorflow_hub as hub
except ImportError as exc:  # pragma: no cover
    raise ImportError("YAMNet inference requires tensorflow_hub.") from exc

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    raise ImportError("YAMNet inference requires xgboost.") from exc


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def latest_json_model(root: Path) -> Path:
    candidates = [p for p in root.glob("yamnet_xgboost_*.json") if p.is_file() and not p.name.endswith("_labels.json")]
    if not candidates:
        raise FileNotFoundError(f"No XGBoost model files found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained YAMNet -> XGBoost model.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path or latest_json_model(ARTIFACTS_DIR)
    stem = model_path.stem
    labels_path = model_path.with_name(f"{stem}_labels.json")
    metadata = json.loads(labels_path.read_text(encoding="utf-8"))

    sample_rate, data = wavfile.read(args.audio_path)
    if sample_rate != metadata["sample_rate"]:
        raise ValueError(f"Expected sample rate {metadata['sample_rate']}, got {sample_rate}")
    if data.ndim > 1:
        data = data[:, 0]
    waveform = data.astype(np.float32) / np.iinfo(np.int16).max
    target_samples = int(metadata["sample_rate"] * metadata["duration_seconds"])
    if waveform.shape[0] < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.shape[0]))
    else:
        waveform = waveform[:target_samples]

    yamnet_model = hub.load(metadata["yamnet_handle"])
    scores, embeddings, spectrogram = yamnet_model(waveform)
    clip_embedding = np.mean(np.asarray(embeddings), axis=0, keepdims=True).astype(np.float32)

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    probs = model.predict_proba(clip_embedding)[0]
    pred_idx = int(np.argmax(probs))

    print(f"model_path={model_path}")
    print(f"predicted_label={metadata['label_names'][pred_idx]}")
    print(f"predicted_index={pred_idx}")
    print("probabilities=")
    for idx, label in enumerate(metadata["label_names"]):
        print(f"  {label}: {float(probs[idx]):.6f}")


if __name__ == "__main__":
    main()
