from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "YAMNet training requires `tensorflow` and `tensorflow_hub`. "
        "Install them first, for example from YAMNet_XGBoost/requirements.txt."
    ) from exc

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This trainer requires `xgboost`. Install it first, for example from "
        "YAMNet_XGBoost/requirements.txt."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YAMNet embeddings -> XGBoost classifier.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-mode", type=str, default="machine_status", choices=["status", "machine_status", "machine_id_status"])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--xgb-estimators", type=int, default=200)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class Logger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log_path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        tqdm.write(line)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def make_label_name(machine_type: str, machine_id: str, status: str, label_mode: str) -> str:
    if label_mode == "status":
        return status
    if label_mode == "machine_status":
        return f"{status}_{machine_type}"
    if label_mode == "machine_id_status":
        return f"{status}_{machine_type}_{machine_id}"
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def scan_examples(data_root: Path, label_mode: str) -> tuple[list[dict], list[str]]:
    machine_types: list[str] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("id_*/*/*.wav")):
            machine_types.append(child.name)
    pending: list[dict] = []
    label_names: set[str] = set()

    for machine_type in machine_types:
        base = data_root / machine_type
        if not base.exists():
            continue

        for wav_path in sorted(base.rglob("*.wav")):
            rel = wav_path.relative_to(data_root)
            parts = rel.parts
            if len(parts) < 4:
                continue

            status = parts[2]
            label_name = make_label_name(parts[0], parts[1], status, label_mode)
            stratify_key = f"{parts[0]}|{parts[1]}|{parts[2]}"
            pending.append(
                {
                    "path": wav_path,
                    "machine_type": parts[0],
                    "machine_id": parts[1],
                    "status": status,
                    "label_name": label_name,
                    "stratify_key": stratify_key,
                }
            )
            label_names.add(label_name)

    ordered_labels = sorted(label_names)
    label_to_id = {name: idx for idx, name in enumerate(ordered_labels)}
    examples = []
    for item in pending:
        item = dict(item)
        item["label"] = label_to_id[item["label_name"]]
        examples.append(item)
    return examples, ordered_labels


def choose_stratify_labels(examples: list[dict]) -> list[str] | None:
    stratify_counts = Counter(item["stratify_key"] for item in examples)
    if all(count >= 2 for count in stratify_counts.values()):
        return [item["stratify_key"] for item in examples]

    label_counts = Counter(item["label_name"] for item in examples)
    if all(count >= 2 for count in label_counts.values()):
        return [item["label_name"] for item in examples]

    return None


def format_counts(examples: list[dict]) -> str:
    counts = Counter(item["label_name"] for item in examples)
    return " ".join(f"{label}={counts[label]}" for label in sorted(counts))


def load_waveform(path: Path, sample_rate: int, duration_seconds: float) -> np.ndarray:
    sr, data = wavfile.read(path)
    if sr != sample_rate:
        raise ValueError(f"Unexpected sample rate for {path}: {sr}, expected {sample_rate}")

    if data.ndim > 1:
        data = data[:, 0]

    waveform = data.astype(np.float32) / np.iinfo(np.int16).max
    target_samples = int(sample_rate * duration_seconds)
    if waveform.shape[0] < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.shape[0]))
    else:
        waveform = waveform[:target_samples]
    return waveform


def extract_embeddings(
    yamnet_model,
    examples: list[dict],
    sample_rate: int,
    duration_seconds: float,
    logger: Logger,
    phase: str,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings_list: list[np.ndarray] = []
    labels: list[int] = []

    progress = tqdm(examples, desc=f"extract [{phase}]", dynamic_ncols=True, leave=False)
    for item in progress:
        waveform = load_waveform(item["path"], sample_rate=sample_rate, duration_seconds=duration_seconds)
        scores, embeddings, spectrogram = yamnet_model(waveform)
        clip_embedding = tf.reduce_mean(embeddings, axis=0).numpy().astype(np.float32)
        embeddings_list.append(clip_embedding)
        labels.append(item["label"])

    x = np.stack(embeddings_list, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    logger.log(f"{phase}_embeddings shape={tuple(x.shape)}")
    return x, y


def make_classifier(args: argparse.Namespace, num_classes: int):
    objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
    classifier = xgb.XGBClassifier(
        objective=objective,
        num_class=None if num_classes == 2 else num_classes,
        n_estimators=args.xgb_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        reg_lambda=args.xgb_reg_lambda,
        random_state=args.seed,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=1,
    )
    return classifier


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.logs_dir / f"yamnet_xgboost_{timestamp}.txt"
    logger = Logger(log_path)

    logger.log("device=cpu_or_tensorflow_runtime")
    logger.log(f"yamnet_handle={YAMNET_HANDLE}")

    examples, label_names = scan_examples(args.data_root, label_mode=args.label_mode)
    if not examples:
        raise RuntimeError("No wav files found in the expected fan/pump/slider folders.")

    if args.max_files is not None:
        rng = random.Random(args.seed)
        rng.shuffle(examples)
        examples = examples[:args.max_files]
        logger.log(f"max_files_applied={args.max_files}")

    stratify_labels = choose_stratify_labels(examples)
    train_examples, val_examples = train_test_split(
        examples,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=stratify_labels,
    )

    logger.log(
        f"dataset_summary total={len(examples)} train={len(train_examples)} val={len(val_examples)} "
        f"label_mode={args.label_mode} num_classes={len(label_names)}"
    )
    logger.log(f"labels={' | '.join(label_names)}")
    logger.log(f"train_label_counts {format_counts(train_examples)}")
    logger.log(f"val_label_counts {format_counts(val_examples)}")
    logger.log(
        "training_config "
        f"xgb_estimators={args.xgb_estimators} xgb_max_depth={args.xgb_max_depth} "
        f"xgb_learning_rate={args.xgb_learning_rate} dry_run={args.dry_run}"
    )

    tf.keras.backend.clear_session()
    start_load = time.perf_counter()
    yamnet_model = hub.load(YAMNET_HANDLE)
    logger.log(f"yamnet_loaded_seconds={time.perf_counter() - start_load:.1f}")

    x_train, y_train = extract_embeddings(
        yamnet_model, train_examples, sample_rate=args.sample_rate, duration_seconds=args.duration_seconds, logger=logger, phase="train"
    )
    x_val, y_val = extract_embeddings(
        yamnet_model, val_examples, sample_rate=args.sample_rate, duration_seconds=args.duration_seconds, logger=logger, phase="val"
    )

    classifier = make_classifier(args, num_classes=len(label_names))

    start_fit = time.perf_counter()
    classifier.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    fit_seconds = time.perf_counter() - start_fit

    val_pred = classifier.predict(x_val)
    val_acc = accuracy_score(y_val, val_pred)
    logger.log(f"fit_seconds={fit_seconds:.1f} val_acc={val_acc:.4f}")
    logger.log("classification_report_start")
    for line in classification_report(y_val, val_pred, target_names=label_names, zero_division=0).splitlines():
        logger.log(line)
    logger.log("classification_report_end")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"yamnet_xgboost_{timestamp}.json"
    meta_path = args.output_dir / f"yamnet_xgboost_{timestamp}_labels.json"
    embed_path = args.output_dir / f"yamnet_xgboost_{timestamp}_yamnet.pkl"

    classifier.save_model(model_path)
    meta_path.write_text(
        json.dumps(
            {
                "label_names": label_names,
                "label_mode": args.label_mode,
                "sample_rate": args.sample_rate,
                "duration_seconds": args.duration_seconds,
                "yamnet_handle": YAMNET_HANDLE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with embed_path.open("wb") as handle:
        pickle.dump({"yamnet_handle": YAMNET_HANDLE}, handle)

    logger.log(f"model_path={model_path}")
    logger.log(f"labels_path={meta_path}")
    logger.log(f"yamnet_meta_path={embed_path}")
    logger.log(f"log_file={log_path}")
    logger.close()


if __name__ == "__main__":
    main()
