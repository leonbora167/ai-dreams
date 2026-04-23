from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


@dataclass
class Example:
    path: Path
    machine_type: str
    machine_id: str
    status: str
    label_name: str
    label: int
    stratify_key: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_machine_types(data_root: Path) -> list[str]:
    machine_types: list[str] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("id_*/*/*.wav")):
            machine_types.append(child.name)
    return machine_types


def make_label_name(machine_type: str, machine_id: str, status: str, label_mode: str) -> str:
    if label_mode == "status":
        return status
    if label_mode == "machine_status":
        return f"{status}_{machine_type}"
    if label_mode == "machine_id_status":
        return f"{status}_{machine_type}_{machine_id}"
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def scan_examples(data_root: Path, label_mode: str) -> tuple[list[Example], list[str]]:
    machine_types = discover_machine_types(data_root)
    pending: list[dict[str, str | Path]] = []
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
    examples = [
        Example(
            path=item["path"],
            machine_type=item["machine_type"],
            machine_id=item["machine_id"],
            status=item["status"],
            label_name=item["label_name"],
            label=label_to_id[item["label_name"]],
            stratify_key=item["stratify_key"],
        )
        for item in pending
    ]
    return examples, ordered_labels


def choose_stratify_labels(examples: list[Example]) -> list[str] | None:
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex.stratify_key] = counts.get(ex.stratify_key, 0) + 1

    if all(count >= 2 for count in counts.values()):
        return [ex.stratify_key for ex in examples]

    label_counts: dict[str, int] = {}
    for ex in examples:
        label_counts[ex.label_name] = label_counts.get(ex.label_name, 0) + 1

    if all(count >= 2 for count in label_counts.values()):
        return [ex.label_name for ex in examples]

    return None


def summarize_examples(examples: Iterable[Example]) -> dict[str, int]:
    summary: dict[str, int] = {"total": 0}
    for ex in examples:
        summary["total"] += 1
        summary[ex.label_name] = summary.get(ex.label_name, 0) + 1
    return summary


def format_label_summary(summary: dict[str, int]) -> str:
    items = [f"{k}={v}" for k, v in sorted(summary.items()) if k != "total"]
    return " ".join(items)


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


def gpu_memory_report(device: torch.device) -> str:
    if device.type != "cuda":
        return "gpu_memory_mb=0 reserved_mb=0 max_allocated_mb=0"

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return (
        f"gpu_memory_mb={allocated:.1f} "
        f"reserved_mb={reserved:.1f} "
        f"max_allocated_mb={max_allocated:.1f}"
    )


class Wav2Vec2Dataset(Dataset):
    def __init__(self, examples: list[Example], sample_rate: int, duration_seconds: float) -> None:
        self.examples = examples
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * duration_seconds)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        sample_rate, data = wavfile.read(example.path)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Unexpected sample rate for {example.path}: {sample_rate}, expected {self.sample_rate}"
            )

        if data.ndim > 1:
            data = data[:, 0]

        waveform = torch.from_numpy(data.astype(np.float32) / np.iinfo(np.int16).max)
        length = min(waveform.numel(), self.target_samples)
        if waveform.numel() < self.target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_samples - waveform.numel()))
        else:
            waveform = waveform[: self.target_samples]

        attention_mask = torch.zeros(self.target_samples, dtype=torch.long)
        attention_mask[:length] = 1

        return {
            "input_values": waveform,
            "attention_mask": attention_mask,
            "labels": torch.tensor(example.label, dtype=torch.long),
        }


def collate_waveform_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_values": torch.stack([item["input_values"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def build_dataloaders(
    *,
    data_root: Path,
    label_mode: str,
    sample_rate: int,
    duration_seconds: float,
    val_size: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_files: int | None,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, list[Example], list[Example], list[str]]:
    examples, label_names = scan_examples(data_root, label_mode=label_mode)
    if not examples:
        raise RuntimeError("No wav files found in the expected fan/pump/slider folders.")

    if max_files is not None:
        rng = random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:max_files]

    stratify_labels = choose_stratify_labels(examples)
    train_examples, val_examples = train_test_split(
        examples,
        test_size=val_size,
        random_state=seed,
        stratify=stratify_labels,
    )

    train_dataset = Wav2Vec2Dataset(train_examples, sample_rate=sample_rate, duration_seconds=duration_seconds)
    val_dataset = Wav2Vec2Dataset(val_examples, sample_rate=sample_rate, duration_seconds=duration_seconds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_waveform_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_waveform_batch,
    )
    return train_loader, val_loader, train_examples, val_examples, label_names


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    phase: str,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    running_correct = 0
    total = 0

    progress = tqdm(loader, desc=f"epoch {epoch} [{phase}]", leave=False, dynamic_ncols=True)

    for batch in progress:
        inputs = {
            "input_values": batch["input_values"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
            "labels": batch["labels"].to(device, non_blocking=True),
        }

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(**inputs)
            loss = outputs.loss
            logits = outputs.logits
            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs["labels"].size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == inputs["labels"]).sum().item()
        total += inputs["labels"].size(0)

        avg_loss = running_loss / max(total, 1)
        avg_acc = running_correct / max(total, 1)
        progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    return running_loss / max(total, 1), running_correct / max(total, 1)


def train_model(
    *,
    model_name: str,
    model_builder,
    model_saver=None,
    args,
) -> Path:
    set_seed(args.seed)

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.logs_dir / f"{model_name}_{timestamp}.txt"
    logger = Logger(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"device={device}")
    if device.type == "cuda":
        logger.log(f"gpu_name={torch.cuda.get_device_name(0)}")

    train_loader, val_loader, train_examples, val_examples, label_names = build_dataloaders(
        data_root=args.data_root,
        label_mode=args.label_mode,
        sample_rate=args.sample_rate,
        duration_seconds=args.duration_seconds,
        val_size=args.val_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_files=args.max_files,
        device=device,
    )

    train_summary = summarize_examples(train_examples)
    val_summary = summarize_examples(val_examples)
    logger.log(
        "dataset_summary "
        f"total={len(train_examples) + len(val_examples)} train={len(train_examples)} val={len(val_examples)} "
        f"label_mode={args.label_mode} num_classes={len(label_names)}"
    )
    logger.log(f"labels={' | '.join(label_names)}")
    logger.log(f"train_label_counts {format_label_summary(train_summary)}")
    logger.log(f"val_label_counts {format_label_summary(val_summary)}")
    logger.log(
        "training_config "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.learning_rate} "
        f"sample_rate={args.sample_rate} duration_seconds={args.duration_seconds} "
        f"freeze_feature_encoder={args.freeze_feature_encoder} dry_run={args.dry_run}"
    )
    if args.max_files is not None:
        logger.log(f"max_files_applied={args.max_files}")

    model = model_builder(len(label_names), label_names).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    param_count = sum(p.numel() for p in model.parameters())
    logger.log(f"model={model_name} num_parameters={param_count}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    max_epochs = 1 if args.dry_run else args.epochs
    best_val_acc = -math.inf

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
            phase="train",
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            epoch=epoch,
            phase="val",
        )

        epoch_seconds = time.perf_counter() - epoch_start
        logger.log(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"epoch_seconds={epoch_seconds:.1f} {gpu_memory_report(device)}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if args.dry_run:
            logger.log("dry_run_complete=True")
            break

    logger.log(f"best_val_acc={best_val_acc:.4f}")
    if hasattr(args, "output_dir"):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = args.output_dir / f"{model_name}_{timestamp}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_name": model_name,
            "label_names": label_names,
            "label_mode": args.label_mode,
            "sample_rate": args.sample_rate,
            "duration_seconds": args.duration_seconds,
            "freeze_feature_encoder": getattr(args, "freeze_feature_encoder", False),
            "hf_model_name": getattr(args, "model_name", None),
        }
        (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if model_saver is not None:
            model_saver(model, artifact_dir, label_names, args)
        logger.log(f"artifact_dir={artifact_dir}")
    logger.log(f"log_file={log_path}")
    logger.close()
    return log_path
