from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def choose_stratify_labels(examples: list[Example]) -> list[str]:
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex.stratify_key] = counts.get(ex.stratify_key, 0) + 1

    if all(count >= 2 for count in counts.values()):
        return [ex.stratify_key for ex in examples]

    return [ex.label_name for ex in examples]


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


class LogMelTransform:
    def __init__(
        self,
        sample_rate: int,
        duration_seconds: float,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * duration_seconds)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self._mel_cache: dict[str, torch.Tensor] = {}
        self._window_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + freq / 700.0)

    @staticmethod
    def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _window(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._window_cache:
            self._window_cache[key] = torch.hann_window(self.win_length, device=device)
        return self._window_cache[key]

    def _mel_filter(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key in self._mel_cache:
            return self._mel_cache[key]

        n_freqs = self.n_fft // 2 + 1
        fft_freqs = torch.linspace(0, self.sample_rate / 2, n_freqs, device=device)
        mel_min = self._hz_to_mel(torch.tensor(self.f_min, device=device))
        mel_max = self._hz_to_mel(torch.tensor(self.f_max, device=device))
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2, device=device)
        hz_points = self._mel_to_hz(mel_points)

        filter_bank = torch.zeros(self.n_mels, n_freqs, device=device)
        for i in range(self.n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            left_slope = (fft_freqs - left) / (center - left + 1e-8)
            right_slope = (right - fft_freqs) / (right - center + 1e-8)
            filter_bank[i] = torch.clamp(torch.minimum(left_slope, right_slope), min=0.0)

        self._mel_cache[key] = filter_bank
        return filter_bank

    def __call__(self, waveform: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(waveform).float()
        if x.numel() < self.target_samples:
            x = F.pad(x, (0, self.target_samples - x.numel()))
        else:
            x = x[: self.target_samples]

        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window(x.device),
            return_complex=True,
        )
        power = stft.abs().pow(2.0)
        mel = self._mel_filter(x.device) @ power
        log_mel = 10.0 * torch.log10(mel + 1e-10)
        max_db = log_mel.max()
        log_mel = torch.maximum(log_mel, max_db - 80.0)
        mean = log_mel.mean()
        std = log_mel.std().clamp_min(1e-6)
        return (log_mel - mean) / std


class MIMIIDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        transform: LogMelTransform,
        feature_adapter: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.examples = examples
        self.transform = transform
        self.feature_adapter = feature_adapter

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        sample_rate, data = wavfile.read(example.path)
        if sample_rate != self.transform.sample_rate:
            raise ValueError(
                f"Unexpected sample rate for {example.path}: {sample_rate}, expected {self.transform.sample_rate}"
            )

        if data.ndim > 1:
            data = data[:, 0]

        waveform = data.astype(np.float32) / np.iinfo(np.int16).max
        features = self.transform(waveform)
        features = self.feature_adapter(features)
        label = torch.tensor(example.label, dtype=torch.long)
        return features, label


def build_dataloaders(
    *,
    data_root: Path,
    label_mode: str,
    feature_adapter: Callable[[torch.Tensor], torch.Tensor],
    sample_rate: int,
    duration_seconds: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    f_min: float,
    f_max: float,
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

    transform = LogMelTransform(
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )

    train_dataset = MIMIIDataset(train_examples, transform, feature_adapter)
    val_dataset = MIMIIDataset(val_examples, transform, feature_adapter)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return train_loader, val_loader, train_examples, val_examples, label_names


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
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

    progress = tqdm(
        loader,
        desc=f"epoch {epoch} [{phase}]",
        leave=False,
        dynamic_ncols=True,
    )

    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(logits, targets)
            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == targets).sum().item()
        total += targets.size(0)

        avg_loss = running_loss / max(total, 1)
        avg_acc = running_correct / max(total, 1)
        progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    return running_loss / max(total, 1), running_correct / max(total, 1)


def train_model(
    *,
    model_name: str,
    model_builder: Callable[[int, list[str]], nn.Module],
    feature_adapter: Callable[[torch.Tensor], torch.Tensor],
    model_saver: Callable[[nn.Module, Path, list[str], object], None] | None = None,
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
        feature_adapter=feature_adapter,
        sample_rate=args.sample_rate,
        duration_seconds=args.duration_seconds,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        f_min=args.f_min,
        f_max=args.f_max,
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
        f"n_mels={args.n_mels} n_fft={args.n_fft} hop_length={args.hop_length} "
        f"num_workers={args.num_workers} dry_run={args.dry_run}"
    )
    if args.max_files is not None:
        logger.log(f"max_files_applied={args.max_files}")

    model = model_builder(len(label_names), label_names).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

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
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
            phase="train",
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
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
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "win_length": args.win_length,
            "n_mels": args.n_mels,
            "f_min": args.f_min,
            "f_max": args.f_max,
        }
        (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if model_saver is not None:
            model_saver(model, artifact_dir, label_names, args)
        logger.log(f"artifact_dir={artifact_dir}")
    logger.log(f"log_file={log_path}")
    logger.close()
    return log_path
