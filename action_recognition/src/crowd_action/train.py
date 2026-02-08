import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from crowd_action.config import load_config
from crowd_action.data import build_dataloader
from crowd_action.models import MultiModalSwinClassifier
from crowd_action.utils import accuracy, seed_everything


def resolve_class_names(manifest_path: str, class_names: list[str]) -> list[str]:
    if class_names:
        return class_names
    df = pd.read_csv(manifest_path)
    inferred = sorted(df["label"].dropna().astype(str).unique().tolist())
    if not inferred:
        raise ValueError(f"No labels found in manifest: {manifest_path}")
    return inferred


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    train: bool,
    grad_clip_norm: float,
    grad_accum_steps: int,
    skip_oom_batches: bool,
    epoch: int,
    total_epochs: int,
):
    model.train(train)
    losses = []
    accs = []
    split_name = "train" if train else "val"
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"{split_name} epoch {epoch}/{total_epochs}",
        leave=True,
        mininterval=1.0,
        dynamic_ncols=True,
    )
    if train:
        optimizer.zero_grad(set_to_none=True)
    step = 0
    for step, batch in enumerate(progress, start=1):
        rgb = batch["rgb"].to(device, non_blocking=True)
        flow = batch["flow"].to(device, non_blocking=True)
        crowd = batch["crowd"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        try:
            with torch.set_grad_enabled(train):
                with torch.autocast(device_type=device.type, enabled=scaler is not None):
                    logits = model(rgb=rgb, flow=flow, crowd=crowd)
                    loss = criterion(logits, labels)
                if train:
                    loss_for_backward = loss / max(1, grad_accum_steps)
                    if scaler is not None:
                        scaler.scale(loss_for_backward).backward()
                        if step % grad_accum_steps == 0:
                            if grad_clip_norm > 0:
                                scaler.unscale_(optimizer)
                                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                    else:
                        loss_for_backward.backward()
                        if step % grad_accum_steps == 0:
                            if grad_clip_norm > 0:
                                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
        except RuntimeError as e:
            if skip_oom_batches and device.type == "cuda" and "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                progress.set_postfix(oom="1")
                continue
            raise

        losses.append(loss.item())
        accs.append(accuracy(logits.detach(), labels))
        progress.set_postfix(loss=f"{losses[-1]:.4f}", acc=f"{accs[-1]:.4f}")

    if train and step > 0 and (step % max(1, grad_accum_steps) != 0):
        if scaler is not None:
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if not losses:
        return float("nan"), float("nan")
    return sum(losses) / len(losses), sum(accs) / len(accs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(cfg.train.gpu_memory_fraction)

    base_output_dir = Path(cfg.train.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir={run_dir}")

    class_names = resolve_class_names(cfg.data.manifest_path, cfg.data.class_names)
    model_num_classes = cfg.model.num_classes or len(class_names)
    if model_num_classes != len(class_names):
        raise ValueError(
            f"num_classes={model_num_classes} does not match detected classes={len(class_names)}: {class_names}"
        )
    print(f"detected classes ({len(class_names)}): {class_names}")

    train_loader = build_dataloader(
        manifest_path=cfg.data.manifest_path,
        split=cfg.data.split_train,
        frames_per_clip=cfg.data.frames_per_clip,
        image_size=cfg.data.image_size,
        class_names=class_names,
        aux_dir=cfg.data.aux_dir,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
    )
    val_loader = build_dataloader(
        manifest_path=cfg.data.manifest_path,
        split=cfg.data.split_val,
        frames_per_clip=cfg.data.frames_per_clip,
        image_size=cfg.data.image_size,
        class_names=class_names,
        aux_dir=cfg.data.aux_dir,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )
    if len(train_loader.dataset) == 0:
        raise ValueError(f"No samples found for split '{cfg.data.split_train}' in {cfg.data.manifest_path}")
    if len(val_loader.dataset) == 0:
        raise ValueError(f"No samples found for split '{cfg.data.split_val}' in {cfg.data.manifest_path}")

    model = MultiModalSwinClassifier(
        num_classes=model_num_classes,
        dropout=cfg.model.dropout,
        flow_weight=cfg.model.flow_weight,
        crowd_weight=cfg.model.crowd_weight,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (cfg.train.mixed_precision and device.type == "cuda") else None

    best_val_acc = -1.0
    history_rows: list[dict[str, float | int]] = []
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            train=True,
            grad_clip_norm=cfg.train.grad_clip_norm,
            grad_accum_steps=cfg.train.grad_accum_steps,
            skip_oom_batches=cfg.train.skip_oom_batches,
            epoch=epoch,
            total_epochs=cfg.train.epochs,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            train=False,
            grad_clip_norm=0.0,
            grad_accum_steps=1,
            skip_oom_batches=cfg.train.skip_oom_batches,
            epoch=epoch,
            total_epochs=cfg.train.epochs,
        )
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = run_dir / "best.pt"
            torch.save({"model_state": model.state_dict()}, ckpt)
            print(f"saved checkpoint: {ckpt}")

    last_ckpt = run_dir / "last.pt"
    torch.save({"model_state": model.state_dict()}, last_ckpt)
    print(f"saved checkpoint: {last_ckpt}")

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history_rows)
    print(f"saved metrics: {metrics_path}")

    summary_path = run_dir / "summary.json"
    summary = {
        "run_dir": str(run_dir),
        "device": device.type,
        "epochs": cfg.train.epochs,
        "best_val_acc": best_val_acc,
        "class_names": class_names,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
