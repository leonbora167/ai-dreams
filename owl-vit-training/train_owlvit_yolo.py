from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from src.owlvit_dataset import (
    OwlViTYoloDataset,
    collate_fn,
    default_coco8_config,
    load_dataset_config,
    owlvit_detection_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_text_tower(model: OwlViTForObjectDetection) -> None:
    for parameter in model.owlvit.text_model.parameters():
        parameter.requires_grad = False
    for parameter in model.owlvit.text_projection.parameters():
        parameter.requires_grad = False


def move_batch_to_device(batch: dict, device: torch.device) -> tuple[dict, list[dict]]:
    targets = batch.pop("targets")
    model_inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    return model_inputs, targets


def run_epoch(model, loader, optimizer, device, train: bool) -> dict[str, float]:
    model.train(train)
    running = {"loss": 0.0, "loss_cls": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0, "matched_objects": 0.0}
    steps = 0

    for batch in loader:
        model_inputs, targets = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(train):
            outputs = model(**model_inputs)
            loss, metrics = owlvit_detection_loss(outputs, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        for key in running:
            running[key] += metrics[key]
        steps += 1

    if steps == 0:
        return running
    return {key: value / steps for key, value in running.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune OWL-ViT on a YOLO-style detection dataset.")
    parser.add_argument(
        "--dataset-config",
        default="data/coco8/dataset.yaml",
        help="YAML config with dataset paths and class names.",
    )
    parser.add_argument("--model-name", default="google/owlvit-base-patch32", help="HF model checkpoint.")
    parser.add_argument("--output-dir", default="artifacts/owlvit-yolo", help="Where checkpoints and logs go.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=0, help="PyTorch dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--freeze-text", action="store_true", help="Freeze the OWL-ViT text tower.")
    parser.add_argument(
        "--tensorboard-dir",
        default="runs/owlvit-yolo",
        help="Base directory for TensorBoard event files.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_run_dir = Path(args.tensorboard_dir) / run_name
    writer = SummaryWriter(log_dir=str(tb_run_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"TensorBoard run dir: {tb_run_dir}")

    dataset_config_path = Path(args.dataset_config)
    if dataset_config_path.exists():
        dataset_config = load_dataset_config(dataset_config_path)
    else:
        dataset_config = default_coco8_config()
        print(f"Dataset config not found at {dataset_config_path}, falling back to COCO8 defaults.")

    processor = OwlViTProcessor.from_pretrained(args.model_name)
    model = OwlViTForObjectDetection.from_pretrained(args.model_name)
    if args.freeze_text:
        freeze_text_tower(model)
    model.to(device)

    train_dataset = OwlViTYoloDataset(dataset_config, split="train")
    val_dataset = OwlViTYoloDataset(dataset_config, split="val")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": lambda batch: collate_fn(batch, processor),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    best_val_loss = float("inf")
    best_path = output_dir / "best.pt"

    writer.add_text("run/model_name", args.model_name)
    writer.add_text("run/dataset_root", str(dataset_config.root))
    writer.add_text("run/dataset_config", args.dataset_config)
    writer.add_text("run/device", str(device))
    writer.add_text("run/output_dir", str(output_dir))
    writer.add_text("run/freeze_text", str(args.freeze_text))
    writer.add_text("run/run_name", run_name)
    writer.add_text("run/class_names", ", ".join(dataset_config.class_names))
    writer.add_text("run/prompt_template", dataset_config.prompt_template)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"val loss={val_metrics['loss']:.4f} "
            f"val cls={val_metrics['loss_cls']:.4f} "
            f"val bbox={val_metrics['loss_bbox']:.4f} "
            f"val giou={val_metrics['loss_giou']:.4f}"
        )

        writer.add_scalar("loss/train_total", train_metrics["loss"], epoch)
        writer.add_scalar("loss/val_total", val_metrics["loss"], epoch)
        writer.add_scalar("loss/train_cls", train_metrics["loss_cls"], epoch)
        writer.add_scalar("loss/val_cls", val_metrics["loss_cls"], epoch)
        writer.add_scalar("loss/train_bbox", train_metrics["loss_bbox"], epoch)
        writer.add_scalar("loss/val_bbox", val_metrics["loss_bbox"], epoch)
        writer.add_scalar("loss/train_giou", train_metrics["loss_giou"], epoch)
        writer.add_scalar("loss/val_giou", val_metrics["loss_giou"], epoch)
        writer.add_scalar("matching/train_objects", train_metrics["matched_objects"], epoch)
        writer.add_scalar("matching/val_objects", val_metrics["matched_objects"], epoch)
        writer.add_scalar("optim/learning_rate", optimizer.param_groups[0]["lr"], epoch)

        checkpoint = {
            "epoch": epoch,
            "model_name": args.model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "args": vars(args),
            "tensorboard_run_dir": str(tb_run_dir),
            "class_names": dataset_config.class_names,
            "text_queries": dataset_config.text_queries,
            "prompt_template": dataset_config.prompt_template,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, best_path)
            processor.save_pretrained(output_dir / "processor")
            writer.add_scalar("best/val_loss", best_val_loss, epoch)

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    writer.add_hparams(
        {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "freeze_text": int(args.freeze_text),
            "seed": args.seed,
        },
        {
            "hparam/best_val_loss": best_val_loss,
            "hparam/final_train_loss": history[-1]["train"]["loss"],
            "hparam/final_val_loss": history[-1]["val"]["loss"],
        },
    )
    writer.close()
    print(f"Training finished. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
