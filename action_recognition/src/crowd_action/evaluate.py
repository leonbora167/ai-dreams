import argparse

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from crowd_action.config import load_config
from crowd_action.data import build_dataloader
from crowd_action.models import MultiModalSwinClassifier


def resolve_class_names(manifest_path: str, class_names: list[str]) -> list[str]:
    if class_names:
        return class_names
    df = pd.read_csv(manifest_path)
    inferred = sorted(df["label"].dropna().astype(str).unique().tolist())
    if not inferred:
        raise ValueError(f"No labels found in manifest: {manifest_path}")
    return inferred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    class_names = resolve_class_names(cfg.data.manifest_path, cfg.data.class_names)
    model_num_classes = cfg.model.num_classes or len(class_names)
    if model_num_classes != len(class_names):
        raise ValueError(
            f"num_classes={model_num_classes} does not match detected classes={len(class_names)}: {class_names}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_dataloader(
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

    model = MultiModalSwinClassifier(
        num_classes=model_num_classes,
        dropout=cfg.model.dropout,
        flow_weight=cfg.model.flow_weight,
        crowd_weight=cfg.model.crowd_weight,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            logits = model(
                rgb=batch["rgb"].to(device),
                flow=batch["flow"].to(device),
                crowd=batch["crowd"].to(device),
            )
            preds = logits.argmax(dim=1).cpu().tolist()
            labels = batch["label"].tolist()
            y_pred.extend(preds)
            y_true.extend(labels)
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=class_names))


if __name__ == "__main__":
    main()
