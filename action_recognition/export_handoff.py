import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def latest_run_dir(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    dirs = [d for d in runs_root.iterdir() if d.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def manifest_stats(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"exists": False}
    df = pd.read_csv(manifest_path)
    payload: dict[str, Any] = {"exists": True, "rows": int(len(df))}
    if "split" in df.columns:
        payload["split_counts"] = {k: int(v) for k, v in df["split"].value_counts().to_dict().items()}
    if "label" in df.columns:
        labels = sorted(df["label"].dropna().astype(str).unique().tolist())
        payload["class_names"] = labels
        payload["num_classes"] = len(labels)
    return payload


def run_artifacts(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {"latest_run_exists": False}
    out = {
        "latest_run_exists": True,
        "latest_run_dir": str(run_dir).replace("\\", "/"),
        "files_present": [],
    }
    expected = ["best.pt", "last.pt", "metrics.csv", "summary.json"]
    out["files_present"] = [name for name in expected if (run_dir / name).exists()]
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                out["summary"] = json.load(f)
        except Exception:
            out["summary"] = {"error": "failed_to_parse_summary"}
    return out


def build_context(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    manifest_path = root / data_cfg.get("manifest_path", "data_training/manifest.csv")
    runs_root = root / train_cfg.get("output_dir", "runs")
    latest = latest_run_dir(runs_root)

    return {
        "project": {
            "name": root.name,
            "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        },
        "config": {
            "path": safe_rel(config_path, root),
            "data": data_cfg,
            "train": train_cfg,
            "model": model_cfg,
        },
        "paths": {
            "root": str(root).replace("\\", "/"),
            "manifest": safe_rel(manifest_path, root),
            "runs_root": safe_rel(runs_root, root),
            "dataset_layout": "data_training/train/<class>/... and data_training/val/<class>/...",
        },
        "entrypoints": {
            "dataset_bootstrap": "python data.py",
            "full_pipeline": "python scripts/run_training.py",
            "train_only": f"python -m crowd_action.train --config {safe_rel(config_path, root)}",
            "evaluate": f"python -m crowd_action.evaluate --config {safe_rel(config_path, root)} --checkpoint runs/<timestamp>/best.pt",
            "inference_video": f"python test.py --video <input.mp4> --checkpoint runs/<timestamp>/best.pt --output <output.mp4> --config {safe_rel(config_path, root)}",
        },
        "manifest_stats": manifest_stats(manifest_path),
        "latest_run": run_artifacts(latest),
        "inference_contract": {
            "overlay": "Top-left green label + confidence in output mp4",
            "class_resolution_order": [
                "--class-names",
                "runs/<timestamp>/summary.json",
                "config class_names",
                "manifest label inference",
            ],
            "progress_bars": ["inferring clips", "writing output"],
        },
        "git_hygiene": {"ignored": ["data/", "data_training/", "runs/", "outputs/", "*.log"]},
    }


def merge_existing(existing_path: Path, new_context: dict[str, Any]) -> dict[str, Any]:
    if not existing_path.exists():
        return new_context
    try:
        with existing_path.open("r", encoding="utf-8") as f:
            old = json.load(f)
    except Exception:
        return new_context
    for key in ["major_changes_made_in_session", "known_operational_notes", "pending_todos", "recommended_resume_prompt_template"]:
        if key in old and key not in new_context:
            new_context[key] = old[key]
    return new_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_example.yaml")
    parser.add_argument("--output", type=str, default="SESSION_CONTEXT.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = (root / args.config).resolve()
    output_path = (root / args.output).resolve()

    context = build_context(root=root, config_path=config_path)
    context = merge_existing(output_path, context)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)
    print(f"wrote handoff context: {output_path}")


if __name__ == "__main__":
    main()
