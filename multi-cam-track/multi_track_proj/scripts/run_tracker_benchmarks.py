"""Benchmark comparison script for all integrated single-camera trackers.

Evaluates:
- ByteTrack
- OC-SORT
- Deep OC-SORT
- Strong-SORT
- DeepSort
- Norfair
- BoT-SORT
on PETS 2009 using RF-DETR (Nano) and Strategy 1 Pose Feature Extractor.
"""
import copy
import csv
from pathlib import Path
import subprocess
import sys
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "pets2009.yaml"
GT_PATH = BASE_DIR.parent / "data" / "pets2009" / "ground_truth" / "cam01_gt.json"
OUTPUT_ROOT = BASE_DIR / "outputs"
PYTHON_EXEC = sys.executable

TRACKERS = [
    ("bytetrack", "ByteTrack"),
    ("ocsort", "OC-SORT"),
    ("deepocsort", "Deep OC-SORT"),
    ("strongsort", "Strong-SORT"),
    ("deepsort", "DeepSort"),
    ("norfair", "Norfair"),
    ("botsort", "BoT-SORT"),
]


def run_benchmark():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # Ensure RF-DETR and Pose are active
    base_cfg.setdefault("detector", {})["name"] = "rfdetr"
    base_cfg["detector"]["model"] = "rf-detr-nano"
    base_cfg.setdefault("features", {}).setdefault("pose", {})["enabled"] = True

    results = []

    for tracker_id, tracker_label in TRACKERS:
        out_dir = OUTPUT_ROOT / f"pets2009_rfdetr_{tracker_id}"
        eval_csv = out_dir / "evaluation" / "summary_metrics.csv"

        # Check if bytetrack was already run as v1
        existing_v1_csv = OUTPUT_ROOT / "pets2009_rfdetr_v1" / "evaluation" / "summary_metrics.csv"
        if tracker_id == "bytetrack" and existing_v1_csv.exists() and not eval_csv.exists():
            print(f"\n[{tracker_label}] Found existing baseline run in pets2009_rfdetr_v1, linking results...")
            import shutil
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(OUTPUT_ROOT / "pets2009_rfdetr_v1", out_dir, dirs_exist_ok=True)

        if not eval_csv.exists():
            print(f"\n==================================================================")
            print(f"Running Sequential Pipeline with Tracker: {tracker_label} ({tracker_id})")
            print(f"Target directory: {out_dir}")
            print(f"==================================================================")

            # Create specific tracker config
            cfg = copy.deepcopy(base_cfg)
            cfg.setdefault("tracker", {})["name"] = tracker_id
            temp_config = BASE_DIR / "configs" / f"_temp_pets2009_{tracker_id}.yaml"
            with open(temp_config, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f)

            # 1. Run tracking pipeline
            cmd_run = [
                PYTHON_EXEC,
                str(BASE_DIR / "run.py"),
                "--config", str(temp_config),
                "--output-dir", str(out_dir)
            ]
            print(f"Executing: {' '.join(cmd_run)}")
            res_run = subprocess.run(cmd_run)
            if res_run.returncode != 0:
                print(f"Error: Tracking failed for {tracker_label}")
                if temp_config.exists():
                    temp_config.unlink()
                continue

            # 2. Run evaluation
            cmd_eval = [
                PYTHON_EXEC,
                str(BASE_DIR / "scripts" / "evaluate.py"),
                "--output-dir", str(out_dir),
                "--gt-path", str(GT_PATH),
                "--gt-camera", "cam01"
            ]
            print(f"Evaluating: {' '.join(cmd_eval)}")
            res_eval = subprocess.run(cmd_eval)
            if res_eval.returncode != 0:
                print(f"Error: Evaluation failed for {tracker_label}")

            if temp_config.exists():
                temp_config.unlink()

        # Parse metrics
        if eval_csv.exists():
            metrics = {"Tracker": tracker_label, "Tracker_ID": tracker_id}
            with open(eval_csv, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 2 and row[1] != "---":
                        metrics[row[0]] = row[1]
            results.append(metrics)
            print(f"Successfully loaded metrics for {tracker_label}: MOTA={metrics.get('mota')}, IDF1={metrics.get('idf1')}")

    # Output consolidated comparison CSV
    summary_path = BASE_DIR / "benchmark_trackers_comparison.csv"
    if results:
        headers = [
            "Tracker", "mota", "motp", "precision", "recall", "f1",
            "idsw", "idf1", "total_global_ids", "overlap_conflicts",
            "total_tracklets", "mean_tracklet_length_frames"
        ]
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        print(f"\nConsolidated Comparison CSV saved to: {summary_path}")

        # Print formatted Markdown table
        print("\n" + "=" * 80)
        print("BENCHMARK COMPARISON OF ALL TRACKERS ON PETS 2009 (RF-DETR + POSE)")
        print("=" * 80)
        print(f"| {'Tracker':<15} | {'MOTA':<8} | {'MOTP':<8} | {'Precision':<10} | {'Recall':<8} | {'IDF1':<8} | {'IDSW':<6} | {'Global IDs':<10} | {'Conflicts':<9} |")
        print(f"|{'-'*17}|{'-'*10}|{'-'*10}|{'-'*12}|{'-'*10}|{'-'*10}|{'-'*8}|{'-'*12}|{'-'*11}|")
        for r in results:
            def fmt_pct(val):
                try:
                    return f"{float(val)*100:.2f}%"
                except Exception:
                    return str(val)

            def fmt_flt(val):
                try:
                    return f"{float(val):.4f}"
                except Exception:
                    return str(val)

            print(
                f"| {r.get('Tracker', ''):<15} "
                f"| {fmt_pct(r.get('mota', '')):<8} "
                f"| {fmt_pct(r.get('motp', '')):<8} "
                f"| {fmt_pct(r.get('precision', '')):<10} "
                f"| {fmt_pct(r.get('recall', '')):<8} "
                f"| {fmt_pct(r.get('idf1', '')):<8} "
                f"| {r.get('idsw', ''):<6} "
                f"| {r.get('total_global_ids', ''):<10} "
                f"| {r.get('overlap_conflicts', ''):<9} |"
            )
        print("=" * 80 + "\n")


if __name__ == "__main__":
    run_benchmark()

