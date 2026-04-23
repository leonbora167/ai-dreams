from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def make_label_name(machine_type: str, machine_id: str, status: str, label_mode: str) -> str:
    if label_mode == "status":
        return status
    if label_mode == "machine_status":
        return f"{status}_{machine_type}"
    if label_mode == "machine_id_status":
        return f"{status}_{machine_type}_{machine_id}"
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def discover_dataset_roots(data_root: Path) -> list[Path]:
    roots: list[Path] = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("id_*/*/*.wav")):
            roots.append(child)
    return roots


def build_mapping(data_root: Path, label_mode: str) -> dict[str, int]:
    label_names: set[str] = set()
    for machine_root in discover_dataset_roots(data_root):
        machine_type = machine_root.name
        for wav_path in machine_root.glob("id_*/*/*.wav"):
            rel = wav_path.relative_to(data_root)
            parts = rel.parts
            machine_id = parts[1]
            status = parts[2]
            label_names.add(make_label_name(machine_type, machine_id, status, label_mode))

    ordered_labels = sorted(label_names)
    return {label: idx for idx, label in enumerate(ordered_labels)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create class mapping from dataset folders automatically.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--label-mode",
        type=str,
        default="machine_status",
        choices=["status", "machine_status", "machine_id_status"],
    )
    parser.add_argument("--output", type=Path, default=Path("class_mapping.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = build_mapping(args.data_root, args.label_mode)
    payload = {
        "data_root": str(args.data_root.resolve()),
        "label_mode": args.label_mode,
        "num_classes": len(mapping),
        "class_to_index": mapping,
        "index_to_class": {str(idx): label for label, idx in mapping.items()},
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved_mapping={args.output}")
    print(f"num_classes={len(mapping)}")
    print(f"classes={list(mapping.keys())}")


if __name__ == "__main__":
    main()
