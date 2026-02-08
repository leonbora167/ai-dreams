import argparse
import hashlib
from pathlib import Path

import pandas as pd


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def stable_video_id(path: Path) -> str:
    return hashlib.md5(str(path).encode("utf-8")).hexdigest()


def infer_split(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "train" in parts:
        return "train"
    if "val" in parts or "valid" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return "train"


def infer_label(path: Path) -> str:
    parts = list(path.parts)
    lower_parts = [p.lower() for p in parts]
    for split_name in ("train", "val", "valid", "test"):
        if split_name in lower_parts:
            split_idx = lower_parts.index(split_name)
            if split_idx + 1 < len(parts) - 1:
                return parts[split_idx + 1]
    return path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.data_root)
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        rows.append(
            {
                "video_id": stable_video_id(p.resolve()),
                "video_path": str(p.resolve()),
                "label": infer_label(p),
                "split": infer_split(p),
            }
        )

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
