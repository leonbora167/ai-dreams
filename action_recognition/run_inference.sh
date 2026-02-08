#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage:"
  echo "  ./run_inference.sh <video_path> <checkpoint_path> <output_mp4> [config_path] [stride]"
  echo
  echo "Example:"
  echo "  ./run_inference.sh data_training/val/violent/cam1__96.mp4 runs/20260208_190708/best.pt runs/infer_cam1_96.mp4 configs/train_example.yaml 5"
  exit 1
fi

VIDEO_PATH="$1"
CHECKPOINT_PATH="$2"
OUTPUT_PATH="$3"
CONFIG_PATH="${4:-configs/train_example.yaml}"
STRIDE="${5:-5}"

python test.py \
  --video "$VIDEO_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --output "$OUTPUT_PATH" \
  --config "$CONFIG_PATH" \
  --stride "$STRIDE"
