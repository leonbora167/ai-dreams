#!/usr/bin/env bash
# Run this instead of a plain `pip install -r requirements.txt`.
# Order matters: torch and numpy must exist in the environment BEFORE
# torchreid is built, and torchreid must be built with --no-build-isolation
# or its setup.py fails with a false "no module named numpy" error.

set -e

echo "Step 1/3: installing PyTorch (edit the index-url below for your CUDA version)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

echo "Step 2/3: installing base requirements"
pip install -r requirements.txt

echo "Step 2.5/3: installing build-time deps torchreid's setup.py needs directly"
pip install Cython

echo "Step 3/3: installing torchreid (no build isolation)"
pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git

echo "Done."