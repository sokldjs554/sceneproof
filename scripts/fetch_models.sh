#!/usr/bin/env bash
# Downloads the official YOLOX COCO checkpoints and exports ONNX. Idempotent.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p models
BASE="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
for v in ${VARIANTS:-yolox_nano yolox_tiny yolox_s}; do
  [ -s "models/$v.pth" ] || { echo "[fetch] $v.pth"; curl -fsSL --retry 6 --retry-all-errors --retry-delay 5 -o "models/$v.pth" "$BASE/$v.pth"; }
  [ -s "models/$v.onnx" ] || python scripts/export_onnx.py --variant "$v" --weights "models/$v.pth" --out "models/$v.onnx"
done
ls -la models/
