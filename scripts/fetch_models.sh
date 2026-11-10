#!/usr/bin/env bash
# Downloads the official YOLOX COCO checkpoints and exports ONNX. Idempotent.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p models
BASE="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
get() { curl -fsSL --retry 6 --retry-all-errors --retry-delay 5 -o "$2.part" "$1" && mv "$2.part" "$2"; }
if python -c "import torch" 2>/dev/null; then HAVE_TORCH=1; else HAVE_TORCH=0
  echo "[fetch] PyTorch not installed: using the official YOLOX ONNX files (same detections as our export)"; fi
for v in ${VARIANTS:-yolox_nano yolox_tiny yolox_s}; do
  if [ "$HAVE_TORCH" = 1 ]; then
    [ -s "models/$v.pth" ] || { echo "[fetch] $v.pth"; get "$BASE/$v.pth" "models/$v.pth"; }
    [ -s "models/$v.onnx" ] || python scripts/export_onnx.py --variant "$v" --weights "models/$v.pth" --out "models/$v.onnx"
  else
    [ -s "models/$v.onnx" ] || { echo "[fetch] $v.onnx (official)"; get "$BASE/$v.onnx" "models/$v.onnx"; }
  fi
done
ls -la models/
