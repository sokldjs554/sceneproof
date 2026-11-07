#!/usr/bin/env python
"""Export an official YOLOX checkpoint to ONNX (raw head outputs, decoded in numpy).

Usage: python scripts/export_onnx.py --variant yolox_tiny --weights models/yolox_tiny.pth --out models/yolox_tiny.onnx
Records weights sha256, opset, torch/onnx versions next to the model (<out>.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="yolox_tiny")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--input-size", type=int, nargs=2, default=None, metavar=("W", "H"))
    args = ap.parse_args()

    import onnx
    import torch

    from sceneproof.edge.yolox_model import YOLOX_VARIANTS, file_sha256, load_yolox

    model = load_yolox(args.variant, args.weights)
    default_hw = YOLOX_VARIANTS[args.variant][3]
    w, h = args.input_size if args.input_size else (default_hw[1], default_hw[0])
    dummy = torch.zeros(1, 3, h, w)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    torch.onnx.export(
        model, dummy, str(out), input_names=["images"], output_names=["output"], opset_version=args.opset,
        dynamo=False, do_constant_folding=True,
    )
    export_s = time.perf_counter() - t0
    m = onnx.load(str(out))
    onnx.checker.check_model(m)

    # parity check on random input: torch vs onnxruntime
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = np.random.RandomState(0).randint(0, 255, size=(1, 3, h, w)).astype(np.float32)
    with torch.inference_mode():
        ref = model(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"images": x})[0]
    max_abs = float(np.abs(ref - got).max())
    meta = {
        "variant": args.variant, "weights": str(args.weights), "weights_sha256_12": file_sha256(args.weights),
        "onnx_sha256_12": file_sha256(out), "opset": args.opset, "input_wh": [w, h], "torch": torch.__version__,
        "onnx": onnx.__version__, "onnxruntime": ort.__version__, "export_seconds": round(export_s, 2),
        "raw_output_max_abs_diff_random_input": max_abs, "output_shape": list(got.shape),
        "note": "output is the raw head tensor [1,N,85]; grid decoding + NMS happen in sceneproof.edge.detector",
    }
    out.with_suffix(".onnx.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0 if max_abs < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
