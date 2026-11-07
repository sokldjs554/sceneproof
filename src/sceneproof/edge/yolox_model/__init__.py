"""Vendored YOLOX (Apache-2.0, Megvii) — inference-only model definitions.

Only the model graph is vendored (darknet / PAFPN / head). Training code, data
augmentation and the loguru dependency were removed. Weights are the official
COCO checkpoints from the YOLOX GitHub releases (see scripts/fetch_models.sh).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# name -> (depth, width, depthwise, input_size)
YOLOX_VARIANTS: dict[str, tuple[float, float, bool, tuple[int, int]]] = {
    "yolox_nano": (0.33, 0.25, True, (416, 416)),
    "yolox_tiny": (0.33, 0.375, False, (416, 416)),
    "yolox_s": (0.33, 0.50, False, (640, 640)),
}

COCO_PERSON_CLASS_ID = 0
NUM_COCO_CLASSES = 80


def build_yolox(name: str, num_classes: int = NUM_COCO_CLASSES):
    """Builds the YOLOX graph for a variant name; mirrors exps/default/yolox_*.py."""
    import torch.nn as nn

    from .yolo_head import YOLOXHead
    from .yolo_pafpn import YOLOPAFPN
    from .yolox import YOLOX

    if name not in YOLOX_VARIANTS:
        raise ValueError(f"unknown YOLOX variant {name!r}; choose from {sorted(YOLOX_VARIANTS)}")
    depth, width, depthwise, _ = YOLOX_VARIANTS[name]
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act="silu", depthwise=depthwise)
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act="silu", depthwise=depthwise)
    model = YOLOX(backbone, head)
    for m in model.modules():  # init_yolo() in the official exps
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
    model.head.initialize_biases(1e-2)
    return model


def load_yolox(name: str, weights: str | Path, device: str = "cpu"):
    """Loads an official checkpoint. ``weights_only=True`` so the pickle cannot run code."""
    import torch

    model = build_yolox(name)
    ckpt = torch.load(str(weights), map_location=device, weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch for {name}: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()
    model.head.decode_in_inference = False  # we decode grids in numpy, same as the official ONNX export
    return model.to(device)


def file_sha256(path: str | Path, length: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]
