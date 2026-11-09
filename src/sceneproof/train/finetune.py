"""Fine-tune YOLOX on hand-labelled person boxes from the benchmark TRAIN split; pick the epoch on DEV.

Why: on the DEV split the COCO-pretrained detector misses people seen from above by a wide-angle lobby camera
(docs/eval/bench/README.md). The TRAIN split has the same kinds of views with boxes drawn by people.

What it does (CPU only, no GPU needed for a tiny model):
  * frames: every --stride-th frame of each TRAIN video (MOT16-04 every --mot-stride-th), decoded from the
    re-encoded benchmark videos, boxes from the datasets' ground truth (people only);
  * augmentation: random scale 0.6-1.4 + random placement on the 416x416 letterbox canvas (pad 114), horizontal
    flip, HSV jitter — the same "BGR 0-255, no normalisation, pad 114" input the product feeds the model;
  * model: the COCO checkpoint, all 80 class outputs kept (only class 0 "person" gets targets), backbone at
    0.1x learning rate, SGD + warm-up + cosine, EMA of weights;
  * selection: after every epoch, person AP on DEV frames (every 5th) with the product's exact
    letterbox/decode/NMS; the best epoch is kept. TEST is never touched here.

    python -m sceneproof.train.finetune --variant yolox_tiny --epochs 10 --out models/finetune/yolox_tiny_bench
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from sceneproof.edge.detector import decode_raw, grid_tables, letterbox, postprocess
from sceneproof.edge.yolox_model import YOLOX_VARIANTS, load_yolox
from sceneproof.eval.bench.data import ROOT, load_boxes, load_sequences
from sceneproof.eval.bench.detection import APAccumulator
from sceneproof.eval.bench.split import load_split

MAX_LABELS = 120


def _read_frames(video: Path, idxs: list[int], max_side: int) -> dict[int, np.ndarray]:
    want, out = set(idxs), {}
    cap = cv2.VideoCapture(str(video))
    i = 0
    while want:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            h, w = f.shape[:2]
            s = min(1.0, max_side / max(h, w))
            out[i] = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else f
            want.discard(i)
        i += 1
    cap.release()
    return out


def build_samples(part: str, stride: int, mot_stride: int, max_side: int = 960, min_vis: float = 0.1):
    """[(image_bgr, boxes_xyxy[n,4] in that image's pixels)] for one split."""
    seqs = load_sequences()
    ids = load_split()["sequences"][part]
    samples, n_boxes = [], 0
    for sid in ids:
        s = seqs.get(sid)
        if s is None:
            continue
        boxes = load_boxes(s)
        st = mot_stride if s.dataset.startswith("mot") else stride
        idxs = list(range(0, s.frames, st))
        frames = _read_frames(s.video, idxs, max_side)
        for i, img in frames.items():
            scale = img.shape[1] / s.width
            bl = [b for b in boxes.get(i, []) if b.label == "person" and b.visibility >= min_vis]
            bx = np.array([b.xyxy for b in bl], np.float32).reshape(-1, 4) * scale
            samples.append((img, bx, sid))
            n_boxes += len(bx)
    return samples, n_boxes


def augment(img: np.ndarray, boxes: np.ndarray, size: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    r = min(size / h, size / w) * rng.uniform(0.6, 1.4)
    nw, nh = max(8, int(w * r)), max(8, int(h * r))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    b = boxes * r
    if rng.random() < 0.5:
        im = im[:, ::-1]
        b = b.copy()
        b[:, [0, 2]] = nw - b[:, [2, 0]]
    # HSV jitter
    hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-5, 5)) % 180
    hsv[..., 1] *= rng.uniform(0.7, 1.3)
    hsv[..., 2] *= rng.uniform(0.7, 1.3)
    im = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    ox = rng.randint(min(0, size - nw), max(0, size - nw))  # negative offset = crop
    oy = rng.randint(min(0, size - nh), max(0, size - nh))
    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    cw, ch = min(nw - sx0, size - dx0), min(nh - sy0, size - dy0)
    canvas[dy0:dy0 + ch, dx0:dx0 + cw] = im[sy0:sy0 + ch, sx0:sx0 + cw]
    if len(b):
        area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        b = b + np.array([ox, oy, ox, oy], np.float32)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, size)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, size)
        area1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        keep = (area1 >= 0.4 * np.maximum(area0, 1e-6)) & (b[:, 2] - b[:, 0] >= 3) & (b[:, 3] - b[:, 1] >= 6)
        b = b[keep]
    return canvas, b


def to_batch(items) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = np.stack([np.ascontiguousarray(c.transpose(2, 0, 1)) for c, _ in items]).astype(np.float32)
    labels = np.zeros((len(items), MAX_LABELS, 5), np.float32)
    for k, (_, b) in enumerate(items):
        b = b[:MAX_LABELS]
        if len(b):
            labels[k, :len(b), 0] = 0  # COCO class 0 = person
            labels[k, :len(b), 1] = (b[:, 0] + b[:, 2]) / 2
            labels[k, :len(b), 2] = (b[:, 1] + b[:, 3]) / 2
            labels[k, :len(b), 3] = b[:, 2] - b[:, 0]
            labels[k, :len(b), 4] = b[:, 3] - b[:, 1]
    return torch.from_numpy(imgs), torch.from_numpy(labels)


@torch.no_grad()
def evaluate_dev(model, frames, input_hw, op_thresh: float = 0.4) -> dict:
    """Person AP with the product's letterbox -> raw head -> numpy decode -> NMS path."""
    model.eval()
    model.head.decode_in_inference = False
    grids, strides = grid_tables(input_hw)
    acc = APAccumulator()
    for img, gt_boxes in frames:
        blob, ratio = letterbox(img, input_hw)
        raw = model(torch.from_numpy(blob[None]))[0].numpy()
        boxes, scores = postprocess(decode_raw(raw, grids, strides), ratio, 0.05, 0.45, frame_wh=(img.shape[1], img.shape[0]))
        acc.add(boxes, scores, gt_boxes, op_thresh)
    return acc.result()


def dev_frames(stride: int = 5):
    seqs = load_sequences()
    out = []
    for sid in load_split()["sequences"]["dev"]:
        s = seqs.get(sid)
        if s is None:
            continue
        boxes = load_boxes(s)
        frames = _read_frames(s.video, list(range(0, s.frames, stride)), 10_000)
        out += [(img, boxes.get(i, [])) for i, img in frames.items()]
    return out


class EMA:
    def __init__(self, model, decay: float = 0.999) -> None:
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 500))  # YOLOX-style warm-up of the decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="yolox_tiny")
    ap.add_argument("--weights", default="")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=10, help="CAVIAR frame stride for training samples")
    ap.add_argument("--mot-stride", type=int, default=3)
    ap.add_argument("--dev-stride", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/finetune/yolox_tiny_bench")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    weights = args.weights or f"models/{args.variant}.pth"
    size = YOLOX_VARIANTS[args.variant][3][0]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = {"args": vars(args), "weights": weights, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "epochs": []}

    t0 = time.time()
    train, n_boxes = build_samples("train", args.stride, args.mot_stride)
    dev = dev_frames(args.dev_stride)
    log["data"] = {"train_images": len(train), "train_boxes": n_boxes,
                   "train_sequences": sorted({sid for *_, sid in train}), "dev_frames": len(dev),
                   "train_digest": hashlib.sha256("".join(f"{sid}:{len(b)}" for _, b, sid in train).encode()).hexdigest()[:12]}
    print(f"[ft] {len(train)} train images ({n_boxes} boxes), {len(dev)} dev frames, loaded in {time.time() - t0:.0f}s", flush=True)

    model = load_yolox(args.variant, weights)
    input_hw = YOLOX_VARIANTS[args.variant][3]
    base = evaluate_dev(model, dev, input_hw)
    log["dev_before"] = base
    print(f"[ft] dev before: AP50={base.get('ap50')} AP50:95={base.get('ap50_95')} recall@0.4={base.get('recall_at_op_by_height')}", flush=True)

    model.train()
    model.head.decode_in_inference = True
    bb_params = list(model.backbone.backbone.parameters())
    bb_ids = {id(p) for p in bb_params}
    rest = [p for p in model.parameters() if id(p) not in bb_ids]

    def groups(ps, lr):
        decay = [p for p in ps if p.ndim > 1]
        no_decay = [p for p in ps if p.ndim <= 1]
        return [{"params": decay, "lr": lr, "weight_decay": 5e-4, "base_lr": lr},
                {"params": no_decay, "lr": lr, "weight_decay": 0.0, "base_lr": lr}]

    opt = torch.optim.SGD(groups(bb_params, args.lr * args.backbone_lr_mult) + groups(rest, args.lr), lr=args.lr, momentum=0.9, nesterov=True)
    ema = EMA(model)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total, warm = steps_per_epoch * args.epochs, steps_per_epoch
    step, best = 0, (-1.0, -1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        te, sums = time.time(), {"total_loss": 0.0, "iou_loss": 0.0, "conf_loss": 0.0, "cls_loss": 0.0}
        for bi in range(steps_per_epoch):
            items = [augment(train[k][0], train[k][1], size, rng) for k in order[bi * args.batch:(bi + 1) * args.batch]]
            x, y = to_batch(items)
            f = (step + 1) / warm if step < warm else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            outd = model(x, y)
            opt.zero_grad()
            outd["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            ema.update(model)
            step += 1
            for k in sums:
                sums[k] += float(outd[k])
            if bi % 100 == 0:
                print(f"[ft] epoch {epoch} {bi}/{steps_per_epoch} loss={float(outd['total_loss']):.3f} lr={opt.param_groups[-1]['lr']:.2e}", flush=True)
        res = evaluate_dev(ema.ema, dev, input_hw)
        rec = {"epoch": epoch, "train_s": round(time.time() - te, 1), **{k: round(v / steps_per_epoch, 4) for k, v in sums.items()},
               "dev": res}
        log["epochs"].append(rec)
        torch.save({"model": ema.ema.state_dict(), "epoch": epoch, "dev": res}, out / f"epoch{epoch:02d}.pth")
        score = res.get("ap50_95") or 0.0
        if score > best[0]:
            best = (score, epoch)
            torch.save({"model": ema.ema.state_dict(), "epoch": epoch, "dev": res}, out / "best.pth")
        print(f"[ft] epoch {epoch} done in {rec['train_s']}s loss={rec['total_loss']} dev AP50={res.get('ap50')} "
              f"AP50:95={res.get('ap50_95')} (best epoch {best[1]})", flush=True)
        log["best_epoch"] = best[1]
        (out / "train_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False))
    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "train_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False))
    print(f"[ft] best epoch {best[1]} (dev AP50:95={best[0]:.4f}) -> {out / 'best.pth'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
