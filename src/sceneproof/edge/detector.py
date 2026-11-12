"""Person detector: YOLOX pre/post-processing shared by the ONNX Runtime and PyTorch backends.

The two backends must produce the same numbers for the same input; that property is
what tests/test_detector_parity.py and eval/bench_detector.py check. Keeping
pre/post-processing in one place (here, in numpy) is what makes the comparison fair.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .yolox_model import COCO_PERSON_CLASS_ID, YOLOX_VARIANTS, file_sha256

STRIDES = (8, 16, 32)


# --------------------------------------------------------------------------- pre / post


def letterbox(img_bgr: np.ndarray, input_hw: tuple[int, int]) -> tuple[np.ndarray, float]:
    """YOLOX ``preproc``: scale to fit, pad bottom/right with 114, CHW float32, no normalisation."""
    ih, iw = input_hw
    h, w = img_bgr.shape[:2]
    r = min(ih / h, iw / w)
    nw, nh = int(w * r), int(h * r)
    padded = np.full((ih, iw, 3), 114, dtype=np.uint8)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded[:nh, :nw] = resized
    blob = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return blob, r


def grid_tables(input_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(grids[N,2], strides[N,1]) in the exact anchor order the YOLOX head flattens its outputs."""
    grids, strides = [], []
    ih, iw = input_hw
    for s in STRIDES:
        hs, ws = ih // s, iw // s
        yv, xv = np.meshgrid(np.arange(hs), np.arange(ws), indexing="ij")
        g = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(g)
        strides.append(np.full((g.shape[0], 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


def decode_raw(raw: np.ndarray, grids: np.ndarray, strides: np.ndarray) -> np.ndarray:
    """Raw head output [N, 5+C] (grid units) -> [N, 5+C] with cx,cy,w,h in input pixels."""
    out = raw.copy()
    out[:, :2] = (out[:, :2] + grids) * strides
    out[:, 2:4] = np.exp(out[:, 2:4]) * strides
    return out


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Greedy NMS; returns kept indices sorted by score desc."""
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return np.asarray(keep, dtype=np.int64)


def postprocess(
    decoded: np.ndarray,
    ratio: float,
    conf_thresh: float,
    nms_thresh: float,
    class_id: int = COCO_PERSON_CLASS_ID,
    frame_wh: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """YOLOX ``postprocess`` restricted to one class. Returns (boxes_xyxy[n,4] in frame px, scores[n])."""
    obj = decoded[:, 4]
    cls_scores = decoded[:, 5:]
    cls_idx = cls_scores.argmax(1)
    cls_conf = cls_scores[np.arange(decoded.shape[0]), cls_idx]
    conf = obj * cls_conf
    mask = (conf >= conf_thresh) & (cls_idx == class_id)
    if not mask.any():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    d = decoded[mask]
    scores = conf[mask].astype(np.float32)
    boxes = np.empty((d.shape[0], 4), np.float32)
    boxes[:, 0] = d[:, 0] - d[:, 2] / 2
    boxes[:, 1] = d[:, 1] - d[:, 3] / 2
    boxes[:, 2] = d[:, 0] + d[:, 2] / 2
    boxes[:, 3] = d[:, 1] + d[:, 3] / 2
    keep = nms_xyxy(boxes, scores, nms_thresh)
    boxes, scores = boxes[keep] / ratio, scores[keep]
    if frame_wh is not None:
        w, h = frame_wh
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes, scores


# --------------------------------------------------------------------------- backends


class Backend(Protocol):
    name: str
    version: str

    def run(self, blob_nchw: np.ndarray) -> np.ndarray:  # -> raw [N, 5+C]
        ...


class OnnxRuntimeBackend:
    def __init__(self, model_path: str | Path, providers: list[Any] | None = None, intra_op_threads: int = 0,
                 graph_optimization: str = "all") -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        if intra_op_threads:
            so.intra_op_num_threads = intra_op_threads
        so.graph_optimization_level = {
            "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }[graph_optimization]
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.active_providers = self.session.get_providers()
        self.name = "onnxruntime-" + self.active_providers[0].replace("ExecutionProvider", "").lower()
        self.version = ort.__version__
        self.model_path = str(model_path)

    def run(self, blob_nchw: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: blob_nchw})[0][0]


class TorchBackend:
    def __init__(self, variant: str, weights: str | Path, device: str = "cpu", threads: int = 0) -> None:
        import torch

        from .yolox_model import load_yolox

        if threads:
            torch.set_num_threads(threads)
        self.model = load_yolox(variant, weights, device=device)
        self.device = device
        self.name = f"torch-{device}"
        self.version = torch.__version__
        self._torch = torch

    def run(self, blob_nchw: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(blob_nchw).to(self.device)
            return self.model(x)[0].cpu().numpy()


class TensorRTBackend:
    """NVIDIA TensorRT engine built from the same ONNX file, FP32 or FP16, batch 1.

    Pre/post-processing stay in PersonDetector, so only the network itself changes and the outputs can be compared
    with the ONNX Runtime CPU path frame by frame (scripts/gpu_bench.py). GPU buffers are PyTorch CUDA tensors —
    PyTorch is present wherever this is used here (Colab, Jetson images). An engine only runs on the GPU model and
    TensorRT version it was built with, so the cache file name carries both; the first run builds it (tens of
    seconds), later runs load it.
    """

    def __init__(self, model_path: str | Path, precision: str = "fp16", engine_dir: str | Path | None = None,
                 workspace_mb: int = 1024) -> None:
        import tensorrt as trt
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT backend needs an NVIDIA GPU (torch.cuda.is_available() is False)")
        precision = precision.lower()
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision must be fp32 or fp16, not {precision!r}")
        self._torch = torch
        gpu = torch.cuda.get_device_name(0)
        tag = re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
        engine_dir = Path(engine_dir) if engine_dir else Path(model_path).parent / "trt"
        engine_path = engine_dir / f"{Path(model_path).stem}.{precision}.trt{trt.__version__}.{tag}.engine"
        logger = trt.Logger(trt.Logger.WARNING)
        self.build_s: float | None = None
        fp16_note = engine_path.with_suffix(".fp16-method")
        if engine_path.exists():
            data = engine_path.read_bytes()
            self.fp16_method = fp16_note.read_text().strip() if fp16_note.exists() else ""
        else:
            t0 = time.perf_counter()
            data, self.fp16_method = _build_trt_engine(trt, logger, Path(model_path), precision, workspace_mb)
            self.build_s = time.perf_counter() - t0
            engine_dir.mkdir(parents=True, exist_ok=True)
            engine_path.write_bytes(data)
            fp16_note.write_text(self.fp16_method)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not load {engine_path} (delete it to rebuild)")
        self.context = self.engine.create_execution_context()
        self.inputs: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"tensor {name} has a dynamic shape {shape}; export the ONNX with a fixed input size")
            np_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            buf = torch.empty(shape, dtype=getattr(torch, np_dtype.name), device="cuda")  # float32 -> torch.float32 ...
            self.context.set_tensor_address(name, buf.data_ptr())
            (self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs)[name] = buf
        if len(self.inputs) != 1 or not self.outputs:
            raise RuntimeError(f"expected 1 input and >=1 output, got {list(self.inputs)} / {list(self.outputs)}")
        self.input = next(iter(self.inputs.values()))
        self.output = next(iter(self.outputs.values()))
        self.stream = torch.cuda.Stream()
        self.name = f"tensorrt-{precision}"
        self.version = trt.__version__
        self.precision = precision
        self.gpu = gpu
        self.engine_path = str(engine_path)
        self.model_path = str(model_path)

    def run(self, blob_nchw: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.cuda.stream(self.stream):
            self.input.copy_(torch.from_numpy(np.ascontiguousarray(blob_nchw)).to(self.input.dtype))
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            out = self.output.float()
        self.stream.synchronize()
        return out.cpu().numpy()[0]


def onnx_to_fp16(model_path: str | Path) -> bytes:
    """FP16 weights and activations, FP32 input and output (Cast nodes at the edges). Resize stays FP32: its
    `scales` input must be float32. Used for TensorRT >= 11, whose networks are strongly typed (precision comes
    from the ONNX tensor types; the FP16 builder flag is gone)."""
    import warnings

    import onnx
    from onnxconverter_common import float16

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "float32 number ... will be truncated" for tiny weights
        m16 = float16.convert_float_to_float16(onnx.load(str(model_path)), keep_io_types=True, op_block_list=["Resize"])
    onnx.checker.check_model(m16)
    return m16.SerializeToString()


def _build_trt_engine(trt: Any, logger: Any, model_path: Path, precision: str, workspace_mb: int) -> tuple[bytes, str]:
    """-> (serialized engine, how FP16 was obtained: "builder-flag" (TensorRT <= 10), "fp16-onnx" (strongly typed),
    or "" for FP32)."""
    builder = trt.Builder(logger)
    major = int(trt.__version__.split(".")[0])
    flags = 0 if major >= 10 else 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    onnx_bytes = model_path.read_bytes()
    method = ""
    if precision == "fp16":
        if hasattr(trt.BuilderFlag, "FP16"):
            method = "builder-flag"
        else:
            onnx_bytes, method = onnx_to_fp16(model_path), "fp16-onnx"
            strongly_typed = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
            if strongly_typed is not None:
                flags |= 1 << int(strongly_typed)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {model_path}: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if method == "builder-flag":
        config.set_flag(trt.BuilderFlag.FP16)
    data = builder.build_serialized_network(network, config)
    if data is None:
        raise RuntimeError("TensorRT engine build failed (see the TensorRT log above)")
    return bytes(data), method


# --------------------------------------------------------------------------- detector


@dataclass
class Detection:
    boxes: np.ndarray  # [n,4] xyxy in frame pixels
    scores: np.ndarray  # [n]
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class DetectorInfo:
    model_name: str
    model_version: str
    backend: str
    backend_version: str
    input_width: int
    input_height: int
    conf_thresh: float
    nms_thresh: float


class PersonDetector:
    def __init__(self, backend: Backend, variant: str, input_hw: tuple[int, int], conf_thresh: float = 0.4,
                 nms_thresh: float = 0.45, model_version: str = "", min_box_height_px: int = 0) -> None:
        self.backend = backend
        self.variant = variant
        self.input_hw = input_hw
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.min_box_height_px = min_box_height_px
        self.grids, self.strides = grid_tables(input_hw)
        self.model_version = model_version
        self.info = DetectorInfo(variant, model_version, backend.name, backend.version, input_hw[1], input_hw[0],
                                 conf_thresh, nms_thresh)

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        t0 = time.perf_counter()
        blob, ratio = letterbox(frame_bgr, self.input_hw)
        t1 = time.perf_counter()
        raw = self.backend.run(blob[None])
        t2 = time.perf_counter()
        decoded = decode_raw(raw, self.grids, self.strides)
        h, w = frame_bgr.shape[:2]
        boxes, scores = postprocess(decoded, ratio, self.conf_thresh, self.nms_thresh, frame_wh=(w, h))
        if self.min_box_height_px > 0:
            keep = (boxes[:, 3] - boxes[:, 1]) >= self.min_box_height_px
            boxes, scores = boxes[keep], scores[keep]
        t3 = time.perf_counter()
        return Detection(boxes, scores, {"preprocess": (t1 - t0) * 1e3, "infer": (t2 - t1) * 1e3,
                                         "postprocess": (t3 - t2) * 1e3})

    def warmup(self, n: int = 3, hw: tuple[int, int] = (480, 640)) -> None:
        dummy = np.zeros((hw[0], hw[1], 3), np.uint8)
        for _ in range(n):
            self.detect(dummy)


def build_detector(cfg: dict[str, Any]) -> PersonDetector:
    """cfg keys: variant, backend (onnxruntime|torch|tensorrt), model (path), input_size [w,h], conf_threshold,
    nms_threshold, providers, intra_op_threads, graph_optimization, device, precision (tensorrt: fp32|fp16),
    engine_dir, workspace_mb, min_box_height_px."""
    variant = cfg.get("variant", "yolox_tiny")
    default_hw = YOLOX_VARIANTS[variant][3]
    if cfg.get("input_size"):
        w, h = cfg["input_size"]
        input_hw = (int(h), int(w))
    else:
        input_hw = default_hw
    model_path = cfg["model"]
    version = file_sha256(model_path) if Path(model_path).exists() else "missing"
    backend_name = cfg.get("backend", "onnxruntime")
    if backend_name == "onnxruntime":
        backend: Backend = OnnxRuntimeBackend(model_path, cfg.get("providers"), int(cfg.get("intra_op_threads", 0)),
                                              cfg.get("graph_optimization", "all"))
    elif backend_name == "torch":
        backend = TorchBackend(variant, model_path, cfg.get("device", "cpu"), int(cfg.get("intra_op_threads", 0)))
    elif backend_name == "tensorrt":
        backend = TensorRTBackend(model_path, cfg.get("precision") or "fp16", cfg.get("engine_dir"),
                                  int(cfg.get("workspace_mb") or 1024))
    else:
        raise ValueError(f"unknown backend {backend_name!r}")
    return PersonDetector(backend, variant, input_hw, float(cfg.get("conf_threshold", 0.4)),
                          float(cfg.get("nms_threshold", 0.45)), version, int(cfg.get("min_box_height_px", 0)))
