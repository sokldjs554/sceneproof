"""Edge configuration (YAML) with validation. See configs/edge.yaml for a documented example."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from .zones import ZoneSpec


class ZoneConfig(BaseModel):
    zone_id: str
    name: str
    polygon: list[tuple[float, float]] = Field(min_length=3)
    entry_confirm_frames: int = 3
    exit_confirm_frames: int = 5
    dwell_seconds: float = 5.0
    lost_grace_seconds: float = 1.5
    entry_confirm_seconds: float | None = None
    exit_confirm_seconds: float | None = None
    # person-shaped objects inside the zone (mannequin, signboard): feet there never open an episode
    exclude: list[list[tuple[float, float]]] = Field(default_factory=list)
    # continue an episode when the tracker starts a new id for the same person (crowds, occlusion)
    rejoin: bool = True
    rejoin_dist_rel: float = Field(default=1.0, gt=0)

    @field_validator("polygon")
    @classmethod
    def _norm(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for x, y in v:
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("polygon points must be normalized to [0,1]")
        return v

    @field_validator("exclude")
    @classmethod
    def _norm_exclude(cls, v: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
        for poly in v:
            if len(poly) < 3:
                raise ValueError("an exclusion polygon needs at least 3 points")
            cls._norm(poly)
        return v

    def to_spec(self) -> ZoneSpec:
        return ZoneSpec(self.zone_id, self.name, [tuple(p) for p in self.polygon], self.entry_confirm_frames,
                        self.exit_confirm_frames, self.dwell_seconds, self.lost_grace_seconds,
                        self.entry_confirm_seconds, self.exit_confirm_seconds,
                        exclude=[[tuple(p) for p in poly] for poly in self.exclude],
                        rejoin=self.rejoin, rejoin_dist_rel=self.rejoin_dist_rel)


class CameraConfig(BaseModel):
    camera_id: str
    name: str = ""
    source: str
    autostart: bool = False
    realtime: bool = True
    loop: bool = False
    speed: float = 1.0
    fps_hint: float = 25.0
    max_width: int = 0
    zones: list[ZoneConfig] = Field(default_factory=list)
    split: str = "dev"  # dev | eval  (documentation only; the evaluator enforces the split)


class DetectorConfig(BaseModel):
    variant: str = "yolox_tiny"
    backend: str = "onnxruntime"
    model: str = "models/yolox_tiny.onnx"
    conf_threshold: float = 0.4
    nms_threshold: float = 0.45
    intra_op_threads: int = 0
    graph_optimization: str = "all"
    providers: list[Any] | None = None
    device: str = "cpu"
    input_size: list[int] | None = None


class TrackerConfig(BaseModel):
    track_thresh: float = 0.5
    low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8
    track_buffer_seconds: float = 1.5
    fuse_score: bool = True


class PipelineConfig(BaseModel):
    source_backend: str = "auto"
    queue_capacity: int = 4
    ring_seconds: float = 12.0
    ring_max_mb: int = 128
    ring_max_side: int = 960
    ring_jpeg_quality: int = 80
    pre_roll_s: float = 3.0
    post_roll_s: float = 3.0
    clip_fps: float = 10.0
    clip_max_frames: int = 300
    trajectory_stride: int = 3
    rtsp_latency_ms: int = 200
    rtsp_protocols: str = "tcp"


class EdgeConfig(BaseModel):
    device_id: str = "edge-01"
    gateway: str = "127.0.0.1:50051"
    status_interval_s: float = 2.0
    evidence_root: str = "data/evidence"
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    pipeline: PipelineConfig = PipelineConfig()
    cameras: list[CameraConfig] = Field(default_factory=list)

    def camera(self, camera_id: str) -> CameraConfig | None:
        return next((c for c in self.cameras if c.camera_id == camera_id), None)

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_config(path: str | Path) -> EdgeConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return EdgeConfig.model_validate(data)
