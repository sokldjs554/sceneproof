"""Benchmark sources, manifest and hand-labelled box loaders (CAVIAR CVML XML, MOTChallenge gt.txt)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
BENCH_DIR = ROOT / "data" / "bench"
CONFIG_DIR = ROOT / "configs" / "bench"

# MOTChallenge class ids (MOT16 paper, Table 3). Persons count as people; distractors/reflections are
# person-like but not people -> "ignore" (a detection there is neither right nor wrong); the rest is skipped.
MOT_PERSON = {1: "pedestrian", 2: "person_on_vehicle", 7: "static_person"}
MOT_IGNORE = {8: "distractor", 12: "reflection"}


@dataclass
class GTBox:
    track_id: int
    xyxy: tuple[float, float, float, float]
    label: str = "person"  # person | ignore
    visibility: float = 1.0
    consider: bool = True  # MOTChallenge "conf" column: 0 = not used in the official detection evaluation


@dataclass
class Sequence:
    id: str
    dataset: str
    site: str
    group: str
    fps: float
    video: Path
    gt_path: Path
    frames: int = 0
    width: int = 0
    height: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


def load_sources() -> dict:
    return yaml.safe_load((CONFIG_DIR / "sources.yaml").read_text())


def load_sequences(require_data: bool = True) -> dict[str, Sequence]:
    src = load_sources()
    man_path = BENCH_DIR / "manifest.json"
    man = json.loads(man_path.read_text())["sequences"] if man_path.exists() else {}
    out = {}
    for s in src["sequences"]:
        ext = "xml" if s["dataset"] == "caviar" else "txt"
        m = man.get(s["id"], {})
        seq = Sequence(s["id"], s["dataset"], s["site"], s["group"], float(s["fps"]),
                       BENCH_DIR / "videos" / f"{s['id']}.mp4", BENCH_DIR / "gt" / f"{s['id']}.{ext}",
                       int(m.get("frames", 0)), int(m.get("width", 0)), int(m.get("height", 0)), {"manifest": m})
        if require_data and not (seq.video.exists() and seq.gt_path.exists()):
            continue
        if seq.video.exists() and not (seq.frames and seq.width and seq.height):
            _probe(seq)
        out[s["id"]] = seq
    return out


def _probe(seq: Sequence) -> None:
    """Frame count and size from the re-encoded video (constant frame rate, so the container count is exact)."""
    import cv2

    cap = cv2.VideoCapture(str(seq.video))
    seq.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seq.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    seq.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()


def load_boxes(seq: Sequence) -> dict[int, list[GTBox]]:
    """frame index (0-based, == decoded video frame) -> hand-labelled boxes in pixels."""
    return _caviar(seq.gt_path) if seq.dataset == "caviar" else _mot(seq.gt_path)


def _caviar(path: Path) -> dict[int, list[GTBox]]:
    out: dict[int, list[GTBox]] = defaultdict(list)
    for fr in ET.parse(path).getroot().findall("frame"):
        n = int(fr.get("number"))
        for o in fr.findall("objectlist/object"):  # individuals only; <grouplist> boxes are groups, not people
            b = o.find("box")
            xc, yc, w, h = (float(b.get(k)) for k in ("xc", "yc", "w", "h"))
            if w <= 0 or h <= 0:
                continue
            out[n].append(GTBox(int(o.get("id")), (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)))
    return out


def _mot(path: Path) -> dict[int, list[GTBox]]:
    out: dict[int, list[GTBox]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        f = line.split(",")
        frame, tid, x, y, w, h = int(f[0]), int(f[1]), *map(float, f[2:6])
        cls = int(float(f[7])) if len(f) > 7 else 1
        vis = float(f[8]) if len(f) > 8 else 1.0
        if cls in MOT_PERSON:
            label = "person"
        elif cls in MOT_IGNORE:
            label = "ignore"
        else:
            continue
        consider = (float(f[6]) != 0) if len(f) > 6 else True
        out[frame - 1].append(GTBox(tid, (x, y, x + w, y + h), label, vis, consider))
    return out
