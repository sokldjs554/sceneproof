"""Ground-truth format for event-level evaluation.

One YAML per evaluation video (data/ground_truth/<video>.yaml):

    video: campus-walkway.mp4
    fps: 10
    zones:
      - zone_id: z-grass
        name: 잔디 출입금지
        polygon: [[0.0,0.44],[0.36,0.50],...]
    rules:
      entry_confirm_frames: 3
      exit_confirm_frames: 5
      dwell_seconds: 5
    events:                      # true episodes (one person inside the zone)
      - id: e01
        start_s: 12.3            # first frame the person's feet are inside the polygon
        end_s: 18.9              # last frame inside (or when they leave the view)
        label: intrusion         # intrusion | boundary_pass | occluded | unsure
        dwell: true              # stayed >= rules.dwell_seconds
        note: "..."
    negatives:                   # situations that must NOT raise an event (for the report only)
      - {start_s: 30.0, end_s: 40.0, note: "boundary walk, feet stay outside"}
    annotation:
      method: "..."              # how the file was produced
      status: draft | reviewed   # reviewed == a human verified every row
      reviewer: ""
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GTEvent:
    id: str
    start_s: float
    end_s: float
    label: str = "intrusion"
    dwell: bool = False
    note: str = ""
    zone_id: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class GroundTruth:
    video: str
    fps: float
    zones: list[dict]
    rules: dict
    events: list[GTEvent]
    negatives: list[dict] = field(default_factory=list)
    annotation: dict = field(default_factory=dict)
    duration_s: float = 0.0

    @classmethod
    def load(cls, path: str | Path) -> GroundTruth:
        d = yaml.safe_load(Path(path).read_text())
        events = [GTEvent(e["id"], float(e["start_s"]), float(e["end_s"]), e.get("label", "intrusion"),
                          bool(e.get("dwell", False)), e.get("note", ""), e.get("zone_id", "")) for e in d.get("events", [])]
        return cls(d["video"], float(d.get("fps", 10)), d.get("zones", []), d.get("rules", {}), events,
                   d.get("negatives", []), d.get("annotation", {}), float(d.get("duration_s", 0.0)))

    def positives(self) -> list[GTEvent]:
        """Events that count as true positives: intrusion (and boundary passes are *not* positives)."""
        return [e for e in self.events if e.label == "intrusion"]
