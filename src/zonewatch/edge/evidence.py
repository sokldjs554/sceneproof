"""Evidence writer: key frames (raw + annotated) and a pre/post-roll clip per episode.

Files are written under ``<root>/<event_id>/`` with stable names so the gateway can upload them
by relative path and the server can address them without a lookup table:
    context_before.jpg, entry.jpg, dwell.jpg, exit.jpg, context_after.jpg, *_annot.jpg, clip.mp4
The clip is encoded by piping the ring buffer's JPEGs into ffmpeg (``image2pipe``), so no frame
is decoded twice. H.264 is used when libx264 is present (browser-playable); mpeg4 is the fallback.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .frames import FrameRing, RingEntry
from .zones import Episode, ZoneSpec

CONTEXT_OFFSET_S = 1.5


@dataclass
class KeyFrameOut:
    role: str
    index: int
    media_s: float
    wall_ts: float
    path: str
    annot_path: str
    box: tuple[float, float, float, float] | None
    width: int
    height: int


@dataclass
class ClipOut:
    path: str
    start_media_s: float
    end_media_s: float
    fps: float
    frames: int
    encode_ms: float
    codec: str


def _ffmpeg_codec() -> str:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return "none"
    if "libx264" in out:
        return "libx264"
    if "mpeg4" in out:
        return "mpeg4"
    return "none"


def draw_annotations(img: np.ndarray, zone: ZoneSpec | None, box: tuple | None, label: str = "",
                     scale_from: tuple[int, int] | None = None) -> np.ndarray:
    """Draw zone polygon + target box. ``scale_from`` = (w,h) of the analysed frame if the JPEG was downscaled."""
    out = img.copy()
    h, w = out.shape[:2]
    if zone is not None:
        pts = np.array([[int(x * w), int(y * h)] for x, y in zone.polygon], dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
        cv2.polylines(out, [pts], True, (0, 0, 255), 2)
        cv2.putText(out, zone.name, (pts[:, 0].min() + 4, max(16, pts[:, 1].min() - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
    if box is not None:
        sx = sy = 1.0
        if scale_from:
            sx, sy = w / scale_from[0], h / scale_from[1]
        x1, y1, x2, y2 = int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if label:
            cv2.putText(out, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return out


class EvidenceWriter:
    def __init__(self, root: str | Path, pre_roll_s: float = 3.0, post_roll_s: float = 3.0, clip_fps: float = 10.0,
                 clip_max_frames: int = 300) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pre_roll_s = pre_roll_s
        self.post_roll_s = post_roll_s
        self.clip_fps = clip_fps
        self.clip_max_frames = clip_max_frames
        self.codec = _ffmpeg_codec()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ key frames
    def write_keyframe(self, event_id: str, role: str, entry: RingEntry, zone: ZoneSpec | None,
                       box: tuple | None, frame_wh: tuple[int, int], label: str = "") -> KeyFrameOut:
        d = self.root / event_id
        d.mkdir(parents=True, exist_ok=True)
        raw_path = d / f"{role}.jpg"
        annot_path = d / f"{role}_annot.jpg"
        with self._lock:
            raw_path.write_bytes(entry.jpeg)
            img = cv2.imdecode(np.frombuffer(entry.jpeg, np.uint8), cv2.IMREAD_COLOR)
            annot = draw_annotations(img, zone, box, label, scale_from=frame_wh)
            cv2.imwrite(str(annot_path), annot, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return KeyFrameOut(role, entry.index, entry.media_s, entry.wall_ts, f"{event_id}/{role}.jpg",
                           f"{event_id}/{role}_annot.jpg", tuple(box) if box is not None else None, entry.width,
                           entry.height)

    def keyframes_for(self, ep: Episode, ring: FrameRing, frame_wh: tuple[int, int], roles: list[str]) -> list[KeyFrameOut]:
        """roles ⊆ {context_before, entry, dwell, exit, context_after}; missing ring frames are skipped."""
        out: list[KeyFrameOut] = []
        eid = ep.episode_id
        want: list[tuple[str, float, tuple | None]] = []
        for role in roles:
            if role == "context_before":
                want.append((role, ep.entry_media_s - CONTEXT_OFFSET_S, None))
            elif role == "entry":
                want.append((role, ep.entry_media_s, ep.entry_xyxy))
            elif role == "dwell" and ep.dwell_media_s is not None:
                want.append((role, ep.dwell_media_s, ep.last_xyxy))
            elif role == "exit" and ep.exit_media_s is not None:
                want.append((role, ep.exit_media_s, ep.last_xyxy))
            elif role == "context_after" and ep.exit_media_s is not None:
                want.append((role, ep.exit_media_s + CONTEXT_OFFSET_S, None))
        for role, t, box in want:
            e = ring.nearest(t)
            if e is None or abs(e.media_s - t) > max(1.0, CONTEXT_OFFSET_S):
                continue
            label = f"track {ep.track_id}" if box is not None else ""
            out.append(self.write_keyframe(eid, role, e, ep.zone, box, frame_wh, label))
        return out

    # ------------------------------------------------------------------ clip
    def clip_window(self, ep: Episode) -> tuple[float, float]:
        end = ep.exit_media_s if ep.exit_media_s is not None else ep.last_media_s
        return ep.entry_media_s - self.pre_roll_s, end + self.post_roll_s

    def write_clip(self, ep: Episode, ring: FrameRing) -> ClipOut | None:
        start, end = self.clip_window(ep)
        entries = ring.slice(start, end)
        if not entries:
            return None
        # downsample to clip_fps and cap the frame count
        picked: list[RingEntry] = []
        next_t = entries[0].media_s
        for e in entries:
            if e.media_s + 1e-6 >= next_t:
                picked.append(e)
                next_t = e.media_s + 1.0 / self.clip_fps
        if len(picked) > self.clip_max_frames:
            step = len(picked) / self.clip_max_frames
            picked = [picked[int(i * step)] for i in range(self.clip_max_frames)]
        d = self.root / ep.episode_id
        d.mkdir(parents=True, exist_ok=True)
        path = d / "clip.mp4"
        t0 = time.perf_counter()
        if self.codec == "none" or shutil.which("ffmpeg") is None:
            # last resort: MJPEG via OpenCV (not browser-playable but keeps the evidence)
            path = d / "clip.avi"
            w, h = picked[0].width, picked[0].height
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), self.clip_fps, (w, h))
            for e in picked:
                vw.write(cv2.imdecode(np.frombuffer(e.jpeg, np.uint8), cv2.IMREAD_COLOR))
            vw.release()
            codec = "mjpeg"
        else:
            codec = self.codec
            vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "image2pipe", "-vcodec", "mjpeg",
                   "-framerate", f"{self.clip_fps}", "-i", "-", "-vf", vf, "-c:v", codec]
            if codec == "libx264":
                cmd += ["-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
            else:
                cmd += ["-q:v", "5"]
            cmd.append(str(path))
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            assert proc.stdin is not None
            for e in picked:
                proc.stdin.write(e.jpeg)
            proc.stdin.close()
            err = proc.stderr.read().decode(errors="ignore") if proc.stderr else ""
            if proc.wait(timeout=120) != 0:
                raise RuntimeError(f"ffmpeg failed: {err[-400:]}")
        ms = (time.perf_counter() - t0) * 1e3
        meta = {"start_media_s": picked[0].media_s, "end_media_s": picked[-1].media_s, "fps": self.clip_fps,
                "frames": len(picked), "codec": codec, "encode_ms": round(ms, 1)}
        (d / "clip.json").write_text(json.dumps(meta))
        return ClipOut(f"{ep.episode_id}/{path.name}", picked[0].media_s, picked[-1].media_s, self.clip_fps,
                       len(picked), ms, codec)
