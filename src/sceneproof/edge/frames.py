"""Frame container, bounded latest-frame queue, and the evidence ring buffer.

Two different buffers with two different policies on purpose:

* ``FrameQueue`` sits between the decoder thread and the analyser thread. When input arrives
  faster than the analyser can process, it **drops the oldest** waiting frame. For a live
  camera the newest frame is the one that matters; letting the queue grow would only push the
  analysis lag up without bound. Drops are counted and shown to the operator.
* ``FrameRing`` keeps the last N seconds of frames as JPEG so that, when an episode opens or
  closes, the pre-roll context is still available. It is bounded by **both** seconds and
  bytes and never drops frames that an open episode will need (eviction only trims the past).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    media_s: float      # seconds in the source media / stream time
    wall_ts: float      # time.time() at capture
    image: np.ndarray   # BGR
    source_pts_s: float | None = None

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass
class QueueStats:
    capacity: int
    depth: int
    received: int
    dropped: int
    taken: int


class FrameQueue:
    """Thread-safe bounded queue.

    ``lossless=False`` (live): ``put`` never blocks; it evicts the oldest frame when full.
    ``lossless=True`` (offline/evaluation): ``put`` blocks the producer until there is room, so
    every frame is analysed and results are reproducible. The same bound applies in both modes;
    only the policy differs.
    """

    def __init__(self, capacity: int = 4, lossless: bool = False) -> None:
        self.capacity = max(1, capacity)
        self.lossless = lossless
        self._dq: deque[Frame] = deque()
        self._cv = threading.Condition()
        self.received = 0
        self.dropped = 0
        self.taken = 0
        self._closed = False

    def put(self, frame: Frame) -> bool:
        """Returns False if an older frame had to be dropped to make room (never in lossless mode)."""
        with self._cv:
            self.received += 1
            if self.lossless:
                while len(self._dq) >= self.capacity and not self._closed:
                    self._cv.wait(0.5)
                if self._closed:
                    return False
                self._dq.append(frame)
                self._cv.notify_all()
                return True
            dropped = False
            while len(self._dq) >= self.capacity:
                self._dq.popleft()
                self.dropped += 1
                dropped = True
            self._dq.append(frame)
            self._cv.notify_all()
            return not dropped

    def get(self, timeout: float | None = 0.5) -> Frame | None:
        with self._cv:
            if not self._dq and not self._closed:
                self._cv.wait(timeout)
            if not self._dq:
                return None
            self.taken += 1
            f = self._dq.popleft()
            self._cv.notify_all()  # wake a blocked lossless producer
            return f

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def stats(self) -> QueueStats:
        with self._cv:
            return QueueStats(self.capacity, len(self._dq), self.received, self.dropped, self.taken)

    def __len__(self) -> int:
        with self._cv:
            return len(self._dq)


@dataclass
class RingEntry:
    index: int
    media_s: float
    wall_ts: float
    jpeg: bytes
    width: int
    height: int


@dataclass
class RingStats:
    entries: int
    bytes: int
    seconds: float
    max_seconds: float
    max_bytes: int
    evicted_by_time: int = 0
    evicted_by_bytes: int = 0


class FrameRing:
    """Time-bounded, byte-bounded JPEG ring of recent frames for pre-roll evidence.

    ``pinned_from`` lets the pipeline protect frames that an open episode still needs
    (eviction stops at the pin), trading memory for evidence completeness; the byte cap is the
    hard limit either way.
    """

    def __init__(self, max_seconds: float = 12.0, max_bytes: int = 128 << 20, max_side: int = 960,
                 jpeg_quality: int = 80) -> None:
        self.max_seconds = max_seconds
        self.max_bytes = max_bytes
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality
        self._dq: deque[RingEntry] = deque()
        self._bytes = 0
        self._lock = threading.Lock()
        self.pinned_from: float | None = None
        self.evicted_by_time = 0
        self.evicted_by_bytes = 0
        self._last_encode_ms = 0.0

    def encode(self, image: np.ndarray) -> tuple[bytes, int, int]:
        h, w = image.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes(), w, h

    def append(self, frame: Frame) -> RingEntry:
        t0 = time.perf_counter()
        jpeg, w, h = self.encode(frame.image)
        self._last_encode_ms = (time.perf_counter() - t0) * 1e3
        e = RingEntry(frame.index, frame.media_s, frame.wall_ts, jpeg, w, h)
        with self._lock:
            self._dq.append(e)
            self._bytes += len(jpeg)
            self._evict_locked(frame.media_s)
        return e

    def _evict_locked(self, now_media_s: float) -> None:
        floor = now_media_s - self.max_seconds
        if self.pinned_from is not None:
            floor = min(floor, self.pinned_from)
        while self._dq and self._dq[0].media_s < floor:
            self._bytes -= len(self._dq.popleft().jpeg)
            self.evicted_by_time += 1
        while self._dq and self._bytes > self.max_bytes:  # hard cap wins over the pin
            self._bytes -= len(self._dq.popleft().jpeg)
            self.evicted_by_bytes += 1

    def slice(self, start_media_s: float, end_media_s: float) -> list[RingEntry]:
        with self._lock:
            return [e for e in self._dq if start_media_s <= e.media_s <= end_media_s]

    def nearest(self, media_s: float) -> RingEntry | None:
        with self._lock:
            if not self._dq:
                return None
            return min(self._dq, key=lambda e: abs(e.media_s - media_s))

    def stats(self) -> RingStats:
        with self._lock:
            secs = (self._dq[-1].media_s - self._dq[0].media_s) if len(self._dq) > 1 else 0.0
            return RingStats(len(self._dq), self._bytes, secs, self.max_seconds, self.max_bytes,
                             self.evicted_by_time, self.evicted_by_bytes)

    @property
    def last_encode_ms(self) -> float:
        return self._last_encode_ms


@dataclass
class StageTimer:
    """Rolling per-stage timings (ms) with p50/p95 over a window."""

    window: int = 300
    samples: dict[str, deque] = field(default_factory=dict)

    def add(self, stage: str, ms: float) -> None:
        dq = self.samples.setdefault(stage, deque(maxlen=self.window))
        dq.append(ms)

    def mean(self, stage: str) -> float:
        dq = self.samples.get(stage)
        return float(sum(dq) / len(dq)) if dq else 0.0

    def percentile(self, stage: str, q: float) -> float:
        dq = self.samples.get(stage)
        if not dq:
            return 0.0
        return float(np.percentile(np.fromiter(dq, dtype=np.float64), q))

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {k: {"mean": self.mean(k), "p50": self.percentile(k, 50), "p95": self.percentile(k, 95), "n": len(v)}
                for k, v in self.samples.items()}
