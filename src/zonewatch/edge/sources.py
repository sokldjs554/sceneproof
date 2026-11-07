"""Video sources: OpenCV (file / any FFmpeg URI) and GStreamer (file / RTSP with reconnection).

Both produce ``Frame`` objects into a ``FrameQueue`` from their own thread and expose a
``SourceStatus`` the status reporter publishes. The GStreamer source is the production path:
``rtspsrc`` handles RTSP/RTP, ``decodebin`` picks the decoder, and ``appsink`` with
``drop=true max-buffers=2`` already discards frames at the decoder when the consumer is slow,
before they are even copied into Python.

Reconnection: a stream error or EOS on an RTSP URI puts the source into RECONNECTING; it
rebuilds the pipeline after an exponential backoff (0.5s .. 10s). A file URI ending normally
goes to ENDED (or loops when ``loop=True``).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .frames import Frame, FrameQueue

log = logging.getLogger(__name__)


class SourceState(Enum):
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    ENDED = "ended"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SourceStatus:
    state: SourceState = SourceState.CONNECTING
    frames: int = 0
    reconnects: int = 0
    last_error: str = ""
    last_frame_wall_ts: float = 0.0
    width: int = 0
    height: int = 0
    fps_nominal: float = 0.0
    _fps_window: list[float] = field(default_factory=list)

    def input_fps(self) -> float:
        now = time.time()
        self._fps_window = [t for t in self._fps_window if now - t <= 2.0]
        return len(self._fps_window) / 2.0


class VideoSource:
    """Base class. Subclasses implement ``_run`` and push frames via ``_emit``."""

    def __init__(self, uri: str, queue: FrameQueue, realtime: bool = True, loop: bool = False,
                 speed: float = 1.0) -> None:
        self.uri = uri
        self.queue = queue
        self.realtime = realtime
        self.loop = loop
        self.speed = speed
        self.status = SourceStatus()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._index = 0
        self._media_offset = 0.0  # accumulates across loops / reconnects so media_s is monotonic

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_guarded, name=f"source:{self.uri[:40]}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        if self.status.state not in (SourceState.ENDED, SourceState.ERROR):
            self.status.state = SourceState.STOPPED

    def _run_guarded(self) -> None:
        try:
            self._run()
        except Exception as e:
            log.exception("source crashed: %s", self.uri)
            self.status.state = SourceState.ERROR
            self.status.last_error = str(e)

    def _run(self) -> None:
        raise NotImplementedError

    def _emit(self, image: np.ndarray, media_s: float) -> None:
        now = time.time()
        self.status.frames += 1
        self.status.last_frame_wall_ts = now
        self.status._fps_window.append(now)
        self.status.height, self.status.width = image.shape[:2]
        self.queue.put(Frame(self._index, self._media_offset + media_s, now, image, media_s))
        self._index += 1

    @property
    def is_live(self) -> bool:
        return urlparse(self.uri).scheme in ("rtsp", "rtsps", "rtmp", "udp", "srt")


def _uri_to_path(uri: str) -> str:
    """file://relative/or/absolute path → path (relative file URIs are allowed in configs)."""
    if uri.startswith("file://"):
        return uri[len("file://"):]
    return uri


class OpenCVSource(VideoSource):
    """FFmpeg-backed decode via cv2.VideoCapture. Paces playback to the file's fps when ``realtime``."""

    def _run(self) -> None:
        import cv2

        backoff = 0.5
        while not self._stop.is_set():
            path = _uri_to_path(self.uri)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.status.last_error = f"cannot open {self.uri}"
                if self.is_live:
                    self.status.state = SourceState.RECONNECTING
                    self.status.reconnects += 1
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 10.0)
                    continue
                self.status.state = SourceState.ERROR
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.status.fps_nominal = fps
            self.status.state = SourceState.STREAMING
            backoff = 0.5
            t_start = time.perf_counter()
            local_index = 0
            last_media = 0.0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                media_s = local_index / fps
                if self.realtime and not self.is_live:
                    target = t_start + media_s / self.speed
                    delay = target - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                self._emit(frame, media_s)
                last_media = media_s
                local_index += 1
            cap.release()
            if self._stop.is_set():
                return
            if self.is_live:
                self.status.state = SourceState.RECONNECTING
                self.status.reconnects += 1
                self.status.last_error = "stream ended/disconnected"
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
                continue
            self._media_offset += last_media + 1.0 / fps
            if self.loop:
                continue
            self.status.state = SourceState.ENDED
            return


class GStreamerSource(VideoSource):
    """uridecodebin/rtspsrc → videoconvert → appsink(BGR). Requires PyGObject + GStreamer 1.x."""

    def __init__(self, uri: str, queue: FrameQueue, realtime: bool = True, loop: bool = False, speed: float = 1.0,
                 rtsp_latency_ms: int = 200, rtsp_protocols: str = "tcp", max_width: int = 0) -> None:
        super().__init__(uri, queue, realtime, loop, speed)
        self.rtsp_latency_ms = rtsp_latency_ms
        self.rtsp_protocols = rtsp_protocols
        self.max_width = max_width
        self._pipeline = None
        self._mainloop = None

    def _launch_string(self) -> str:
        u = urlparse(self.uri)
        caps = "video/x-raw,format=BGR"
        if u.scheme in ("rtsp", "rtsps"):
            src = (f'rtspsrc location="{self.uri}" latency={self.rtsp_latency_ms} protocols={self.rtsp_protocols} '
                   f"tcp-timeout=5000000 teardown-timeout=1000000 ! decodebin")
            # live: never block the RTP path; appsink drops stale decoded frames itself
            sink = "appsink name=sink emit-signals=true max-buffers=2 drop=true sync=false"
            q = "queue max-size-buffers=4 leaky=downstream"
        else:
            path = _uri_to_path(self.uri)
            src = f'filesrc location="{Path(path).resolve()}" ! decodebin'
            # file: pace with the pipeline clock (sync=true). drop=true must NOT be combined with
            # sync=true here: the decoder runs ahead, appsink would keep replacing the buffer it is
            # waiting to render with a newer one and almost nothing gets delivered (observed: 2 frames
            # in 4 s). With drop=false the decoder simply blocks on the sink, which is the pacing we want.
            sync = "true" if self.realtime else "false"
            sink = f"appsink name=sink emit-signals=true max-buffers=4 drop=false sync={sync}"
            q = "queue max-size-buffers=4"
        scale = ""
        if self.max_width:
            scale = f"! videoscale ! video/x-raw,width={self.max_width},pixel-aspect-ratio=1/1 "
        return f"{src} ! {q} ! videoconvert {scale}! {caps} ! {sink}"

    def _run_once(self, Gst, GLib) -> tuple[dict, float]:
        """Builds one pipeline, runs it until EOS/ERROR/stop. Returns ({eos, error}, last_media_s)."""
        pipeline = Gst.parse_launch(self._launch_string())
        sink = pipeline.get_by_name("sink")
        bus = pipeline.get_bus()
        done = threading.Event()
        result = {"eos": False, "error": ""}
        counters = {"index": 0, "last_media": 0.0, "first_pts": None}

        def on_sample(appsink):
            sample = appsink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.OK
            buf = sample.get_buffer()
            caps_ = sample.get_caps().get_structure(0)
            w, h = caps_.get_value("width"), caps_.get_value("height")
            ok, mi = buf.map(Gst.MapFlags.READ)
            if not ok:
                return Gst.FlowReturn.OK
            try:
                arr = np.frombuffer(mi.data, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3).copy()
            finally:
                buf.unmap(mi)
            pts = buf.pts / 1e9 if buf.pts != Gst.CLOCK_TIME_NONE else counters["index"] / 25.0
            if counters["first_pts"] is None:
                counters["first_pts"] = pts
            media_s = max(0.0, pts - counters["first_pts"])
            if self.status.state != SourceState.STREAMING:
                self.status.state = SourceState.STREAMING
            self._emit(arr, media_s)
            counters["last_media"] = media_s
            counters["index"] += 1
            return Gst.FlowReturn.OK

        def on_bus(bus_, msg):
            t = msg.type
            if t == Gst.MessageType.EOS:
                result["eos"] = True
                done.set()
            elif t == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                result["error"] = f"{err.message} ({dbg or ''})"[:300]
                done.set()
            return True

        sink.connect("new-sample", on_sample)
        bus.add_signal_watch()
        bus.connect("message", on_bus)
        pipeline.set_state(Gst.State.PLAYING)
        self._pipeline = pipeline
        loop = GLib.MainLoop()  # GLib main loop in this thread so bus signals are dispatched
        self._mainloop = loop

        def poll():
            if done.is_set() or self._stop.is_set():
                loop.quit()
                return False
            return True

        GLib.timeout_add(100, poll)
        loop.run()
        pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()
        return result, counters["last_media"]

    def _run(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import GLib, Gst, GstApp  # noqa: F401

        Gst.init(None)
        backoff = 0.5
        while not self._stop.is_set():
            self.status.state = SourceState.CONNECTING if self.status.reconnects == 0 else SourceState.RECONNECTING
            result, last_media = self._run_once(Gst, GLib)
            if self._stop.is_set():
                return
            if result["error"]:
                self.status.last_error = result["error"]
                log.warning("gst error on %s: %s", self.uri, result["error"])
            if self.is_live or result["error"]:
                if not self.is_live and not self.loop:
                    self.status.state = SourceState.ERROR
                    return
                self.status.state = SourceState.RECONNECTING
                self.status.reconnects += 1
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
                self._media_offset += last_media
                continue
            backoff = 0.5
            self._media_offset += last_media + 0.04
            if self.loop:
                continue
            self.status.state = SourceState.ENDED
            return


def gstreamer_available() -> bool:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # noqa: F401

        return True
    except (ImportError, ValueError):
        return False


def open_source(uri: str, queue: FrameQueue, backend: str = "auto", **kw) -> VideoSource:
    """backend: auto (GStreamer if importable, else OpenCV) | gstreamer | opencv"""
    if backend == "auto":
        backend = "gstreamer" if gstreamer_available() else "opencv"
    if backend == "gstreamer":
        return GStreamerSource(uri, queue, **kw)
    if backend == "opencv":
        kw.pop("rtsp_latency_ms", None)
        kw.pop("rtsp_protocols", None)
        kw.pop("max_width", None)
        return OpenCVSource(uri, queue, **kw)
    raise ValueError(f"unknown source backend {backend!r}")
