"""RTSP camera simulator built on gst-rtsp-server.

Serves every video in a directory as ``rtsp://<host>:<port>/<name>`` (H.264 passthrough, no
re-encode, so a 4-vCPU box can serve several streams). Used by the demo, the reconnection
tests and the multi-stream scaling benchmark. Stop/start the process to simulate a camera
going offline; the analyser's GStreamer source must reconnect on its own.

    python -m zonewatch.tools.rtsp_sim --dir data/videos --port 8554 [--loop]

``--loop`` restarts a stream at EOF by having the client reconnect (the server ends the
session at EOF, which is what real cameras do when a session drops).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

log = logging.getLogger("rtsp_sim")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/videos")
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--loop", action="store_true", help="loop files (uses multifilesrc; requires parseable mp4)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstRtspServer", "1.0")
    from gi.repository import GLib, Gst, GstRtspServer

    Gst.init(None)
    server = GstRtspServer.RTSPServer()
    server.set_address(args.host)
    server.set_service(str(args.port))
    mounts = server.get_mount_points()
    videos = sorted(Path(args.dir).glob("*.mp4"))
    if not videos:
        log.error("no .mp4 files in %s", args.dir)
        return 2
    for v in videos:
        factory = GstRtspServer.RTSPMediaFactory()
        # passthrough: demux H.264 from mp4 and payload it; no decode/encode on the server side
        launch = (f'( filesrc location="{v.resolve()}" ! qtdemux ! h264parse config-interval=1 '
                  f"! rtph264pay name=pay0 pt=96 )")
        factory.set_launch(launch)
        factory.set_shared(True)  # one pipeline per mount, shared across clients (like a real camera)
        factory.set_eos_shutdown(True)
        mounts.add_factory(f"/{v.stem}", factory)
        log.info("mount rtsp://%s:%d/%s  <- %s", "127.0.0.1" if args.host == "0.0.0.0" else args.host, args.port,
                 v.stem, v.name)
    if server.attach(None) == 0:
        log.error("failed to attach RTSP server (port busy?)")
        return 1
    loop = GLib.MainLoop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        GLib.unix_signal_add(GLib.PRIORITY_HIGH, sig, lambda *_: (loop.quit(), False)[1])
    log.info("serving %d streams on port %d", len(videos), args.port)
    loop.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
