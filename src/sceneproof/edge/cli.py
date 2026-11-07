"""sceneproof-edge: run the analyser against the gateway, or process a video offline."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sceneproof-edge")
    ap.add_argument("--config", default="configs/edge.yaml")
    ap.add_argument("--log-level", default="INFO")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="live mode: cameras with autostart, gateway sink, command subscription")
    r.add_argument("--camera", action="append", default=[], help="start these cameras (in addition to autostart)")
    r.add_argument("--gateway", default=None)
    o = sub.add_parser("offline", help="process one camera's video to the end, write events to JSONL")
    o.add_argument("--camera", required=True)
    o.add_argument("--source", default=None)
    o.add_argument("--out", required=True)
    o.add_argument("--realtime", action="store_true")
    o.add_argument("--run-id", default=None)
    o.add_argument("--summary", default=None, help="write run summary JSON here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.cmd == "offline":
        from .agent import run_offline

        summary = run_offline(cfg, args.camera, args.out, args.source, args.realtime, args.run_id)
        text = json.dumps(summary, indent=2, default=str)
        if args.summary:
            with open(args.summary, "w") as f:
                f.write(text)
        print(text)
        return 0
    from .agent import EdgeAgent

    if args.gateway:
        cfg = cfg.model_copy(update={"gateway": args.gateway})
    agent = EdgeAgent(cfg)
    agent.autostart()
    for cid in args.camera:
        agent.start_camera(cid)

    def _sig(*_):
        agent.shutdown()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    agent.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
