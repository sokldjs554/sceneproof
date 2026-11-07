"""zonewatch-server: run the review server, import/export replay bundles."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import ServerSettings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="zonewatch-server")
    ap.add_argument("--log-level", default="INFO")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="serve HTTP (+console) and gRPC ingest")
    r.add_argument("--port", type=int, default=None)
    r.add_argument("--grpc", default=None)
    r.add_argument("--mode", default=None, choices=["live", "replay"])
    r.add_argument("--vlm", default=None, help="mock|anthropic|openai_compat|transformers|none")
    r.add_argument("--replay-bundle", default=None)
    r.add_argument("--data-dir", default=None)
    i = sub.add_parser("import", help="import a recorded run bundle into the DB")
    i.add_argument("bundle")
    i.add_argument("--data-dir", default=None)
    i.add_argument("--source", default="replay")
    e = sub.add_parser("export", help="export the DB to a bundle dir")
    e.add_argument("out")
    e.add_argument("--data-dir", default=None)
    e.add_argument("--camera", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    overrides = {k: v for k, v in {
        "http_port": getattr(args, "port", None), "grpc_listen": getattr(args, "grpc", None),
        "mode": getattr(args, "mode", None), "vlm_provider": getattr(args, "vlm", None),
        "replay_bundle": getattr(args, "replay_bundle", None), "data_dir": getattr(args, "data_dir", None),
    }.items() if v is not None}
    s = ServerSettings(**overrides)
    if args.cmd == "run":
        from .app import run

        run(s)
        return 0
    from .db import Store

    store = Store(s.resolved_db())
    if args.cmd == "import":
        from .replay import import_bundle

        print(json.dumps(import_bundle(store, args.bundle, s.resolved_evidence(), None, args.source), indent=2))
    elif args.cmd == "export":
        from .replay import export_bundle

        print(json.dumps(export_bundle(store, s.resolved_evidence(), args.out, args.camera), indent=2))
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
