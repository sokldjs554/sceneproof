#!/usr/bin/env python
"""Build a static (GitHub Pages) replay of the console from a recorded bundle.

The console normally talks to /api/*. For a static host we serve the *same JSON the API would
return*, captured through FastAPI's TestClient from a temporary replay-mode server, under
<out>/data/, and point the console at it with window.SP_STATIC_BASE. Nothing is mocked: the
JSON is the real API output for the bundle. Write actions are disabled in static mode.

    python scripts/build_static_demo.py --bundle demo/bundle --out site
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="demo/bundle")
    ap.add_argument("--out", default="site")
    ap.add_argument("--videos", default="demo/videos", help="optional dir with the original mp4s for '원본 영상에서 보기'")
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "events").mkdir(parents=True)

    from fastapi.testclient import TestClient

    from sceneproof.server.app import create_app
    from sceneproof.server.config import ServerSettings

    tmp = Path(tempfile.mkdtemp(prefix="sp-static-"))
    settings = ServerSettings(data_dir=str(tmp), mode="replay", vlm_provider="none", replay_bundle=str(ROOT / args.bundle),
                              videos_dir=str(ROOT / args.videos), console_dir=str(ROOT / "console"), public_url="static")
    app = create_app(settings)
    with TestClient(app) as c:
        def save(path: str, name: str):
            r = c.get(path)
            r.raise_for_status()
            (out / "data" / name).write_text(json.dumps(r.json(), ensure_ascii=False))
            return r.json()

        save("/api/status", "status.json")
        events = save("/api/events?limit=2000", "events.json")
        save("/api/timeline", "timeline.json")
        save("/api/audit", "audit.json")
        save("/api/devices", "devices.json")
        save("/api/videos", "videos.json")
        save("/api/export?format=json", "export.json")
        for it in events["items"]:
            save(f"/api/events/{it['event_id']}", f"events/{it['event_id']}.json")
        # evidence files
        ev_src = tmp / "evidence"
        if ev_src.exists():
            shutil.copytree(ev_src, out / "data" / "evidence")
        vids = ROOT / args.videos
        if vids.exists():
            shutil.copytree(vids, out / "data" / "videos")
    # console files + static switch
    for f in (ROOT / "console").iterdir():
        if f.is_file():
            shutil.copy(f, out / f.name)
    index = (out / "index.html").read_text()
    inject = "<script>window.SP_STATIC_BASE='data';</script>"
    index = index.replace("<head>", "<head>" + inject, 1) if "<head>" in index else inject + index
    (out / "index.html").write_text(index)
    (out / ".nojekyll").write_text("")
    print(json.dumps({"out": str(out), "events": len(events["items"]), "files": sum(1 for _ in out.rglob("*") if _.is_file())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
