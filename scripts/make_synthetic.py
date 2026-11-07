#!/usr/bin/env python
"""Synthetic robustness variants derived from evaluation videos (kept separate from real-data numbers).

  lighting   : brightness/contrast ramp (day→dusk) over the clip
  occlusion  : a static dark pillar covering part of the zone edge
  fast       : 2x frame rate (same frames, half the timestamps) to overload a realtime analyser
Outputs <name>-<variant>.mp4 next to the source and a JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--variants", default="lighting,occlusion,fast")
    args = ap.parse_args()
    src = Path(args.src)
    out_dir = Path(args.out_dir or src.parent)
    manifest = {"source": str(src), "variants": {}}
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-an"]
    for v in args.variants.split(","):
        dst = out_dir / f"{src.stem}-{v}.mp4"
        if v == "lighting":
            # brightness falls from +0 to -0.35 and contrast from 1.0 to 0.7 across the video
            vf = "eq=brightness='-0.35*t/70':contrast='1-0.3*t/70'"
            ffmpeg(["-i", str(src), "-vf", vf, *enc, str(dst)])
            manifest["variants"][v] = {"file": dst.name, "filter": vf}
        elif v == "occlusion":
            vf = "drawbox=x=iw*0.42:y=0:w=iw*0.08:h=ih:color=black@0.95:t=fill"
            ffmpeg(["-i", str(src), "-vf", vf, *enc, str(dst)])
            manifest["variants"][v] = {"file": dst.name, "filter": vf}
        elif v == "fast":
            # keep every frame but double the presentation rate -> same content at 2x input fps
            ffmpeg(["-i", str(src), "-vf", "setpts=0.5*PTS", "-r", "60", *enc, str(dst)])
            manifest["variants"][v] = {"file": dst.name, "filter": "setpts=0.5*PTS, 2x fps"}
        print("wrote", dst)
    (out_dir / f"{src.stem}-synthetic.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
