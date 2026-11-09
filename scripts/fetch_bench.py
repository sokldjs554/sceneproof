"""Download the research-licensed benchmark (configs/bench/sources.yaml) and prepare it for evaluation.

For every sequence:
  * frames come from the ORIGINAL host (CAVIAR JPEG tarballs; MOT16/MOT20 zips read with HTTP range
    requests so only the needed sequences are transferred),
  * frames are re-encoded to H.264 at the dataset frame rate so that decoded video frame i is exactly
    ground-truth frame i (CAVIAR's MPEG files differ from its JPEG/GT numbering by one frame),
  * the hand-made ground truth is stored next to it unchanged,
  * data/bench/manifest.json records sizes, sha256 of what was downloaded, frame counts and licences.

Nothing here is committed: data/bench/ is git-ignored. Usage:
    python scripts/fetch_bench.py [--only caviar,mot16,mot20] [--ids id1,id2] [--jobs 4]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "bench"


def _get(url: str, tries: int = 5) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sceneproof-bench"}), timeout=120) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  retry {i + 1} {url}: {e}", flush=True)
            time.sleep(3 * (i + 1))
    raise RuntimeError("unreachable")


def _encode(frames: list[np.ndarray], fps: float, dst: Path) -> None:
    """Lossless-enough H.264 (crf 18) at a constant frame rate; one encoded frame per input frame."""
    h, w = frames[0].shape[:2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-g", str(int(fps * 2)), "-movflags", "+faststart", "-an", str(dst)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert p.stdin is not None
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {dst}")


def _count_frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return int(out.strip().split(",")[0])


def fetch_caviar(seq: dict, base: str) -> dict:
    vid = OUT / "videos" / f"{seq['id']}.mp4"
    gt = OUT / "gt" / f"{seq['id']}.xml"
    if vid.exists() and gt.exists():
        return {"id": seq["id"], "skipped": True}
    tar_bytes = _get(base + seq["frames"])
    gt_bytes = _get(base + seq["gt"])
    folder = seq["frames"].rsplit("/", 2)[-2]
    frames: dict[int, np.ndarray] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.lower().endswith(".jpg"):
                continue
            stem = Path(m.name).stem  # e.g. Walk1001, Rest_WiggleOnFloor_01190.ppm, OneLeaveShop1cor0001
            rest = stem[len(folder):] if stem.startswith(folder) else stem
            num = re.search(r"(\d+)(?:\.ppm)?$", rest)
            if num is None:
                raise RuntimeError(f"{seq['id']}: cannot read a frame number from {m.name}")
            data = tf.extractfile(m).read()
            frames[int(num.group(1))] = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    idx = sorted(frames)
    n_gt = len(re.findall(rb"<frame number=", gt_bytes))
    note = ""
    run = [i for k, i in enumerate(idx) if i == idx[0] + k]  # contiguous run from the first file
    if len(run) != len(idx) or len(run) > n_gt:
        # Rest_WiggleOnFloor: JPEG 1189 is missing and the hand-made GT stops at frame 1188 -> keep the GT span
        if len(run) < n_gt:
            raise RuntimeError(f"{seq['id']}: JPEG numbering has gaps inside the ground-truth span ({idx[0]}..{idx[-1]})")
        note = f"kept frames {run[0]}..{run[n_gt - 1]} (the ground-truth span) of {len(idx)} JPEGs"
        idx = run[:n_gt]
    gt.parent.mkdir(parents=True, exist_ok=True)
    gt.write_bytes(gt_bytes)
    _encode([frames[i] for i in idx], seq["fps"], vid)
    n_vid = _count_frames(vid)
    h, w = frames[idx[0]].shape[:2]
    return {"id": seq["id"], "frames": n_vid, "gt_frames": n_gt, "jpeg_first_index": idx[0], "width": w, "height": h,
            "fps": seq["fps"], "duration_s": round(n_vid / seq["fps"], 2),
            "sha256_frames_archive": hashlib.sha256(tar_bytes).hexdigest(), "sha256_gt": hashlib.sha256(gt_bytes).hexdigest(),
            "frames_match_gt": n_vid == n_gt, **({"note": note} if note else {})}


def fetch_mot(seq: dict, zip_url: str) -> dict:
    from remotezip import RemoteZip

    vid = OUT / "videos" / f"{seq['id']}.mp4"
    gt = OUT / "gt" / f"{seq['id']}.txt"
    if vid.exists() and gt.exists():
        return {"id": seq["id"], "skipped": True}
    pre = seq["zip_prefix"]
    with RemoteZip(zip_url) as z:
        names = [n for n in z.namelist() if n.startswith(pre)]
        jpgs = sorted(n for n in names if n.startswith(pre + "img1/") and n.endswith(".jpg"))
        gt_bytes = z.read(pre + "gt/gt.txt")
        ini = z.read(pre + "seqinfo.ini").decode()
        h = hashlib.sha256()
        frames = []
        for n in jpgs:
            data = z.read(n)
            h.update(data)
            frames.append(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))
    seqlen = int(re.search(r"seqLength=(\d+)", ini).group(1))
    if len(frames) != seqlen:
        raise RuntimeError(f"{seq['id']}: {len(frames)} frames but seqinfo says {seqlen}")
    gt.parent.mkdir(parents=True, exist_ok=True)
    gt.write_bytes(gt_bytes)
    (OUT / "gt" / f"{seq['id']}.seqinfo.ini").write_text(ini)
    _encode(frames, seq["fps"], vid)
    n_vid = _count_frames(vid)
    hh, ww = frames[0].shape[:2]
    return {"id": seq["id"], "frames": n_vid, "gt_frames": seqlen, "width": ww, "height": hh, "fps": seq["fps"],
            "duration_s": round(n_vid / seq["fps"], 2), "sha256_frames": h.hexdigest(),
            "sha256_gt": hashlib.sha256(gt_bytes).hexdigest(), "frames_match_gt": n_vid == seqlen}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(ROOT / "configs" / "bench" / "sources.yaml"))
    ap.add_argument("--only", default="", help="comma-separated datasets (caviar,mot16,mot20)")
    ap.add_argument("--ids", default="", help="comma-separated sequence ids")
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    src = yaml.safe_load(Path(args.sources).read_text())
    seqs = [s for s in src["sequences"] if (not args.only or s["dataset"] in args.only.split(","))
            and (not args.ids or s["id"] in args.ids.split(","))]
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sequences": {}}
    manifest["datasets"] = src["datasets"]

    def work(seq: dict) -> dict:
        t0 = time.time()
        ds = src["datasets"][seq["dataset"]]
        try:
            r = fetch_caviar(seq, ds["base_url"]) if seq["dataset"] == "caviar" else fetch_mot(seq, ds["zip_url"])
        except Exception as e:
            r = {"id": seq["id"], "error": str(e)[:300]}
        r.update({"site": seq["site"], "dataset": seq["dataset"], "group": seq["group"]})
        print(f"[bench] {seq['id']}: {'skipped (exists)' if r.get('skipped') else r.get('error') or str(r.get('frames')) + ' frames'}"
              f" ({time.time() - t0:.0f}s)", flush=True)
        return r

    # MOT sequences are large single-zip reads: run them one at a time; CAVIAR in parallel
    cav = [s for s in seqs if s["dataset"] == "caviar"]
    mot = [s for s in seqs if s["dataset"] != "caviar"]
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        results += list(ex.map(work, cav))
    results += [work(s) for s in mot]
    for r in results:
        if not r.get("skipped"):
            manifest["sequences"][r["id"]] = r
    manifest["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    bad = [r["id"] for r in results if r.get("error") or (not r.get("skipped") and not r.get("frames_match_gt", True))]
    print(f"[bench] done: {len(results)} sequences, problems: {bad or 'none'} -> {manifest_path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
