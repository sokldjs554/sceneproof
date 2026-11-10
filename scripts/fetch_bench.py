"""Download the research-licensed benchmark (configs/bench/sources.yaml) and prepare it for evaluation.

For every sequence:
  * frames come from the ORIGINAL host (CAVIAR JPEG tarballs; MOT16/MOT20 zips read with HTTP range
    requests so only the needed sequences are transferred),
  * frames are re-encoded to H.264 at the dataset frame rate so that decoded video frame i is exactly
    ground-truth frame i (CAVIAR's MPEG files differ from its JPEG/GT numbering by one frame),
  * the hand-made ground truth is stored next to it unchanged,
  * data/bench/manifest.json records sizes, sha256 of what was downloaded, frame counts and licences;
    a sequence is skipped on a re-run only if its manifest entry is error-free and its frame count matches.

Nothing here is committed: data/bench/ is git-ignored. Usage:
    python scripts/fetch_bench.py [--only caviar,mot16,mot20] [--ids id1,id2] [--jobs 4]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import ssl
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

from sceneproof.ffmpeg import ffmpeg_exe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "bench"


def _ssl_context() -> ssl.SSLContext:
    """System trust store PLUS certifi's: Python from python.org on macOS ships without access to the system
    certificates until 'Install Certificates.command' is run, so plain urllib fails with CERTIFICATE_VERIFY_FAILED
    (curl, which uses the system store, works on the same machine)."""
    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx.load_verify_locations(certifi.where())
    except Exception:
        pass
    return ctx


_SSL = _ssl_context()


def _get(url: str, tries: int = 5) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sceneproof-bench"}), timeout=120,
                                        context=_SSL) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  retry {i + 1} {url}: {e}", flush=True)
            time.sleep(3 * (i + 1))
    raise RuntimeError("unreachable")


def _encode(frames, fps: float, dst: Path) -> tuple[int, int]:
    """Lossless-enough H.264 (crf 18) at a constant frame rate; one encoded frame per input frame.
    ``frames`` may be a list or a generator (decoded lazily, so a 2,800-frame 1080p sequence never sits in RAM)."""
    it = iter(frames)
    first = next(it)
    h, w = first.shape[:2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    # encode into a temporary file and rename only on success: an interrupted run never leaves a short video that a
    # later run would mistake for a finished one
    tmp = dst.with_name(dst.stem + ".partial.mp4")
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg not found: install ffmpeg or `pip install imageio-ffmpeg`")
    cmd = [exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-g", str(int(fps * 2)), "-movflags", "+faststart", "-an", str(tmp)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert p.stdin is not None
    try:
        p.stdin.write(first.tobytes())
        for f in it:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            p.stdin.write(f.tobytes())
        p.stdin.close()
        if p.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {dst}")
    except BaseException:
        p.kill()
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dst)
    return w, h


def _count_frames(path: Path) -> int:
    """Decoded frame count: ffprobe when installed, else decode with OpenCV (the bundled ffmpeg has no ffprobe)."""
    if shutil.which("ffprobe"):
        out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                              "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
        return int(out.strip().split(",")[0])
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def fetch_caviar(seq: dict, base: str) -> dict:
    vid = OUT / "videos" / f"{seq['id']}.mp4"
    gt = OUT / "gt" / f"{seq['id']}.xml"
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
    _encode([frames[i] for i in idx], seq["fps"], vid)
    gt.parent.mkdir(parents=True, exist_ok=True)
    gt.write_bytes(gt_bytes)
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
    pre = seq["zip_prefix"]
    with RemoteZip(zip_url) as z:
        names = [n for n in z.namelist() if n.startswith(pre)]
    jpgs = sorted(n for n in names if n.startswith(pre + "img1/") and n.endswith(".jpg"))

    # each worker keeps its own range-request reader; a dropped connection reopens it and retries the file
    import threading

    local = threading.local()

    def read(name: str, tries: int = 6) -> bytes:
        for i in range(tries):
            try:
                if getattr(local, "z", None) is None:
                    local.z = RemoteZip(zip_url)
                return local.z.read(name)
            except Exception as e:
                local.z = None
                if i == tries - 1:
                    raise
                print(f"  retry {i + 1} {name}: {str(e)[:80]}", flush=True)
                time.sleep(2 * (i + 1))
        raise RuntimeError("unreachable")

    gt_bytes = read(pre + "gt/gt.txt")
    ini = read(pre + "seqinfo.ini").decode()
    with ThreadPoolExecutor(max_workers=8) as ex:
        blobs = list(ex.map(read, jpgs))
    h = hashlib.sha256()
    for data in blobs:
        h.update(data)
    seqlen = int(re.search(r"seqLength=(\d+)", ini).group(1))
    if len(blobs) != seqlen:
        raise RuntimeError(f"{seq['id']}: {len(blobs)} frames but seqinfo says {seqlen}")
    # keep only the compressed JPEGs in memory (~0.4 MB each) and decode one frame at a time into ffmpeg
    ww, hh = _encode((cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in blobs), seq["fps"], vid)
    del blobs
    gt.parent.mkdir(parents=True, exist_ok=True)
    gt.write_bytes(gt_bytes)
    (OUT / "gt" / f"{seq['id']}.seqinfo.ini").write_text(ini)
    n_vid = _count_frames(vid)
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

    import threading

    lock = threading.Lock()

    def save(r: dict) -> None:
        with lock:
            cur = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sequences": {}}
            cur["datasets"] = src["datasets"]
            if not r.get("skipped"):
                cur["sequences"][r["id"]] = r
            cur["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = manifest_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cur, indent=1, ensure_ascii=False))
            os.replace(tmp, manifest_path)  # a crash mid-write cannot leave a truncated manifest

    def verified(seq: dict) -> bool:
        """Skip only a sequence whose video and GT exist AND whose manifest entry says it decoded completely."""
        ext = "xml" if seq["dataset"] == "caviar" else "txt"
        if not ((OUT / "videos" / f"{seq['id']}.mp4").exists() and (OUT / "gt" / f"{seq['id']}.{ext}").exists()):
            return False
        cur = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sequences": {}}
        m = cur["sequences"].get(seq["id"], {})
        return bool(m) and not m.get("error") and bool(m.get("frames_match_gt"))

    def work(seq: dict) -> dict:
        t0 = time.time()
        ds = src["datasets"][seq["dataset"]]
        if verified(seq):
            r = {"id": seq["id"], "skipped": True}
            print(f"[bench] {seq['id']}: skipped (verified in manifest)", flush=True)
            return r
        try:
            r = fetch_caviar(seq, ds["base_url"]) if seq["dataset"] == "caviar" else fetch_mot(seq, ds["zip_url"])
        except Exception as e:
            r = {"id": seq["id"], "error": str(e)[:300]}
        r.update({"site": seq["site"], "dataset": seq["dataset"], "group": seq["group"]})
        print(f"[bench] {seq['id']}: {r.get('error') or str(r.get('frames')) + ' frames'}"
              f" ({time.time() - t0:.0f}s)", flush=True)
        save(r)
        return r

    # MOT sequences are large single-zip reads: run them one at a time; CAVIAR in parallel
    cav = [s for s in seqs if s["dataset"] == "caviar"]
    mot = [s for s in seqs if s["dataset"] != "caviar"]
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        results += list(ex.map(work, cav))
    results += [work(s) for s in mot]
    bad = [r for r in results if r.get("error") or (not r.get("skipped") and not r.get("frames_match_gt", True))]
    print(f"[bench] done: {len(results)} sequences, problems: {[r['id'] for r in bad] or 'none'} -> {manifest_path}")
    for r in bad:
        print(f"[bench]   {r['id']}: {r.get('error') or 'video frame count differs from ground truth'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
