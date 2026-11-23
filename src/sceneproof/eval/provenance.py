from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info(ffmpeg: str = "ffmpeg") -> dict[str, str | None]:
    import cv2
    import numpy
    import onnxruntime

    from sceneproof.edge.sources import gstreamer_available

    gst_version = None
    if gstreamer_available():
        from gi.repository import Gst

        gst_version = Gst.version_string()

    ffmpeg_version = None
    try:
        result = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True,
                                errors="replace", check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass
    else:
        lines = result.stdout.splitlines()
        if lines:
            ffmpeg_version = lines[0].strip() or None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "onnxruntime": onnxruntime.__version__,
        "ffmpeg": ffmpeg_version,
        "gstreamer": gst_version,
    }


def collect_inputs(paths: Iterable[str | Path], root: str | Path = ".") -> dict[str, dict]:
    base = Path(root)
    files = {}
    for name in paths:
        path = base / name
        files[str(name)] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return files


def diff_inputs(locked: dict[str, str], actual: dict[str, dict]) -> list[str]:
    problems = []
    for name, expected in locked.items():
        if name not in actual:
            problems.append(f"{name}: missing")
        elif actual[name]["sha256"] != expected:
            problems.append(f"{name}: expected {expected}, found {actual[name]['sha256']}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock", help="write input file hashes")
    lock.add_argument("--root", type=Path, default=Path("."))
    lock.add_argument("--out", type=Path, required=True)
    lock.add_argument("paths", nargs="+")
    verify = commands.add_parser("verify", help="compare input files with a lock file")
    verify.add_argument("--root", type=Path, default=Path("."))
    verify.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "lock":
            files = collect_inputs(args.paths, args.root)
            hashes = {name: info["sha256"] for name, info in files.items()}
            with NamedTemporaryFile(dir=args.out.parent, delete=False) as temp:
                temp_path = Path(temp.name)
            try:
                temp_path.write_text(json.dumps({"files": hashes}, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
                temp_path.replace(args.out)
            finally:
                temp_path.unlink(missing_ok=True)
            return 0

        manifest = json.loads(args.lock.read_text(encoding="utf-8"))
        locked = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(locked, dict) or not all(isinstance(v, str) for v in locked.values()):
            raise ValueError(f"{args.lock}: expected a 'files' mapping of paths to SHA-256 hashes")
        actual = {}
        for name in locked:
            try:
                actual[name] = {"sha256": file_sha256(args.root / name)}
            except FileNotFoundError:
                continue
        problems = diff_inputs(locked, actual)
        for problem in problems:
            print(problem, file=sys.stderr)
        return int(bool(problems))
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
