"""Where the ffmpeg program is: the one on PATH, else the copy bundled with the `imageio-ffmpeg` package.

Homebrew builds ffmpeg from source on Macs it no longer supports (hours); `pip install imageio-ffmpeg` brings a
ready-made binary with H.264 (libx264). Both the evidence writer and the dataset scripts use this lookup.
"""

from __future__ import annotations

import shutil


def ffmpeg_exe() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # not installed, or no binary for this platform
        return None


if __name__ == "__main__":  # used by shell scripts: FFMPEG=$(python -m sceneproof.ffmpeg)
    exe = ffmpeg_exe()
    if exe is None:
        raise SystemExit("ffmpeg not found: install it, or `pip install imageio-ffmpeg`")
    print(exe)
