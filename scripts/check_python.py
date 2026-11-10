"""Checks, before `make setup` creates the venv, that this Python can install the analyzer's inference runtime.

onnxruntime publishes install files (wheels) only for some Python version x OS x CPU combinations. When none
fits, pip prints only "No matching distribution found for onnxruntime", which does not say what to change.
This script says it in plain words. It uses only the standard library (it runs before anything is installed).

    python3 scripts/check_python.py          # exit 0 = fine, 1 = this Python cannot install onnxruntime
"""

from __future__ import annotations

import platform
import sys

TESTED = (3, 12)  # the version CI runs


def problem() -> str | None:
    v = sys.version_info[:2]
    system, machine = platform.system(), platform.machine().lower()
    bits = 64 if sys.maxsize > 2**32 else 32
    if v < (3, 11):
        return f"Python {v[0]}.{v[1]}은 너무 오래되었습니다(3.11 이상 필요)."
    if v > (3, 14):
        return f"Python {v[0]}.{v[1]}용 onnxruntime 설치 파일이 아직 없습니다(3.11–3.14 지원)."
    if bits == 32:
        return "32비트 Python입니다. onnxruntime은 64비트 Python만 지원합니다."
    if system == "Darwin":
        mac = tuple(int(x) for x in (platform.mac_ver()[0] or "0").split(".")[:2])
        if machine in ("x86_64", "amd64"):
            if v > (3, 13):
                return (f"인텔 Mac에서 Python {v[0]}.{v[1]}용 onnxruntime 설치 파일이 없습니다"
                        "(인텔 Mac은 onnxruntime 1.23까지만 지원, Python 3.13 이하).")
            if mac and mac < (13, 0):
                return f"macOS {platform.mac_ver()[0]}입니다. 인텔 Mac용 onnxruntime은 macOS 13 이상이 필요합니다."
        elif mac and mac < (14, 0) and v > (3, 13):
            return f"macOS {platform.mac_ver()[0]}에서 Python {v[0]}.{v[1]}용 onnxruntime 설치 파일이 없습니다(macOS 14 이상 필요)."
    return None


def main() -> int:
    v = sys.version_info
    where = f"Python {v.major}.{v.minor}.{v.micro} · {platform.system()} {platform.mac_ver()[0] or platform.release()} · {platform.machine()}"
    p = problem()
    if p is None:
        note = "" if (v.major, v.minor) == TESTED else f" (자동 검사는 Python {TESTED[0]}.{TESTED[1]}로 합니다)"
        print(f"[check_python] OK: {where}{note}")
        return 0
    print(f"[check_python] 이 Python으로는 설치할 수 없습니다: {where}\n  이유: {p}\n"
          f"  해결: Python {TESTED[0]}.{TESTED[1]}을 설치한 뒤 그 Python으로 다시 실행하세요.\n"
          f"    make setup PYTHON=python{TESTED[0]}.{TESTED[1]}\n"
          "  (Mac: brew install python@3.12 · Windows/Mac: python.org 설치 파일 · 공통: uv python install 3.12)",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
