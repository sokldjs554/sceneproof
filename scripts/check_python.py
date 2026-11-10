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
        intel = machine in ("x86_64", "amd64")
        # newest Python that has onnxruntime (and OpenCV) install files for this macOS / CPU
        if intel:
            if mac and mac < (12, 0):
                return f"macOS {platform.mac_ver()[0]}입니다. 인텔 Mac은 macOS 12 이상이 필요합니다(영상 라이브러리 OpenCV)."
            newest = (3, 12) if mac and mac < (13, 0) else (3, 13)
        else:
            newest = (3, 12) if mac and mac < (13, 0) else (3, 13) if mac and mac < (14, 0) else (3, 14)
        if v > newest:
            return (f"{'인텔' if intel else 'Apple 칩'} Mac, macOS {platform.mac_ver()[0]}에서는 onnxruntime 설치 파일이 "
                    f"Python {newest[0]}.{newest[1]}까지만 있습니다(지금 {v[0]}.{v[1]}).")
    return None


def main() -> int:
    v = sys.version_info
    where = f"Python {v.major}.{v.minor}.{v.micro} · {platform.system()} {platform.mac_ver()[0] or platform.release()} · {platform.machine()}"
    p = problem()
    if p is None:
        note = "" if (v.major, v.minor) == TESTED else f" (자동 검사는 Python {TESTED[0]}.{TESTED[1]}로 합니다)"
        print(f"[check_python] OK: {where}{note}")
        if platform.system() == "Darwin" and platform.machine().lower() in ("x86_64", "amd64"):
            print("[check_python] 인텔 Mac: PyTorch 설치 파일이 없어 추가 학습·PyTorch 비교는 못 합니다. "
                  "나머지(분석·평가·검토 도구)는 공식 ONNX 모델로 동작합니다.")
        return 0
    print(f"[check_python] 이 Python으로는 설치할 수 없습니다: {where}\n  이유: {p}\n"
          f"  해결: Python {TESTED[0]}.{TESTED[1]}을 설치한 뒤 그 Python으로 다시 실행하세요.\n"
          f"    make setup PYTHON=python{TESTED[0]}.{TESTED[1]}\n"
          "  (가장 빠른 방법: python.org의 Python 3.12 설치 파일. Homebrew는 지원 밖(Tier 3) Mac에서 직접 빌드해 오래 걸립니다.)",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
