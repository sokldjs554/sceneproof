"""Do two event-evaluation results describe the same events?

    python -m sceneproof.eval.run_eval --variant yolox_tiny --out before.json     # on main
    python -m sceneproof.eval.run_eval --variant yolox_tiny --out after.json      # on your branch
    python scripts/compare_event_eval.py before.json after.json

Compares the totals and, per video, every count, timing error and predicted event. Fields that describe the run
itself (when, how fast, which commit) are ignored. Exit code 0 = same, 1 = different.
"""

from __future__ import annotations

import json
import sys

IGNORED_PER_VIDEO = {"run"}  # run conditions and speed: expected to differ between two runs


def flatten(value, prefix: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f"{prefix}{key}.")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from flatten(item, f"{prefix}{i}.")
    else:
        yield prefix[:-1], value


def comparable(result: dict) -> dict:
    videos = {name: {k: v for k, v in video.items() if k not in IGNORED_PER_VIDEO}
              for name, video in result["per_video"].items()}
    return dict(flatten({"aggregate": result["aggregate"], "per_video": videos}))


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    with open(sys.argv[1]) as f:
        before = comparable(json.load(f))
    with open(sys.argv[2]) as f:
        after = comparable(json.load(f))
    different = [k for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)]
    if not different:
        print(f"같음: 비교한 값 {len(before)}개가 모두 같습니다.")
        return 0
    print(f"다름: {len(different)}개 값이 다릅니다 (앞 20개).")
    for key in different[:20]:
        print(f"  {key}: {before.get(key)!r} -> {after.get(key)!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
