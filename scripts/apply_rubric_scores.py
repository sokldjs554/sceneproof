"""Writes the scores exported from tools/review/vlm_rubric.html into docs/eval/raw/vlm_eval-rubric.csv (grader columns).

    python scripts/apply_rubric_scores.py ~/Downloads/vlm_rubric.<name>.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "docs" / "eval" / "raw" / "vlm_eval-rubric.csv"
FIELDS = {"observations_match_frames": "grader_observations_match_frames_0_1", "hallucination": "grader_hallucination_0_1",
          "uncertainties_sensible": "grader_uncertainties_sensible_0_1", "readability": "grader_readability_1_3"}


def main() -> int:
    data = json.loads(Path(sys.argv[1]).read_text())
    if data.get("page") != "vlm_rubric":
        print("not a vlm_rubric export", file=sys.stderr)
        return 2
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    filled = 0
    for r in rows:
        a = data["answers"].get(r["event_id"], {})
        if not a:
            continue
        for k, col in FIELDS.items():
            if k in a:
                r[col] = a[k].split()[0]  # "1 맞음" -> "1"
        r["grader_hallucination_note"] = a.get("note", r.get("grader_hallucination_note", ""))
        r["grader_name"], r["grader_date"] = data.get("who", ""), data.get("saved", "")[:10]
        filled += 1
    with CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{filled} / {len(rows)} rows scored -> {CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
