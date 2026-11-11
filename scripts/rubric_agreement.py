"""Compares a second person's scores of the Claude explanations with the first grader's (docs/eval/raw/vlm_eval-rubric.csv).

The second person scores on tools/review/vlm_rubric.html (press 새로 시작 first if the same browser was used before)
and exports a JSON file. The export is stored without the person's name (as "grader2"), and the agreement per
question is written next to it:

    python scripts/rubric_agreement.py ~/Downloads/vlm_rubric.<name>.json
    # -> docs/eval/raw/vlm_rubric_grader2.json, docs/eval/raw/vlm_rubric_agreement.json

Per yes/no question: how many of the events both people answered the same way, and Cohen's kappa only when each
person used both answers at least once. When one person gave the same answer to every event, kappa is 0 or undefined
whatever the other did (it measures agreement beyond chance, and a constant rater has none to offer), so it is left out
and percent agreement is the number to read. Readability (1-3): exact and within-one agreement and each person's mean. Every
disagreement is listed with both people's notes so it can be looked at.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "docs" / "eval" / "raw"
QUESTIONS = {"observations_match_frames": "grader_observations_match_frames_0_1", "hallucination": "grader_hallucination_0_1",
             "uncertainties_sensible": "grader_uncertainties_sensible_0_1", "readability": "grader_readability_1_3"}


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two raters over the same items; None when chance agreement is 1 (both constant, same value)."""
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1.0:
        return None
    return round((po - pe) / (1 - pe), 3)


def compare(rows: list[dict], answers: dict) -> dict:
    out: dict = {"events_scored_by_both": 0, "questions": {}, "disagreements": []}
    both = [r for r in rows if r.get(QUESTIONS["observations_match_frames"], "") != "" and answers.get(r["event_id"])]
    out["events_scored_by_both"] = len(both)
    for q, col in QUESTIONS.items():
        pairs = [(r[col], answers[r["event_id"]][q].split()[0], r) for r in both if q in answers[r["event_id"]]]
        g1, g2 = [p[0] for p in pairs], [p[1] for p in pairs]
        res: dict = {"n": len(pairs), "same": sum(x == y for x, y in zip(g1, g2, strict=True))}
        if q == "readability":
            res["within_one"] = sum(abs(int(x) - int(y)) <= 1 for x, y in zip(g1, g2, strict=True))
            res["mean_grader1"] = round(sum(map(int, g1)) / len(g1), 2) if g1 else None
            res["mean_grader2"] = round(sum(map(int, g2)) / len(g2), 2) if g2 else None
        else:
            res["kappa"] = cohen_kappa(g1, g2) if len(set(g1)) > 1 and len(set(g2)) > 1 else None
            res["grader1_yes"] = sum(x == "1" for x in g1)
            res["grader2_yes"] = sum(x == "1" for x in g2)
        out["questions"][q] = res
        for x, y, r in pairs:
            if x != y:
                out["disagreements"].append({"event_id": r["event_id"], "camera_id": r["camera_id"], "question": q, "grader1": x, "grader2": y,
                                             "note_grader1": r.get("grader_hallucination_note", ""),
                                             "note_grader2": answers[r["event_id"]].get("note", "")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="vlm_rubric.<name>.json exported by the second person")
    ap.add_argument("--label", default="grader2", help="name stored in the repository instead of the person's name")
    ap.add_argument("--out-dir", default=str(RAW))
    args = ap.parse_args()
    data = json.loads(Path(args.export).read_text())
    if data.get("page") != "vlm_rubric":
        print("not a vlm_rubric export", file=sys.stderr)
        return 2
    rows = list(csv.DictReader((RAW / "vlm_eval-rubric.csv").open(encoding="utf-8")))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stored = {"page": "vlm_rubric", "who": args.label, "saved": data.get("saved", "")[:10], "answers": data["answers"]}
    (out_dir / f"vlm_rubric_{args.label}.json").write_text(json.dumps(stored, ensure_ascii=False, indent=1))
    res = {"grader1": rows[0].get("grader_name", ""), "grader2": args.label, **compare(rows, data["answers"])}
    (out_dir / "vlm_rubric_agreement.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for q, r in res["questions"].items():
        extra = f" kappa {r['kappa']}" if "kappa" in r else f" within-one {r['within_one']}/{r['n']} means {r['mean_grader1']} vs {r['mean_grader2']}"
        print(f"{q}: same {r['same']}/{r['n']}{extra}")
    print(f"{len(res['disagreements'])} disagreements -> {out_dir / 'vlm_rubric_agreement.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
