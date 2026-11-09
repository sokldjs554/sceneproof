"""Frozen benchmark split (train / dev / test), decided BEFORE zones were drawn or anything was measured.

Rules:
  * whole groups move together (CAVIAR Lisbon corridor + front views of one take are one group),
  * CAVIAR sites are split by group with a fixed seed: train (fine-tuning) / dev (error analysis, threshold
    and training choices) / test (measured once),
  * MOT16-04 is train only (adds a high-resolution street view to fine-tuning),
  * MOT16-02, MOT16-09, MOT20-01, MOT20-02 are test only: sites nobody tuned or trained on (unseen sites).

    python -m sceneproof.eval.bench.split            # writes configs/bench/split.yaml (refuses to overwrite)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time

import yaml

from .data import CONFIG_DIR, load_sources

SEED = 20261109
CAVIAR_PLAN = {"caviar-lobby": {"test": 10, "dev": 4}, "caviar-mall": {"test": 10, "dev": 4}}
FIXED = {"mot16-04": "train", "mot16-02": "test", "mot16-09": "test", "mot20-01": "test", "mot20-02": "test"}
UNSEEN_SITES = ["mot16-02-square", "mot16-09-lowangle", "mot20-station"]


def make_split() -> dict:
    src = load_sources()
    groups: dict[str, list[str]] = {}
    for s in src["sequences"]:
        groups.setdefault(s["group"], []).append(s["id"])
    assign: dict[str, str] = {}
    rng = random.Random(SEED)
    for family, plan in CAVIAR_PLAN.items():
        gs = sorted(g for g in groups if g.startswith(family + ":"))
        rng.shuffle(gs)
        n_test, n_dev = plan["test"], plan["dev"]
        for i, g in enumerate(gs):
            assign[g] = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"
    for g, part in FIXED.items():
        assign[g] = part
    seqs = {part: sorted(sid for g, ids in groups.items() if assign[g] == part for sid in ids) for part in ("train", "dev", "test")}
    body = {"seed": SEED, "rules": __doc__.split("Rules:")[1].split("python -m")[0].strip(), "unseen_sites": UNSEEN_SITES,
            "groups": dict(sorted(assign.items())), "sequences": seqs}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    return {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "sha256_16": digest, **body}


def load_split() -> dict:
    d = yaml.safe_load((CONFIG_DIR / "split.yaml").read_text())
    body = {k: d[k] for k in ("seed", "rules", "unseen_sites", "groups", "sequences")}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16] != d["sha256_16"]:
        raise RuntimeError("configs/bench/split.yaml was edited after it was frozen (hash mismatch)")
    return d


def split_of(seq_id: str, split: dict | None = None) -> str:
    s = split or load_split()
    return next(p for p, ids in s["sequences"].items() if seq_id in ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    path = CONFIG_DIR / "split.yaml"
    if path.exists() and not args.force:
        print(f"{path} already exists — the split is frozen; refusing to overwrite", file=sys.stderr)
        return 1
    d = make_split()
    path.write_text("# FROZEN — generated once by `python -m sceneproof.eval.bench.split`; edits break the hash check.\n"
                    + yaml.safe_dump(d, allow_unicode=True, sort_keys=False, width=120))
    print({p: len(v) for p, v in d["sequences"].items()}, d["sha256_16"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
