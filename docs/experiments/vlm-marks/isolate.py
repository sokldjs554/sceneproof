"""Run the two missing image-by-prompt conditions on the original 46 queries."""
import json
from pathlib import Path

from sceneproof.eval.vlm_crop import SHORT_PROMPT, SmolVLM, judge, load_queries, region
from sceneproof.server.vlm.marks import MARK_PROMPT, mark_for_vlm


def main():
    model = SmolVLM("HuggingFaceTB/SmolVLM-500M-Instruct", threads=2)
    rows = []
    for query in load_queries():
        marked = mark_for_vlm(query.image, query.box, query.zone)
        conditions = [
            ("image_only", marked, SHORT_PROMPT),
            ("prompt_only", query.image, f"{SHORT_PROMPT} {MARK_PROMPT}"),
        ]
        for name, image, prompt in conditions:
            for strategy in ("full", "crop"):
                pixels, _ = region(query, strategy, image=image)
                response = model(pixels, prompt, False)
                relation, outcome = judge(query.label, response["text"])
                rows.append(dict(event_id=query.event_id, role=query.role, variant=query.variant,
                                 label=query.label, strategy=strategy, condition=name,
                                 zone_relation=relation, outcome=outcome, **response))
        print(f"{len(rows)}/184", flush=True)
    path = Path(__file__).with_name("isolated-results.json")
    path.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
