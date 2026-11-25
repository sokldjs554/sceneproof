"""Evaluate both-feet visibility on unique SceneProof entry and dwell frames."""
import json
import re
from pathlib import Path

import cv2

from sceneproof.eval.vlm_crop import BUNDLE, SmolVLM

HERE = Path(__file__).resolve().parent
PROMPT = (
    "Look at the person in the yellow box. "
    "Are both of their feet visible in this image? Answer yes or no."
)


def scenes():
    for line in (BUNDLE / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        for frame in event["evidence"]["key_frames"]:
            if frame["role"] not in ("entry", "dwell") or not frame.get("box"):
                continue
            path = BUNDLE / "evidence" / frame["path"]
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)

            height, width = image.shape[:2]
            box = frame["box"]
            # Match the original corner rounding, including near the image edge.
            corners = [
                (round(box["x" + n] * width / event["frame_width"]),
                 round(box["y" + n] * height / event["frame_height"]))
                for n in ("1", "2")
            ]
            cv2.rectangle(image, *corners, (0, 255, 255), 2)
            yield event["event_id"], frame["role"], image


def main():
    labels = json.loads((HERE / "labels.json").read_text())
    model = SmolVLM("HuggingFaceTB/SmolVLM-500M-Instruct", threads=2)
    rows = []
    for event, role, image in scenes():
        key = f"{event}/{role}"
        response = model(image, PROMPT, False)
        match = re.fullmatch(r"(yes|no)[.!]?", response["text"].strip(), re.I)
        rows.append({
            "key": key,
            "reference": labels[key],
            "answer": match[1].lower() if match else None,
            **response,
        })
    assert len(rows) == len(labels) == len({r["key"] for r in rows})
    result = {"prompt": PROMPT, "rows": rows}
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
