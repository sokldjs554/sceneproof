"""Compare full and lower-body views of the same designated person."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sceneproof.eval.vlm_crop import SmolVLM

HERE = Path(__file__).resolve().parent
PROMPT = "Look at the person in the yellow box. Are both of their feet visible in this image? Answer yes or no."


def views(frame, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    height, width = frame.shape[:2]
    marked = frame.copy()
    cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 255, 255), 2)
    rect = [max(0, x1 - round(.2*w)), max(0, y1 + round(.45*h)),
            min(width, x2 + round(.2*w) + 1), min(height, y2 + round(.12*h) + 1)]
    l, t, r, b = rect
    return [('full', marked, [0, 0, width, height]), ('lower', marked[t:b, l:r], rect)]


def preview(model, frame, path):
    # Recover actual normalized RGB tensor and its non-padding mask.
    proc = model.processor.image_processor
    rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    out = proc(images=[[rgb]], return_tensors="pt")
    pixels = out['pixel_values'][0, 0].permute(1, 2, 0).numpy()
    mask = out['pixel_attention_mask'][0, 0].numpy()
    y, x = np.where(mask)
    recovered = (pixels * np.asarray(proc.image_std) + np.asarray(proc.image_mean)) * 255
    Image.fromarray(np.rint(recovered).clip(0, 255).astype('uint8')).save(path)
    return list(pixels.shape[:2]), [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = SmolVLM('HuggingFaceTB/SmolVLM-500M-Instruct', threads=2)
    model._configure(False)
    rows = []
    cap = cv2.VideoCapture(str(args.video))
    for i, case in enumerate(json.loads((HERE / 'cases.json').read_text())):
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * case['frame_s'])
        success, frame = cap.read()
        if not success:
            raise ValueError(case)
        for name, pixels, rect in views(frame, case['box']):
            stem = f'{i+1:02}-{name}'
            size, active = preview(model, pixels, args.output / (stem + '.png'))
            response = model(pixels, PROMPT, False)
            match = re.fullmatch(r'(yes|no)[.!]?', response['text'].strip(), re.I)
            rows.append(dict(case=i+1, view=name, rect=rect, tensor_hw=size, active_xyxy=active,
                             answer=match[1].lower() if match else None, **response))
            print(stem, response['text'], flush=True)
    result = dict(prompt=PROMPT, model=model.model_name, video_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(), rows=rows)
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
