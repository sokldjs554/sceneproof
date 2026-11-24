"""Run one alternative VQA model on the same source pixels, rectangles and prompt."""
import argparse
import json
import re

import cv2
import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor
from run import HERE, PROMPT, views

MODEL = 'Salesforce/blip-vqa-base'
REVISION = '787b3d35d57e49572baabd22884b3d5a05acf072'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('output')
    args = ap.parse_args()
    proc = BlipProcessor.from_pretrained(MODEL, revision=REVISION)
    torch.set_num_threads(2)
    net = BlipForQuestionAnswering.from_pretrained(MODEL, revision=REVISION).eval()
    cap = cv2.VideoCapture(args.video)
    rows = []
    for i, case in enumerate(json.loads((HERE/'cases.json').read_text())):
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * case['frame_s'])
        ok, frame = cap.read()
        if not ok:
            raise ValueError(case)
        for name, pixels, rect in views(frame, case['box']):
            rgb = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
            inputs = proc(images=rgb, text=PROMPT, return_tensors='pt')
            with torch.inference_mode():
                ids = net.generate(**inputs, do_sample=False, max_new_tokens=48)
            text = proc.decode(ids[0], skip_special_tokens=True)
            m = re.fullmatch(r'(yes|no)[.!]?', text.strip(), re.I)
            rows.append(dict(case=i+1, view=name, rect=rect, tensor_hw=list(inputs.pixel_values.shape[-2:]),
                             text=text, answer=m[1].lower() if m else None))
    with open(args.output, 'w') as handle:
        json.dump(dict(model=MODEL, revision=REVISION, prompt=PROMPT, rows=rows), handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
