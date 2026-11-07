"""Local open-weights VLM via Hugging Face transformers (CPU-capable small models such as
HuggingFaceTB/SmolVLM-256M/500M-Instruct, Apache-2.0). Slow on CPU (seconds to tens of seconds
per request) which is exactly why the worker runs it off the ingestion path.

Weights are downloaded from huggingface.co on first use (network access required once).
"""

from __future__ import annotations

import io
import threading
import time

from .base import ExplainRequest, ExplainResult
from .prompt import REPAIR_INSTRUCTION, build_user_text, system_prompt


class TransformersProvider:
    name = "transformers"

    def __init__(self, model: str = "HuggingFaceTB/SmolVLM-500M-Instruct", device: str = "cpu", threads: int = 0,
                 max_images: int = 3) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if threads:
            torch.set_num_threads(threads)
        self.model = model
        self.device = device
        self.max_images = max_images
        self.processor = AutoProcessor.from_pretrained(model)
        self.net = AutoModelForImageTextToText.from_pretrained(model, torch_dtype=torch.float32).to(device).eval()
        self._torch = torch
        self._lock = threading.Lock()  # one generation at a time on CPU

    def explain(self, req: ExplainRequest, repair_error: str | None = None, previous_text: str | None = None) -> ExplainResult:
        from PIL import Image

        frames = req.frames[: self.max_images]
        images = [Image.open(io.BytesIO(f.jpeg)).convert("RGB") for f in frames]
        lang = "en"  # small open models answer best in English; the console shows the language as-is
        text = build_user_text(req.ctx, lang)
        if repair_error is not None:
            text += "\n\n" + REPAIR_INSTRUCTION[lang].format(error=repair_error)
        content = [{"type": "text", "text": system_prompt(lang)}]
        for f in frames:
            content.append({"type": "text", "text": f"[frame: {f.role} @ {f.media_s:.1f}s]"})
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        t0 = time.perf_counter()
        with self._lock:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)
            with self._torch.inference_mode():
                out = self.net.generate(**inputs, max_new_tokens=req.max_tokens, do_sample=False)
            gen = out[0, inputs["input_ids"].shape[1]:]
            answer = self.processor.batch_decode([gen], skip_special_tokens=True)[0]
        ms = (time.perf_counter() - t0) * 1e3
        usage = {"input_tokens": int(inputs["input_ids"].shape[1]), "output_tokens": int(gen.shape[0])}
        return ExplainResult(answer, self.model, self.name, ms, usage=usage, repair_attempted=repair_error is not None)
