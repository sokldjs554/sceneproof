"""Asynchronous VLM worker.

Guarantees the rest of the server relies on:
  * ingestion never waits for a VLM call: jobs are rows in SQLite, picked up here
  * bounded concurrency, hard per-call timeout, bounded attempts, bounded spend
  * every outcome is persisted (done / invalid / failed / timeout / skipped) with latency, usage,
    raw text and the validated JSON, plus provider/model/prompt/schema versions
  * a crash mid-call leaves the job 'running'; startup re-queues it
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import ServerSettings
from ..db import Store
from .base import ExplainRequest, FrameInput, VLMProvider
from .prompt import PROMPT_VERSION, EventContext
from .schema import SCHEMA_VERSION, extract_json, validate

log = logging.getLogger(__name__)


def _echoes_example(parsed) -> bool:
    from .prompt import COMPACT_EXAMPLE

    return parsed.scene_summary.strip() == COMPACT_EXAMPLE["scene_summary"]


class VLMWorker:
    def __init__(self, store: Store, provider: VLMProvider | None, settings: ServerSettings) -> None:
        self.store = store
        self.provider = provider
        self.s = settings
        self.evidence_dir = Path(settings.resolved_evidence())
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._jobs: set[asyncio.Task] = set()
        self._inflight = 0
        self.stats = {"done": 0, "invalid": 0, "failed": 0, "timeout": 0, "skipped": 0}

    # ------------------------------------------------------------------ queueing
    def should_trigger(self, ev: dict, accepted: bool) -> bool:
        if self.provider is None or not accepted or self.s.vlm_trigger == "never":
            return False
        status, kind = ev.get("status"), ev.get("kind")
        if status == "EVENT_STATUS_CLOSED":
            return True
        return self.s.vlm_trigger == "dwell_or_closed" and kind == "EVENT_KIND_ZONE_DWELL"

    def enqueue(self, event_id: str, revision: int, force: bool = False) -> int | None:
        if self.provider is None:
            return None
        return self.store.enqueue_explanation(event_id, revision, self.provider.name, self.provider.model,
                                              PROMPT_VERSION, SCHEMA_VERSION, force=force)

    def maybe_enqueue_for_event(self, ev: dict, accepted: bool) -> int | None:
        """Called after an upsert or an evidence upload. Only queues when evidence is complete."""
        if not self.should_trigger(ev, accepted):
            return None
        row = self.store.get_event(ev["event_id"])
        if not row or not row.get("evidence_ready"):
            return None
        return self.enqueue(ev["event_id"], int(row["revision"]))

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        n = self.store.requeue_running()
        if n:
            log.info("re-queued %d explanation jobs left running by a previous process", n)
        b = self.backfill()
        if b:
            log.info("queued %d explanations for closed events that had none", b)
        self._task = asyncio.create_task(self._loop())

    def backfill(self, limit: int = 500) -> int:
        """Closed events with complete evidence and no explanation yet (restart, import, provider enabled later)."""
        if self.provider is None or self.s.vlm_trigger == "never":
            return 0
        rows = self.store.q(
            "SELECT event_id, revision FROM events WHERE status='EVENT_STATUS_CLOSED' AND evidence_ready=1 "
            "AND event_id NOT IN (SELECT event_id FROM explanations) ORDER BY entry_ts DESC LIMIT ?", (limit,))
        n = 0
        for r in rows:
            if self.enqueue(r["event_id"], int(r["revision"])) is not None:
                n += 1
        return n

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await asyncio.wait([self._task], timeout=5)

    async def _loop(self) -> None:
        cap = max(1, self.s.vlm_max_concurrency)
        while not self._stop.is_set():
            if self.provider is None:
                await asyncio.sleep(1.0)
                continue
            free = cap - len(self._jobs)
            if free <= 0:
                await asyncio.sleep(0.2)
                continue
            jobs = self.store.next_queued_explanations(limit=free)
            if not jobs:
                await asyncio.sleep(0.5)
                continue
            for job in jobs:
                job["attempts"] = int(job.get("attempts", 0)) + 1  # attempt token for this run
                # 'running' means executing right now (one task per job, bounded by cap)
                self.store.update_explanation(job["id"], status="running", started_at=time.time(), attempts=job["attempts"])
                task = asyncio.create_task(self._run_job_task(job))
                self._jobs.add(task)
                task.add_done_callback(self._jobs.discard)
            await asyncio.sleep(0.05)

    async def _run_job_task(self, job: dict) -> None:
        self._inflight += 1
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(loop.run_in_executor(None, self._run_job, job), timeout=self.s.vlm_timeout_s)
        except TimeoutError:
            self._on_timeout(job)
        except Exception as e:
            log.exception("explanation job %s crashed", job["id"])
            self._finish(job, "failed", error=f"{type(e).__name__}: {e}"[:300])
        finally:
            self._inflight -= 1

    # ------------------------------------------------------------------ job
    def _load_frames(self, ev: dict) -> list[FrameInput]:
        roles = [r.strip() for r in self.s.vlm_frames.split(",") if r.strip()]
        by_role = {k["role"]: k for k in ev.get("evidence", {}).get("key_frames", [])}
        out: list[FrameInput] = []
        for role in roles:
            key = f"{role}_annot" if self.s.vlm_use_annotated and f"{role}_annot" in by_role else role
            kf = by_role.get(key)
            if not kf:
                continue
            p = self.evidence_dir / kf["path"]
            if not p.exists():
                continue
            data = p.read_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = min(1.0, self.s.vlm_frame_max_side / max(h, w))
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                _ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                data = buf.tobytes()
            out.append(FrameInput(role, data, float(kf.get("media_time_s", 0.0)), img.shape[1], img.shape[0]))
        return out

    def _spent(self) -> float:
        return float(self.store.get_meta("vlm_spend_usd") or 0.0)

    def _run_job(self, job: dict) -> None:
        assert self.provider is not None
        row = self.store.get_event(job["event_id"])
        if row is None:
            self._finish(job, "failed", error="event missing")
            return
        ev = json.loads(row["event_json"])
        if self.provider.name in ("anthropic",) and self._spent() >= self.s.vlm_budget_usd:
            self._finish(job, "skipped", error=f"budget exhausted ({self._spent():.2f} >= {self.s.vlm_budget_usd} USD)")
            return
        frames = self._load_frames(ev)
        if not frames:
            self._finish(job, "failed", error="no evidence frames available")
            return
        ctx = EventContext(
            camera_name=row.get("camera_name") or ev.get("camera_id", ""), zone_name=ev.get("zone_name", ""),
            kind=ev.get("kind", ""), dwell_seconds=float(ev.get("dwell_seconds", 0.0)),
            entry_media_s=float(ev.get("media_entry_s", 0.0)), exit_media_s=ev.get("media_exit_s"),
            close_reason=ev.get("close_reason", ""), frame_roles=[f.role for f in frames],
        )
        req = ExplainRequest(ev["event_id"], ctx, frames, lang="ko", max_tokens=self.s.vlm_max_tokens)
        input_meta = {"frames": [{"role": f.role, "bytes": len(f.jpeg), "w": f.width, "h": f.height} for f in frames],
                      "lang": req.lang, "annotated": self.s.vlm_use_annotated}
        attempts = int(job.get("attempts", 1))
        t0 = time.perf_counter()
        res = self.provider.explain(req)
        total_cost = res.cost_usd
        if res.error:
            self._bump_spend(total_cost)
            self._finish(job, "failed", error=res.error, raw_text=res.text, latency_ms=res.latency_ms,
                         usage=res.usage, attempts=attempts, input_meta=input_meta, model=res.model)
            return
        parsed, err = validate(extract_json(res.text))
        if parsed is not None and _echoes_example(parsed):
            parsed, err = None, "echoed the format example instead of describing the frames"
        raw = res.text
        repaired = False
        if parsed is None:
            res2 = self.provider.explain(req, repair_error=err, previous_text=res.text)
            total_cost += res2.cost_usd
            repaired = True
            raw = res.text + "\n\n--- repair ---\n\n" + res2.text
            parsed, err = validate(extract_json(res2.text))
            if parsed is not None and _echoes_example(parsed):
                parsed, err = None, "echoed the format example instead of describing the frames"
            res.usage = {k: res.usage.get(k, 0) + res2.usage.get(k, 0) for k in set(res.usage) | set(res2.usage)}
        ms = (time.perf_counter() - t0) * 1e3
        self._bump_spend(total_cost)
        input_meta["repair_attempted"] = repaired
        input_meta["cost_usd"] = round(total_cost, 5)
        if parsed is None:
            self._finish(job, "invalid", error=err, raw_text=raw, latency_ms=ms, usage=res.usage, attempts=attempts,
                         input_meta=input_meta, model=res.model)
            return
        self._finish(job, "done", raw_text=raw, latency_ms=ms, usage=res.usage, attempts=attempts,
                     result=parsed.model_dump(), input_meta=input_meta, model=res.model)

    def _bump_spend(self, cost: float) -> None:
        if cost:
            self.store.set_meta("vlm_spend_usd", f"{self._spent() + cost:.6f}")

    def _on_timeout(self, job: dict) -> None:
        attempts = int(job.get("attempts", 1))
        if attempts < self.s.vlm_max_attempts:
            self.store.update_explanation(job["id"], status="queued", attempts=attempts,
                                          error=f"timeout after {self.s.vlm_timeout_s}s (attempt {attempts})")
            log.warning("explanation %s timed out; re-queued (attempt %d)", job["id"], attempts)
        else:
            self._finish(job, "timeout", error=f"timeout after {self.s.vlm_timeout_s}s x{attempts}", attempts=attempts)

    def _is_current(self, job: dict) -> bool:
        """A thread that outlived its timeout must not overwrite the job's state: the loop re-queued the job
        (status 'queued') or started a newer attempt (attempts > ours)."""
        row = self.store.q1("SELECT status, attempts FROM explanations WHERE id=?", (job["id"],))
        return bool(row) and row["status"] == "running" and int(row["attempts"]) == int(job.get("attempts", 1))

    def _finish(self, job: dict, status: str, error: str = "", raw_text: str = "", latency_ms: float | None = None,
                usage: dict | None = None, attempts: int | None = None, result: dict | None = None,
                input_meta: dict | None = None, model: str | None = None) -> None:
        if status != "timeout" and not self._is_current(job):
            log.info("explanation %s attempt %s finished late (%s); result discarded", job["id"], job.get("attempts"), status)
            return
        if status == "failed" and (attempts or 1) < self.s.vlm_max_attempts and "budget" not in error:
            self.store.update_explanation(job["id"], status="queued", attempts=attempts or 1, error=error)
            log.warning("explanation %s failed (%s); re-queued", job["id"], error)
            return
        fields = dict(status=status, completed_at=time.time(), error=error or None, raw_text=raw_text or None,
                      latency_ms=latency_ms, usage_json=json.dumps(usage or {}), attempts=attempts or job.get("attempts", 0),
                      result_json=json.dumps(result, ensure_ascii=False) if result else None,
                      input_json=json.dumps(input_meta or {}))
        if model:
            fields["model"] = model
        self.store.update_explanation(job["id"], **fields)
        self.stats[status] = self.stats.get(status, 0) + 1
        log.info("explanation %s -> %s (%s)", job["id"], status, error or f"{latency_ms:.0f} ms" if latency_ms else "")

    def snapshot(self) -> dict:
        return {"provider": self.provider.name if self.provider else "none",
                "model": self.provider.model if self.provider else "", "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION, "inflight": self._inflight, "spend_usd": round(self._spent(), 4),
                "budget_usd": self.s.vlm_budget_usd, "trigger": self.s.vlm_trigger, "db": self.store.explanation_stats(),
                "timeout_s": self.s.vlm_timeout_s, "max_attempts": self.s.vlm_max_attempts}
