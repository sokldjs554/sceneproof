#!/usr/bin/env python
"""Failure-scenario runner: starts an isolated stack on dedicated ports, injects faults, records
status snapshots, asserts the expected behaviour and writes a JSON report per scenario.

Scenarios (see docs/ops/failure-modes.md for the expected behaviour of each):
  overload        input faster than analysis (30 fps 720p stream, 1 inference thread)
  stream_loss     RTSP camera disappears and comes back
  vlm_faults      VLM provider slow / failing while ingestion keeps going; retry path
  uplink_loss     server hangs (SIGSTOP) and is restarted (SIGKILL): gateway queues, resends, no duplicates
  duplicates      the same event revision delivered repeatedly, older revision after newer
  restart         server and gateway restarted: reviews, explanations and queued jobs survive

Usage: python scripts/failure_scenarios.py --out docs/evidence/failure-scenarios [--only overload,stream_loss]
Ports: rtsp 8564, server http 8010 / grpc 50062, gateway grpc 50061 / http 9110.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PORTS = {"rtsp": 8564, "http": 8010, "grpc": 50062, "gw_grpc": 50061, "gw_http": 9110}
SRV = f"http://127.0.0.1:{PORTS['http']}"
GW = f"http://127.0.0.1:{PORTS['gw_http']}"


class Stack:
    def __init__(self, work: Path, log_dir: Path) -> None:
        self.work = work
        self.log_dir = log_dir
        self.procs: dict[str, subprocess.Popen] = {}
        self.env = dict(os.environ, PYTHONUNBUFFERED="1")

    def _spawn(self, name: str, cmd: list[str], env: dict | None = None) -> subprocess.Popen:
        log = (self.log_dir / f"{name}.log").open("ab")
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**self.env, **(env or {})})
        self.procs[name] = p
        return p

    def start_rtsp(self) -> None:
        self._spawn("rtsp", [PY, "-m", "sceneproof.tools.rtsp_sim", "--dir", "data/videos", "--port", str(PORTS["rtsp"])])
        import socket

        for _ in range(40):  # wait until the RTSP port accepts connections (slow under CPU load)
            try:
                with socket.create_connection(("127.0.0.1", PORTS["rtsp"]), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.25)

    def start_server(self, extra_env: dict | None = None) -> None:
        env = {"SP_SERVER_DATA_DIR": str(self.work / "server"), "SP_SERVER_GRPC_LISTEN": f"0.0.0.0:{PORTS['grpc']}",
               "SP_SERVER_HTTP_PORT": str(PORTS["http"]), "SP_SERVER_VLM_PROVIDER": "mock", **(extra_env or {})}
        self._spawn("server", [PY, "-m", "sceneproof.server.cli", "run"], env)
        wait_http(f"{SRV}/api/health", 30)

    def start_gateway(self) -> None:
        self._spawn("gateway", [str(ROOT / "gateway/bin/sceneproof-gateway"), "--listen", f":{PORTS['gw_grpc']}",
                                "--server", f"127.0.0.1:{PORTS['grpc']}", "--device-id", "edge-sc",
                                "--queue-dir", str(self.work / "gateway"), "--evidence-root", str(self.work / "evidence"),
                                "--http", f":{PORTS['gw_http']}", "--insecure"])
        wait_http(f"{GW}/healthz", 20)

    def start_edge(self, cfg: dict, cameras: list[str]) -> None:
        cfg_path = self.work / "edge.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
        self._spawn("edge", [PY, "-m", "sceneproof.edge.cli", "--config", str(cfg_path), "run", *[a for c in cameras for a in ("--camera", c)]])

    def stop(self, name: str, sig: int = signal.SIGTERM, wait: float = 10.0) -> None:
        p = self.procs.pop(name, None)
        if p is None:
            return
        if p.poll() is None:
            p.send_signal(sig)
            try:
                p.wait(wait)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(5)

    def stop_all(self) -> None:
        for n in list(self.procs):
            self.stop(n)


def wait_http(url: str, timeout: float) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"{url} not up after {timeout}s")


def get(url: str, **kw):
    return httpx.get(url, timeout=5, **kw).json()


def edge_config(work: Path, cameras: list[dict], threads: int = 4, queue: int = 4) -> dict:
    base = yaml.safe_load((ROOT / "configs/edge.yaml").read_text())
    base["device_id"] = "edge-sc"
    base["gateway"] = f"127.0.0.1:{PORTS['gw_grpc']}"
    base["evidence_root"] = str(work / "evidence")
    base["detector"]["intra_op_threads"] = threads
    base["pipeline"]["queue_capacity"] = queue
    base["cameras"] = cameras
    return base


def cam(camera_id: str, stream: str, zones: list[dict], fps_hint: float) -> dict:
    return {"camera_id": camera_id, "name": camera_id, "source": f"rtsp://127.0.0.1:{PORTS['rtsp']}/{stream}",
            "autostart": False, "realtime": True, "fps_hint": fps_hint, "zones": zones}


ZONE_TAPE = [{"zone_id": "z-tape", "name": "황색 테이프 통제구역", "polygon": [[0.52, 0.49], [0.66, 0.49], [0.98, 0.80], [0.98, 1.0], [0.22, 1.0]],
              "entry_confirm_frames": 5, "exit_confirm_frames": 8, "dwell_seconds": 5, "lost_grace_seconds": 1.5}]
ZONE_BENCH = [{"zone_id": "z-bench", "name": "작업대 반경 통제구역", "polygon": [[0.40, 0.58], [0.76, 0.58], [0.80, 1.0], [0.36, 1.0]],
               "entry_confirm_frames": 3, "exit_confirm_frames": 4, "dwell_seconds": 5, "lost_grace_seconds": 1.5}]


def _num(v, default=0):
    """protobuf JSON renders int64 as strings and omits zero-valued fields."""
    if v is None:
        return default
    if isinstance(v, str):
        try:
            return float(v) if "." in v else int(v)
        except ValueError:
            return default
    return v


def camera_snapshot(camera_id: str) -> dict | None:
    try:
        st = get(f"{SRV}/api/status")
    except Exception:
        return None
    for dev in st["devices"].values():
        for c in dev["status"].get("cameras", []):
            if c["camera_id"] == camera_id:
                out = {k: _num(c.get(k)) for k in ("input_fps", "processed_fps", "analysis_lag_ms", "queue_depth",
                                                   "dropped_frames", "frames_processed", "reconnect_count", "open_episodes",
                                                   "infer_ms_p50")}
                out["state"] = c.get("state")
                out["last_error"] = c.get("last_error")
                out["device_age_s"] = dev["age_s"]
                return out
    return None


def gateway_snapshot() -> dict | None:
    try:
        s = get(f"{GW}/status")
    except Exception:
        return None
    u = s.get("uplink", {})
    q = s.get("queue", {}) or {}

    def pick(d, *names, default=0):
        for n in names:
            if n in d and d[n] is not None:
                return d[n]
        return default

    return {"uplink_state": u.get("state"), "pending": pick(q, "pending", "Pending", "live", "Live", "pending_events"),
            "sent": u.get("sent_events"), "acked": u.get("acked_events"), "dup_acks": u.get("duplicate_acks"),
            "reconnects": u.get("reconnect_count"), "dropped": pick(q, "dropped", "Dropped", "evicted", "Evicted"),
            "queue_bytes": pick(q, "bytes", "Bytes", "live_bytes", "LiveBytes"), "uploads_pending": s.get("uploads", {}).get("pending"),
            "queue_raw": q}


def observe(seconds: float, camera_id: str | None, every: float = 1.0) -> list[dict]:
    out = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        snap = {"t": round(time.time() - t0, 1), "camera": camera_snapshot(camera_id) if camera_id else None,
                "gateway": gateway_snapshot()}
        out.append(snap)
        time.sleep(every)
    return out


def live_events() -> list[dict]:
    try:
        return [x for x in get(f"{SRV}/api/events?limit=500")["items"] if x["ingest_source"] == "live"]
    except Exception:
        return []


# ----------------------------------------------------------------------------- scenarios


def sc_overload(st: Stack, work: Path) -> dict:
    """30 fps 720p stream analysed with one inference thread: the queue must drop, lag must stay bounded."""
    st.start_rtsp()
    st.start_server()
    st.start_gateway()
    st.start_edge(edge_config(work, [cam("cam-over", "worker-zone", ZONE_TAPE, 30)], threads=1, queue=4), ["cam-over"])
    snaps = observe(40, "cam-over")
    st.stop_all()
    cams = [s["camera"] for s in snaps if s["camera"] and s["camera"]["state"] == "CAMERA_STATE_STREAMING"]
    lag = [c["analysis_lag_ms"] for c in cams]
    checks = {
        "input_faster_than_processing": any(c["input_fps"] > c["processed_fps"] + 3 for c in cams),
        "frames_dropped_by_policy": cams[-1]["dropped_frames"] > 0 if cams else False,
        "queue_depth_bounded": all(c["queue_depth"] <= 4 for c in cams),
        "lag_bounded_under_2s": max(lag) < 2000 if lag else False,
        "analysis_kept_running": cams[-1]["frames_processed"] > 200 if cams else False,
    }
    return {"snapshots": snaps, "checks": checks, "summary": {"max_lag_ms": max(lag) if lag else None,
            "median_input_fps": sorted(c["input_fps"] for c in cams)[len(cams) // 2] if cams else None,
            "median_processed_fps": sorted(c["processed_fps"] for c in cams)[len(cams) // 2] if cams else None,
            "dropped_frames": cams[-1]["dropped_frames"] if cams else None, "frames_processed": cams[-1]["frames_processed"] if cams else None}}


def sc_stream_loss(st: Stack, work: Path) -> dict:
    st.start_rtsp()
    st.start_server()
    st.start_gateway()
    st.start_edge(edge_config(work, [cam("cam-loss", "one-by-one", ZONE_BENCH, 10)]), ["cam-loss"])
    phase1 = observe(20, "cam-loss")
    st.stop("rtsp", signal.SIGKILL)
    t_kill = time.time()
    phase2 = observe(12, "cam-loss")
    st.start_rtsp()
    t_back = time.time()
    phase3 = observe(25, "cam-loss")
    st.stop_all()
    states2 = [s["camera"]["state"] for s in phase2 if s["camera"]]
    states3 = [s["camera"]["state"] for s in phase3 if s["camera"]]
    last = next((s["camera"] for s in reversed(phase3) if s["camera"]), {})
    first = next((s["camera"] for s in reversed(phase1) if s["camera"]), {})
    checks = {
        "streaming_before_loss": first.get("state") == "CAMERA_STATE_STREAMING",
        "reconnecting_observed": "CAMERA_STATE_RECONNECTING" in states2 + states3,
        "streaming_after_recovery": states3[-1] == "CAMERA_STATE_STREAMING" if states3 else False,
        "reconnect_count_incremented": (last.get("reconnect_count") or 0) >= 1,
        "frames_resumed": (last.get("frames_processed") or 0) > (first.get("frames_processed") or 0),
    }
    recov = next((s["t"] for s in phase3 if s["camera"] and s["camera"]["state"] == "CAMERA_STATE_STREAMING"), None)
    return {"phases": {"before": phase1, "down": phase2, "after": phase3}, "checks": checks,
            "summary": {"rtsp_down_s": round(t_back - t_kill, 1), "recovered_after_s": recov,
                        "reconnect_count": last.get("reconnect_count"), "last_error_seen": [s["camera"]["last_error"] for s in phase2 if s["camera"] and s["camera"]["last_error"]][:1]}}


def _import_bundle(st: Stack, work: Path) -> int:
    bundle = ROOT / "data/runs/scratch/bundle-dev"
    if not bundle.exists():
        raise RuntimeError("dev bundle missing: run the offline dev runs first")
    env = {"SP_SERVER_DATA_DIR": str(work / "server")}
    subprocess.run([PY, "-m", "sceneproof.server.cli", "import", str(bundle)], cwd=ROOT, check=True, capture_output=True,
                   env={**st.env, **env})
    return len(get(f"{SRV}/api/events?limit=500")["items"]) if False else 0


def sc_vlm_faults(st: Stack, work: Path) -> dict:
    """Mock VLM: 3 s per call with a 1.5 s timeout and every 2nd call failing. Ingestion must stay fast."""
    _import_bundle(st, work)
    st.start_server({"SP_SERVER_VLM_MOCK_DELAY_S": "3", "SP_SERVER_VLM_TIMEOUT_S": "1.5", "SP_SERVER_VLM_MAX_ATTEMPTS": "2",
                     "SP_SERVER_VLM_MOCK_FAIL_EVERY": "2", "SP_SERVER_VLM_MAX_CONCURRENCY": "1"})
    # while the VLM is stuck, publish fresh events straight to the ingest port and time the ack
    import grpc

    from sceneproof.proto.sceneproof.v1 import sceneproof_pb2 as pb
    from sceneproof.proto.sceneproof.v1 import sceneproof_pb2_grpc as pbg

    ch = grpc.insecure_channel(f"127.0.0.1:{PORTS['grpc']}")
    stub = pbg.IngestStub(ch)
    ack_ms = []
    timeline = []
    t0 = time.time()
    while time.time() - t0 < 60:  # 16 queued jobs x up to 2 attempts x 1.5 s timeout, concurrency 1
        eid = str(uuid.uuid4())
        ev = pb.Event(event_id=eid, revision=1, device_id="edge-sc", camera_id="cam-x", zone_id="z", kind=pb.EVENT_KIND_ZONE_ENTRY,
                      status=pb.EVENT_STATUS_OPEN)
        t = time.perf_counter()
        a = stub.PublishEvent(ev, timeout=5)
        ack_ms.append((time.perf_counter() - t) * 1e3)
        vl = get(f"{SRV}/api/status")["vlm"]
        timeline.append({"t": round(time.time() - t0, 1), "vlm_db": vl["db"], "inflight": vl["inflight"], "ack_ms": round(ack_ms[-1], 1),
                         "accepted": a.accepted})
        time.sleep(1.0)
    final = get(f"{SRV}/api/status")["vlm"]["db"]
    # retry path: restart the server with a healthy mock and re-request one failed/timeout event
    st.stop("server")
    st.start_server({"SP_SERVER_VLM_MOCK_DELAY_S": "0.2"})
    items = get(f"{SRV}/api/events?limit=500")["items"]
    bad = [x for x in items if x["explanation"] and x["explanation"]["status"] in ("failed", "timeout", "invalid")]
    retried = None
    if bad:
        r = httpx.post(f"{SRV}/api/events/{bad[0]['event_id']}/explain", timeout=5).json()
        status_after = None
        for _ in range(60):  # the healthy server first drains the jobs re-queued at startup
            status_after = get(f"{SRV}/api/events/{bad[0]['event_id']}")["explanation"]["status"]
            if status_after in ("done", "failed", "invalid", "timeout"):
                break
            time.sleep(0.5)
        retried = {"event": bad[0]["event_id"], "queued": r, "status_after_retry": status_after, "bad_before_retry": len(bad)}
    st.stop_all()
    checks = {
        "ingest_ack_p95_under_500ms_while_vlm_stuck": sorted(ack_ms)[int(len(ack_ms) * 0.95) - 1] < 500 if ack_ms else False,
        "timeouts_or_failures_recorded": any(k in final for k in ("timeout", "failed")),
        "no_job_left_running_or_queued_forever": final.get("running", 0) == 0,
        "retry_after_recovery_succeeds": bool(retried) and retried["status_after_retry"] == "done",
    }
    return {"timeline": timeline, "final_vlm_db": final, "retry": retried, "checks": checks,
            "summary": {"ack_p50_ms": round(sorted(ack_ms)[len(ack_ms) // 2], 1) if ack_ms else None,
                        "ack_p95_ms": round(sorted(ack_ms)[int(len(ack_ms) * 0.95) - 1], 1) if ack_ms else None}}


def sc_uplink_loss(st: Stack, work: Path) -> dict:
    st.start_rtsp()
    st.start_server()
    st.start_gateway()
    st.start_edge(edge_config(work, [cam("cam-up", "one-by-one", ZONE_BENCH, 10)]), ["cam-up"])
    before = observe(25, "cam-up")
    n_before = len(live_events())
    # a review made before the outage must survive the restart
    ev0 = live_events()
    reviewed = None
    if ev0:
        httpx.post(f"{SRV}/api/events/{ev0[0]['event_id']}/review", json={"state": "in_review", "actor": "scenario", "note": "before outage"}, timeout=5)
        reviewed = ev0[0]["event_id"]
    # phase A: server hangs (SIGSTOP) -> gateway sees deadline exceeded, queues
    st.procs["server"].send_signal(signal.SIGSTOP)
    hang = [{"t": i, "gateway": gateway_snapshot()} for i in range(12) if not time.sleep(1)]
    st.procs["server"].send_signal(signal.SIGCONT)
    time.sleep(6)
    after_hang = gateway_snapshot()
    # phase B: server killed and restarted -> pending events resent, no duplicates
    st.stop("server", signal.SIGKILL)
    down = [{"t": i, "gateway": gateway_snapshot()} for i in range(15) if not time.sleep(1)]
    st.start_server()
    recovery = observe(25, "cam-up")
    items = live_events()  # collect while the server is still up
    ids = [x["event_id"] for x in items]
    status = get_safe(f"{SRV}/api/status") or {}
    st.stop_all()
    reviewed_state = next((x["review"]["state"] for x in items if x["event_id"] == reviewed), None) if reviewed else None
    gw_last = recovery[-1]["gateway"] if recovery else None
    checks = {
        "gateway_disconnected_during_hang": any(h["gateway"] and h["gateway"]["uplink_state"] != "CONNECTED" for h in hang),
        "events_queued_while_down": any(d["gateway"] and (d["gateway"]["pending"] or 0) > 0 for d in down) or
                                    any(h["gateway"] and (h["gateway"]["pending"] or 0) > 0 for h in hang),
        "reconnected_after_restart": bool(gw_last) and gw_last["uplink_state"] == "CONNECTED",
        "queue_drained": bool(gw_last) and (gw_last["pending"] or 0) == 0,
        "no_duplicate_event_rows": len(ids) == len(set(ids)),
        "events_survived_restart": len(items) >= n_before,
        "review_survived_restart": reviewed_state == "in_review" if reviewed else None,
    }
    return {"before": before, "hang": hang, "after_hang": after_hang, "down": down, "recovery": recovery, "checks": checks,
            "summary": {"events_before": n_before, "events_after": len(items), "ingest_counters": status.get("ingest"),
                        "gateway_final": gw_last}}


def get_safe(url: str):
    try:
        return get(url)
    except Exception:
        return None


def sc_duplicates(st: Stack, work: Path) -> dict:
    st.start_server()
    import grpc

    from sceneproof.proto.sceneproof.v1 import sceneproof_pb2 as pb
    from sceneproof.proto.sceneproof.v1 import sceneproof_pb2_grpc as pbg

    stub = pbg.IngestStub(grpc.insecure_channel(f"127.0.0.1:{PORTS['grpc']}"))
    eid = str(uuid.uuid4())

    def send(rev: int):
        ev = pb.Event(event_id=eid, revision=rev, device_id="edge-sc", camera_id="cam-d", zone_id="z", kind=pb.EVENT_KIND_ZONE_ENTRY,
                      status=pb.EVENT_STATUS_OPEN if rev < 3 else pb.EVENT_STATUS_CLOSED)
        a = stub.PublishEvent(ev, timeout=5)
        return {"rev": rev, "accepted": a.accepted, "duplicate": a.duplicate, "stored": a.revision}

    log = [send(1), send(1), send(1), send(3), send(2), send(3), send(4), send(4)]
    rows = [x for x in get(f"{SRV}/api/events?limit=500")["items"] if x["event_id"] == eid]
    st.stop_all()
    checks = {
        "first_delivery_accepted": log[0]["accepted"] and not log[0]["duplicate"],
        "repeat_same_revision_is_duplicate": all(x["duplicate"] and not x["accepted"] for x in (log[1], log[2])),
        "newer_revision_accepted": log[3]["accepted"],
        "older_revision_after_newer_ignored": log[4]["duplicate"] and log[4]["stored"] == 3,
        "single_row_with_max_revision": len(rows) == 1 and rows[0]["revision"] == 4,
    }
    return {"log": log, "rows": [{"event_id": r["event_id"], "revision": r["revision"]} for r in rows], "checks": checks}


def sc_restart(st: Stack, work: Path) -> dict:
    """Server + gateway restarted mid-run: DB state and the gateway WAL carry everything over."""
    st.start_rtsp()
    st.start_server({"SP_SERVER_VLM_MOCK_DELAY_S": "2.5", "SP_SERVER_VLM_MAX_CONCURRENCY": "1"})
    st.start_gateway()
    st.start_edge(edge_config(work, [cam("cam-rs", "one-by-one", ZONE_BENCH, 10)]), ["cam-rs"])
    observe(30, "cam-rs")
    pre = get(f"{SRV}/api/status")
    pre_events = live_events()
    if pre_events:
        httpx.post(f"{SRV}/api/events/{pre_events[0]['event_id']}/review", json={"state": "resolved", "resolution": "dismissed",
                   "actor": "scenario", "note": "restart test"}, timeout=5)
    st.stop("gateway", signal.SIGKILL)
    st.stop("server", signal.SIGKILL)
    time.sleep(3)
    st.start_server({"SP_SERVER_VLM_MOCK_DELAY_S": "0.2"})
    st.start_gateway()
    observe(25, "cam-rs")
    post = get(f"{SRV}/api/status")
    post_events = live_events()
    st.stop_all()
    ids = [x["event_id"] for x in post_events]
    first = next((x for x in post_events if pre_events and x["event_id"] == pre_events[0]["event_id"]), None)
    checks = {
        "events_persisted": len(post_events) >= len(pre_events),
        "no_duplicate_rows": len(ids) == len(set(ids)),
        "review_persisted": bool(first) and first["review"]["state"] == "resolved" and first["review"]["resolution"] == "dismissed",
        "vlm_queue_resumed_after_restart": (post["vlm"]["db"].get("done", 0) >= pre["vlm"]["db"].get("done", 0)) and post["vlm"]["db"].get("running", 0) == 0,
        "gateway_reconnected": post["devices"].get("edge-sc", {}).get("online", False),
    }
    return {"pre": {"events": len(pre_events), "vlm": pre["vlm"]["db"]}, "post": {"events": len(post_events), "vlm": post["vlm"]["db"],
            "ingest": post["ingest"]}, "checks": checks}


SCENARIOS = {"overload": sc_overload, "stream_loss": sc_stream_loss, "vlm_faults": sc_vlm_faults, "uplink_loss": sc_uplink_loss,
             "duplicates": sc_duplicates, "restart": sc_restart}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/evidence/failure-scenarios")
    ap.add_argument("--only", default="")
    ap.add_argument("--work", default="data/runs/scenarios")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in args.only.split(",") if n] or list(SCENARIOS)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    summary = {}
    for name in names:
        work = ROOT / args.work / name
        shutil.rmtree(work, ignore_errors=True)
        (work / "evidence").mkdir(parents=True)
        logs = out / "logs" / name
        logs.mkdir(parents=True, exist_ok=True)
        st = Stack(work, logs)
        t0 = time.time()
        print(f"=== {name} ===", flush=True)
        try:
            res = SCENARIOS[name](st, work)
            res["ok"] = all(v for v in res["checks"].values() if v is not None)
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}", "checks": {}, "ok": False}
            import traceback

            traceback.print_exc()
        finally:
            st.stop_all()
        res["seconds"] = round(time.time() - t0, 1)
        res["git"] = git
        res["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (out / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
        summary[name] = {"ok": res["ok"], "checks": res["checks"], "summary": res.get("summary"), "seconds": res["seconds"]}
        print(json.dumps({name: summary[name]}, ensure_ascii=False, default=str), flush=True)
    # summary covers every scenario that has a result file, so partial (--only) runs don't erase older results
    merged = {}
    for name in SCENARIOS:
        f = out / f"{name}.json"
        if f.exists():
            r = json.loads(f.read_text())
            merged[name] = {"ok": r.get("ok", False), "checks": r.get("checks", {}), "summary": r.get("summary"),
                            "seconds": r.get("seconds"), "git": r.get("git"), "time": r.get("time")}
    (out / "summary.json").write_text(json.dumps({"git": git, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "scenarios": merged},
                                                 indent=1, ensure_ascii=False, default=str))
    return 0 if all(s["ok"] for s in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
