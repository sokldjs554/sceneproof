"""Several cameras in one analyser: shared detector, track ids, evidence-frame rate, and scripts/gpu_channels.py."""

import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gpu_channels as gc  # noqa: E402
from sceneproof.edge.detector import build_detector  # noqa: E402
from sceneproof.edge.pipeline import CameraPipeline  # noqa: E402
from sceneproof.edge.tracker import STrack  # noqa: E402
from sceneproof.eval.bench_detector import load_frames  # noqa: E402

MODEL = ROOT / "models" / "yolox_nano.onnx"
VIDEOS = ROOT / "data" / "videos"
needs_data = pytest.mark.skipif(not (MODEL.exists() and (VIDEOS / "people-indoor.mp4").exists()),
                                reason="needs scripts/fetch_data.sh and fetch_models.sh")


def test_a_channel_count_passes_only_when_every_channel_keeps_up():
    ok = [{"processed_ratio": 0.99}, {"processed_ratio": 0.981}]
    assert gc.judge(ok, 0.98)
    assert not gc.judge([*ok, {"processed_ratio": 0.97}], 0.98)
    assert not gc.judge([], 0.98)


def test_capacity_is_the_largest_passing_count_below_the_first_failure():
    rows = [{"channels": n, "pass": p} for n, p in ((1, True), (2, True), (4, False), (6, True), (3, True))]
    assert gc.capacity(rows) == (3, 4)  # 6 passed, but only after 4 had failed
    assert gc.capacity([{"channels": 1, "pass": True}]) == (1, None)
    assert gc.capacity([{"channels": 1, "pass": False}]) == (0, 1)


def _kept(fps: float, ring_fps: float, seconds: int = 10) -> list[int]:
    class P:  # the two fields CameraPipeline._ring_due uses
        _ring_due = CameraPipeline._ring_due
        _ring_every_s = 1.0 / ring_fps if ring_fps > 0 else 0.0
        _ring_next_s = float("-inf")

    p = P()
    return [i for i in range(int(fps * seconds)) if p._ring_due(i / fps)]


def test_ring_fps_keeps_that_many_evidence_frames_per_second():
    assert _kept(30, 0) == list(range(300))             # 0 = every analysed frame (the default)
    assert _kept(30, 10) == list(range(0, 300, 3))      # every third frame of a 30 fps camera
    assert len(_kept(25, 10)) == 100                    # 10 per second on average, not every third (8.3)
    assert _kept(10, 10) == list(range(100))            # a camera at or below the cap keeps everything
    assert _kept(10, 30) == list(range(100))


def test_ring_fps_restarts_after_a_gap_in_the_stream():
    class P:
        _ring_due = CameraPipeline._ring_due
        _ring_every_s = 0.1
        _ring_next_s = float("-inf")

    p = P()
    assert [p._ring_due(t) for t in (0.0, 0.05, 0.1)] == [True, False, True]
    assert p._ring_due(60.0)        # reconnect / loop: the stream time jumped ahead
    assert not p._ring_due(60.05)   # ...and the 0.1 s spacing continues from there
    assert p._ring_due(60.1)


def test_track_ids_stay_unique_when_cameras_start_tracks_at_the_same_time():
    ids: list[int] = []
    lock = threading.Lock()

    def worker():
        mine = [STrack.next_id() for _ in range(3000)]
        with lock:
            ids.extend(mine)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(ids)) == len(ids) == 8 * 3000


@needs_data
def test_cameras_sharing_one_detector_get_their_own_detections():
    det = build_detector(gc.detector_dict("yolox_nano", "onnxruntime", None))
    frames = load_frames(str(VIDEOS / "people-indoor.mp4"), 12) + load_frames(str(VIDEOS / "parking-lot.mp4"), 12)
    alone = [det.detect(f) for f in frames]
    together: dict[int, list] = {}

    def camera(k: int):  # each "camera" analyses its own frames, six times over, while the others do the same
        together[k] = [det.detect(frames[i]) for _ in range(6) for i in range(k, len(frames), 4)]

    threads = [threading.Thread(target=camera, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for k in range(4):
        expected = [alone[i] for _ in range(6) for i in range(k, len(frames), 4)]
        assert len(together[k]) == len(expected)
        for got, want in zip(together[k], expected, strict=True):
            assert np.array_equal(got.boxes, want.boxes) and np.array_equal(got.scores, want.scores)


@needs_data
def test_videos_analysed_together_give_the_same_events_as_one_at_a_time(tmp_path):
    det_cfg = gc.detector_dict("yolox_nano", "onnxruntime", None)
    res = gc.isolation(build_detector(det_cfg), det_cfg, ["people-indoor", "parking-lot"], tmp_path, "test")
    assert res["all_same"], res
    assert all(v["frames_together"] > 500 for v in res["videos"].values())
    assert res["videos"]["people-indoor"]["events_together"] >= 1


@needs_data
def test_fewer_evidence_frames_keep_the_events_and_the_entry_picture(tmp_path):
    det_cfg = gc.detector_dict("yolox_nano", "onnxruntime", None)
    res = gc.evidence_check(build_detector(det_cfg), det_cfg, ["people-indoor"], 4.0, tmp_path, "test")  # a 12 fps video
    v = res["videos"]["people-indoor"]
    every, fewer = v["every_frame"], v["ring_fps_4"]
    assert v["same_events"] and every["evidence"]["events"] >= 1
    assert fewer["ring_entries_at_end"] < every["ring_entries_at_end"] * 0.5
    assert fewer["evidence"]["clips"] == every["evidence"]["clips"]
    assert fewer["evidence"]["key_frames"]["entry"] == every["evidence"]["key_frames"]["entry"]
    assert fewer["evidence"]["entry_picture_offset_s_max"] == 0.0   # the frame an event opens on is always kept
    assert fewer["evidence"]["exit_picture_offset_s_max"] <= 0.5 / 4 + 1e-6  # otherwise the nearest kept frame


@pytest.mark.skipif(not ((ROOT / "models" / "yolox_tiny.onnx").exists() and (VIDEOS / "worker-zone.mp4").exists()),
                    reason="needs scripts/fetch_data.sh and fetch_models.sh")
def test_script_records_what_it_cannot_run_and_measures_the_cpu_reference(tmp_path):
    out = tmp_path / "channels.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gpu_channels.py"), "--channels", "1", "--seconds", "2",
                        "--warmup", "1", "--out", str(out)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(out.read_text())
    assert res["overrides"] == {"channels": [1], "measure_s": 2.0, "warmup_s": 1.0}
    assert [(s["role"], s["variant"]) for s in res["sweeps"]] == [("main", "as-built"), ("main", "ring-fps-10"),
                                                                  ("cpu_reference", "as-built")]
    cpu = res["sweeps"][2]
    assert cpu["status"] == "ok" and [row["channels"] for row in cpu["rows"]] == [1, 2]
    for row in cpu["rows"]:
        assert len(row["per_channel"]) == row["channels"] and row["processed_fps_total"] > 0
        assert row["pass"] == all(c["processed_ratio"] >= 0.98 for c in row["per_channel"])
    for s in res["sweeps"][:2]:  # TensorRT: runs on a GPU machine, otherwise the reason is recorded
        assert s["status"] in ("ok", "failed") and (s["status"] == "ok" or s["error"])
    if res["sweeps"][0]["status"] == "failed":
        assert res["isolation"]["status"] == res["evidence_check"]["status"] == "skipped"
