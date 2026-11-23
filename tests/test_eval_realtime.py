"""Evaluation in real time: the video is played at its own speed and the analyser drops frames when it falls behind.

Offline (the default) every frame is analysed, so the result depends only on the inputs. In real time the result also
depends on how much CPU the analyser gets: that is what the CPU-limit measurement on Kubernetes looks at. The result
file must say which mode it was made in, how many frames were dropped, and how much CPU the container was allowed
to use. `--threads` sets the detector's thread count, so it can be matched to that limit.

The totals that scripts/compare_event_eval.py compares ("aggregate", and "per_video" without "run") must not get new
keys: a result made offline has to stay comparable, value for value, with the results recorded before this change.

The analyser itself is replaced by a stand-in here: these tests check what run_eval passes on and what it records.
"""

import json
import os
import sys

import pytest

import sceneproof.eval.run_eval as run_eval
from sceneproof.eval import provenance

AGGREGATE_KEYS = {"tp", "fp", "missed", "duplicates", "boundary_hits", "duration_s", "gt_positives", "precision",
                  "recall", "f1", "false_alarms_per_hour"}

CAMERAS = "eval-indoor,eval-store"
DROPPED = {"eval-indoor": 7, "eval-store": 3}


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_run_offline(cfg, camera_id, out_jsonl, source=None, realtime=False, run_id=None):
        seen.append({"camera": camera_id, "realtime": realtime, "threads": cfg.detector.intra_op_threads})
        open(out_jsonl, "w").close()  # no events
        return {"run_id": run_id, "frames": 100, "wall_s": 10.0, "fps": 10.0, "events_written": 0, "timings": {},
                "dropped": DROPPED[camera_id] if realtime else 0, "detector": {"model_name": "stand-in"}}

    monkeypatch.setattr(run_eval, "run_offline", fake_run_offline)
    monkeypatch.setattr(run_eval, "environment_info", lambda *a, **k: {"python": "test"})
    monkeypatch.setattr(run_eval, "collect_inputs", lambda *a, **k: {})
    return seen


def run(tmp_path, monkeypatch, *extra):
    out = tmp_path / "result.json"
    argv = ["run_eval", "--cameras", CAMERAS, "--out", str(out), "--runs-dir", str(tmp_path / "runs"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    assert run_eval.main() == 0
    return json.loads(out.read_text())


def test_offline_is_still_the_default(tmp_path, monkeypatch, calls):
    result = run(tmp_path, monkeypatch)
    assert [c["realtime"] for c in calls] == [False, False]
    assert result["realtime"] is False
    assert result["dropped_frames"] == 0
    assert all(v["run"]["dropped"] == 0 for v in result["per_video"].values())


def test_realtime_is_passed_to_the_analyser_and_recorded(tmp_path, monkeypatch, calls):
    result = run(tmp_path, monkeypatch, "--realtime")
    assert [c["realtime"] for c in calls] == [True, True]
    assert result["realtime"] is True


def test_dropped_frames_are_recorded_per_video_and_in_total(tmp_path, monkeypatch, calls):
    result = run(tmp_path, monkeypatch, "--realtime")
    assert {name: v["run"]["dropped"] for name, v in result["per_video"].items()} == DROPPED
    assert result["dropped_frames"] == sum(DROPPED.values())


def test_the_compared_totals_get_no_new_keys(tmp_path, monkeypatch, calls):
    for extra in ([], ["--realtime"]):
        result = run(tmp_path, monkeypatch, *extra)
        assert set(result["aggregate"]) == AGGREGATE_KEYS


def test_threads_sets_the_detector_thread_count(tmp_path, monkeypatch, calls):
    default = run(tmp_path, monkeypatch)
    configured = default["config"]["detector"]["intra_op_threads"]
    assert {c["threads"] for c in calls} == {configured}       # unchanged without the option
    calls.clear()
    result = run(tmp_path, monkeypatch, "--threads", "1")
    assert {c["threads"] for c in calls} == {1}
    assert result["config"]["detector"]["intra_op_threads"] == 1  # the result says what was used


@pytest.mark.parametrize("bad", ["0", "-2", "two"])
def test_threads_must_be_a_positive_whole_number(tmp_path, monkeypatch, calls, bad):
    monkeypatch.setattr(sys, "argv", ["run_eval", "--out", str(tmp_path / "r.json"), "--threads", bad])
    with pytest.raises(SystemExit) as stop:
        run_eval.main()
    assert stop.value.code != 0
    assert calls == []


# --- how much CPU the container may use -------------------------------------------------------------------------
# Inside a container, Linux (cgroup v2) says this in the file cpu.max: "<quota> <period>" in microseconds, or
# "max <period>" when there is no limit. A Kubernetes limit of cpu: 500m shows up as "50000 100000".

def cgroup(tmp_path, text):
    (tmp_path / "cpu.max").write_text(text)
    return tmp_path


@pytest.mark.parametrize("text, cpus", [("50000 100000\n", 0.5), ("100000 100000\n", 1.0), ("200000 100000\n", 2.0),
                                        ("150000 100000", 1.5), ("25000 50000\n", 0.5)])
def test_cpu_limit_is_quota_divided_by_period(tmp_path, text, cpus):
    assert provenance.cpu_limit(cgroup(tmp_path, text)) == pytest.approx(cpus)


def test_cpu_limit_is_none_when_there_is_no_limit(tmp_path):
    assert provenance.cpu_limit(cgroup(tmp_path, "max 100000\n")) is None


@pytest.mark.parametrize("text", ["", "\n", "garbage", "100000", "100000 0", "abc 100000", "1 2 3"])
def test_cpu_limit_is_none_when_the_file_cannot_be_understood(tmp_path, text):
    assert provenance.cpu_limit(cgroup(tmp_path, text)) is None


def test_cpu_limit_is_none_when_the_file_is_missing(tmp_path):
    assert provenance.cpu_limit(tmp_path) is None            # not Linux, or an older kind of cgroup
    assert provenance.cpu_limit(tmp_path / "no-such-folder") is None


def test_cpu_limit_reads_the_real_place_by_default():
    limit = provenance.cpu_limit()
    assert limit is None or (isinstance(limit, float) and limit > 0)


def test_environment_names_the_cpu_limit_and_the_cpu_count():
    env = provenance.environment_info()
    assert "cpu_limit" in env and env["cpu_limit"] == provenance.cpu_limit()
    assert env["cpu_count"] == os.cpu_count()
