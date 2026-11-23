"""An evaluation result must say which input files and which tools produced it.

The same settings gave different results on different computers (docs/experiments/rerun-20261121). One cause was
measured: ffmpeg writes a different video file depending on how many CPU cores the machine has, unless the number
of threads is fixed. Another: the analyser reads video through GStreamer when it is installed and through OpenCV
when it is not, and the two give different frames. So a result is only comparable with another when both name the
exact files they read (SHA-256) and the tools they ran with. `sceneproof.eval.provenance` collects that; `run_eval` stores it.
"""

import hashlib
import json
import os
import platform
import stat
from types import SimpleNamespace

import cv2
import numpy
import onnxruntime
import pytest

from sceneproof.edge.sources import gstreamer_available
from sceneproof.eval import provenance
from sceneproof.eval.run_eval import eval_inputs


def write(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def fake_tool(tmp_path, name, script):
    tool = tmp_path / name
    tool.write_text("#!/bin/sh\n" + script)
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return str(tool)


# ----------------------------------------------------------------------------- hashing a file

def test_sha256_is_the_hash_of_the_file_contents(tmp_path):
    data = os.urandom(5000)
    f = write(tmp_path / "a.bin", data)
    assert provenance.file_sha256(f) == hashlib.sha256(data).hexdigest()
    assert provenance.file_sha256(str(f)) == hashlib.sha256(data).hexdigest()      # a str path works too


def test_sha256_does_not_depend_on_the_chunk_size(tmp_path):
    data = os.urandom(1000)
    f = write(tmp_path / "a.bin", data)
    want = hashlib.sha256(data).hexdigest()
    assert provenance.file_sha256(f, chunk_size=7) == want     # read in many small pieces
    assert provenance.file_sha256(f, chunk_size=1 << 20) == want
    assert provenance.file_sha256(write(tmp_path / "empty.bin", b"")) == hashlib.sha256(b"").hexdigest()


def test_sha256_of_a_missing_file_is_an_error_not_an_empty_hash(tmp_path):
    with pytest.raises(FileNotFoundError):
        provenance.file_sha256(tmp_path / "nope.mp4")


# ----------------------------------------------------------------------------- the tools that ran

def test_environment_names_the_libraries_that_decide_the_result():
    env = provenance.environment_info()
    assert env["python"] == platform.python_version()
    assert env["numpy"] == numpy.__version__
    assert env["opencv"] == cv2.__version__
    assert env["onnxruntime"] == onnxruntime.__version__
    assert isinstance(env["platform"], str) and env["platform"]
    assert "ffmpeg" in env                       # None when there is no ffmpeg, never a missing key
    # reading a video through GStreamer or through OpenCV gives different frames, hence different events:
    # the result must say whether GStreamer was there (its version) or not (None)
    assert "gstreamer" in env and (env["gstreamer"] is None or (isinstance(env["gstreamer"], str) and env["gstreamer"]))
    assert (env["gstreamer"] is not None) == gstreamer_available()
    json.dumps(env)                              # must be storable in the result file as it is


def test_environment_ffmpeg_is_the_first_line_the_tool_prints(tmp_path):
    tool = fake_tool(tmp_path, "ffmpeg", 'echo "ffmpeg version 9.9-test Copyright (c) the FFmpeg developers"\necho "built with gcc"\n')
    assert provenance.environment_info(ffmpeg=tool)["ffmpeg"] == "ffmpeg version 9.9-test Copyright (c) the FFmpeg developers"


def test_environment_ffmpeg_is_none_when_the_tool_is_missing_or_fails(tmp_path):
    assert provenance.environment_info(ffmpeg=str(tmp_path / "no-such-ffmpeg"))["ffmpeg"] is None
    broken = fake_tool(tmp_path, "ffmpeg-broken", "echo oops >&2\nexit 3\n")
    assert provenance.environment_info(ffmpeg=broken)["ffmpeg"] is None


# ----------------------------------------------------------------------------- the files that were read

def test_collect_inputs_records_hash_and_size_under_the_path_as_given(tmp_path):
    a = write(tmp_path / "videos" / "a.mp4", b"aaaa")
    b = write(tmp_path / "models" / "m.onnx", b"bbbbbbbb")
    got = provenance.collect_inputs(["videos/a.mp4", "models/m.onnx"], root=tmp_path)
    assert got == {"videos/a.mp4": {"sha256": hashlib.sha256(b"aaaa").hexdigest(), "bytes": 4},
                   "models/m.onnx": {"sha256": hashlib.sha256(b"bbbbbbbb").hexdigest(), "bytes": 8}}
    assert list(got) == ["videos/a.mp4", "models/m.onnx"]        # same order as asked
    assert provenance.collect_inputs([str(a)]) == {str(a): {"sha256": hashlib.sha256(b"aaaa").hexdigest(), "bytes": 4}}
    assert b.exists()


def test_collect_inputs_fails_on_a_missing_file(tmp_path):
    write(tmp_path / "a.mp4", b"x")
    with pytest.raises(FileNotFoundError):
        provenance.collect_inputs(["a.mp4", "gone.mp4"], root=tmp_path)


def test_diff_inputs_is_empty_when_every_locked_file_has_the_locked_hash():
    locked = {"a.mp4": "11", "m.onnx": "22"}
    actual = {"a.mp4": {"sha256": "11", "bytes": 1}, "m.onnx": {"sha256": "22", "bytes": 2}, "extra.mp4": {"sha256": "99", "bytes": 3}}
    assert provenance.diff_inputs(locked, actual) == []          # a file that is not in the lock is not a difference


def test_diff_inputs_names_each_file_that_changed_or_is_missing():
    locked = {"a.mp4": "11", "b.mp4": "22", "m.onnx": "33"}
    actual = {"a.mp4": {"sha256": "11", "bytes": 1}, "b.mp4": {"sha256": "ff", "bytes": 2}}
    problems = provenance.diff_inputs(locked, actual)
    assert len(problems) == 2
    assert any("b.mp4" in p and "22" in p and "ff" in p for p in problems)      # says which file, expected and found
    assert any("m.onnx" in p for p in problems)
    assert not any("a.mp4" in p for p in problems)


# ----------------------------------------------------------------------------- command line: lock / verify

def test_lock_then_verify_passes_and_a_changed_file_fails(tmp_path, capsys):
    write(tmp_path / "data" / "a.mp4", b"video-a")
    write(tmp_path / "models" / "m.onnx", b"model")
    lock = tmp_path / "inputs.lock.json"
    assert provenance.main(["lock", "--root", str(tmp_path), "--out", str(lock), "data/a.mp4", "models/m.onnx"]) == 0
    saved = json.loads(lock.read_text())
    assert saved == {"files": {"data/a.mp4": hashlib.sha256(b"video-a").hexdigest(), "models/m.onnx": hashlib.sha256(b"model").hexdigest()}}
    assert provenance.main(["verify", "--root", str(tmp_path), "--lock", str(lock)]) == 0

    write(tmp_path / "data" / "a.mp4", b"video-a, encoded on another machine")
    capsys.readouterr()
    assert provenance.main(["verify", "--root", str(tmp_path), "--lock", str(lock)]) == 1
    out = capsys.readouterr()
    assert "data/a.mp4" in out.out + out.err

    (tmp_path / "models" / "m.onnx").unlink()
    assert provenance.main(["verify", "--root", str(tmp_path), "--lock", str(lock)]) == 1      # a missing file is a failure, not a crash


# ----------------------------------------------------------------------------- which files an evaluation reads

def test_eval_inputs_are_the_model_and_each_local_video_once():
    cfg = SimpleNamespace(detector=SimpleNamespace(model="models/yolox_tiny.onnx"), cameras=[
        SimpleNamespace(source="file://data/videos/a.mp4"), SimpleNamespace(source="rtsp://camera.local/stream"),
        SimpleNamespace(source="file://data/videos/b.mp4"), SimpleNamespace(source="file://data/videos/a.mp4")])
    assert eval_inputs(cfg) == ["models/yolox_tiny.onnx", "data/videos/a.mp4", "data/videos/b.mp4"]
