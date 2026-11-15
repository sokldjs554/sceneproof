"""Every-frame (offline) analysis writes the same evidence however fast the machine or slow the clip encoder is."""

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gpu_channels as gc  # noqa: E402
from sceneproof.edge.detector import build_detector  # noqa: E402
from sceneproof.edge.evidence import EvidenceWriter  # noqa: E402

VIDEO = "people-indoor"


@pytest.mark.skipif(not ((ROOT / "models" / "yolox_nano.onnx").exists() and (ROOT / "data" / "videos" / f"{VIDEO}.mp4").exists()),
                    reason="needs scripts/fetch_data.sh and fetch_models.sh")
def test_a_slow_clip_encoder_does_not_shorten_offline_clips(tmp_path, monkeypatch):
    det_cfg = gc.detector_dict("yolox_nano", "onnxruntime", None)
    det = build_detector(det_cfg)
    normal = gc.run_lossless(det, det_cfg, {}, [VIDEO], tmp_path, "normal")["videos"][VIDEO]

    write_clip = EvidenceWriter.write_clip

    def slow_write_clip(self, ep, ring):
        time.sleep(0.4)  # the analyser would get seconds of video ahead in this time and push frames out of the ring
        return write_clip(self, ep, ring)

    monkeypatch.setattr(EvidenceWriter, "write_clip", slow_write_clip)
    slow = gc.run_lossless(det, det_cfg, {}, [VIDEO], tmp_path, "slow")["videos"][VIDEO]
    assert normal["evidence"]["clips"] >= 2
    assert slow["events"] == normal["events"]
    assert slow["evidence"] == normal["evidence"]
