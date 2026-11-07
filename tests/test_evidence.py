import numpy as np

from sceneproof.edge.evidence import EvidenceWriter, draw_annotations
from sceneproof.edge.frames import Frame, FrameRing
from sceneproof.edge.zones import CloseReason, Episode, ZoneSpec


def ring_with(n, fps=10):
    r = FrameRing(max_seconds=60, max_bytes=50 << 20, max_side=160)
    for i in range(n):
        img = np.full((90, 160, 3), (i * 7) % 255, np.uint8)
        r.append(Frame(i, i / fps, 1000 + i / fps, img))
    return r


def episode(zone):
    ep = Episode("ev-1", zone, 5, 30, 3.0, 1003.0, (10, 10, 40, 80), 60, 6.0, 1006.0, (12, 10, 42, 80))
    ep.closed = True
    ep.close_reason = CloseReason.EXITED
    ep.exit_frame, ep.exit_media_s, ep.exit_wall_ts = 60, 6.0, 1006.0
    ep.dwell_escalated, ep.dwell_frame, ep.dwell_media_s = True, 45, 4.5
    return ep


def test_keyframes_and_clip(tmp_path):
    zone = ZoneSpec("z", "zone", [(0.2, 0.2), (0.8, 0.2), (0.8, 0.9), (0.2, 0.9)])
    ring = ring_with(100)
    w = EvidenceWriter(tmp_path, pre_roll_s=1.0, post_roll_s=1.0, clip_fps=5)
    ep = episode(zone)
    kfs = w.keyframes_for(ep, ring, (160, 90), ["context_before", "entry", "dwell", "exit", "context_after"])
    roles = {k.role for k in kfs}
    assert roles == {"context_before", "entry", "dwell", "exit", "context_after"}
    for k in kfs:
        assert (tmp_path / k.path).exists() and (tmp_path / k.annot_path).exists()
    entry = next(k for k in kfs if k.role == "entry")
    assert entry.index == 30 and entry.box == (10, 10, 40, 80)
    clip = w.write_clip(ep, ring)
    assert clip is not None and (tmp_path / clip.path).exists()
    assert abs(clip.start_media_s - 2.0) < 0.11 and abs(clip.end_media_s - 7.0) < 0.21
    assert clip.frames <= 5 * 5 + 2 and clip.codec in ("libx264", "mpeg4", "mjpeg")


def test_draw_annotations_does_not_mutate_input():
    img = np.zeros((90, 160, 3), np.uint8)
    zone = ZoneSpec("z", "zone", [(0.2, 0.2), (0.8, 0.2), (0.8, 0.9), (0.2, 0.9)])
    out = draw_annotations(img, zone, (10, 10, 40, 80), "track 1")
    assert img.sum() == 0 and out.sum() > 0
