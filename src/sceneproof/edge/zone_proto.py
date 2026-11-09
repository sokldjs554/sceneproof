"""ZoneSpec <-> protobuf ZoneSpec (status reports and START_CAMERA commands use the same conversion)."""

from __future__ import annotations

from sceneproof.proto.sceneproof.v1 import sceneproof_pb2 as pb

from .zones import ZoneSpec


def fill_pb(z: ZoneSpec, zs: pb.ZoneSpec) -> pb.ZoneSpec:
    zs.zone_id, zs.name = z.zone_id, z.name
    zs.entry_confirm_frames, zs.exit_confirm_frames = z.entry_confirm_frames, z.exit_confirm_frames
    zs.dwell_seconds, zs.lost_grace_seconds = z.dwell_seconds, z.lost_grace_seconds
    zs.entry_confirm_seconds = z.entry_confirm_seconds or 0.0
    zs.exit_confirm_seconds = z.exit_confirm_seconds or 0.0
    zs.rejoin, zs.rejoin_dist_rel = z.rejoin, z.rejoin_dist_rel
    for x, y in z.polygon:
        zs.polygon.add(x=float(x), y=float(y))
    for poly in z.exclude:
        ex = zs.exclude.add()
        for x, y in poly:
            ex.points.add(x=float(x), y=float(y))
    return zs


def from_pb(z: pb.ZoneSpec) -> ZoneSpec:
    """Zero values mean 'not set' (proto3): fall back to the same defaults as the YAML config."""
    return ZoneSpec(z.zone_id, z.name, [(p.x, p.y) for p in z.polygon], z.entry_confirm_frames or 3,
                    z.exit_confirm_frames or 5, z.dwell_seconds or 5.0, z.lost_grace_seconds or 1.5,
                    z.entry_confirm_seconds or None, z.exit_confirm_seconds or None,
                    exclude=[[(p.x, p.y) for p in e.points] for e in z.exclude if len(e.points) >= 3],
                    rejoin=z.rejoin, rejoin_dist_rel=z.rejoin_dist_rel or 1.0)
