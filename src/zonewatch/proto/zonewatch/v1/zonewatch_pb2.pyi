import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EventKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EVENT_KIND_UNSPECIFIED: _ClassVar[EventKind]
    EVENT_KIND_ZONE_ENTRY: _ClassVar[EventKind]
    EVENT_KIND_ZONE_DWELL: _ClassVar[EventKind]

class EventStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EVENT_STATUS_UNSPECIFIED: _ClassVar[EventStatus]
    EVENT_STATUS_OPEN: _ClassVar[EventStatus]
    EVENT_STATUS_CLOSED: _ClassVar[EventStatus]

class CloseReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CLOSE_REASON_UNSPECIFIED: _ClassVar[CloseReason]
    CLOSE_REASON_EXITED: _ClassVar[CloseReason]
    CLOSE_REASON_TRACK_LOST: _ClassVar[CloseReason]
    CLOSE_REASON_STREAM_ENDED: _ClassVar[CloseReason]
    CLOSE_REASON_SHUTDOWN: _ClassVar[CloseReason]

class CameraState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CAMERA_STATE_UNSPECIFIED: _ClassVar[CameraState]
    CAMERA_STATE_CONNECTING: _ClassVar[CameraState]
    CAMERA_STATE_STREAMING: _ClassVar[CameraState]
    CAMERA_STATE_RECONNECTING: _ClassVar[CameraState]
    CAMERA_STATE_ENDED: _ClassVar[CameraState]
    CAMERA_STATE_STOPPED: _ClassVar[CameraState]
    CAMERA_STATE_ERROR: _ClassVar[CameraState]

class CommandType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COMMAND_TYPE_UNSPECIFIED: _ClassVar[CommandType]
    COMMAND_TYPE_START_CAMERA: _ClassVar[CommandType]
    COMMAND_TYPE_STOP_CAMERA: _ClassVar[CommandType]
EVENT_KIND_UNSPECIFIED: EventKind
EVENT_KIND_ZONE_ENTRY: EventKind
EVENT_KIND_ZONE_DWELL: EventKind
EVENT_STATUS_UNSPECIFIED: EventStatus
EVENT_STATUS_OPEN: EventStatus
EVENT_STATUS_CLOSED: EventStatus
CLOSE_REASON_UNSPECIFIED: CloseReason
CLOSE_REASON_EXITED: CloseReason
CLOSE_REASON_TRACK_LOST: CloseReason
CLOSE_REASON_STREAM_ENDED: CloseReason
CLOSE_REASON_SHUTDOWN: CloseReason
CAMERA_STATE_UNSPECIFIED: CameraState
CAMERA_STATE_CONNECTING: CameraState
CAMERA_STATE_STREAMING: CameraState
CAMERA_STATE_RECONNECTING: CameraState
CAMERA_STATE_ENDED: CameraState
CAMERA_STATE_STOPPED: CameraState
CAMERA_STATE_ERROR: CameraState
COMMAND_TYPE_UNSPECIFIED: CommandType
COMMAND_TYPE_START_CAMERA: CommandType
COMMAND_TYPE_STOP_CAMERA: CommandType

class Point(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ...) -> None: ...

class BBox(_message.Message):
    __slots__ = ("x1", "y1", "x2", "y2")
    X1_FIELD_NUMBER: _ClassVar[int]
    Y1_FIELD_NUMBER: _ClassVar[int]
    X2_FIELD_NUMBER: _ClassVar[int]
    Y2_FIELD_NUMBER: _ClassVar[int]
    x1: float
    y1: float
    x2: float
    y2: float
    def __init__(self, x1: _Optional[float] = ..., y1: _Optional[float] = ..., x2: _Optional[float] = ..., y2: _Optional[float] = ...) -> None: ...

class TrackPoint(_message.Message):
    __slots__ = ("ts", "frame_index", "media_time_s", "box", "score", "in_zone")
    TS_FIELD_NUMBER: _ClassVar[int]
    FRAME_INDEX_FIELD_NUMBER: _ClassVar[int]
    MEDIA_TIME_S_FIELD_NUMBER: _ClassVar[int]
    BOX_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    IN_ZONE_FIELD_NUMBER: _ClassVar[int]
    ts: _timestamp_pb2.Timestamp
    frame_index: int
    media_time_s: float
    box: BBox
    score: float
    in_zone: bool
    def __init__(self, ts: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., frame_index: _Optional[int] = ..., media_time_s: _Optional[float] = ..., box: _Optional[_Union[BBox, _Mapping]] = ..., score: _Optional[float] = ..., in_zone: _Optional[bool] = ...) -> None: ...

class KeyFrame(_message.Message):
    __slots__ = ("role", "ts", "frame_index", "media_time_s", "path", "box", "width", "height")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    TS_FIELD_NUMBER: _ClassVar[int]
    FRAME_INDEX_FIELD_NUMBER: _ClassVar[int]
    MEDIA_TIME_S_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    BOX_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    role: str
    ts: _timestamp_pb2.Timestamp
    frame_index: int
    media_time_s: float
    path: str
    box: BBox
    width: int
    height: int
    def __init__(self, role: _Optional[str] = ..., ts: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., frame_index: _Optional[int] = ..., media_time_s: _Optional[float] = ..., path: _Optional[str] = ..., box: _Optional[_Union[BBox, _Mapping]] = ..., width: _Optional[int] = ..., height: _Optional[int] = ...) -> None: ...

class Evidence(_message.Message):
    __slots__ = ("key_frames", "clip_path", "clip_start_media_s", "clip_end_media_s", "clip_fps", "clip_frames")
    KEY_FRAMES_FIELD_NUMBER: _ClassVar[int]
    CLIP_PATH_FIELD_NUMBER: _ClassVar[int]
    CLIP_START_MEDIA_S_FIELD_NUMBER: _ClassVar[int]
    CLIP_END_MEDIA_S_FIELD_NUMBER: _ClassVar[int]
    CLIP_FPS_FIELD_NUMBER: _ClassVar[int]
    CLIP_FRAMES_FIELD_NUMBER: _ClassVar[int]
    key_frames: _containers.RepeatedCompositeFieldContainer[KeyFrame]
    clip_path: str
    clip_start_media_s: float
    clip_end_media_s: float
    clip_fps: float
    clip_frames: int
    def __init__(self, key_frames: _Optional[_Iterable[_Union[KeyFrame, _Mapping]]] = ..., clip_path: _Optional[str] = ..., clip_start_media_s: _Optional[float] = ..., clip_end_media_s: _Optional[float] = ..., clip_fps: _Optional[float] = ..., clip_frames: _Optional[int] = ...) -> None: ...

class RunInfo(_message.Message):
    __slots__ = ("run_id", "edge_version", "model_name", "model_version", "backend", "tracker", "config_hash", "input_width", "input_height")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    EDGE_VERSION_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    TRACKER_FIELD_NUMBER: _ClassVar[int]
    CONFIG_HASH_FIELD_NUMBER: _ClassVar[int]
    INPUT_WIDTH_FIELD_NUMBER: _ClassVar[int]
    INPUT_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    edge_version: str
    model_name: str
    model_version: str
    backend: str
    tracker: str
    config_hash: str
    input_width: int
    input_height: int
    def __init__(self, run_id: _Optional[str] = ..., edge_version: _Optional[str] = ..., model_name: _Optional[str] = ..., model_version: _Optional[str] = ..., backend: _Optional[str] = ..., tracker: _Optional[str] = ..., config_hash: _Optional[str] = ..., input_width: _Optional[int] = ..., input_height: _Optional[int] = ...) -> None: ...

class Event(_message.Message):
    __slots__ = ("event_id", "revision", "device_id", "camera_id", "zone_id", "zone_name", "kind", "status", "close_reason", "track_id", "entry_ts", "exit_ts", "detected_at", "entry_frame_index", "exit_frame_index", "media_entry_s", "media_exit_s", "dwell_seconds", "trajectory", "evidence", "run", "zone_polygon", "frame_width", "frame_height", "source")
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    ZONE_ID_FIELD_NUMBER: _ClassVar[int]
    ZONE_NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CLOSE_REASON_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    ENTRY_TS_FIELD_NUMBER: _ClassVar[int]
    EXIT_TS_FIELD_NUMBER: _ClassVar[int]
    DETECTED_AT_FIELD_NUMBER: _ClassVar[int]
    ENTRY_FRAME_INDEX_FIELD_NUMBER: _ClassVar[int]
    EXIT_FRAME_INDEX_FIELD_NUMBER: _ClassVar[int]
    MEDIA_ENTRY_S_FIELD_NUMBER: _ClassVar[int]
    MEDIA_EXIT_S_FIELD_NUMBER: _ClassVar[int]
    DWELL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TRAJECTORY_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_FIELD_NUMBER: _ClassVar[int]
    RUN_FIELD_NUMBER: _ClassVar[int]
    ZONE_POLYGON_FIELD_NUMBER: _ClassVar[int]
    FRAME_WIDTH_FIELD_NUMBER: _ClassVar[int]
    FRAME_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    event_id: str
    revision: int
    device_id: str
    camera_id: str
    zone_id: str
    zone_name: str
    kind: EventKind
    status: EventStatus
    close_reason: CloseReason
    track_id: int
    entry_ts: _timestamp_pb2.Timestamp
    exit_ts: _timestamp_pb2.Timestamp
    detected_at: _timestamp_pb2.Timestamp
    entry_frame_index: int
    exit_frame_index: int
    media_entry_s: float
    media_exit_s: float
    dwell_seconds: float
    trajectory: _containers.RepeatedCompositeFieldContainer[TrackPoint]
    evidence: Evidence
    run: RunInfo
    zone_polygon: _containers.RepeatedCompositeFieldContainer[Point]
    frame_width: int
    frame_height: int
    source: str
    def __init__(self, event_id: _Optional[str] = ..., revision: _Optional[int] = ..., device_id: _Optional[str] = ..., camera_id: _Optional[str] = ..., zone_id: _Optional[str] = ..., zone_name: _Optional[str] = ..., kind: _Optional[_Union[EventKind, str]] = ..., status: _Optional[_Union[EventStatus, str]] = ..., close_reason: _Optional[_Union[CloseReason, str]] = ..., track_id: _Optional[int] = ..., entry_ts: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., exit_ts: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., detected_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., entry_frame_index: _Optional[int] = ..., exit_frame_index: _Optional[int] = ..., media_entry_s: _Optional[float] = ..., media_exit_s: _Optional[float] = ..., dwell_seconds: _Optional[float] = ..., trajectory: _Optional[_Iterable[_Union[TrackPoint, _Mapping]]] = ..., evidence: _Optional[_Union[Evidence, _Mapping]] = ..., run: _Optional[_Union[RunInfo, _Mapping]] = ..., zone_polygon: _Optional[_Iterable[_Union[Point, _Mapping]]] = ..., frame_width: _Optional[int] = ..., frame_height: _Optional[int] = ..., source: _Optional[str] = ...) -> None: ...

class CameraStatus(_message.Message):
    __slots__ = ("camera_id", "source", "name", "zones", "state", "input_fps", "processed_fps", "analysis_lag_ms", "queue_depth", "queue_capacity", "dropped_frames", "frames_received", "frames_processed", "infer_ms_p50", "infer_ms_p95", "stage_ms_decode", "stage_ms_preprocess", "stage_ms_infer", "stage_ms_postprocess", "stage_ms_track", "stage_ms_evidence", "reconnect_count", "last_error", "last_frame_ts", "open_episodes", "ring_buffer_bytes", "media_time_s")
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ZONES_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    INPUT_FPS_FIELD_NUMBER: _ClassVar[int]
    PROCESSED_FPS_FIELD_NUMBER: _ClassVar[int]
    ANALYSIS_LAG_MS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    QUEUE_CAPACITY_FIELD_NUMBER: _ClassVar[int]
    DROPPED_FRAMES_FIELD_NUMBER: _ClassVar[int]
    FRAMES_RECEIVED_FIELD_NUMBER: _ClassVar[int]
    FRAMES_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    INFER_MS_P50_FIELD_NUMBER: _ClassVar[int]
    INFER_MS_P95_FIELD_NUMBER: _ClassVar[int]
    STAGE_MS_DECODE_FIELD_NUMBER: _ClassVar[int]
    STAGE_MS_PREPROCESS_FIELD_NUMBER: _ClassVar[int]
    STAGE_MS_INFER_FIELD_NUMBER: _ClassVar[int]
    STAGE_MS_POSTPROCESS_FIELD_NUMBER: _ClassVar[int]
    STAGE_MS_TRACK_FIELD_NUMBER: _ClassVar[int]
    STAGE_MS_EVIDENCE_FIELD_NUMBER: _ClassVar[int]
    RECONNECT_COUNT_FIELD_NUMBER: _ClassVar[int]
    LAST_ERROR_FIELD_NUMBER: _ClassVar[int]
    LAST_FRAME_TS_FIELD_NUMBER: _ClassVar[int]
    OPEN_EPISODES_FIELD_NUMBER: _ClassVar[int]
    RING_BUFFER_BYTES_FIELD_NUMBER: _ClassVar[int]
    MEDIA_TIME_S_FIELD_NUMBER: _ClassVar[int]
    camera_id: str
    source: str
    name: str
    zones: _containers.RepeatedCompositeFieldContainer[ZoneSpec]
    state: CameraState
    input_fps: float
    processed_fps: float
    analysis_lag_ms: float
    queue_depth: int
    queue_capacity: int
    dropped_frames: int
    frames_received: int
    frames_processed: int
    infer_ms_p50: float
    infer_ms_p95: float
    stage_ms_decode: float
    stage_ms_preprocess: float
    stage_ms_infer: float
    stage_ms_postprocess: float
    stage_ms_track: float
    stage_ms_evidence: float
    reconnect_count: int
    last_error: str
    last_frame_ts: _timestamp_pb2.Timestamp
    open_episodes: int
    ring_buffer_bytes: int
    media_time_s: float
    def __init__(self, camera_id: _Optional[str] = ..., source: _Optional[str] = ..., name: _Optional[str] = ..., zones: _Optional[_Iterable[_Union[ZoneSpec, _Mapping]]] = ..., state: _Optional[_Union[CameraState, str]] = ..., input_fps: _Optional[float] = ..., processed_fps: _Optional[float] = ..., analysis_lag_ms: _Optional[float] = ..., queue_depth: _Optional[int] = ..., queue_capacity: _Optional[int] = ..., dropped_frames: _Optional[int] = ..., frames_received: _Optional[int] = ..., frames_processed: _Optional[int] = ..., infer_ms_p50: _Optional[float] = ..., infer_ms_p95: _Optional[float] = ..., stage_ms_decode: _Optional[float] = ..., stage_ms_preprocess: _Optional[float] = ..., stage_ms_infer: _Optional[float] = ..., stage_ms_postprocess: _Optional[float] = ..., stage_ms_track: _Optional[float] = ..., stage_ms_evidence: _Optional[float] = ..., reconnect_count: _Optional[int] = ..., last_error: _Optional[str] = ..., last_frame_ts: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., open_episodes: _Optional[int] = ..., ring_buffer_bytes: _Optional[int] = ..., media_time_s: _Optional[float] = ...) -> None: ...

class UplinkStatus(_message.Message):
    __slots__ = ("connected", "pending_events", "sent_events", "acked_events", "dropped_events", "duplicate_acks", "reconnect_count", "last_ack_ts", "last_error", "queue_bytes", "queue_capacity_bytes", "pending_uploads")
    CONNECTED_FIELD_NUMBER: _ClassVar[int]
    PENDING_EVENTS_FIELD_NUMBER: _ClassVar[int]
    SENT_EVENTS_FIELD_NUMBER: _ClassVar[int]
    ACKED_EVENTS_FIELD_NUMBER: _ClassVar[int]
    DROPPED_EVENTS_FIELD_NUMBER: _ClassVar[int]
    DUPLICATE_ACKS_FIELD_NUMBER: _ClassVar[int]
    RECONNECT_COUNT_FIELD_NUMBER: _ClassVar[int]
    LAST_ACK_TS_FIELD_NUMBER: _ClassVar[int]
    LAST_ERROR_FIELD_NUMBER: _ClassVar[int]
    QUEUE_BYTES_FIELD_NUMBER: _ClassVar[int]
    QUEUE_CAPACITY_BYTES_FIELD_NUMBER: _ClassVar[int]
    PENDING_UPLOADS_FIELD_NUMBER: _ClassVar[int]
    connected: bool
    pending_events: int
    sent_events: int
    acked_events: int
    dropped_events: int
    duplicate_acks: int
    reconnect_count: int
    last_ack_ts: _timestamp_pb2.Timestamp
    last_error: str
    queue_bytes: int
    queue_capacity_bytes: int
    pending_uploads: int
    def __init__(self, connected: _Optional[bool] = ..., pending_events: _Optional[int] = ..., sent_events: _Optional[int] = ..., acked_events: _Optional[int] = ..., dropped_events: _Optional[int] = ..., duplicate_acks: _Optional[int] = ..., reconnect_count: _Optional[int] = ..., last_ack_ts: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., last_error: _Optional[str] = ..., queue_bytes: _Optional[int] = ..., queue_capacity_bytes: _Optional[int] = ..., pending_uploads: _Optional[int] = ...) -> None: ...

class DeviceStatus(_message.Message):
    __slots__ = ("device_id", "ts", "cameras", "uplink", "gateway_version", "analyzer_run_id", "cpu_percent", "rss_mb")
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    TS_FIELD_NUMBER: _ClassVar[int]
    CAMERAS_FIELD_NUMBER: _ClassVar[int]
    UPLINK_FIELD_NUMBER: _ClassVar[int]
    GATEWAY_VERSION_FIELD_NUMBER: _ClassVar[int]
    ANALYZER_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    CPU_PERCENT_FIELD_NUMBER: _ClassVar[int]
    RSS_MB_FIELD_NUMBER: _ClassVar[int]
    device_id: str
    ts: _timestamp_pb2.Timestamp
    cameras: _containers.RepeatedCompositeFieldContainer[CameraStatus]
    uplink: UplinkStatus
    gateway_version: str
    analyzer_run_id: str
    cpu_percent: float
    rss_mb: float
    def __init__(self, device_id: _Optional[str] = ..., ts: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., cameras: _Optional[_Iterable[_Union[CameraStatus, _Mapping]]] = ..., uplink: _Optional[_Union[UplinkStatus, _Mapping]] = ..., gateway_version: _Optional[str] = ..., analyzer_run_id: _Optional[str] = ..., cpu_percent: _Optional[float] = ..., rss_mb: _Optional[float] = ...) -> None: ...

class Ack(_message.Message):
    __slots__ = ("event_id", "revision", "accepted", "duplicate", "message")
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DUPLICATE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    event_id: str
    revision: int
    accepted: bool
    duplicate: bool
    message: str
    def __init__(self, event_id: _Optional[str] = ..., revision: _Optional[int] = ..., accepted: _Optional[bool] = ..., duplicate: _Optional[bool] = ..., message: _Optional[str] = ...) -> None: ...

class EvidenceChunk(_message.Message):
    __slots__ = ("event_id", "path", "data", "offset", "last", "sha256", "total_size")
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    LAST_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    TOTAL_SIZE_FIELD_NUMBER: _ClassVar[int]
    event_id: str
    path: str
    data: bytes
    offset: int
    last: bool
    sha256: str
    total_size: int
    def __init__(self, event_id: _Optional[str] = ..., path: _Optional[str] = ..., data: _Optional[bytes] = ..., offset: _Optional[int] = ..., last: _Optional[bool] = ..., sha256: _Optional[str] = ..., total_size: _Optional[int] = ...) -> None: ...

class UploadResult(_message.Message):
    __slots__ = ("event_id", "path", "ok", "already_present", "size", "message")
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    OK_FIELD_NUMBER: _ClassVar[int]
    ALREADY_PRESENT_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    event_id: str
    path: str
    ok: bool
    already_present: bool
    size: int
    message: str
    def __init__(self, event_id: _Optional[str] = ..., path: _Optional[str] = ..., ok: _Optional[bool] = ..., already_present: _Optional[bool] = ..., size: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class HeartbeatAck(_message.Message):
    __slots__ = ("server_ts",)
    SERVER_TS_FIELD_NUMBER: _ClassVar[int]
    server_ts: _timestamp_pb2.Timestamp
    def __init__(self, server_ts: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ZoneSpec(_message.Message):
    __slots__ = ("zone_id", "name", "polygon", "entry_confirm_frames", "exit_confirm_frames", "dwell_seconds", "lost_grace_seconds")
    ZONE_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    POLYGON_FIELD_NUMBER: _ClassVar[int]
    ENTRY_CONFIRM_FRAMES_FIELD_NUMBER: _ClassVar[int]
    EXIT_CONFIRM_FRAMES_FIELD_NUMBER: _ClassVar[int]
    DWELL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    LOST_GRACE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    zone_id: str
    name: str
    polygon: _containers.RepeatedCompositeFieldContainer[Point]
    entry_confirm_frames: int
    exit_confirm_frames: int
    dwell_seconds: float
    lost_grace_seconds: float
    def __init__(self, zone_id: _Optional[str] = ..., name: _Optional[str] = ..., polygon: _Optional[_Iterable[_Union[Point, _Mapping]]] = ..., entry_confirm_frames: _Optional[int] = ..., exit_confirm_frames: _Optional[int] = ..., dwell_seconds: _Optional[float] = ..., lost_grace_seconds: _Optional[float] = ...) -> None: ...

class Command(_message.Message):
    __slots__ = ("command_id", "type", "device_id", "camera_id", "source", "zones", "issued_at", "issued_by")
    COMMAND_ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    ZONES_FIELD_NUMBER: _ClassVar[int]
    ISSUED_AT_FIELD_NUMBER: _ClassVar[int]
    ISSUED_BY_FIELD_NUMBER: _ClassVar[int]
    command_id: str
    type: CommandType
    device_id: str
    camera_id: str
    source: str
    zones: _containers.RepeatedCompositeFieldContainer[ZoneSpec]
    issued_at: _timestamp_pb2.Timestamp
    issued_by: str
    def __init__(self, command_id: _Optional[str] = ..., type: _Optional[_Union[CommandType, str]] = ..., device_id: _Optional[str] = ..., camera_id: _Optional[str] = ..., source: _Optional[str] = ..., zones: _Optional[_Iterable[_Union[ZoneSpec, _Mapping]]] = ..., issued_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., issued_by: _Optional[str] = ...) -> None: ...

class CommandResult(_message.Message):
    __slots__ = ("command_id", "device_id", "ok", "message", "completed_at")
    COMMAND_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    OK_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_AT_FIELD_NUMBER: _ClassVar[int]
    command_id: str
    device_id: str
    ok: bool
    message: str
    completed_at: _timestamp_pb2.Timestamp
    def __init__(self, command_id: _Optional[str] = ..., device_id: _Optional[str] = ..., ok: _Optional[bool] = ..., message: _Optional[str] = ..., completed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class WatchRequest(_message.Message):
    __slots__ = ("device_id",)
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    device_id: str
    def __init__(self, device_id: _Optional[str] = ...) -> None: ...
