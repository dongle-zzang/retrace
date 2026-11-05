"""Detached vehicle/frame records and bounded latest-frame handoff storage.

PersonMetadata is kept as the shared detection record base and persons as an
always-empty legacy field; person detection (PeopleNet) was removed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import threading

UNTRACKED_OBJECT_ID = (1 << 64) - 1


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class PersonMetadata:
    camera_id: str
    timestamp: str
    track_id: int | None
    class_id: int
    confidence: float | None
    bbox: BoundingBox
    tracker_confidence: float | None

    def to_dict(self):
        return {**asdict(self), "class": "person"}


@dataclass(frozen=True)
class VehicleMetadata(PersonMetadata):
    # Closed polygon ((x, y), ...) in the same mux pixels as bbox; None without a mask or with seg off.
    outline: tuple[tuple[float, float], ...] | None = None

    def to_dict(self):
        return {**asdict(self), "class": "vehicle"}


def vehicle_from_object(camera_id, timestamp, obj, class_id, outline=None):
    person = person_from_object(camera_id, timestamp, obj)
    return VehicleMetadata(camera_id, timestamp, person.track_id, class_id, person.confidence,
                           person.bbox, person.tracker_confidence, outline)


@dataclass(frozen=True)
class FrameMetadata:
    source_id: int
    camera_id: str
    frame_number: int
    timestamp: str
    pts_ns: int | None
    inference_done: bool
    persons: tuple[PersonMetadata, ...]
    generation: int = 0
    pipeline_source_id: int = 0
    bbox_width: int = 1920
    bbox_height: int = 1080
    vehicles: tuple[VehicleMetadata, ...] = ()
    vehicle_detection_enabled: bool = False
    vehicle_inference_done: bool = False

    def to_dict(self):
        result = asdict(self)
        result["persons"] = [person.to_dict() for person in self.persons]
        result["vehicles"] = [vehicle.to_dict() for vehicle in self.vehicles]
        return result


def utc_timestamp():
    # Processing time, not the camera's capture/NTP time.
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def valid_confidence(value):
    value = float(value)
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def person_from_object(camera_id, timestamp, obj):
    """Copy values while NvDsObjectMeta is valid; retain no PyDS references.

    rect_params is the current tracker/OSD box in mux coordinates. Negative
    detector confidence (tracker-only objects, some clustering modes) is null.
    """
    rect = obj.rect_params
    track_id = int(obj.object_id)
    return PersonMetadata(
        camera_id, timestamp, None if track_id == UNTRACKED_OBJECT_ID else track_id,
        int(obj.class_id), valid_confidence(obj.confidence),
        BoundingBox(float(rect.left), float(rect.top), float(rect.width), float(rect.height)),
        valid_confidence(obj.tracker_confidence),
    )


class MetadataStore:
    """One immutable frame per source. Consumers poll outside the GPU probe.

    This is a latest snapshot, not a lossless event queue. An empty frame clears
    previous vehicles. No I/O or backend calls run on streaming threads.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._latest = {}

    def put(self, frame):
        with self._lock:
            self._latest[frame.source_id] = frame

    def clear(self, source_id):
        with self._lock:
            self._latest.pop(source_id, None)

    def snapshot(self):
        with self._lock:
            return dict(self._latest)
