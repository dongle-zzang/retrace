from datetime import datetime, timezone
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

State = Literal["connecting", "online", "degraded", "offline", "reconnecting"]


class StatusOut(BaseModel):
    camera_id: str
    state: State
    fps: float
    last_frame_at: datetime | None
    last_error: str | None
    reconnect_count: int
    generation: int
    runtime_session: str | None
    observed_at: datetime
    stale: bool


class CameraOut(BaseModel):
    camera_id: str
    floor: int | str
    name: str
    enabled: bool
    source_id: int | None
    preview_path: str | None
    metadata_path: str = "/ws"
    status: StatusOut


class HealthOut(BaseModel):
    status: Literal["ok", "unavailable"]
    database: Literal["ok", "unavailable"]
    preview: Literal["reachable", "unreachable", "unknown"]


class StatusSummary(BaseModel):
    counts: dict[State, int]
    cameras: list[StatusOut]


class BBox(BaseModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    width: float = Field(ge=0, allow_inf_nan=False)
    height: float = Field(ge=0, allow_inf_nan=False)


class PersonIn(BaseModel):
    track_id: int | None = Field(default=None, ge=0, lt=(1 << 64) - 1)
    class_id: Literal[0]
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    tracker_confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    bbox: BBox


class VehicleIn(PersonIn):
    class_id: int = Field(ge=0)
    # Pixel polygon in bbox coordinates. An invalid outline only nulls itself: one invalid frame
    # would make the poller drop every camera's frame.
    outline: list[tuple[float, float]] | None = None

    @field_validator("outline", mode="before")
    @classmethod
    def drop_invalid_outline(cls, value):
        def finite(number):
            return isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number)
        if (isinstance(value, (list, tuple)) and 3 <= len(value) <= 32
                and all(isinstance(p, (list, tuple)) and len(p) == 2 and all(map(finite, p)) for p in value)):
            return value
        return None


class FrameIn(BaseModel):
    camera_id: str
    source_id: int = Field(ge=0)
    generation: int = Field(ge=0)
    timestamp: datetime
    frame_number: int = Field(ge=0)
    bbox_width: int = Field(gt=0)
    bbox_height: int = Field(gt=0)
    # Legacy person field: Preview now always sends [], and older senders may omit it.
    persons: list[PersonIn] = Field(default_factory=list, max_length=1000)
    vehicles: list[VehicleIn] = Field(default_factory=list, max_length=1000)
    vehicle_detection_enabled: bool = False
    vehicle_inference_done: bool = False

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError("Timezone required")
        return value


class LiveTrackOut(BaseModel):
    track_id: str | None
    class_id: int
    person_class: Literal["person"] = Field(default="person", serialization_alias="class")
    confidence: float | None
    tracker_confidence: float | None
    bbox: BBox


class TrackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    camera_id: str
    runtime_session: str
    generation: int
    track_id: str
    first_seen_at: datetime
    last_seen_at: datetime
    max_confidence: float | None


class TracksOut(BaseModel):
    camera_id: str
    runtime_session: str | None
    generation: int
    timestamp: datetime | None
    bbox_width: int | None
    bbox_height: int | None
    current: list[LiveTrackOut]
    recent: list[TrackOut]


Occupancy = Literal["occupied", "empty", "unknown"]


class PolygonPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float = Field(ge=0, le=1, allow_inf_nan=False, strict=True)
    y: float = Field(ge=0, le=1, allow_inf_nan=False, strict=True)


def validate_polygon(points):
    """Accept either winding of a simple polygon, including concave polygons."""
    vertices = [(p.x, p.y) for p in points]
    if len(set(vertices)) != len(vertices):
        raise ValueError("Polygon vertices must be distinct; omit the closing vertex")

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, p):
        return (cross(a, b, p) == 0 and min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))

    def intersects(a, b, c, d):
        ca, cb, cc, cd = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
        return ((ca * cb < 0 and cc * cd < 0) or on_segment(a, b, c)
                or on_segment(a, b, d) or on_segment(c, d, a) or on_segment(c, d, b))

    n = len(vertices)
    area = sum(vertices[i][0] * vertices[(i + 1) % n][1]
               - vertices[(i + 1) % n][0] * vertices[i][1] for i in range(n))
    if abs(area) <= 1e-12:
        raise ValueError("Polygon must have positive area")
    for i in range(n):
        a, b = vertices[i], vertices[(i + 1) % n]
        c = vertices[(i + 2) % n]
        if on_segment(a, b, c) or on_segment(b, c, a):
            raise ValueError("Polygon edges must not backtrack")
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if intersects(a, b, vertices[j], vertices[(j + 1) % n]):
                raise ValueError("Polygon must not self-intersect")
    return points


class ParkingSpaceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=128)
    polygon: list[PolygonPoint] = Field(min_length=3, max_length=64)

    @field_validator("polygon")
    @classmethod
    def simple_polygon(cls, value):
        return validate_polygon(value)


class ParkingSpaceUpdate(ParkingSpaceCreate):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    polygon: list[PolygonPoint] | None = Field(default=None, min_length=3, max_length=64)

    @field_validator("name", "polygon", mode="before")
    @classmethod
    def reject_explicit_null(cls, value):
        if value is None:
            raise ValueError("Null is not an update value")
        return value


class ParkingSpaceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    space_id: str
    camera_id: str
    name: str
    polygon: list[PolygonPoint]
    occupancy: Occupancy
    revision: int
    occupancy_observed_at: datetime | None
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at", "occupancy_observed_at")
    @classmethod
    def utc_dates(cls, value):
        # Match PostgreSQL UTC output when CPU tests use SQLite naive timestamps.
        return value.replace(tzinfo=timezone.utc) if value is not None and value.tzinfo is None else value


class ParkingSummary(BaseModel):
    camera_id: str
    total: int
    occupied: int
    empty: int
    unknown: int
