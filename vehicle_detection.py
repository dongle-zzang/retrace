"""Parking-camera full-frame vehicle ROIs for one shared detector; no GI/PyDS imports."""
import configparser
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np

from pgie_cache import prepare_engine_cache

VEHICLE_GIE_ID = 2
VEHICLE_SEG_GIE_ID = 3
VEHICLE_ROI_ID = 90
SEG_ROI_CLASS = 1  # second ROI on the frames the seg nvinfer should infer (it operates on this class only)
UNTRACKED_OBJECT_ID = (1 << 64) - 1
VEHICLE_TRACKER_CLASS = 100
# Labels tracked as one vehicle class. car is required; truck/bus are optional
# (TrafficCamNet has only car, COCO YOLO models have all three).
VEHICLE_LABELS = ("car", "truck", "bus")
# Mask cells per det box: a one-cell ring is ~2.7 px wide in the 1280-wide JPEG.
OUTLINE_CELL = 4
OUTLINE_MAX_POINTS = 32
MASK_MATCH_IOU = .5
MASK_MIN_FILL = .15  # less of its box than this: the mask is unreliable
# Seg runs on one parking camera per batch, each camera every VEHICLE_SEG_INTERVAL seconds; tracks reuse
# their last ring in between (2026-11-06: seg on every frame saturated the GIL and slowed every camera).
SEG_INTERVAL_DEFAULT, SEG_INTERVAL_RANGE = 2.0, (.2, 60.0)
RING_CELLS_DEFAULT, RING_CELLS_RANGE = 2, (1, 4)
OUTLINE_REUSE_INTERVALS = 3  # a cached ring older than this many intervals is dropped
# Automatic protection: seg Python work above this for SEG_GUARD_WINDOWS consecutive windows stops seg ROIs.
SEG_GUARD_MS_PER_S, SEG_GUARD_WINDOW, SEG_GUARD_WINDOWS = 200.0, 5.0, 3
SEG_RETRY_AFTER, SEG_RETRY_MAX = 600.0, 3600.0  # retry delay doubles per trip, capped


@dataclass(frozen=True)
class VehicleDetector:
    cameras: frozenset[str]
    config_path: str
    car_class_id: int
    vehicle_class_ids: frozenset[int] | None = None

    def __post_init__(self):
        if self.vehicle_class_ids is None:
            object.__setattr__(self, "vehicle_class_ids", frozenset({self.car_class_id}))


@dataclass(frozen=True)
class VehicleSegmenter:
    cameras: frozenset[str]
    config_path: str
    vehicle_class_ids: frozenset[int]
    det_weights_sha256: str | None = None  # det best.pt the seg box branch was grafted from
    box_width: int = 0  # det box border for vehicles that got an outline
    interval: float = SEG_INTERVAL_DEFAULT  # seconds between seg inferences of one camera
    ring_cells: int = RING_CELLS_DEFAULT  # outline thickness in OUTLINE_CELL cells


def prepare_vehicle_detector(cameras, cache_dir, environ=None):
    env = os.environ if environ is None else environ
    requested = frozenset(v.strip() for v in env.get("VEHICLE_CAMERA_IDS", "").split(",") if v.strip())
    # A parking camera removed or disabled in the camera YAML is skipped, not fatal,
    # so the remaining parking cameras keep vehicle detection. IDs are public.
    selected = requested & {c.camera_id for c in cameras}
    for camera_id in sorted(requested - selected):
        print(f"Vehicle camera skipped: camera_id={camera_id} is not an enabled runtime camera", flush=True)
    if not selected:
        return None
    path = Path(env.get("VEHICLE_INFER_CONFIG", ""))
    if not path.is_file():
        raise ValueError("Vehicle detector requires a local native nvinfer model config")
    # Detector threshold: explicit environment override, else the model config's
    # class/all value. Backend min_confidence stays separate, so weaker
    # candidates reach occupancy as uncertain evidence instead of vanishing.
    generated, car, vehicle_ids = _generate_config(path, cache_dir, "vehicle", len(selected), VEHICLE_GIE_ID, "0",
                                                   threshold=env.get("VEHICLE_MIN_CONFIDENCE"))
    return VehicleDetector(selected, generated, car, frozenset(vehicle_ids))


def prepare_vehicle_segmenter(detector, cache_dir, environ=None):
    """Optional instance-seg nvinfer on the detector's ROIs; any failure only disables outlines."""
    env = os.environ if environ is None else environ
    raw = env.get("VEHICLE_SEG_INFER_CONFIG", "").strip()
    if not raw:
        return None
    try:
        path = Path(raw)
        if not path.is_file():
            raise ValueError("Vehicle segmentation config missing")
        # DS 7.0 instance masks require cluster-mode=4; the YOLO-seg ONNX does its own NMS.
        generated, _, vehicle_ids = _generate_config(
            path, cache_dir, "vehicle-seg", len(detector.cameras), VEHICLE_SEG_GIE_ID, "3",
            {"output-instance-mask": "1", "cluster-mode": "4", "operate-on-class-ids": str(SEG_ROI_CLASS)},
            required=("parse-bbox-instance-mask-func-name", "custom-lib-path"))
        interval, ring_cells = outline_settings(env)
        return VehicleSegmenter(detector.cameras, generated, frozenset(vehicle_ids),
                                _manifest_value(path.parent, "det_weights_sha256"), outline_box_width(env),
                                interval, ring_cells)
    except Exception:
        print("Vehicle segmentation unavailable; details hidden; outlines disabled", flush=True)
        return None


def outline_box_width(environ):
    raw = environ.get("VEHICLE_OUTLINE_BOX_WIDTH", "").strip()
    if not raw:
        return 0
    if raw in ("0", "1", "2", "3"):
        return int(raw)
    print("VEHICLE_OUTLINE_BOX_WIDTH must be 0-3; using 0", flush=True)
    return 0


def outline_settings(environ):
    """(VEHICLE_SEG_INTERVAL seconds, VEHICLE_OUTLINE_RING_CELLS); display/rate settings fall back, never fail."""
    interval = SEG_INTERVAL_DEFAULT
    raw = environ.get("VEHICLE_SEG_INTERVAL", "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = math.nan
        if SEG_INTERVAL_RANGE[0] <= value <= SEG_INTERVAL_RANGE[1]:
            interval = value
        else:
            print(f"VEHICLE_SEG_INTERVAL must be {SEG_INTERVAL_RANGE[0]}-{SEG_INTERVAL_RANGE[1]:g} seconds; "
                  f"using {SEG_INTERVAL_DEFAULT}", flush=True)
    cells = RING_CELLS_DEFAULT
    raw = environ.get("VEHICLE_OUTLINE_RING_CELLS", "").strip()
    if raw:
        if raw.isdigit() and RING_CELLS_RANGE[0] <= int(raw) <= RING_CELLS_RANGE[1]:
            cells = int(raw)
        else:
            print(f"VEHICLE_OUTLINE_RING_CELLS must be {RING_CELLS_RANGE[0]}-{RING_CELLS_RANGE[1]}; "
                  f"using {RING_CELLS_DEFAULT}", flush=True)
    return interval, cells


def graft_status(segmenter, det_config):
    """Is the seg box branch grafted from the active detector's weights? None when seg is off."""
    if segmenter is None:
        return None
    active = _manifest_value(Path(det_config).parent, "weights_sha256")
    if not segmenter.det_weights_sha256 or not active:
        return "unknown"
    return "match" if active == segmenter.det_weights_sha256 else "stale"


def _manifest_value(directory, key):
    try:
        value = json.loads((Path(directory) / "manifest.json").read_text()).get(key)
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) and value else None


def _generate_config(path, cache_dir, name, batch, gie_id, network_type, extra=None, required=(), threshold=None):
    """Secondary nvinfer config on the synthetic parking ROIs; returns (generated path, car id, vehicle ids)."""
    config = configparser.ConfigParser(interpolation=None)
    config.read(path)
    props = config["property"]
    if any(not props.get(key, "").strip() for key in required):
        raise ValueError("Vehicle nvinfer config is missing a required key")
    for key in ("onnx-file", "tlt-encoded-model", "model-file", "proto-file", "uff-file",
                "labelfile-path", "custom-lib-path", "mean-file", "int8-calib-file"):
        if props.get(key, "").strip():
            file = (path.parent / props[key].strip()).resolve()
            if not file.is_file():
                raise ValueError("Vehicle model/config dependency missing; no download attempted")
            props[key] = str(file)
    labels = Path(props["labelfile-path"]).read_text().splitlines()
    names = [label.split(";")[0].strip().lower() for label in labels]
    ids = {name: [i for i, value in enumerate(names) if value == name] for name in VEHICLE_LABELS}
    if (len(ids["car"]) != 1 or any(len(v) > 1 for v in ids.values())
            or props.getint("num-detected-classes") != len(labels)):
        raise ValueError("Vehicle labels must identify exactly one car class and match class count")
    car = ids["car"][0]
    vehicle_ids = sorted(v[0] for v in ids.values() if v)
    default = config.get("class-attrs-all", "pre-cluster-threshold", fallback="0.4")
    # Only synthetic full-frame ROIs of selected cameras are inferred, in one batch.
    props.update({"gpu-id": "0", "batch-size": str(batch), "process-mode": "2",
                  "network-type": network_type, "gie-unique-id": str(gie_id),
                  "operate-on-gie-id": str(VEHICLE_ROI_ID), "operate-on-class-ids": "0",
                  "interval": "0", "secondary-reinfer-interval": "0", "classifier-async-mode": "0",
                  "input-object-min-width": "1", "input-object-min-height": "1",
                  "input-object-max-width": "0", "input-object-max-height": "0",
                  "input-tensor-meta": "0", "network-mode": props.get("network-mode", "2"), **(extra or {})})
    props["filter-out-class-ids"] = ";".join(str(i) for i in range(len(labels)) if i not in vehicle_ids)
    for class_id in vehicle_ids:
        section = f"class-attrs-{class_id}"
        if not config.has_section(section):
            config.add_section(section)
        value = float(threshold or config[section].get("pre-cluster-threshold", default))
        if not 0 <= value <= 1:
            raise ValueError("Invalid vehicle confidence threshold")
        config[section]["pre-cluster-threshold"] = str(value)
    directory, digest = prepare_engine_cache(props, Path(cache_dir) / name)
    generated = directory / f"{name.replace('-', '_')}_{digest}.txt"
    with generated.open("w") as target:
        config.write(target, space_around_delimiters=False)
    return str(generated), car, vehicle_ids


def metadata_nodes(node, cast):
    while node is not None:
        value = cast(node.data)
        try:
            node = node.next
        except StopIteration:
            node = None
        yield value


def add_vehicle_rois(batch, detector, camera_for_frame, pyds, outlines=None):
    """Full-frame ROI (class 0) for the detector on every parking frame; with outlines, also the seg ROI
    (class 1) on the frame of the one camera whose seg is due."""
    selected = []
    for frame in metadata_nodes(batch.frame_meta_list, pyds.NvDsFrameMeta.cast):
        camera_id = camera_for_frame(frame).camera_id
        if camera_id in detector.cameras:
            selected.append((frame, camera_id))
    due = outlines.due_cameras([camera_id for _, camera_id in selected]) if outlines is not None else ()
    for frame, camera_id in selected:
        _add_roi(batch, frame, 0, pyds)
        if camera_id in due:
            _add_roi(batch, frame, SEG_ROI_CLASS, pyds)
            outlines.mark_pending(camera_id, frame.frame_num)
        # Secondary detector sets this even for an empty result. Missing vehicle
        # inference is unknown.
        frame.bInferDone = False


def _add_roi(batch, frame, class_id, pyds):
    roi = pyds.nvds_acquire_obj_meta_from_pool(batch)
    roi.unique_component_id, roi.class_id = VEHICLE_ROI_ID, class_id
    roi.object_id, roi.confidence = UNTRACKED_OBJECT_ID, 1.0
    rect = roi.rect_params
    rect.left, rect.top, rect.width, rect.height = 0.0, 0.0, 1920.0, 1080.0
    rect.border_width = 0
    pyds.nvds_add_obj_meta_to_frame(frame, roi, None)


def prepare_vehicle_tracks(batch, detector, camera_for_frame, pyds):
    for frame in metadata_nodes(batch.frame_meta_list, pyds.NvDsFrameMeta.cast):
        selected = camera_for_frame(frame).camera_id in detector.cameras
        remove = []
        for obj in metadata_nodes(frame.obj_meta_list, pyds.NvDsObjectMeta.cast):
            if obj.unique_component_id == VEHICLE_GIE_ID:
                if selected and obj.class_id in detector.vehicle_class_ids:
                    obj.parent = None
                    obj.class_id = VEHICLE_TRACKER_CLASS
                else:
                    remove.append(obj)
            elif obj.unique_component_id in (VEHICLE_ROI_ID, VEHICLE_SEG_GIE_ID):
                remove.append(obj)
        # Children are detached or removed before returning the ROI parent to the meta pool.
        remove.sort(key=lambda obj: obj.unique_component_id == VEHICLE_ROI_ID)
        for obj in remove:
            pyds.nvds_remove_obj_meta_from_frame(frame, obj)


def outline_from_grid(grid, left, top, width, height):
    """Row scan of a box-covering bool mask: outer left/right edges of up to 16 rows, clockwise on screen.

    Gaps between mask pieces on one row are filled; a single polygon cannot hold several pieces.
    """
    shape = outline_shape(grid)
    return None if shape is None else place_outline(shape, left, top, width, height)


def outline_shape(grid):
    """Box-relative (0-1) row-scan outline of grid as (xs, ys) arrays, or None; place_outline puts it on a box."""
    rows = np.flatnonzero(grid.any(1))
    if rows.size < 2:
        return None
    h, w = grid.shape
    rows = rows[np.linspace(0, rows.size - 1, min(OUTLINE_MAX_POINTS // 2, rows.size)).round().astype(int)]
    picked = grid[rows]
    lefts = picked.argmax(1) / w
    rights = (w - picked[:, ::-1].argmax(1)) / w
    ys = (rows + .5) / h
    ys[0], ys[-1] = rows[0] / h, (rows[-1] + 1) / h
    return np.concatenate([rights, lefts[::-1]]), np.concatenate([ys, ys[::-1]])


def place_outline(shape, left, top, width, height):
    xs, ys = shape
    return tuple(zip((left + xs * width).round(1).tolist(), (top + ys * height).round(1).tolist()))


def mask_grid(mask):
    """Bool copy of an NvOSD_MaskParams buffer; the PyDS view must not outlive the probe."""
    if not mask.size:
        return None
    return mask.get_mask_array().reshape(mask.height, mask.width) > mask.threshold


def resample_mask(seg, seg_rect, det_rect):
    """Nearest seg-mask cell for each OUTLINE_CELL cell of the det box; cells outside the seg box stay empty."""
    seg_left, seg_top, seg_width, seg_height = seg_rect
    left, top, width, height = det_rect
    h, w = seg.shape
    columns, rows = max(1, math.ceil(width / OUTLINE_CELL)), max(1, math.ceil(height / OUTLINE_CELL))
    u = np.floor((left - seg_left + (np.arange(columns) + .5) * (width / columns)) * (w / seg_width)).astype(np.intp)
    v = np.floor((top - seg_top + (np.arange(rows) + .5) * (height / rows)) * (h / seg_height)).astype(np.intp)
    grid = seg[np.clip(v, 0, h - 1)][:, np.clip(u, 0, w - 1)]
    grid[(v < 0) | (v >= h)] = False
    grid[:, (u < 0) | (u >= w)] = False
    return grid


def ring_mask(grid, cells=1):
    """Border cells only, `cells` thick (OSD draws them as a line); the outer outline stays the same."""
    inner = grid
    for _ in range(cells):
        padded = np.pad(inner, 1)
        inner = inner & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return grid & ~inner


def vehicle_outline(obj):
    """Outline of the mask attached to obj, stretched over its current (tracker) box; None on any failure."""
    try:
        grid = mask_grid(obj.mask_params)
        rect = obj.rect_params
        return None if grid is None else outline_from_grid(
            grid, float(rect.left), float(rect.top), float(rect.width), float(rect.height))
    except Exception:
        return None


def attach_vehicle_masks(batch, detector, segmenter, camera_for_frame, pyds, due=None, stats=None,
                         timer=time.perf_counter):
    """Seg nvinfer src probe: give each det vehicle the ring of its best-IoU seg mask, on the det box.

    due(camera_id, frame) limits the work to frames that got a seg ROI. stats, when given, collects the
    time split (scan / mask_read / mask_write seconds) and counts for the vehicle_masks diagnostics.
    Returns the number of skipped failures and never raises: a probe exception would stop the
    whole shared pipeline, while a missing mask only leaves that vehicle without an outline.
    """
    errors = 0
    cells = getattr(segmenter, "ring_cells", 1)
    for frame in metadata_nodes(batch.frame_meta_list, pyds.NvDsFrameMeta.cast):
        started = timer()
        try:
            camera_id = camera_for_frame(frame).camera_id
            if camera_id not in segmenter.cameras or (due is not None and not due(camera_id, frame)):
                continue
            dets, segs = [], []
            for obj in metadata_nodes(frame.obj_meta_list, pyds.NvDsObjectMeta.cast):
                component = obj.unique_component_id
                if component == VEHICLE_GIE_ID and obj.class_id in detector.vehicle_class_ids:
                    dets.append((obj, _rect(obj)))
                elif (component == VEHICLE_SEG_GIE_ID and obj.class_id in segmenter.vehicle_class_ids
                        and obj.mask_params.size):
                    segs.append((obj, _rect(obj)))
            pairs = sorted(((_iou(d[1], s[1]), i, j) for i, d in enumerate(dets) for j, s in enumerate(segs)),
                           reverse=True) if dets and segs else ()
        except Exception:
            errors += 1
            continue
        finally:
            scanned = timer()
            if stats is not None:
                stats["scan"] += scanned - started
        if stats is not None:
            stats["seg_objects"] += len(segs)
        used_dets, used_segs = set(), set()
        for iou, i, j in pairs:
            if iou < MASK_MATCH_IOU:
                break
            if i in used_dets or j in used_segs:
                continue
            used_dets.add(i)
            used_segs.add(j)
            read = timer()
            try:
                (det, det_rect), (seg, seg_rect) = dets[i], segs[j]
                params = seg.mask_params
                if stats is not None:
                    stats["paired"] += 1
                    stats["seg_mask_floats"] += params.width * params.height
                grid = resample_mask(mask_grid(params), seg_rect, det_rect)
                written = timer()
                if stats is not None:
                    stats["mask_read"] += written - read
                if grid.mean() < MASK_MIN_FILL:
                    continue
                mask = det.mask_params
                mask.height, mask.width = grid.shape
                mask.threshold = .5
                mask.alloc_mask_array()[:] = ring_mask(grid, cells).ravel()
                if stats is not None:
                    stats["attached"] += 1
                    stats["mask_write"] += timer() - written
            except Exception:
                errors += 1
    return errors


class OutlineRuntime:
    """Seg rate limit (A), per-track outline reuse, automatic protection (E) and the diagnostics split (0).

    Called from four threads: det sink (due_cameras/mark_pending), seg src (attach), tracker src
    (track_outline) and the 5 s stats printer (snapshot). Counters and the guard share one lock; the
    track cache is only touched by the tracker thread.
    """
    TIMES = ("scan", "mask_read", "mask_write", "outline", "reattach")
    COUNTS = ("seg_frames", "seg_objects", "seg_mask_floats", "paired", "attached", "reattached", "errors")

    def __init__(self, segmenter, detector, clock=time.monotonic, timer=time.perf_counter, log=None):
        self.segmenter, self.detector = segmenter, detector
        self.clock, self.timer = clock, timer
        self.log = log or (lambda line: print(line, flush=True))
        self.interval = float(getattr(segmenter, "interval", SEG_INTERVAL_DEFAULT))
        self.max_age = OUTLINE_REUSE_INTERVALS * self.interval
        self.lock = threading.Lock()
        now = clock()
        self.last_seg = {}
        self.pending = {}
        self.cache = {}  # (camera_id, track id) -> (ring float32, width, height, shape, time)
        self.cache_cleared = True
        self.next_prune = now
        self.window = self._empty()
        self.window_start = now
        self.guard_start, self.guard_seconds, self.strikes = now, 0.0, 0
        self.suspended_until, self.trips, self.retry_after = None, 0, SEG_RETRY_AFTER

    def _empty(self):
        return {**{key: 0.0 for key in self.TIMES}, **{key: 0 for key in self.COUNTS}, "attach_max": 0.0}

    @property
    def suspended(self):
        return self.suspended_until is not None

    def record_cost(self, seconds):
        """Seg Python work in seconds; the guard compares it per window with SEG_GUARD_MS_PER_S."""
        with self.lock:
            self.guard_seconds += seconds
            self._roll(self.clock())

    def _roll(self, now):
        # Lock held. Resume when the retry time came, else close the guard window if it ended.
        if self.suspended_until is not None and now >= self.suspended_until:
            self.suspended_until, self.strikes = None, 0
            self.guard_start, self.guard_seconds = now, 0.0
            self.log("vehicle_segmentation_resumed " + json.dumps({"trips": self.trips}, separators=(",", ":")))
        elapsed = now - self.guard_start
        if elapsed < SEG_GUARD_WINDOW:
            return
        rate = self.guard_seconds * 1000 / elapsed
        self.guard_start, self.guard_seconds = now, 0.0
        if self.suspended_until is not None:
            return
        self.strikes = self.strikes + 1 if rate > SEG_GUARD_MS_PER_S else 0
        if self.strikes >= SEG_GUARD_WINDOWS:
            self.trips += 1
            retry = self.retry_after
            self.retry_after = min(self.retry_after * 2, SEG_RETRY_MAX)
            self.suspended_until, self.strikes = now + retry, 0
            self.pending.clear()
            self.log("vehicle_segmentation_suspended " + json.dumps(
                {"ms_per_s": round(rate, 1), "windows": SEG_GUARD_WINDOWS, "trips": self.trips,
                 "retry_after_s": int(retry)}, separators=(",", ":")))

    def due_cameras(self, camera_ids):
        """At most one camera per batch, the most overdue one, so seg work is spread over batches."""
        now = self.clock()
        with self.lock:
            self._roll(now)
            if self.suspended_until is not None:
                return ()
            due = [(self.last_seg.get(c, -math.inf), c) for c in camera_ids
                   if now - self.last_seg.get(c, -math.inf) >= self.interval]
            if not due:
                return ()
            camera_id = min(due)[1]
            self.last_seg[camera_id] = now
            self.window["seg_frames"] += 1
            return (camera_id,)

    def mark_pending(self, camera_id, frame_num):
        with self.lock:
            self.pending[camera_id] = frame_num

    def _take_pending(self, camera_id, frame):
        with self.lock:
            if self.pending.get(camera_id) != frame.frame_num:
                return False
            del self.pending[camera_id]
            return True

    def attach(self, batch, camera_for_frame, pyds):
        started = self.timer()
        stats = {key: 0 for key in ("seg_objects", "seg_mask_floats", "paired", "attached")}
        stats.update(scan=0.0, mask_read=0.0, mask_write=0.0)
        errors = attach_vehicle_masks(batch, self.detector, self.segmenter, camera_for_frame, pyds,
                                      self._take_pending, stats, self.timer)
        seconds = self.timer() - started
        with self.lock:
            for key, value in stats.items():
                self.window[key] += value
            self.window["errors"] += errors
            self.window["attach_max"] = max(self.window["attach_max"], seconds)
        self.record_cost(seconds)
        return errors

    def track_outline(self, obj, camera_id):
        """Tracker src: outline on obj's current box; caches seg-frame rings per track and re-attaches them."""
        if self.suspended_until is not None:
            if not self.cache_cleared:
                self.cache.clear()
                self.cache_cleared = True
            return None
        started = self.timer()
        now = self.clock()
        kind, outline = "outline", None
        try:
            key = (camera_id, obj.object_id)
            mask, rect = obj.mask_params, obj.rect_params
            box = float(rect.left), float(rect.top), float(rect.width), float(rect.height)
            if mask.size:
                grid = mask_grid(mask)
                shape = outline_shape(grid)
                if shape is not None:
                    outline = place_outline(shape, *box)
                    if obj.object_id != UNTRACKED_OBJECT_ID:
                        self.cache[key] = (grid.astype(np.float32).ravel(), grid.shape[1], grid.shape[0], shape, now)
                        self.cache_cleared = False
            else:
                entry = self.cache.get(key)
                if entry is not None and now - entry[4] > self.max_age:
                    del self.cache[key]
                elif entry is not None:
                    kind = "reattach"
                    ring, width, height, shape, _ = entry
                    mask.width, mask.height, mask.threshold = width, height, .5
                    mask.alloc_mask_array()[:] = ring
                    outline = place_outline(shape, *box)
            if now >= self.next_prune:
                self.next_prune = now + 1.0
                for stale in [k for k, entry in self.cache.items() if now - entry[4] > self.max_age]:
                    del self.cache[stale]
        except Exception:
            outline = None
            with self.lock:
                self.window["errors"] += 1
        seconds = self.timer() - started
        with self.lock:
            self.window[kind] += seconds
            if kind == "reattach" and outline is not None:
                self.window["reattached"] += 1
        self.record_cost(seconds)
        return outline

    def snapshot(self):
        """Diagnostics for one report window (then reset): rates in ms per wall second."""
        now = self.clock()
        with self.lock:
            self._roll(now)
            window, elapsed = self.window, max(now - self.window_start, 1e-9)
            self.window, self.window_start = self._empty(), now
            state = "suspended" if self.suspended_until is not None else "active"
        rates = {f"{key}_ms_per_s": round(window[key] * 1000 / elapsed, 2) for key in self.TIMES}
        return {"state": state, "interval_s": self.interval, "trips": self.trips,
                **{key: window[key] for key in self.COUNTS}, "cache_tracks": len(self.cache), **rates,
                "probe_ms_per_s": round(sum(window[key] for key in self.TIMES) * 1000 / elapsed, 2),
                "attach_max_ms": round(window["attach_max"] * 1000, 2)}


def _rect(obj):
    rect = obj.rect_params
    return float(rect.left), float(rect.top), float(rect.width), float(rect.height)


def _iou(a, b):
    width = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    height = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)
