"""Parking-camera full-frame vehicle ROIs for one shared detector; no GI/PyDS imports."""
import configparser
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import numpy as np

from pgie_cache import prepare_engine_cache

VEHICLE_GIE_ID = 2
VEHICLE_SEG_GIE_ID = 3
VEHICLE_ROI_ID = 90
VEHICLE_TRACKER_CLASS = 100
# Labels tracked as one vehicle class. car is required; truck/bus are optional
# (TrafficCamNet has only car, COCO YOLO models have all three).
VEHICLE_LABELS = ("car", "truck", "bus")
# Mask cells per det box: a one-cell ring is ~2.7 px wide in the 1280-wide JPEG.
OUTLINE_CELL = 4
OUTLINE_MAX_POINTS = 32
MASK_MATCH_IOU = .5
MASK_MIN_FILL = .15  # less of its box than this: the mask is unreliable


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
    box_width: int = 1  # det box border for vehicles that got an outline


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
            {"output-instance-mask": "1", "cluster-mode": "4"},
            required=("parse-bbox-instance-mask-func-name", "custom-lib-path"))
        return VehicleSegmenter(detector.cameras, generated, frozenset(vehicle_ids),
                                _manifest_value(path.parent, "det_weights_sha256"), outline_box_width(env))
    except Exception:
        print("Vehicle segmentation unavailable; details hidden; outlines disabled", flush=True)
        return None


def outline_box_width(environ):
    raw = environ.get("VEHICLE_OUTLINE_BOX_WIDTH", "").strip()
    if not raw:
        return 1
    if raw in ("0", "1", "2", "3"):
        return int(raw)
    print("VEHICLE_OUTLINE_BOX_WIDTH must be 0-3; using 1", flush=True)
    return 1


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


def add_vehicle_rois(batch, detector, camera_for_frame, pyds):
    for frame in metadata_nodes(batch.frame_meta_list, pyds.NvDsFrameMeta.cast):
        if camera_for_frame(frame).camera_id not in detector.cameras:
            continue
        roi = pyds.nvds_acquire_obj_meta_from_pool(batch)
        roi.unique_component_id, roi.class_id = VEHICLE_ROI_ID, 0
        roi.object_id, roi.confidence = (1 << 64) - 1, 1.0
        rect = roi.rect_params
        rect.left, rect.top, rect.width, rect.height = 0.0, 0.0, 1920.0, 1080.0
        rect.border_width = 0
        pyds.nvds_add_obj_meta_to_frame(frame, roi, None)
        # Secondary detector sets this even for an empty result. Missing vehicle
        # inference is unknown.
        frame.bInferDone = False


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
    rows = np.flatnonzero(grid.any(1))
    if rows.size < 2:
        return None
    h, w = grid.shape
    rows = rows[np.linspace(0, rows.size - 1, min(OUTLINE_MAX_POINTS // 2, rows.size)).round().astype(int)]
    picked = grid[rows]
    sx, sy = width / w, height / h
    lefts = (left + picked.argmax(1) * sx).round(1).tolist()
    rights = (left + (w - picked[:, ::-1].argmax(1)) * sx).round(1).tolist()
    ys = top + (rows + .5) * sy
    ys[0], ys[-1] = top + rows[0] * sy, top + (rows[-1] + 1) * sy
    ys = ys.round(1).tolist()
    return tuple(zip(rights, ys)) + tuple(zip(lefts[::-1], ys[::-1]))


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


def ring_mask(grid):
    """Border cells only (OSD draws them as a line); the outer outline stays the same."""
    padded = np.pad(grid, 1)
    return grid & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])


def vehicle_outline(obj):
    """Outline of the mask attached to obj, stretched over its current (tracker) box; None on any failure."""
    try:
        grid = mask_grid(obj.mask_params)
        rect = obj.rect_params
        return None if grid is None else outline_from_grid(
            grid, float(rect.left), float(rect.top), float(rect.width), float(rect.height))
    except Exception:
        return None


def attach_vehicle_masks(batch, detector, segmenter, camera_for_frame, pyds):
    """Seg nvinfer src probe: give each det vehicle the ring of its best-IoU seg mask, on the det box.

    Returns the number of skipped failures and never raises: a probe exception would stop the
    whole shared pipeline, while a missing mask only leaves that vehicle without an outline.
    """
    errors = 0
    for frame in metadata_nodes(batch.frame_meta_list, pyds.NvDsFrameMeta.cast):
        try:
            if camera_for_frame(frame).camera_id not in segmenter.cameras:
                continue
            dets, segs = [], []
            for obj in metadata_nodes(frame.obj_meta_list, pyds.NvDsObjectMeta.cast):
                if obj.unique_component_id == VEHICLE_GIE_ID and obj.class_id in detector.vehicle_class_ids:
                    dets.append((obj, _rect(obj)))
                elif (obj.unique_component_id == VEHICLE_SEG_GIE_ID and obj.class_id in segmenter.vehicle_class_ids
                        and obj.mask_params.size):
                    segs.append((obj, _rect(obj)))
            pairs = sorted(((_iou(d[1], s[1]), i, j) for i, d in enumerate(dets) for j, s in enumerate(segs)),
                           reverse=True)
        except Exception:
            errors += 1
            continue
        used_dets, used_segs = set(), set()
        for iou, i, j in pairs:
            if iou < MASK_MATCH_IOU:
                break
            if i in used_dets or j in used_segs:
                continue
            used_dets.add(i)
            used_segs.add(j)
            try:
                (det, det_rect), (seg, seg_rect) = dets[i], segs[j]
                grid = resample_mask(mask_grid(seg.mask_params), seg_rect, det_rect)
                if grid.mean() < MASK_MIN_FILL:
                    continue
                mask = det.mask_params
                mask.height, mask.width = grid.shape
                mask.threshold = .5
                mask.alloc_mask_array()[:] = ring_mask(grid).ravel()
            except Exception:
                errors += 1
    return errors


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
