#!/usr/bin/env python3
"""Batched DeepStream parking-camera vehicle detection/tracking with MJPEG preview over HTTP and WebSocket."""

import argparse
import contextlib
import faulthandler
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import uuid

PROCESS_ENTRY = time.monotonic()
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

gst_debug = "0"
sys.dont_write_bytecode = True

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtsp", "1.0")
from gi.repository import GLib, Gst, GstRtsp
import pyds
from app import create_source_bin
from pgie_cache import configure_pgie
from camera_config import CameraConfigError, load_cameras
from camera_runtime import CameraRuntime, CameraRuntimeManager
from person_metadata import FrameMetadata, MetadataStore, vehicle_from_object, utc_timestamp
from vehicle_detection import (prepare_vehicle_detector, add_vehicle_rois, prepare_vehicle_tracks,
                               prepare_vehicle_segmenter, attach_vehicle_masks, vehicle_outline, graft_status,
                               VEHICLE_GIE_ID, VEHICLE_TRACKER_CLASS)
from preview_diagnostics import InferenceDiagnostics, log_safe_traceback, log_settings
from startup_timing import StartupTiming, startup_origin, source_observer, bus_observer
from preview_timing import FrameTiming
from preview_mjpeg import FrameStore, serve_mjpeg
from preview_capture import RawCapture, CaptureBusy, CaptureUnavailable, add_capture_branch
from preview_web import FrontendFiles, BackendAPIProxy
from preview_socket import RealtimeHub
from parking_relay import ParkingRelay
from rtsp_diagnostics import classify_bus_error, safe_factory, source_failure_detail

# app.py sets its own debug default at import; retain preview's requested level.
os.environ["GST_DEBUG"] = gst_debug

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DEEPSTREAM_ROOT = Path("/opt/nvidia/deepstream/deepstream")
TRACKER_CONFIG = DEEPSTREAM_ROOT / "samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
TRACKER_LIBRARY = DEEPSTREAM_ROOT / "lib/libnvds_nvmultiobjecttracker.so"
# A recovery worker still running after HUNG_AFTER s gets one flush_start kick, then KICK_WAIT s before abandon.
RECOVERY_HUNG_AFTER = 30
RECOVERY_KICK_WAIT = 10
RECOVERY_JOIN_TIMEOUT = 5


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, "Invalid arguments; values hidden. Use --help.\n")


def parse_args():
    parser = SafeParser(description=__doc__)
    parser.add_argument("--input", action="append", metavar="RTSP_URI",
                        help="Compatibility input; repeat for each camera (overrides camera config)")
    parser.add_argument("--cameras", type=Path, default=ROOT / "configs/cameras.yaml")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--port", type=int, default=os.environ.get("WEB_PORT", "40225"),
                        help="HTTP listen port (default: WEB_PORT environment variable or 40225)")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--rtsp-latency", type=int, default=1000, metavar="MS",
        help="RTSP jitter-buffer latency in milliseconds (default: 1000)",
    )
    parser.add_argument(
        "--rtsp-drop-on-latency", action=argparse.BooleanOptionalAction, default=True,
        help="Discard RTSP jitter-buffer frames beyond latency; default: enabled",
    )
    parser.add_argument(
        "--source-recovery", action=argparse.BooleanOptionalAction, default=True,
        help="Reset a source bin in place after consecutive frameless attempts; default: enabled",
    )
    parser.add_argument(
        "--source-recovery-rebuild", action=argparse.BooleanOptionalAction, default=True,
        help="From the second recovery on, replace the source bin and its mux sink pad; default: enabled",
    )
    parser.add_argument(
        "--mux-live-source", type=int, choices=(0, 1), default=1,
        help="Legacy mux timestamp mode; 0 is an experiment for matching-FPS sources",
    )
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="Log startup milestones and per-stage arrival gaps/PTS every five seconds",
    )
    parser.add_argument("--startup-source-count", type=int, choices=(1, 2, 4, 8),
                        help="Diagnostics experiment: first N enabled cameras in config order")
    parser.add_argument("--camera-degraded-after", type=float, default=3.0, metavar="SECONDS")
    parser.add_argument("--camera-offline-after", type=float, default=10.0, metavar="SECONDS")
    parser.add_argument("--camera-connect-timeout", type=float, default=30.0, metavar="SECONDS")
    parser.add_argument("--camera-min-fps", type=float, default=1.0)
    args = parser.parse_args()
    try:
        CameraRuntime("validation", args.camera_degraded_after, args.camera_offline_after,
                      args.camera_connect_timeout, args.camera_min_fps)
    except ValueError:
        parser.error("Invalid camera health thresholds")
    try:
        args.cameras = load_cameras(args.cameras, args.env_file, args.input)
    except CameraConfigError as error:
        parser.exit(2, f"{error}\n")
    if args.startup_source_count is not None:
        if not args.diagnostics or args.startup_source_count > len(args.cameras):
            parser.error("Source-count experiment requires diagnostics and enough enabled cameras")
        args.cameras = args.cameras[:args.startup_source_count]
    args.input = [camera.rtsp_url for camera in args.cameras]
    if not 1 <= args.port <= 65535:
        parser.error("port")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("JPEG quality must be between 1 and 100")
    if not 0 <= args.rtsp_latency <= (1 << 32) - 1:
        parser.error("RTSP latency must be a non-negative guint in milliseconds")
    return args


class PreviewHandler(BaseHTTPRequestHandler):
    # WebSocket bytes must not be prefetched into the HTTP file buffer.
    rbufsize = 0

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def finish(self):
        # A disconnect may also occur when the base handler flushes on cleanup.
        try:
            super().finish()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/ws':
            hub = getattr(self.server, 'realtime_hub', None)
            if hub is None:
                self.send_error(503)
            else:
                hub.serve(self)
            return
        if path == '/api' or path.startswith('/api/'):
            proxy = getattr(self.server, 'backend_api_proxy', None)
            if proxy is None:
                self.send_error(503)
            else:
                proxy.serve(self)
            return
        if path.startswith('/training-snapshots/') and path.endswith('.jpg'):
            camera_id = path[len('/training-snapshots/'):-4]
            stream = next((s for s in self.server.stream_snapshot() if s['camera_id'] == camera_id), None)
            capture = getattr(self.server, 'raw_captures', {}).get(camera_id)
            if capture is None or stream is None or stream['runtime']['state'] not in ('online', 'degraded') or self.command == 'HEAD':
                self.send_error(503)
                return
            identity = (stream['runtime_session'], stream['runtime']['generation'])
            try:
                jpeg, captured_at = capture.request()
            except CaptureBusy:
                self.send_error(409)
                return
            except CaptureUnavailable:
                self.send_error(503)
                return
            current = next((s for s in self.server.stream_snapshot() if s['camera_id'] == camera_id), None)
            if current is None or current['runtime']['state'] not in ('online', 'degraded') or identity != (current['runtime_session'], current['runtime']['generation']):
                self.send_error(503)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpeg)))
            self.send_header('X-Capture-Clean', 'pre-osd-v1')
            self.send_header('X-Captured-At', captured_at)
            self.end_headers()
            self.wfile.write(jpeg)
            return
        if path.startswith('/snapshots/') and path.endswith('.jpg'):
            camera_id = path[len('/snapshots/'):-4]
            stream = next((s for s in self.server.stream_snapshot() if s['camera_id'] == camera_id), None)
            store = self.server.mjpeg_streams.get(stream['url']) if stream else None
            sequence, jpeg, age = store.snapshot() if store else (0, None, 0)
            if jpeg is None or stream['runtime']['state'] not in ('online', 'degraded'):
                self.send_error(503)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpeg)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Frame-Age', str(age))
            self.send_header('X-Frame-Identity', f"{stream['runtime_session']}:{stream['runtime']['generation']}:{sequence}")
            self.end_headers()
            if getattr(self, 'command', 'GET') != 'HEAD':
                self.wfile.write(jpeg)
            return
        # Without an uploaded build, "/" stays 404; the check page is only at /diagnostics.
        assets = {"/preview.js": ("preview.js", "text/javascript; charset=utf-8"),
                  "/diagnostics": ("index.html", "text/html; charset=utf-8")}
        if path in self.server.mjpeg_streams:
            if getattr(self, 'command', 'GET') == 'HEAD':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True
            else:
                serve_mjpeg(self, self.server.mjpeg_streams[path])
            return
        if path in ("/streams.json", "/metadata.json"):
            if path == "/metadata.json":
                if not hasattr(self.server, "metadata_snapshot"):
                    self.send_error(404)
                    return
                payload = self.server.metadata_snapshot()
            else:
                payload = self.server.stream_snapshot() if hasattr(self.server, "stream_snapshot") else self.server.streams
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if getattr(self, 'command', 'GET') != 'HEAD':
                self.wfile.write(body)
            return
        if path.startswith('/mjpeg/'):
            self.send_error(404)
            return
        frontend = getattr(self.server, 'frontend_files', None)
        if path not in ('/preview.js', '/diagnostics') and frontend is not None and frontend.serve(self, path):
            return
        if path in assets:
            filename, content_type = assets[path]
            try:
                body = (WEB_ROOT / filename).read_bytes()
            except OSError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if getattr(self, 'command', 'GET') != 'HEAD':
                self.wfile.write(body)
            return
        self.send_error(404)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        proxy = getattr(self.server, 'backend_api_proxy', None)
        if proxy is None:
            self.send_error(503)
        else:
            proxy.serve(self)

    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):
        # Do not echo arbitrary request URLs into terminal logs.
        pass

    def log_request(self, code="-", size="-"):
        # Keep real HTTP failures visible without logging request URLs.
        if isinstance(code, int) and code >= 400:
            print(f"HTTP error status={code}", file=sys.stderr, flush=True)


def make_element(factory, name):
    element = Gst.ElementFactory.make(factory, name)
    if element is None:
        raise RuntimeError(f"Required GStreamer element missing: {factory}")
    return element


def create_preview_source_bin(index, uri, latency, fail, drop_on_latency=True, reconnect_interval=10, observer=None):
    """Reuse app.py's NVMM source, with preview-specific transport/buffering."""
    options = {"observer": observer} if observer is not None else {}
    source_bin = create_source_bin(index, uri, fail, source_factory="nvurisrcbin", **options)
    decoder = source_bin.get_by_name(f"uri-decode-bin-{index}")
    if decoder is None:
        raise RuntimeError(f"source={index}: nvurisrcbin not found in source bin")
    # DS 7.0 native source reconnection; mux/demux pads stay registered.
    for name, value in (("select-rtp-protocol", 4), ("latency", latency),
                        ("rtsp-reconnect-interval", reconnect_interval),
                        ("rtsp-reconnect-attempts", -1), ("num-extra-surfaces", 4),
                        ("cudadec-memtype", 0), ("async-handling", True)):
        if decoder.find_property(name) is None:
            raise RuntimeError(f"nvurisrcbin missing required property: {name}")
        decoder.set_property(name, value)
    # low-latency-mode: decoder output is not held back for reordering (~170 ms -> ~3 ms measured
    # 2026-10-20, same fps/p95 on IPPP H.264 cameras; see docs/report_20261020.md section 3).
    for name, value in (("drop-on-latency", drop_on_latency), ("disable-audio", True),
                        ("low-latency-mode", True)):
        if decoder.find_property(name) is not None:
            decoder.set_property(name, value)

    # Native properties above own RTSP/decoder configuration, including reconnects.
    # Child traversal is only needed to observe diagnostics, never to set properties.
    if observer is not None:
        def on_child_added(proxy, child, name):
            observer("child", child)
            if isinstance(child, Gst.ChildProxy):
                child.connect("child-added", on_child_added)

        decoder.connect("child-added", on_child_added)

        def observe_existing(bin_element):
            if not isinstance(bin_element, Gst.Bin):
                return
            iterator = bin_element.iterate_elements()
            while True:
                result, child = iterator.next()
                if result == Gst.IteratorResult.OK:
                    on_child_added(bin_element, child, child.get_name())
                    observe_existing(child)
                elif result == Gst.IteratorResult.RESYNC:
                    iterator.resync()
                else:
                    break

        observe_existing(decoder)
    return source_bin


def make_tracker():
    """Reuse the SDK's NvDCF perf preset for vehicle track IDs; no Re-ID model is added."""
    for path in (TRACKER_CONFIG, TRACKER_LIBRARY):
        if not path.is_file():
            raise FileNotFoundError(f"Required DeepStream tracker file missing: {path}")
    tracker = make_element("nvtracker", "vehicle-tracker")
    tracker.set_property("ll-lib-file", str(TRACKER_LIBRARY))
    tracker.set_property("ll-config-file", str(TRACKER_CONFIG))
    tracker.set_property("gpu-id", 0)
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)
    # Older plugins expose this switch; DS 7.x uses batched tracking by default.
    if tracker.find_property("enable-batch-process") is not None:
        tracker.set_property("enable-batch-process", True)
    # Track IDs stay in metadata for occupancy; the OSD shows the class only.
    tracker.set_property("display-tracking-id", False)
    print(
        f"Tracker=NvDCF config={TRACKER_CONFIG} library={TRACKER_LIBRARY} "
        "gpu=0 size=640x384 batch-processing=enabled", flush=True,
    )
    return tracker


def detach_vehicle(obj, camera_id, timestamp, class_id, outline=None, box_width=1):
    """Copy one tracked vehicle; its OSD box is box_width when it has an outline, else the usual 3.

    The label stays: nvdsosd places text from text_params, not from the box border.
    """
    obj.rect_params.border_width = 3 if outline is None else box_width
    obj.rect_params.border_color.set(1.0, 0.65, 0.0, 1.0)
    # OSD shows the class only; track_id stays in metadata for occupancy.
    obj.text_params.display_text = "Vehicle"
    return vehicle_from_object(camera_id, timestamp, obj, class_id, outline)


def build_output(index, args, store, fail, display_mask=False):
    """Draw the existing OSD result and publish only the latest encoded JPEG.

    display_mask draws vehicle masks (the outline rings from the seg nvinfer) on parking cameras.
    """
    output_bin = Gst.Bin.new(f"mjpeg-source-{index}")
    if output_bin is None:
        raise RuntimeError(f"source={index}: could not create output bin")
    try:
        queue = make_element("queue", f"queue-{index}")
        queue.set_property("max-size-buffers", 1)
        queue.set_property("max-size-bytes", 0)
        queue.set_property("max-size-time", 0)
        queue.set_property("leaky", 2)
        osd_convert = make_element("nvvideoconvert", f"osd-convert-{index}")
        rgba_caps = make_element("capsfilter", f"osd-rgba-caps-{index}")
        rgba_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))
        osd = make_element("nvdsosd", f"osd-{index}")
        osd.set_property("process-mode", 1)
        osd.set_property("display-bbox", True)
        osd.set_property("display-text", True)
        osd.set_property("display-mask", display_mask)
        converter = make_element("nvvideoconvert", f"convert-mjpeg-{index}")
        raw_caps = make_element("capsfilter", f"raw-caps-mjpeg-{index}")
        raw_caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM),format=I420,width=1280,height=720"
        ))
        encoder = make_element("nvjpegenc", f"jpeg-encoder-{index}")
        encoder.set_property("quality", args.jpeg_quality)
        sink = make_element("appsink", f"jpeg-sink-{index}")
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)
        sink.set_property("async", False)  # Offline branches must not hold pipeline preroll.
        sink.set_property("max-buffers", 1)
        sink.set_property("drop", True)
        sink.set_property("wait-on-eos", False)

        def on_sample(appsink):
            try:
                sample = appsink.emit("pull-sample")
                if sample is None:
                    return Gst.FlowReturn.EOS
                buffer = sample.get_buffer()
                mapped, info = buffer.map(Gst.MapFlags.READ)
                if not mapped:
                    raise RuntimeError("Could not map encoded JPEG")
                try:
                    if hasattr(store, "put_frame"):
                        store.put_frame(bytes(info.data), buffer.pts)
                    else:
                        store.put(bytes(info.data))
                finally:
                    buffer.unmap(info)
                return Gst.FlowReturn.OK
            except Exception:
                fail("JPEG output failed; details hidden")
                # A Python JPEG handoff failure must not poison shared upstream flow.
                return Gst.FlowReturn.OK

        sink.connect("new-sample", on_sample)
        capture = getattr(args, 'raw_captures', {}).get(index)
        if capture is not None:
            tee = make_element('tee', f'capture-tee-{index}')
            preview_queue = make_element('queue', f'capture-preview-queue-{index}')
            preview_queue.set_property('max-size-buffers', 1)
            preview_queue.set_property('max-size-bytes', 0)
            preview_queue.set_property('max-size-time', 0)
            preview_queue.set_property('leaky', 2)
            chain = (queue, tee, preview_queue, osd_convert, rgba_caps, osd, converter, raw_caps, encoder, sink)
        else:
            chain = (queue, osd_convert, rgba_caps, osd, converter, raw_caps, encoder, sink)
        for element in chain:
            output_bin.add(element)
        for left, right in zip(chain, chain[1:]):
            if not left.link(right):
                raise RuntimeError(f"Could not link {left.get_name()} -> {right.get_name()}")
        if capture is not None:
            add_capture_branch(output_bin, tee, capture, index, Gst, make_element)
        ghost_pad = Gst.GhostPad.new("sink", queue.get_static_pad("sink"))
        if ghost_pad is None or not output_bin.add_pad(ghost_pad):
            raise RuntimeError("Could not create output sink pad")
        return output_bin
    except Exception:
        output_bin.set_state(Gst.State.NULL)
        raise


def run_shared_pipeline(args, metadata_store, stores, manager, shutdown):
    """One mux/vehicle detector/tracker/demux; source bins own native RTSP reconnection.

    Only parking cameras selected by the vehicle detector get a full-frame ROI, so
    other cameras pass through without inference while sharing RTSP/JPEG output.
    A source bin that stays silent is reset in place from this control loop.
    """
    startup = getattr(args, "startup_timing", None)
    vehicle_detector = getattr(args, "vehicle_detector", None)
    segmenter = getattr(args, "vehicle_segmenter", None) if vehicle_detector is not None else None
    if startup is not None:
        startup.start_gpu_sampling()
    cameras_by_source = manager.cameras_by_slot
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    os.environ["GST_REGISTRY"] = str(cache / "gstreamer-registry.bin")
    Gst.init(None)
    Gst.debug_set_active(False)
    deployment = None
    pipeline = None
    requested_pads = []
    source_bins = {}
    mux_pads = {}
    bus = None
    failed = None
    pipeline_stop = threading.Event()
    setup_stage = "shared-elements"
    counts = [0] * len(args.input)
    vehicle_counts = [None] * len(args.input)
    vehicles_max = [0] * len(args.input)
    previous_counts = [0] * len(args.input)
    counts_lock = threading.Lock()
    previous_time = time.monotonic()
    timings = {}
    inference_stats = ({index: InferenceDiagnostics() for index, camera in cameras_by_source.items()
                        if camera.camera_id in vehicle_detector.cameras}
                       if args.diagnostics and vehicle_detector is not None else {})
    metadata_samples = {}  # At most five vehicles from the latest nonempty frame per report.
    # Seg probe + outline cost and coverage per report window (guarded by counts_lock).
    mask_stats = {"cameras": {}, "seconds": 0.0, "max": 0.0, "errors": 0}

    def record_mask_cost(seconds, errors=0):
        with counts_lock:
            mask_stats["seconds"] += seconds
            mask_stats["max"] = max(mask_stats["max"], seconds)
            mask_stats["errors"] += errors

    def detector_config():
        # The OTA-active detector config sits beside its weights manifest; the original model has none.
        return ROOT / deployment.active["config"] if deployment is not None else vehicle_detector.config_path

    def camera_for_frame(frame):
        return manager.camera_for_frame(int(frame.source_id), int(frame.pad_index))

    def on_inference(_pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        try:
            batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if batch is None:
                raise RuntimeError("Vehicle detector output has no batch metadata")
            frames = batch.frame_meta_list
            while frames is not None:
                frame = pyds.NvDsFrameMeta.cast(frames.data)
                camera_for_frame(frame)
                stats = inference_stats.get(int(frame.pad_index))
                confidences = []
                objects = frame.obj_meta_list if stats is not None else None
                while objects is not None:
                    obj = pyds.NvDsObjectMeta.cast(objects.data)
                    if obj.unique_component_id == VEHICLE_GIE_ID and obj.class_id == VEHICLE_TRACKER_CLASS:
                        confidences.append(float(obj.confidence))
                    try:
                        objects = objects.next
                    except StopIteration:
                        break
                pts = int(frame.buf_pts)
                if stats is not None:
                    stats.observe(bool(frame.bInferDone), confidences,
                                  None if pts == Gst.CLOCK_TIME_NONE else pts)
                try:
                    frames = frames.next
                except StopIteration:
                    break
        except Exception:
            fail("Inference diagnostics failed; details hidden")
        return Gst.PadProbeReturn.OK

    def startup_probe(pad, stage, slot=None, batched=False):
        if startup is None:
            return

        def observe(_pad, info):
            buffer = info.get_buffer()
            if buffer is None:
                return Gst.PadProbeReturn.OK
            startup.mark(stage, slot, None if batched or buffer.pts == Gst.CLOCK_TIME_NONE else buffer.pts)
            if batched:
                try:
                    batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
                    frames = batch.frame_meta_list if batch else None
                    while frames is not None:
                        frame = pyds.NvDsFrameMeta.cast(frames.data)
                        camera_for_frame(frame)
                        index = int(frame.pad_index)
                        startup.mark(stage, index, None if frame.buf_pts == Gst.CLOCK_TIME_NONE else frame.buf_pts)
                        if stage == "first_infer_output" and frame.bInferDone:
                            startup.mark("first_inference_done", index, frame.buf_pts)
                            startup.mark("first_inference_done")
                        try:
                            frames = frames.next
                        except StopIteration:
                            break
                except Exception:
                    startup.mark("batch_observer_unavailable")
                # Only parking cameras are inferred, so completion waits for output only.
                if startup.complete(stage):
                    return Gst.PadProbeReturn.REMOVE
                return Gst.PadProbeReturn.OK
            return Gst.PadProbeReturn.REMOVE

        pad.add_probe(Gst.PadProbeType.BUFFER, observe)

    def startup_report(final=False):
        if startup is None:
            return
        startup.drain()
        summary = startup.summary()
        summary.update(final=final)
        print("startup_summary " + json.dumps(summary, separators=(",", ":")), flush=True)

    def add_timing_probe(pad, stage, source):
        if not args.diagnostics:
            return
        counter = FrameTiming()
        timings[(stage, source)] = counter
        caps_reported = False

        def observe(_pad, info):
            nonlocal caps_reported
            buffer = info.get_buffer()
            if buffer is not None:
                if stage == "decoded" and not caps_reported:
                    caps = _pad.get_current_caps()
                    if caps is not None and caps.get_size() > 0:
                        structure = caps.get_structure(0)
                        ok, numerator, denominator = structure.get_fraction("framerate")
                        print(f"source={cameras_by_source[source].source_id} camera_id={cameras_by_source[source].camera_id} "
                              f"negotiated_source_fps={numerator / denominator if ok and denominator else 'unknown'}",
                              flush=True)
                        caps_reported = True
                counter.observe(None if buffer.pts == Gst.CLOCK_TIME_NONE else int(buffer.pts))
            return Gst.PadProbeReturn.OK

        pad.add_probe(Gst.PadProbeType.BUFFER, observe)

    def stop(reason, error=False, source="all"):
        nonlocal failed
        if source != "all":
            manager.fail_source(int(source), reason)
            return GLib.SOURCE_CONTINUE
        if error and failed is None:
            failed = reason
            for slot in cameras_by_source:
                manager.fail_source(slot, reason)
            print(f"Shared pipeline ERROR code={reason}; details hidden", flush=True)
        pipeline_stop.set()
        return GLib.SOURCE_REMOVE

    def fail(_reason, source="all"):
        stop("rtsp_error" if _reason == "rtsp_error" else "pipeline_error", True, source)

    def source_failed(reason, index):
        stage, code = source_failure_detail(reason)
        manager.record_error_reason(index, code)
        if args.diagnostics:
            camera = cameras_by_source[index]
            print("source_failure " + json.dumps({"camera_id": camera.camera_id, "source_id": camera.source_id,
                  "failure_stage": stage, "reason": code}, separators=(",", ":")), flush=True)
        fail("rtsp_error" if code == "rtsp_error" else "pipeline_error", str(index))

    def report(final=False):
        nonlocal previous_time
        now = time.monotonic()
        elapsed = max(now - previous_time, 1e-9)
        with counts_lock:
            snapshot = counts.copy()
            vehicle_snapshot = vehicle_counts.copy()
            peak_snapshot = vehicles_max.copy()
            samples = dict(metadata_samples)
            metadata_samples.clear()
            vehicles_max[:] = [0] * len(args.input)
        for index, total in enumerate(snapshot):
            latest = vehicle_snapshot[index] if vehicle_snapshot[index] is not None else "NA"
            fps = (total - previous_counts[index]) / elapsed
            print(
                f"{'final' if final else 'stats'} source={cameras_by_source[index].source_id} frames={total} "
                f"camera_id={cameras_by_source[index].camera_id} fps={fps:.2f} vehicles={latest} "
                f"vehicles_max={peak_snapshot[index]}", flush=True,
            )
        if args.diagnostics:
            for index, diagnostics in inference_stats.items():
                print(f"inference source={cameras_by_source[index].source_id} camera_id={cameras_by_source[index].camera_id} "
                      f"{json.dumps(diagnostics.snapshot(), separators=(',', ':'))}", flush=True)
                sample = samples.get(index)
                if sample is not None:
                    print("metadata " + json.dumps(sample.to_dict(), ensure_ascii=False,
                                                   separators=(',', ':'), allow_nan=False), flush=True)
            if segmenter is not None:
                with counts_lock:
                    masks = {**mask_stats, "cameras": dict(mask_stats["cameras"])}
                    mask_stats.update(cameras={}, seconds=0.0, max=0.0, errors=0)
                print("vehicle_masks " + json.dumps({
                    "cameras": {camera: {"vehicles": n, "with_outline": k} for camera, (n, k) in masks["cameras"].items()},
                    "probe_ms_per_s": round(masks["seconds"] * 1000 / elapsed, 2),
                    "probe_max_ms": round(masks["max"] * 1000, 2), "errors": masks["errors"],
                    "graft": graft_status(segmenter, detector_config())}, separators=(",", ":")), flush=True)
        previous_counts[:] = snapshot
        previous_time = now
        for (stage, source), counter in timings.items():
            print(
                f"timing pipeline_source={source} stage={stage} "
                f"{json.dumps(counter.snapshot(), separators=(',', ':'))}", flush=True,
            )
        return GLib.SOURCE_CONTINUE

    def on_batch(pad, info):
        index = "all"
        try:
            buffer = info.get_buffer()
            if buffer is None or pipeline_stop.is_set():
                return Gst.PadProbeReturn.OK
            batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            if batch is None:
                raise RuntimeError("Tracker output has no DeepStream batch metadata")
            frame_list = batch.frame_meta_list
            outline_seconds = 0.0
            while frame_list is not None:
                frame = pyds.NvDsFrameMeta.cast(frame_list.data)
                camera = camera_for_frame(frame)
                index = int(frame.pad_index)
                pts = int(frame.buf_pts)
                generation = manager.frame_generation(index, pts)
                if generation is None:
                    try:
                        frame_list = frame_list.next
                    except StopIteration:
                        break
                    continue
                timestamp = utc_timestamp()
                vehicles = []
                with_outline = 0
                vehicle_enabled = vehicle_detector is not None and camera.camera_id in vehicle_detector.cameras
                object_list = frame.obj_meta_list
                while object_list is not None:
                    obj = pyds.NvDsObjectMeta.cast(object_list.data)
                    # Save the next node before removing any non-vehicle metadata.
                    try:
                        next_object = object_list.next
                    except StopIteration:
                        next_object = None
                    if (vehicle_enabled and obj.unique_component_id == VEHICLE_GIE_ID
                            and obj.class_id == VEHICLE_TRACKER_CLASS):
                        outline = None
                        if segmenter is not None:
                            started = time.perf_counter()
                            outline = vehicle_outline(obj)
                            outline_seconds += time.perf_counter() - started
                            with_outline += outline is not None
                        vehicles.append(detach_vehicle(obj, camera.camera_id, timestamp, vehicle_detector.car_class_id,
                                                       outline, segmenter.box_width if segmenter else 1))
                    else:
                        pyds.nvds_remove_obj_meta_from_frame(frame, obj)
                    object_list = next_object
                pts = int(frame.buf_pts)
                # inference_done/persons stay in the public schema; only the vehicle
                # detector infers now, and person detection is no longer produced.
                inferred = vehicle_enabled and bool(frame.bInferDone)
                detached = FrameMetadata(
                    camera.source_id, camera.camera_id, int(frame.frame_num), timestamp,
                    None if pts == Gst.CLOCK_TIME_NONE else pts, inferred, (),
                    generation=generation, pipeline_source_id=int(frame.source_id),
                    bbox_width=1920, bbox_height=1080,
                    vehicles=tuple(vehicles),
                    vehicle_detection_enabled=vehicle_enabled,
                    vehicle_inference_done=inferred,
                )
                manager.publish(index, pts, lambda: metadata_store.put(detached), generation=generation)
                with counts_lock:
                    if args.diagnostics and vehicles:
                        metadata_samples[index] = FrameMetadata(
                            camera.source_id, camera.camera_id, detached.frame_number, timestamp,
                            detached.pts_ns, inferred, (), generation=generation, pipeline_source_id=int(frame.source_id),
                            vehicles=tuple(vehicles[:5]), vehicle_detection_enabled=True, vehicle_inference_done=inferred,
                        )
                    counts[index] += 1
                    if vehicle_enabled:
                        vehicle_counts[index] = len(vehicles)
                        vehicles_max[index] = max(vehicles_max[index], len(vehicles))
                        if segmenter is not None:
                            seen, outlined = mask_stats["cameras"].get(camera.camera_id, (0, 0))
                            mask_stats["cameras"][camera.camera_id] = (seen + len(vehicles), outlined + with_outline)
                try:
                    frame_list = frame_list.next
                except StopIteration:
                    break
            if segmenter is not None:
                record_mask_cost(outline_seconds)
        except Exception:
            fail("Metadata/OSD preparation failed; details hidden", str(index))
        return Gst.PadProbeReturn.OK

    def message_scope(message):
        element = message.src
        while element is not None:
            name = element.get_name()
            for index in range(len(args.input)):
                if name in (
                    f"source-bin-{index}",
                    f"mjpeg-source-{index}",
                ):
                    return str(index), "pipeline_error" if name == f"mjpeg-source-{index}" else "rtsp_error"
            element = element.get_parent()
        return "all", "pipeline_error"

    def on_message(bus, message):
        source, error_code = message_scope(message)
        if source != "all" and error_code == "rtsp_error" and message.type in (Gst.MessageType.ERROR, Gst.MessageType.WARNING):
            detail = classify_bus_error(message, warning=message.type == Gst.MessageType.WARNING)
            changed = manager.record_error_reason(int(source), detail["reason"])
            if detail["reason"] in ("source_caps_error", "source_link_error", "decoder_error"):
                error_code = "pipeline_error"
            if args.diagnostics and changed:
                camera = cameras_by_source[int(source)]
                print("source_rtsp " + json.dumps({"camera_id": camera.camera_id, "source_id": camera.source_id,
                      **detail}, separators=(",", ":")), flush=True)
        if message.type == Gst.MessageType.ERROR:
            stop(error_code, True, source)
        elif message.type == Gst.MessageType.WARNING:
            if args.diagnostics:
                print(f"pipeline_source={source}: GStreamer WARNING factory={safe_factory(message.src)}; "
                      "details hidden", flush=True)
        elif message.type == Gst.MessageType.EOS:
            # Source EOS is intercepted before the mux. Aggregate EOS is fatal.
            stop("source_eos", True, source)
        elif message.type == Gst.MessageType.ELEMENT:
            structure = message.get_structure()
            if structure is not None and structure.get_name() == "GstRTSPSrcTimeout":
                if source != "all":
                    manager.record_error_reason(int(source), "rtsp_timeout")
                    stop("rtsp_timeout", True, source)
        elif message.type == Gst.MessageType.STATE_CHANGED and message.src == pipeline:
            _old, current, _pending = message.parse_state_changed()
            if current == Gst.State.PLAYING:
                if startup is not None:
                    startup.mark("pipeline_playing_bus_observed")
                print("DeepStream shared pipeline state=PLAYING", flush=True)

    def observe_decoded(_pad, info, index):
        buffer = info.get_buffer()
        if buffer is not None:
            pts = None if buffer.pts == Gst.CLOCK_TIME_NONE else int(buffer.pts)
            if not manager.observe_decoded(index, pts):
                return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    def source_event(_pad, info, index):
        event = info.get_event()
        if event is not None and event.type == Gst.EventType.EOS:
            manager.fail_source(index, "source_eos")
            # A single (or all) offline source must not terminate the shared chain.
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    def demux_event(_pad, info):
        event = info.get_event()
        if event is not None and event.type == Gst.EventType.EOS:
            # Releasing a mux sink pad can end that stream downstream; the JPEG branch must
            # keep accepting buffers from the rebuilt source instead of refusing them after EOS.
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    class SourceAttachError(RuntimeError):
        def __init__(self, stage):
            super().__init__(f"Could not attach source at stage {stage}")
            self.stage = stage

    def attach_source(index):
        """Create source-bin-{index}, add it to the pipeline and link it to a fresh mux sink_{index}.

        Startup and rebuild share this path so the probes, pad bookkeeping and link order
        stay identical. Raises SourceAttachError(stage) for stage request/link; element
        creation errors propagate unchanged so the startup log keeps their type.
        """
        if startup is not None:
            startup.mark("source_create_start", index)
        observer_options = {"observer": source_observer(startup, index, Gst)} if startup is not None else {}
        source_bin = create_preview_source_bin(
            index, args.input[index], args.rtsp_latency, lambda reason, i=index: source_failed(reason, i),
            args.rtsp_drop_on_latency, max(1, int(args.camera_offline_after)), **observer_options,
        )
        if startup is not None:
            startup.mark("source_created", index)
        startup_probe(source_bin.get_static_pad("src"), "first_source_output", index)
        source_bin.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, observe_decoded, index)
        source_bin.get_static_pad("src").add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, source_event, index)
        add_timing_probe(source_bin.get_static_pad("src"), "decoded", index)
        pipeline.add(source_bin)
        source_bins[index] = source_bin
        mux_pad = mux.request_pad_simple(f"sink_{index}")
        if mux_pad is None:
            raise SourceAttachError("request")
        requested_pads.append((mux, mux_pad))
        mux_pads[index] = mux_pad
        startup_probe(mux_pad, "first_mux_input", index)
        if source_bin.get_static_pad("src").link(mux_pad) != Gst.PadLinkReturn.OK:
            raise SourceAttachError("link")

    def reset_source(index):
        """Reset one silent source bin in place (deepstream-app reset_source_pipeline pattern).

        The mux sink pad, the outer ghost pad and its probes stay; the bin's elements go
        NULL -> PLAYING so nvurisrcbin rebuilds RTSP and decoder. Internals are not touched.
        """
        camera = cameras_by_source[index]
        source_bin = source_bins[index]

        def failed(stage):
            manager.recovery_failed(index)
            print("source_reset_failed " + json.dumps({"camera_id": camera.camera_id, "source_id": camera.source_id,
                  "stage": stage}, separators=(",", ":")), flush=True)

        if source_bin.set_state(Gst.State.NULL) == Gst.StateChangeReturn.FAILURE:
            failed("null")
            return
        # runtime_source_add_delete order; also leaves the flushing a hung-recovery kick started.
        mux_pads[index].send_event(Gst.Event.new_flush_stop(False))
        ghost_pad = source_bin.get_static_pad("src")
        target = ghost_pad.get_target()
        if target is not None and target.get_parent() is None:
            # The plugin dropped its pad without pad-removed; pad-added links the new one.
            # A pad that survives NULL keeps the existing ghost link (GStreamer cannot re-add it).
            ghost_pad.set_target(None)
        if not source_bin.sync_state_with_parent():
            failed("sync")
            return
        generation = manager.recovered(index)
        print("source_reset " + json.dumps({"camera_id": camera.camera_id, "source_id": camera.source_id,
              "generation": generation,
              "recovery_count": manager.runtimes[camera.source_id].snapshot().recovery_count},
              separators=(",", ":")), flush=True)

    def detach_source(index):
        """Drop whatever of source-bin-{index} and mux sink_{index} is attached; the slot stays empty."""
        source_bin, mux_pad = source_bins.get(index), mux_pads.get(index)
        if mux_pad is not None:
            # runtime_source_add_delete order: leave flushing, unlink, release the request pad.
            mux_pad.send_event(Gst.Event.new_flush_stop(False))
            if source_bin is not None:
                source_bin.get_static_pad("src").unlink(mux_pad)
            mux.release_request_pad(mux_pad)
            requested_pads.remove((mux, mux_pad))
        if source_bin is not None:
            pipeline.remove(source_bin)
        source_bins[index] = mux_pads[index] = None

    def rebuild_source(index):
        """Replace one source bin and its mux sink pad (NVIDIA runtime_source_add_delete pattern).

        Used from the second recovery on, when the in-place reset left the camera silent: a
        fresh sink pad negotiates from scratch. The slot, source_id and demux branch stay.
        """
        camera = cameras_by_source[index]

        def failed(stage):
            manager.recovery_failed(index)
            print("source_rebuild_failed " + json.dumps({"camera_id": camera.camera_id, "source_id": camera.source_id,
                  "stage": stage}, separators=(",", ":")), flush=True)

        source_bin = source_bins.get(index)
        if source_bin is not None:  # an earlier rebuild may have left the slot empty; then just create
            result = source_bin.set_state(Gst.State.NULL)
            if result == Gst.StateChangeReturn.ASYNC:
                result, current, _pending = source_bin.get_state(5 * Gst.SECOND)
                if current != Gst.State.NULL:
                    result = Gst.StateChangeReturn.FAILURE
            if result == Gst.StateChangeReturn.FAILURE:
                failed("null")  # the bin stays attached; the next recovery tries again
                return
            detach_source(index)
        try:
            attach_source(index)
        except SourceAttachError as error:
            detach_source(index)
            failed(error.stage)
            return
        except Exception:
            detach_source(index)
            failed("create")
            return
        if not source_bins[index].sync_state_with_parent():
            source_bins[index].set_state(Gst.State.NULL)
            detach_source(index)
            failed("sync")
            return
        generation = manager.recovered(index)
        print("source_rebuild " + json.dumps({"camera_id": camera.camera_id, "source_id": camera.source_id,
              "generation": generation,
              "recovery_count": manager.runtimes[camera.source_id].snapshot().recovery_count},
              separators=(",", ":")), flush=True)

    def recover_source(index, mode):
        """Reset in place first, rebuild from the second recovery on; runs on a recovery worker.

        A plugin exception inside a recovery must not take the shared pipeline down with it:
        the slot is deferred like a failed recovery and the control loop goes on.
        """
        try:
            (rebuild_source if mode == "rebuild" else reset_source)(index)
        except Exception:
            manager.recovery_failed(index)
            camera = cameras_by_source[index]
            print("source_recovery_error " + json.dumps({"camera_id": camera.camera_id, "source_id": camera.source_id,
                  "mode": mode}, separators=(",", ":")), flush=True)

    def recovery_log(event, recovery):
        camera = cameras_by_source[recovery["slot"]]
        print(f"source_recovery_{event} " + json.dumps({"camera_id": camera.camera_id, "source_id": camera.source_id,
              "mode": recovery["mode"], "seconds": round(time.monotonic() - recovery["started"], 1)},
              separators=(",", ":")), flush=True)

    def supervise_recovery(due):
        """Run one recovery at a time on a worker so a native call that never returns cannot stop this loop."""
        nonlocal inflight
        if inflight is not None:
            now = time.monotonic()
            if not inflight["thread"].is_alive():
                if inflight["kicked"] is not None:
                    recovery_log("unblocked", inflight)
                inflight = None
            elif inflight["kicked"] is None and now - inflight["started"] >= RECOVERY_HUNG_AFTER:
                recovery_log("hung", inflight)
                faulthandler.dump_traceback(all_threads=True)
                mux_pad = mux_pads.get(inflight["slot"])
                if mux_pad is not None:
                    # A streaming thread blocked in the mux sink pad leaves on FLUSHING and frees the stream lock.
                    with contextlib.suppress(Exception):
                        mux_pad.send_event(Gst.Event.new_flush_start())
                inflight["kicked"] = now
            elif inflight["kicked"] is not None and now - inflight["kicked"] >= RECOVERY_KICK_WAIT:
                recovery_log("abandoned", inflight)
                manager.abandon_recovery(inflight["slot"])
                inflight = None  # the daemon worker is left behind; other slots may recover again
            return  # a slot freed this tick waits for the next tick() so `due` is not stale
        if not due or shutdown.is_set() or pipeline_stop.is_set():
            return
        index = due[0]
        if source_bins.get(index) is None or (rebuild_enabled and manager.recovery_mode(index) == "rebuild"):
            mode = "rebuild"
        else:
            mode = "reset"
        thread = threading.Thread(target=recover_source, args=(index, mode), name=f"source-recovery-{index}",
                                  daemon=True)
        inflight = {"slot": index, "mode": mode, "thread": thread, "started": time.monotonic(), "kicked": None}
        thread.start()

    inflight = None
    rebuild_enabled = getattr(args, "source_recovery_rebuild", True)
    try:
        pipeline = Gst.Pipeline.new("preview-detection")
        if pipeline is None:
            raise RuntimeError("Could not create pipeline")
        mux = make_element("nvstreammux", "stream-muxer")
        mux.set_property("batch-size", manager.batch_size)
        mux.set_property("live-source", bool(args.mux_live_source))
        mux.set_property("width", 1920)
        mux.set_property("height", 1080)
        # 33333 us: a 40 ms timeout capped partial batches at ~25 fps, cutting 30 fps cameras to
        # ~24 fps (measured 24.8 -> 29.7 fps, docs/report_20261020.md section 3). 9 fps cameras unaffected.
        mux.set_property("batched-push-timeout", 33333)
        mux.set_property("gpu-id", 0)
        mux.set_property("nvbuf-memory-type", 0)  # Existing platform-default NVMM memory.
        mux.set_property("sync-inputs", False)  # Partial batches must not wait for offline cameras.
        add_timing_probe(mux.get_static_pad("src"), "mux-batch", "all")
        tracker = make_tracker()
        demux = make_element("nvstreamdemux", "stream-demuxer")
        shared = (mux, tracker, demux)
        vehicle = None
        if vehicle_detector is not None:
            vehicle = make_element("nvinfer", "vehicle-inference")
            configure_pgie(vehicle, vehicle_detector.config_path, len(vehicle_detector.cameras))
            if os.environ.get("VEHICLE_DEPLOYMENT_ENABLED") == "1":
                try:
                    from vehicle_deployment import DeploymentController
                    deployment = DeploymentController(vehicle, vehicle_detector, manager, segmenter=segmenter)
                except Exception:
                    # Opt-in feature: the configured production model keeps running without it.
                    print("Vehicle deployment controller disabled; details hidden", flush=True)

            def vehicle_probe(_pad, info, steps):
                try:
                    buffer = info.get_buffer()
                    if buffer is not None:
                        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
                        if batch is None:
                            raise ValueError("Missing vehicle batch")
                        for step in steps:
                            step(batch)
                except Exception:
                    fail("Vehicle metadata preparation failed; details hidden")
                return Gst.PadProbeReturn.OK

            def add_rois(batch):
                add_vehicle_rois(batch, vehicle_detector, camera_for_frame, pyds)

            def observe_deployment(batch):
                if deployment is not None:
                    try:
                        deployment.observe_batch(batch, camera_for_frame, pyds)
                    except Exception:
                        pass  # verification then sees no inference evidence and rolls back

            def prepare_tracks(batch):
                prepare_vehicle_tracks(batch, vehicle_detector, camera_for_frame, pyds)

            def attach_masks(batch):
                started = time.perf_counter()
                errors = attach_vehicle_masks(batch, vehicle_detector, segmenter, camera_for_frame, pyds)
                record_mask_cost(time.perf_counter() - started, errors)

            vehicle.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, vehicle_probe, (add_rois,))
            # Seg infers the same ROIs after the detector, so ROI removal waits for its output.
            vehicle.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, vehicle_probe,
                                                    (observe_deployment,) if segmenter else (observe_deployment, prepare_tracks))
            add_timing_probe(vehicle.get_static_pad("src"), "vehicle-batch", "all")
            shared = (mux, vehicle, tracker, demux)
            if segmenter is not None:
                seg = make_element("nvinfer", "vehicle-segmentation")
                configure_pgie(seg, segmenter.config_path, len(segmenter.cameras))
                seg.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, vehicle_probe, (attach_masks, prepare_tracks))
                add_timing_probe(seg.get_static_pad("src"), "vehicle-seg-batch", "all")
                shared = (mux, vehicle, seg, tracker, demux)
                graft = graft_status(segmenter, detector_config())
                print("WARNING vehicle_segmentation graft=stale; re-run the graft for the active detector; details hidden"
                      if graft == "stale" else f"vehicle_segmentation graft={graft}", flush=True)
        for element in shared:
            pipeline.add(element)
        for left, right in zip(shared, shared[1:]):
            if not left.link(right):
                raise RuntimeError(f"Could not link {left.get_name()} -> {right.get_name()}")
        if args.diagnostics:
            log_settings(vehicle_detector.config_path if vehicle is not None else None,
                         sorted(vehicle_detector.vehicle_class_ids) if vehicle is not None else (), TRACKER_CONFIG)
            if vehicle is not None:
                vehicle.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, on_inference)
        startup_probe(mux.get_static_pad("src"), "first_mux_output", batched=True)
        if vehicle is not None:
            startup_probe(vehicle.get_static_pad("sink"), "first_infer_input", batched=True)
            startup_probe(vehicle.get_static_pad("src"), "first_infer_output", batched=True)
        startup_probe(tracker.get_static_pad("sink"), "first_tracker_input", batched=True)
        startup_probe(tracker.get_static_pad("src"), "first_tracker_output", batched=True)
        tracker.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, on_batch)

        for index in range(len(args.input)):
            setup_stage = f"source-{index}"
            attach_source(index)

            setup_stage = f"jpeg-branch-{index}"
            output_bin = build_output(
                index, args, CameraFrameSink(stores[cameras_by_source[index].source_id], manager, index),
                lambda reason, i=index: fail("pipeline_error", str(i)),
                display_mask=segmenter is not None and cameras_by_source[index].camera_id in segmenter.cameras,
            )
            startup_probe(output_bin.get_by_name(f"jpeg-encoder-{index}").get_static_pad("src"),
                          "first_jpeg", index)
            startup_probe(output_bin.get_by_name(f"jpeg-sink-{index}").get_static_pad("sink"),
                          "first_appsink_buffer", index)
            add_timing_probe(
                output_bin.get_by_name(f"jpeg-sink-{index}").get_static_pad("sink"),
                "encoded-mjpeg", index,
            )
            pipeline.add(output_bin)
            # nvstreamdemux src_%u is a request pad, requested while in NULL state.
            demux_pad = demux.request_pad_simple(f"src_{index}")
            if demux_pad is None:
                raise RuntimeError(f"Could not request nvstreamdemux src_{index}")
            requested_pads.append((demux, demux_pad))
            demux_pad.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, demux_event)
            startup_probe(demux_pad, "first_demux_output", index)
            add_timing_probe(demux_pad, "demux", index)
            if demux_pad.link(output_bin.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
                raise RuntimeError("Could not link demux source pad to preview output")
            print(f"source={cameras_by_source[index].source_id} camera_id={cameras_by_source[index].camera_id}: "
                  f"output=mjpeg url=/mjpeg/source{cameras_by_source[index].source_id}", flush=True)
        if startup is not None:
            startup.mark("all_source_branches_built")
        bus = pipeline.get_bus()
        if startup is not None:
            observed = {"pipeline": pipeline, "tracker": tracker}
            if vehicle is not None:
                observed["vehicle"] = vehicle
            bus.set_sync_handler(bus_observer(startup, Gst, observed, message_scope), None)
        previous_time = time.monotonic()
        print(f"DeepStream shared pipeline sources={manager.batch_size} "
              f"streammux batch-size={mux.get_property('batch-size')} "
              f"vehicle cameras={len(vehicle_detector.cameras) if vehicle is not None else 0} "
              f"vehicle batch-size={vehicle.get_property('batch-size') if vehicle is not None else 0} tracker=NvDCF "
              f"demux branches={manager.batch_size} live-source={args.mux_live_source}", flush=True)
        setup_stage = "playing"
        startup_report()
        if startup is not None:
            startup.mark("playing_requested")
            startup.drain()
        state_result = pipeline.set_state(Gst.State.PLAYING)
        if startup is not None:
            startup.mark("playing_call_returned")
        if state_result == Gst.StateChangeReturn.FAILURE:
            stop("pipeline_setup_failed", True)
        else:
            manager.arm_watchdogs()
            previous_time = time.monotonic()
            last_report = time.monotonic()
            message_types = (Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS
                             | Gst.MessageType.ELEMENT | Gst.MessageType.STATE_CHANGED)
            while not shutdown.is_set() and not pipeline_stop.is_set():
                message = bus.timed_pop_filtered(100 * Gst.MSECOND, message_types)
                if message is not None:
                    on_message(bus, message)
                supervise_recovery(manager.tick())  # one bin at a time; the rest stay offline until later ticks
                if deployment is not None:
                    try:
                        deployment.tick()
                    except Exception:
                        # Deployment I/O failure must not terminate RTSP or the shared pipeline.
                        deployment.fault()
                if startup is not None:
                    for slot, camera in cameras_by_source.items():
                        if manager.runtimes[camera.source_id].snapshot().state == "online":
                            startup.mark("online", slot)
                    startup.drain()
                if args.diagnostics and time.monotonic() - last_report >= 5:
                    report()
                    startup_report()
                    for camera in args.cameras:
                        print("camera_status " + json.dumps({"source_id": camera.source_id,
                              **manager.runtimes[camera.source_id].snapshot().to_dict()},
                              separators=(",", ":")), flush=True)
                    last_report = time.monotonic()
    except Exception as error:
        print(f"Shared pipeline setup ERROR stage={setup_stage} type={type(error).__name__}; details hidden",
              flush=True)
        if args.diagnostics:
            log_safe_traceback(error)
        stop("pipeline_setup_failed", True)
    finally:
        if startup is not None:
            startup.stop_gpu_sampling()
        startup_report(final=True)
        if bus is not None and startup is not None:
            bus.set_sync_handler(None, None)
        if deployment is not None:
            with contextlib.suppress(Exception):
                deployment.close()
        if inflight is not None:
            # A hung worker may hold its bin; past this wait docker's stop grace period ends the process.
            inflight["thread"].join(RECOVERY_JOIN_TIMEOUT)
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        for owner, pad in reversed(requested_pads):
            owner.release_request_pad(pad)
    return failed


class CameraFrameSink:
    """Keep HTTP/WebSocket stores alive; reject JPEG from a previous observation epoch."""
    def __init__(self, store, manager, slot):
        self.store, self.manager, self.slot = store, manager, slot

    def put_frame(self, jpeg, pts):
        self.manager.publish(self.slot, int(pts), lambda: self.store.put(jpeg), output=True)


def run(args, metadata_store=None, runtime_statuses=None):
    """One HTTP server, one shared GPU pipeline, camera-local health."""
    startup = None
    if args.diagnostics:
        origin, origin_name = startup_origin(PROCESS_ENTRY)
        startup = StartupTiming(args.cameras, origin, origin_name)
        args.startup_timing = startup
        startup.mark("python_entry", at=PROCESS_ENTRY)
        startup.mark("run_entered")
        print("startup_config " + json.dumps({"origin": origin_name, "rtsp_transport": "TCP",
              "rtsp_latency_ms": args.rtsp_latency, "drop_on_latency": args.rtsp_drop_on_latency,
              "reconnect_interval_sec": max(1, int(args.camera_offline_after)), "reconnect_attempts": -1,
              "sources": len(args.cameras), "mux_batch_size": len(args.cameras),
              "mux_live_source": args.mux_live_source, "mux_sync_inputs": False,
              "mux_timeout_us": 40000}, separators=(",", ":")), flush=True)
    metadata_store = MetadataStore() if metadata_store is None else metadata_store
    runtime_statuses = {} if runtime_statuses is None else runtime_statuses
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    os.environ["GST_REGISTRY"] = str(cache / "gstreamer-registry.bin")
    Gst.init(None)
    Gst.debug_set_active(False)
    required = ("nvurisrcbin", "uridecodebin", "rtspsrc", "nvv4l2decoder", "nvvideoconvert", "queue",
                "capsfilter", "nvjpegenc", "appsink", "nvstreammux", "nvinfer", "nvtracker",
                "nvdsosd", "nvstreamdemux")
    missing = [name for name in required if Gst.ElementFactory.find(name) is None]
    if missing:
        print(f"Preview setup ERROR: missing GStreamer elements: {', '.join(missing)}", flush=True)
        return 1
    # Vehicle config and GObject override agree; engines remain cached by batch size.
    if startup is not None:
        startup.mark("infer_config_start")
    try:
        args.vehicle_detector = prepare_vehicle_detector(args.cameras, cache)
        if args.vehicle_detector is not None:
            # Preserve class-specific NvDCF matching without changing its tuning.
            import re
            if not re.search(r"checkClassMatch:\s*1\b", TRACKER_CONFIG.read_text()):
                raise ValueError("Vehicle tracking requires NvDCF checkClassMatch=1")
    except Exception:
        args.vehicle_detector = None
        print("Vehicle detector unavailable; configuration/model details hidden; parking remains unknown", flush=True)
    # Never raises: a broken seg setup only disables outlines.
    args.vehicle_segmenter = (prepare_vehicle_segmenter(args.vehicle_detector, cache)
                              if args.vehicle_detector is not None else None)
    if startup is not None:
        startup.mark("infer_config_ready")
    for camera in args.cameras:
        runtime_statuses[camera.source_id] = CameraRuntime(
            camera.camera_id, args.camera_degraded_after, args.camera_offline_after,
            args.camera_connect_timeout, args.camera_min_fps)
    frontend_files = FrontendFiles(os.environ.get('FRONTEND_DIST_DIR', str(ROOT / 'frontend-dist')))
    backend_api_proxy = BackendAPIProxy(os.environ.get('PREVIEW_BACKEND_URL', 'http://127.0.0.1:8000'))
    server = ThreadingHTTPServer(("0.0.0.0", args.port), PreviewHandler)
    server.daemon_threads = True
    server.frontend_files = frontend_files
    server.backend_api_proxy = backend_api_proxy
    stores = {camera.source_id: FrameStore() for camera in args.cameras}
    if startup is not None:
        for slot, camera in enumerate(args.cameras):
            stores[camera.source_id].startup_observer = lambda stage, i=slot: startup.mark(stage, i)
    server.mjpeg_streams = {f"/mjpeg/source{index}": store for index, store in stores.items()}
    args.raw_captures = ({slot: RawCapture() for slot in range(len(args.cameras))}
                         if os.environ.get('DATASET_CAPTURE_ENABLED', '0') == '1' else {})
    server.raw_captures = {camera.camera_id: args.raw_captures[slot] for slot, camera in enumerate(args.cameras)
                           if slot in args.raw_captures}

    runtime_session = uuid.uuid4().hex  # Track identities must not collide after process restart.

    def stream_snapshot():
        streams = []
        for camera in args.cameras:
            status = runtime_statuses[camera.source_id].snapshot()
            streams.append({**camera.public_info(), "id": camera.source_id, "format": "mjpeg",
                            "status": "connected" if status.state in ("online", "degraded") else "starting",
                            "url": f"/mjpeg/source{camera.source_id}", "metadata_path": "/ws",
                            "runtime": status.to_dict(),
                            "runtime_session": runtime_session})
        return streams

    server.stream_snapshot = stream_snapshot

    def metadata_snapshot():
        frames = metadata_store.snapshot()
        # A concurrent retry may leave an old immutable snapshot; reject it.
        current = []
        for source_id, frame in frames.items():
            status = runtime_statuses[source_id].snapshot()
            if status.generation == frame.generation and status.state in ("online", "degraded"):
                current.append(frame.to_dict())
        return {"runtime_session": runtime_session, "frames": current}

    server.metadata_snapshot = metadata_snapshot
    shutdown = threading.Event()
    server.realtime_hub = RealtimeHub(metadata_snapshot, stream_snapshot, stores, shutdown)
    parking_relay = ParkingRelay(os.environ.get('PREVIEW_BACKEND_URL', 'http://127.0.0.1:8000'),
                                 server.realtime_hub, shutdown)
    manager = CameraRuntimeManager(args.cameras, runtime_statuses, stores, metadata_store,
                                   recovery_enabled=getattr(args, "source_recovery", True))
    manager.begin()
    result = 1
    old_handlers = {}
    http_thread = threading.Thread(target=server.serve_forever, daemon=True)

    def on_signal(_signum, _frame):
        shutdown.set()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, on_signal)
        # `docker compose kill -s SIGUSR1 retrace` dumps every Python thread's line before a restart.
        faulthandler.register(signal.SIGUSR1, all_threads=True)
        http_thread.start()
        parking_relay.thread.start()
        print(f"Preview HTTP listening on 0.0.0.0:{args.port}; shared camera pipeline", flush=True)
        error = run_shared_pipeline(args, metadata_store, stores, manager, shutdown)
        result = int(error is not None)
    finally:
        shutdown.set()
        if parking_relay.thread.is_alive():
            parking_relay.thread.join(timeout=4)
        for store in stores.values():
            store.close()
        if http_thread.is_alive():
            server.shutdown()
            http_thread.join(timeout=2)
        server.server_close()
        for signum, previous in old_handlers.items():
            signal.signal(signum, previous)
        print("Preview stopped", flush=True)
    return result


if __name__ == "__main__":
    try:
        sys.exit(run(parse_args()))
    except Exception:
        print("source=all: Preview setup ERROR", file=sys.stderr)
        sys.exit(1)
