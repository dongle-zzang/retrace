"""Exercise production preview HTTP code without importing GPU dependencies."""
import ast
import io
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from camera_runtime import CameraStatus
from person_metadata import BoundingBox, FrameMetadata, MetadataStore, VehicleMetadata, utc_timestamp

ROOT = Path(__file__).resolve().parents[2]


def node_named(tree, kind, name):
    return next(n for n in ast.walk(tree) if isinstance(n, kind) and n.name == name)


def test_preview_metadata_endpoint_and_generation_filter():
    tree = ast.parse((ROOT / "preview.py").read_text())
    store = MetadataStore()
    outline = ((10.0, 20.0), (40.0, 20.0), (40.0, 70.0), (10.0, 70.0))
    car = VehicleMetadata("first", utc_timestamp(), 7, 2, .9, BoundingBox(10, 20, 30, 50), .8, outline=outline)
    store.put(FrameMetadata(0, "first", 10, utc_timestamp(), None, True, (), generation=2, vehicles=(car,)))
    runtime = SimpleNamespace(snapshot=lambda: CameraStatus("first", state="online", generation=2))
    namespace = {"metadata_store": store, "runtime_statuses": {0: runtime}, "runtime_session": "a" * 32}
    function = node_named(tree, ast.FunctionDef, "metadata_snapshot")
    exec(compile(ast.Module(body=[function], type_ignores=[]), "preview.py", "exec"), namespace)
    snapshot = namespace["metadata_snapshot"]
    assert len(snapshot()["frames"]) == 1
    runtime.snapshot = lambda: CameraStatus("first", state="online", generation=3)
    assert snapshot()["frames"] == []
    runtime.snapshot = lambda: CameraStatus("first", state="offline", generation=2)
    assert snapshot()["frames"] == []
    runtime.snapshot = lambda: CameraStatus("first", state="degraded", generation=2)
    handler_namespace = {"BaseHTTPRequestHandler": BaseHTTPRequestHandler, "urlsplit": urlsplit,
                         "json": json}
    klass = node_named(tree, ast.ClassDef, "PreviewHandler")
    exec(compile(ast.Module(body=[klass], type_ignores=[]), "preview.py", "exec"), handler_namespace)
    handler = handler_namespace["PreviewHandler"].__new__(handler_namespace["PreviewHandler"])
    handler.path = "/metadata.json"
    handler.server = SimpleNamespace(mjpeg_streams={}, metadata_snapshot=snapshot)
    handler.wfile = io.BytesIO()
    handler.send_response = lambda code: None
    handler.send_header = lambda name, value: None
    handler.end_headers = lambda: None
    handler.do_GET()
    body = json.loads(handler.wfile.getvalue())
    assert body["runtime_session"] == "a" * 32
    assert body["frames"][0]["generation"] == 2
    assert body["frames"][0]["vehicles"][0]["outline"] == [list(point) for point in outline]
    assert "rtsp" not in handler.wfile.getvalue().decode().lower()
