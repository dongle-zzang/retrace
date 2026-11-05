"""Prebuild the optional vehicle segmentation engine in a separate process; never touches live elements.

Run in the retrace container before enabling seg, so the first restart deserializes the engine
instead of stopping every camera while TensorRT builds it (design 2026-11-05 §4-10):
  docker compose exec -T -e VEHICLE_SEG_INFER_CONFIG=models/yolo11m-seg/vehicle_seg_infer.txt retrace \\
      python3 tools/build_vehicle_seg_engine.py
"""
import configparser
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_vehicle_engine import serialize_engine  # noqa: E402
from camera_config import load_public_cameras  # noqa: E402
from vehicle_deployment import DIRECTORY, read_json  # noqa: E402
from vehicle_detection import graft_status, prepare_vehicle_detector, prepare_vehicle_segmenter  # noqa: E402


def main():
    cache = ROOT / '.cache'
    cameras = [SimpleNamespace(camera_id=camera['camera_id']) for camera in load_public_cameras() if camera['enabled']]
    try:
        detector = prepare_vehicle_detector(cameras, cache)
    except Exception:
        detector = None
    segmenter = detector and prepare_vehicle_segmenter(detector, cache)
    if not segmenter:
        print('Vehicle segmentation is not configured or not available; details hidden', flush=True)
        return 2
    active = read_json(DIRECTORY / 'committed.json', {}).get('active') or {}
    if os.environ.get('VEHICLE_DEPLOYMENT_ENABLED') == '1' and active.get('config'):
        detector_config = ROOT / active['config']
    else:
        detector_config = detector.config_path
    print(f'vehicle_segmentation graft={graft_status(segmenter, detector_config)}', flush=True)
    config = configparser.ConfigParser(interpolation=None)
    config.read(segmenter.config_path)
    engine = Path(config['property']['model-engine-file'])
    if not engine.is_file() and not serialize_engine(segmenter.config_path):
        print('nvinfer could not build the segmentation engine; details hidden', flush=True)
        return 1
    if not engine.is_file() or not engine.stat().st_size:
        print('nvinfer did not serialize the expected engine', flush=True)
        return 1
    print(f'Engine ready: {engine.relative_to(ROOT)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
