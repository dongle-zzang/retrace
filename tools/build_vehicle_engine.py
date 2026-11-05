"""Build a candidate vehicle engine with the runtime's own nvinfer/TensorRT; never touches live elements.

Run by vehicle_deployment.DeploymentController as a separate process, so hashing,
staging and TensorRT serialization never block Preview's control loop.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vehicle_deployment import candidate_config, contract, configuration, digest, read_json, relative, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True, help='running detector config')
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--engines', type=Path, required=True)
    args = p.parse_args()
    onnx = args.job / 'best.onnx'
    manifest = read_json(args.job / 'manifest.json', {})
    onnx_sha = digest(onnx)
    if manifest.get('sha256') != onnx_sha or manifest.get('contract') != contract(configuration(args.base)):
        print('Candidate ONNX does not match its manifest/runtime contract', flush=True)
        return 2
    config = candidate_config(args.base, onnx, args.engines)
    path = args.job / 'infer.txt'
    with path.open('w') as stream:
        config.write(stream, space_around_delimiters=False)
    engine = Path(config['property']['model-engine-file'])
    if not engine.is_file() and not serialize_engine(path):
        return 1
    if not engine.is_file() or not engine.stat().st_size:
        print('nvinfer did not serialize the expected engine', flush=True)
        return 1
    write_json(args.job / 'build.json', {'engine': relative(engine), 'sha256': digest(engine), 'onnx_sha256': onnx_sha})
    print(f'Engine ready: {relative(engine)}', flush=True)
    return 0


def serialize_engine(config_path, timeout=None):
    """Standalone nvinfer element, never a live one; True unless the state change failed."""
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)
    element = Gst.ElementFactory.make('nvinfer', 'candidate-builder')
    if element is None:
        return False
    element.set_property('config-file-path', str(config_path))
    try:
        # READY -> PAUSED creates the NvDsInfer context, which serializes the engine.
        if element.set_state(Gst.State.PAUSED) == Gst.StateChangeReturn.FAILURE:
            return False
        return element.get_state((timeout or BUILD_TIMEOUT) * Gst.SECOND)[0] != Gst.StateChangeReturn.FAILURE
    finally:
        element.set_state(Gst.State.NULL)


BUILD_TIMEOUT = 1800

if __name__ == '__main__':
    raise SystemExit(main())
