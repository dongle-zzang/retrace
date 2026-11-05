"""Graft fine-tuned YOLO11m detector weights into YOLO11m-seg, keeping the COCO mask branch.

The official YOLO11m-seg detection branch is tensor-identical to YOLO11m, so copying the
fine-tuned detector's tensors reproduces its boxes and adds COCO masks (design 2026-11-05 §2-5).
Training venv, from training/:
  .venv/bin/python graft_seg.py --seg yolo11m-seg.pt --det <active best.pt> --out-dir ../models/yolo11m-seg
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vehicle_deployment import digest, write_json  # noqa: E402


def graft_state(seg_state, det_state):
    """New seg state dict with every det tensor copied in; a 1-class or other head layout is refused."""
    if any(key not in seg_state or seg_state[key].shape != value.shape for key, value in det_state.items()):
        raise ValueError('det head layout differs from seg')
    return {**seg_state, **{key: value.to(seg_state[key].dtype) for key, value in det_state.items()}}


def write_manifest(out_dir, det, seg):
    """Provenance read by the runtime to warn when OTA activates other detector weights."""
    manifest = {'det_weights_sha256': digest(det), 'seg_weights_sha256': digest(seg)}
    write_json(Path(out_dir) / 'manifest.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seg', type=Path, required=True, help='COCO YOLO11m-seg weights')
    parser.add_argument('--det', type=Path, required=True, help='fine-tuned YOLO11m detector best.pt')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    from ultralytics import YOLO
    seg, det = YOLO(str(args.seg)), YOLO(str(args.det))
    seg.model.load_state_dict(graft_state(seg.model.state_dict(), det.model.state_dict()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seg.save(str(args.out_dir / 'graft.pt'))
    manifest = write_manifest(args.out_dir, args.det, args.seg)
    print(f"graft.pt written; det_weights_sha256={manifest['det_weights_sha256']}", flush=True)
    print("It must equal weights_sha256 in the active deployment's manifest.json.", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
