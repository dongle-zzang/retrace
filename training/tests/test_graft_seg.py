"""Graft of fine-tuned detector weights into YOLO11-seg, without Ultralytics/GPU (torch only).

cd training && .venv/bin/python -m unittest discover -s tests -p 'test_graft_seg.py'
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TRAINING.parent))
import torch  # noqa: E402

from graft_seg import graft_state, write_manifest  # noqa: E402


class GraftTests(unittest.TestCase):
    def test_graft_copies_det_tensors_and_keeps_seg_only_tensors(self):
        seg = {'box': torch.zeros(2, dtype=torch.float16), 'mask': torch.ones(3)}
        det = {'box': torch.tensor([1.0, 2.0])}
        out = graft_state(seg, det)
        self.assertTrue(torch.equal(out['box'], torch.tensor([1.0, 2.0], dtype=torch.float16)))
        self.assertTrue(torch.equal(out['mask'], seg['mask']))
        self.assertEqual(float(seg['box'].sum()), 0.0)  # inputs are not modified

    def test_graft_rejects_a_different_head_layout(self):
        for det in ({'box': torch.zeros(3)}, {'other': torch.zeros(2)}):
            with self.assertRaises(ValueError):
                graft_state({'box': torch.zeros(2)}, det)

    def test_manifest_records_both_weight_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            det, seg = root / 'best.pt', root / 'yolo11m-seg.pt'
            det.write_bytes(b'det weights')
            seg.write_bytes(b'seg weights')
            manifest = write_manifest(root / 'out', det, seg)
            self.assertEqual(json.loads((root / 'out' / 'manifest.json').read_text()), manifest)
            self.assertEqual(manifest, {'det_weights_sha256': hashlib.sha256(b'det weights').hexdigest(),
                                        'seg_weights_sha256': hashlib.sha256(b'seg weights').hexdigest()})


if __name__ == '__main__':
    unittest.main()
