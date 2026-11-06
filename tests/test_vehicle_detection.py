import configparser
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from person_metadata import vehicle_from_object, FrameMetadata
from preview_socket import detection_message
from vehicle_detection import (prepare_vehicle_detector, add_vehicle_rois, prepare_vehicle_tracks,
                               VehicleDetector, VEHICLE_TRACKER_CLASS,
                               outline_from_grid, resample_mask, ring_mask, vehicle_outline,
                               prepare_vehicle_segmenter, outline_box_width, graft_status, VehicleSegmenter,
                               OutlineRuntime, outline_settings, SEG_ROI_CLASS, VEHICLE_ROI_ID)


def nodes(values):
    node = None
    for value in reversed(values):
        node = NS(data=value, next=node)
    return node


def car_grid():
    """50 x 75 cells: an oblique car body with a roof, like the design prototype."""
    yy, xx = np.mgrid[0:50, 0:75]
    car = (xx > 5 + yy * .3) & (xx < 60 + yy * .25) & (yy > 8) & (yy < 48)
    return car | (((xx - 35) ** 2 / 400 + (yy - 14) ** 2 / 60) < 1)


def signed_area(points):
    # Shoelace in screen coordinates (y down): clockwise on screen is positive.
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])) / 2


def polygon_mask_iou(points, grid, rect):
    left, top, width, height = rect
    h, w = grid.shape
    ys, xs = np.mgrid[0:h, 0:w]
    px, py = left + (xs + .5) * width / w, top + (ys + .5) * height / h
    inside = np.zeros_like(grid)
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        if y1 != y2:
            inside ^= ((y1 > py) != (y2 > py)) & (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1)
    return (inside & grid).sum() / (inside | grid).sum()


COCO8 = 'person\nbicycle\ncar\nmotorcycle\nairplane\nbus\ntrain\ntruck\n'
# Generated detector config captured from the implementation before the seg refactor (2026-11-05).
DETECTOR_GOLDEN = (
    '[property]\nonnx-file=<root>/model.onnx\nlabelfile-path=<root>/labels.txt\ncustom-lib-path=<root>/parser.so\n'
    'parse-bbox-func-name=NvDsInferParseYolo\nnum-detected-classes=8\nmaintain-aspect-ratio=1\ncluster-mode=2\n'
    'gpu-id=0\nbatch-size=2\nprocess-mode=2\nnetwork-type=0\ngie-unique-id=2\noperate-on-gie-id=90\n'
    'operate-on-class-ids=0\ninterval=0\nsecondary-reinfer-interval=0\nclassifier-async-mode=0\n'
    'input-object-min-width=1\ninput-object-min-height=1\ninput-object-max-width=0\ninput-object-max-height=0\n'
    'input-tensor-meta=0\nnetwork-mode=2\nfilter-out-class-ids=0;1;3;4;6\n'
    'model-engine-file=<root>/vehicle/vehicle.engine\n\n[class-attrs-all]\npre-cluster-threshold=0.25\n\n'
    '[class-attrs-2]\npre-cluster-threshold=0.25\n\n[class-attrs-5]\npre-cluster-threshold=0.25\n\n'
    '[class-attrs-7]\npre-cluster-threshold=0.25\n\n')


class SegmenterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = root = Path(temporary.name)
        (root / 'model.onnx').write_bytes(b'fake model')
        (root / 'parser.so').write_bytes(b'parser')
        (root / 'labels.txt').write_text(COCO8)
        self.det_config = root / 'infer.txt'
        self.det_config.write_text('[property]\nonnx-file=model.onnx\nlabelfile-path=labels.txt\ncustom-lib-path=parser.so\n'
                                   'parse-bbox-func-name=NvDsInferParseYolo\nnum-detected-classes=8\n'
                                   'maintain-aspect-ratio=1\ncluster-mode=2\n[class-attrs-all]\npre-cluster-threshold=0.25\n')
        seg = root / 'seg'
        seg.mkdir()
        (seg / 'graft.onnx').write_bytes(b'fake seg model')
        (seg / 'parser_seg.so').write_bytes(b'seg parser')
        (seg / 'labels.txt').write_text(COCO8)
        self.seg_config = seg / 'vehicle_seg_infer.txt'
        self.seg_text = ('[property]\nonnx-file=graft.onnx\nlabelfile-path=labels.txt\ncustom-lib-path=parser_seg.so\n'
                         'parse-bbox-instance-mask-func-name=NvDsInferParseYoloSeg\nnum-detected-classes=8\n'
                         'network-type=3\ncluster-mode=4\noutput-instance-mask=1\nsegmentation-threshold=0.5\n'
                         'process-mode=1\ngie-unique-id=1\n[class-attrs-all]\npre-cluster-threshold=0.1\n')
        self.seg_config.write_text(self.seg_text)
        self.targets = []
        def cache(props, target):
            self.targets.append(str(target))
            props['model-engine-file'] = str(Path(target) / 'vehicle.engine')
            return root, 'fixture'
        patcher = patch('vehicle_detection.prepare_engine_cache', cache)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.env = {'VEHICLE_CAMERA_IDS': 'p1,p2', 'VEHICLE_INFER_CONFIG': str(self.det_config)}
        self.detector = prepare_vehicle_detector([NS(camera_id='p1'), NS(camera_id='p2'), NS(camera_id='lobby')],
                                                 root, self.env)

    def segmenter(self, **env):
        return prepare_vehicle_segmenter(self.detector, self.root,
                                         {**self.env, 'VEHICLE_SEG_INFER_CONFIG': str(self.seg_config), **env})

    def test_detector_generated_config_is_unchanged_by_refactor(self):
        self.assertEqual(Path(self.detector.config_path).read_text().replace(str(self.root), '<root>'), DETECTOR_GOLDEN)

    def test_segmenter_disabled_without_env(self):
        self.assertIsNone(prepare_vehicle_segmenter(self.detector, self.root, self.env))
        self.assertIsNone(prepare_vehicle_segmenter(self.detector, self.root, {**self.env, 'VEHICLE_SEG_INFER_CONFIG': ' '}))

    def test_segmenter_forces_secondary_instance_segmentation_on_parking_roi(self):
        seg = self.segmenter(VEHICLE_MIN_CONFIDENCE='0.3')
        parser = configparser.ConfigParser()
        parser.read(seg.config_path)
        props = parser['property']
        # Seg infers only the seg ROI (class 1) that OutlineRuntime adds every VEHICLE_SEG_INTERVAL seconds.
        for key, value in {'process-mode': '2', 'network-type': '3', 'output-instance-mask': '1', 'cluster-mode': '4',
                           'gie-unique-id': '3', 'operate-on-gie-id': '90', 'operate-on-class-ids': '1',
                           'batch-size': '2', 'interval': '0', 'gpu-id': '0',
                           'filter-out-class-ids': '0;1;3;4;6'}.items():
            self.assertEqual(props[key], value, key)
        self.assertEqual(props['custom-lib-path'], str(self.root / 'seg' / 'parser_seg.so'))
        # Seg keeps its own low threshold; VEHICLE_MIN_CONFIDENCE belongs to the detector.
        self.assertEqual(parser['class-attrs-7']['pre-cluster-threshold'], '0.1')
        self.assertEqual(self.targets[-1], str(self.root / 'vehicle-seg'))
        self.assertEqual(Path(seg.config_path).name, 'vehicle_seg_fixture.txt')
        self.assertEqual((seg.cameras, seg.vehicle_class_ids, seg.box_width, seg.det_weights_sha256),
                         (frozenset({'p1', 'p2'}), frozenset({2, 5, 7}), 0, None))
        self.assertEqual((seg.interval, seg.ring_cells), (2.0, 2))
        self.assertEqual(self.segmenter(VEHICLE_OUTLINE_BOX_WIDTH='1').box_width, 1)
        tuned = self.segmenter(VEHICLE_SEG_INTERVAL='0.5', VEHICLE_OUTLINE_RING_CELLS='3')
        self.assertEqual((tuned.interval, tuned.ring_cells), (.5, 3))
        # The detector's config keeps operating on the per-frame ROI (class 0).
        parser.read(self.detector.config_path)
        self.assertEqual(parser['property']['operate-on-class-ids'], '0')

    def test_segmenter_reads_graft_manifest(self):
        (self.root / 'seg' / 'manifest.json').write_text('{"det_weights_sha256": "%s"}' % ('a' * 64))
        self.assertEqual(self.segmenter().det_weights_sha256, 'a' * 64)

    def test_broken_segmenter_config_disables_outlines_without_raising(self):
        for key in ('parse-bbox-instance-mask-func-name', 'custom-lib-path'):
            self.seg_config.write_text('\n'.join(line for line in self.seg_text.splitlines()
                                                 if not line.startswith(key)) + '\n')
            with patch('builtins.print') as printed:
                self.assertIsNone(self.segmenter())
            self.assertEqual(printed.call_args.args[0], 'Vehicle segmentation unavailable; details hidden; outlines disabled')
        self.seg_config.unlink()
        with patch('builtins.print') as printed:
            self.assertIsNone(self.segmenter())
        self.assertNotIn(str(self.root), printed.call_args.args[0])

    def test_outline_box_width(self):
        # Default 0 (2026-11-06): a vehicle with an outline shows only its outline and label.
        for raw, expected in ((None, 0), ('0', 0), ('3', 3), (' 2 ', 2), ('4', 0), ('-1', 0), ('abc', 0), ('1.5', 0)):
            with patch('builtins.print'):
                self.assertEqual(outline_box_width({} if raw is None else {'VEHICLE_OUTLINE_BOX_WIDTH': raw}), expected)

    def test_outline_settings_fall_back_on_invalid_values(self):
        cases = (({}, (2.0, 2)), ({'VEHICLE_SEG_INTERVAL': '1'}, (1.0, 2)), ({'VEHICLE_SEG_INTERVAL': ' 0.2 '}, (.2, 2)),
                 ({'VEHICLE_SEG_INTERVAL': '0.1'}, (2.0, 2)), ({'VEHICLE_SEG_INTERVAL': 'nan'}, (2.0, 2)),
                 ({'VEHICLE_SEG_INTERVAL': '61'}, (2.0, 2)), ({'VEHICLE_OUTLINE_RING_CELLS': '1'}, (2.0, 1)),
                 ({'VEHICLE_OUTLINE_RING_CELLS': '4'}, (2.0, 4)), ({'VEHICLE_OUTLINE_RING_CELLS': '5'}, (2.0, 2)),
                 ({'VEHICLE_OUTLINE_RING_CELLS': 'x'}, (2.0, 2)))
        for env, expected in cases:
            with patch('builtins.print'):
                self.assertEqual(outline_settings(env), expected, env)

    def test_graft_status_compares_the_graft_with_the_active_detector_weights(self):
        seg = VehicleSegmenter(frozenset({'p1'}), 'unused', frozenset({2}), det_weights_sha256='a' * 64)
        active = self.root / 'active'
        active.mkdir()
        config = active / 'infer.txt'
        self.assertIsNone(graft_status(None, config))
        self.assertEqual(graft_status(seg, config), 'unknown')  # original model: no manifest beside its config
        (active / 'manifest.json').write_text('{"weights_sha256": "%s"}' % ('a' * 64))
        self.assertEqual(graft_status(seg, config), 'match')
        (active / 'manifest.json').write_text('{"weights_sha256": "%s"}' % ('b' * 64))
        self.assertEqual(graft_status(seg, config), 'stale')
        self.assertEqual(graft_status(VehicleSegmenter(frozenset({'p1'}), 'unused', frozenset({2})), config), 'unknown')
        (active / 'manifest.json').write_text('not json')
        self.assertEqual(graft_status(seg, config), 'unknown')


class FakeMask:
    """NvOSD_MaskParams: get_mask_array is a flat float view, alloc_mask_array allocates width*height floats."""
    def __init__(self, width=0, height=0, data=None, threshold=.5):
        self.width, self.height, self.threshold, self.data = width, height, threshold, data
        self.size = 0 if data is None else data.nbytes
        self.allocated = None

    def get_mask_array(self):
        return self.data

    def alloc_mask_array(self):
        self.size = self.width * self.height * 4
        self.allocated = np.zeros(self.width * self.height, np.float32)
        return self.allocated


def box(component, left, top, width, height, class_id=2, mask=None):
    return NS(unique_component_id=component, class_id=class_id, parent=None, mask_params=mask or FakeMask(),
              rect_params=NS(left=left, top=top, width=width, height=height))


def det(left, top, width=300, height=200, class_id=2):
    return box(2, left, top, width, height, class_id)


def seg(left, top, width=300, height=200, fill=1.0, class_id=2):
    data = np.zeros((160, 160), np.float32)
    data[:round(160 * fill)] = .9
    return box(3, left, top, width, height, class_id, FakeMask(160, 160, data.ravel()))


PYDS = NS(NvDsFrameMeta=NS(cast=lambda v: v), NvDsObjectMeta=NS(cast=lambda v: v))
PARKING_DETECTOR = VehicleDetector(frozenset({'parking'}), 'unused', 2, frozenset({2, 5, 7}))
PARKING_SEGMENTER = VehicleSegmenter(frozenset({'parking'}), 'unused', frozenset({2, 5, 7}))


def batch_with(**frames):
    return NS(frame_meta_list=nodes([NS(camera_id=camera, frame_num=1, obj_meta_list=nodes(objects))
                                     for camera, objects in frames.items()]))


def attach(batch):
    from vehicle_detection import attach_vehicle_masks
    return attach_vehicle_masks(batch, PARKING_DETECTOR, PARKING_SEGMENTER, lambda frame: frame, PYDS)


class MaskAttachTests(unittest.TestCase):
    def test_masks_attach_to_the_best_iou_det_one_to_one_as_a_ring(self):
        det_a, det_b = det(800, 500), det(1200, 600)
        exact, shifted, far = seg(800, 500), seg(830, 500), seg(100, 100, 50, 50)
        self.assertEqual(attach(batch_with(parking=[det_a, det_b, shifted, exact, far])), 0)
        mask = det_a.mask_params
        self.assertEqual((mask.width, mask.height, mask.threshold), (75, 50, .5))
        grid = mask.allocated.reshape(50, 75)
        self.assertTrue(grid[0].all())  # the exact seg won: the shifted one leaves the left cells empty
        self.assertTrue(grid[1].all())  # default ring is two cells thick
        self.assertFalse(grid[2, 30])
        self.assertFalse(grid[25, 30])  # ring: interior removed
        self.assertIsNone(det_b.mask_params.allocated)  # nothing overlaps it by IoU >= 0.5
        for unused in (shifted, far):
            self.assertIsNone(unused.mask_params.allocated)

    def test_low_fill_masks_and_non_parking_frames_get_no_mask(self):
        sparse_det, lobby_det = det(800, 500), det(0, 0)
        attach(batch_with(parking=[sparse_det, seg(800, 500, fill=.1)], lobby=[lobby_det, seg(0, 0)]))
        self.assertIsNone(sparse_det.mask_params.allocated)
        self.assertIsNone(lobby_det.mask_params.allocated)

    def test_broken_pair_is_skipped_and_other_pairs_still_attach(self):
        broken = seg(800, 500)
        broken.mask_params.width = 7  # size != width * height from a broken parser
        good_det = det(1200, 600)
        self.assertEqual(attach(batch_with(parking=[det(800, 500), broken, good_det, seg(1200, 600)])), 1)
        self.assertIsNotNone(good_det.mask_params.allocated)

    def test_seg_only_frames_and_wrong_classes_attach_nothing(self):
        person = det(800, 500, class_id=0)
        self.assertEqual(attach(batch_with(parking=[person, seg(800, 500), seg(0, 0, class_id=0)])), 0)
        self.assertIsNone(person.mask_params.allocated)

    def test_prepare_removes_seg_objects_and_returns_the_roi_parent_last(self):
        roi = NS(unique_component_id=90, class_id=0)
        car, seg_car = det(800, 500), seg(800, 500)
        car.parent = seg_car.parent = roi
        seg_only = seg(0, 0)
        removed = []
        pyds = NS(NvDsFrameMeta=NS(cast=lambda v: v), NvDsObjectMeta=NS(cast=lambda v: v),
                  nvds_remove_obj_meta_from_frame=lambda frame, obj: removed.append(obj))
        prepare_vehicle_tracks(batch_with(parking=[roi, car, seg_car], other=[seg_only]),
                               VehicleDetector(frozenset({'parking'}), 'unused', 2, frozenset({2})), lambda f: f, pyds)
        self.assertEqual(removed, [seg_car, roi, seg_only])
        self.assertIsNone(car.parent)
        self.assertEqual(car.class_id, VEHICLE_TRACKER_CLASS)


class OutlineTests(unittest.TestCase):
    def test_outline_follows_mask_within_box(self):
        grid, rect = car_grid(), (800.0, 500.0, 300.0, 200.0)
        points = outline_from_grid(grid, *rect)
        self.assertGreaterEqual(len(points), 3)
        self.assertLessEqual(len(points), 32)
        self.assertNotEqual(points[0], points[-1])
        self.assertTrue(all(800 <= x <= 1100 and 500 <= y <= 700 for x, y in points))
        self.assertTrue(all(round(value, 1) == value for point in points for value in point))
        self.assertGreater(signed_area(list(points)), 0)
        self.assertGreaterEqual(polygon_mask_iou(list(points), grid, rect), .9)

    def test_ring_outline_equals_filled_outline(self):
        self.assertEqual(outline_from_grid(ring_mask(car_grid()), 0, 0, 300, 200),
                         outline_from_grid(car_grid(), 0, 0, 300, 200))
        ring = ring_mask(car_grid())
        self.assertTrue(ring[car_grid()].any())
        self.assertFalse(ring[25, 30])  # interior removed, the border stays

    def test_thick_ring_keeps_the_outline_and_is_k_cells_wide(self):
        full = np.ones((10, 10), bool)
        self.assertEqual(ring_mask(full).sum(), 100 - 64)
        self.assertEqual(ring_mask(full, 2).sum(), 100 - 36)
        self.assertEqual(ring_mask(full, 4).sum(), 100 - 4)
        for cells in (1, 2, 3, 4):
            self.assertEqual(outline_from_grid(ring_mask(car_grid(), cells), 0, 0, 300, 200),
                             outline_from_grid(car_grid(), 0, 0, 300, 200))

    def test_empty_single_row_and_tiny_masks_have_no_outline(self):
        self.assertIsNone(outline_from_grid(np.zeros((5, 5), bool), 0, 0, 20, 20))
        one = np.zeros((5, 5), bool)
        one[2, 1:4] = True
        self.assertIsNone(outline_from_grid(one, 0, 0, 20, 20))
        tiny = resample_mask(np.ones((160, 160), bool), (0, 0, 3, 3), (0, 0, 3, 3))
        self.assertEqual(tiny.shape, (1, 1))
        self.assertIsNone(outline_from_grid(tiny, 0, 0, 3, 3))

    def test_resample_maps_cells_and_leaves_outside_seg_box_empty(self):
        seg = np.ones((160, 160), bool)
        # The det box is wider than the seg box, like a vehicle cut at the frame edge.
        grid = resample_mask(seg, (100.0, 100.0, 40.0, 40.0), (80.0, 100.0, 80.0, 40.0))
        self.assertEqual(grid.shape, (10, 20))
        self.assertFalse(grid[:, :5].any())
        self.assertFalse(grid[:, 15:].any())
        self.assertTrue(grid[:, 5:15].all())
        half = np.zeros((160, 160), bool)
        half[:, :80] = True
        grid = resample_mask(half, (0.0, 0.0, 400.0, 200.0), (0.0, 0.0, 400.0, 200.0))
        self.assertEqual(grid.shape, (50, 100))
        self.assertTrue(grid[:, :50].all())
        self.assertFalse(grid[:, 50:].any())

    def test_vehicle_outline_reads_mask_params_and_never_raises(self):
        data = car_grid().astype(np.float32).ravel()
        obj = NS(mask_params=NS(size=data.nbytes, width=75, height=50, threshold=.5, get_mask_array=lambda: data),
                 rect_params=NS(left=800.0, top=500.0, width=300.0, height=200.0))
        self.assertEqual(vehicle_outline(obj), outline_from_grid(car_grid(), 800.0, 500.0, 300.0, 200.0))
        obj.mask_params.width = 74  # size mismatch from a broken parser
        self.assertIsNone(vehicle_outline(obj))
        obj.mask_params.size = 0
        self.assertIsNone(vehicle_outline(obj))


class VehicleTests(unittest.TestCase):
    def test_disabled_does_not_require_model_and_unknown_camera_is_skipped(self):
        self.assertIsNone(prepare_vehicle_detector([], '.', {}))
        # Removed/disabled parking cameras are skipped; no model is needed when none remain.
        with patch('builtins.print') as printed:
            self.assertIsNone(prepare_vehicle_detector([NS(camera_id='lobby')], '.', {'VEHICLE_CAMERA_IDS': 'unknown'}))
        self.assertIn('camera_id=unknown', printed.call_args.args[0])

    def test_config_is_camera_roi_detector_and_reuses_native_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model.onnx').write_bytes(b'fake model')
            (root / 'labels.txt').write_text('car\nperson\nsign\nbicycle\n')
            path = root / 'infer.txt'
            path.write_text('[property]\nonnx-file=model.onnx\nlabelfile-path=labels.txt\nnum-detected-classes=4\n')
            captured = {}
            def cache(props, target):
                captured.update(dict(props))
                captured['cache_dir'] = str(target)
                props['model-engine-file'] = str(root / 'vehicle.engine')
                return root, 'fixture'
            with patch('vehicle_detection.prepare_engine_cache', cache), patch('builtins.print'):
                detector = prepare_vehicle_detector([NS(camera_id='parking'), NS(camera_id='lobby')], root,
                    {'VEHICLE_CAMERA_IDS': 'parking,removed_parking', 'VEHICLE_INFER_CONFIG': str(path)})
            self.assertEqual(detector.cameras, frozenset({'parking'}))
            self.assertEqual(captured['process-mode'], '2')
            self.assertEqual(captured['operate-on-gie-id'], '90')
            self.assertEqual(captured['gie-unique-id'], '2')
            self.assertEqual(captured['batch-size'], '1')
            self.assertEqual(captured['filter-out-class-ids'], '1;2;3')
            self.assertEqual(captured['cache_dir'], str(root / 'vehicle'))
            parser = configparser.ConfigParser()
            parser.read(detector.config_path)
            self.assertEqual(parser['class-attrs-0']['pre-cluster-threshold'], '0.4')

    def test_all_parking_cameras_share_one_detector_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model.onnx').write_bytes(b'fake model')
            (root / 'labels.txt').write_text('person\nbicycle\ncar\n')
            path = root / 'infer.txt'
            path.write_text('[property]\nonnx-file=model.onnx\nlabelfile-path=labels.txt\nnum-detected-classes=3\n')
            captured = {}
            def cache(props, target):
                captured.update(dict(props))
                props['model-engine-file'] = str(root / 'vehicle.engine')
                return root, 'fixture'
            parking = ['b1_parking_entrance', 'b1_parking_overview', 'b1_parking_gate',
                       'b1_parking_inner', '1f_outdoor_parking']
            cameras = [NS(camera_id=c) for c in ['b1_lobby', *parking, '1f_lobby']]
            with patch('vehicle_detection.prepare_engine_cache', cache):
                detector = prepare_vehicle_detector(cameras, root, {
                    'VEHICLE_CAMERA_IDS': ' , '.join(parking), 'VEHICLE_INFER_CONFIG': str(path)})
            self.assertEqual(detector.cameras, frozenset(parking))
            self.assertEqual(captured['batch-size'], '5')
            self.assertEqual(captured['process-mode'], '2')

    def test_coco_car_truck_bus_share_tracker_class_and_config_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model.onnx').write_bytes(b'fake model')
            (root / 'labels.txt').write_text('person\nbicycle\ncar\nmotorcycle\nairplane\nbus\ntrain\ntruck\n')
            path = root / 'infer.txt'
            path.write_text('[property]\nonnx-file=model.onnx\nlabelfile-path=labels.txt\nnum-detected-classes=8\n'
                            'maintain-aspect-ratio=1\n[class-attrs-all]\npre-cluster-threshold=0.25\n')
            def cache(props, target):
                props['model-engine-file'] = str(root / 'vehicle.engine')
                return root, 'fixture'
            cameras = [NS(camera_id='parking')]
            env = {'VEHICLE_CAMERA_IDS': 'parking', 'VEHICLE_INFER_CONFIG': str(path)}
            with patch('vehicle_detection.prepare_engine_cache', cache):
                detector = prepare_vehicle_detector(cameras, root, env)
                parser = configparser.ConfigParser()
                parser.read(detector.config_path)
                self.assertEqual((detector.car_class_id, detector.vehicle_class_ids), (2, frozenset({2, 5, 7})))
                self.assertEqual(parser['property']['filter-out-class-ids'], '0;1;3;4;6')
                self.assertEqual(parser['property']['maintain-aspect-ratio'], '1')
                for class_id in (2, 5, 7):
                    self.assertEqual(parser[f'class-attrs-{class_id}']['pre-cluster-threshold'], '0.25')
                parser = configparser.ConfigParser()
                parser.read(prepare_vehicle_detector(cameras, root, {**env, 'VEHICLE_MIN_CONFIDENCE': '0.3'}).config_path)
                self.assertEqual(parser['class-attrs-7']['pre-cluster-threshold'], '0.3')
                (root / 'labels.txt').write_text('person\ncar\ncar\n')
                path.write_text(path.read_text().replace('num-detected-classes=8', 'num-detected-classes=3'))
                with self.assertRaises(ValueError):
                    prepare_vehicle_detector(cameras, root, env)
        frame = NS(camera_id='parking', obj_meta_list=None)
        truck, bus, person = NS(unique_component_id=2, class_id=7, parent=1), NS(unique_component_id=2, class_id=5, parent=1), NS(unique_component_id=2, class_id=0, parent=1)
        frame.obj_meta_list = nodes([truck, bus, person])
        removed = []
        pyds = NS(NvDsFrameMeta=NS(cast=lambda v: v), NvDsObjectMeta=NS(cast=lambda v: v),
                  nvds_remove_obj_meta_from_frame=lambda f, obj: removed.append(obj))
        prepare_vehicle_tracks(NS(frame_meta_list=nodes([frame])),
                               VehicleDetector(frozenset({'parking'}), 'unused', 2, frozenset({2, 5, 7})), lambda f: f, pyds)
        self.assertEqual((truck.class_id, bus.class_id), (VEHICLE_TRACKER_CLASS, VEHICLE_TRACKER_CLASS))
        self.assertEqual(removed, [person])

    def test_only_selected_frame_gets_roi_and_parent_removed_before_tracking(self):
        parking = NS(camera_id='parking', bInferDone=True, obj_meta_list=None, pad_index=0, frame_num=1, buf_pts=1)
        lobby = NS(camera_id='lobby', bInferDone=True, obj_meta_list=None)
        batch = NS(frame_meta_list=nodes([parking, lobby]))
        allocated, removed = [], []
        def allocate(batch):
            obj = NS(rect_params=NS())
            allocated.append(obj)
            return obj
        pyds = NS(NvDsFrameMeta=NS(cast=lambda v: v), NvDsObjectMeta=NS(cast=lambda v: v),
                  nvds_acquire_obj_meta_from_pool=allocate,
                  nvds_add_obj_meta_to_frame=lambda frame, obj, parent: setattr(frame, 'obj_meta_list', nodes([obj])),
                  nvds_remove_obj_meta_from_frame=lambda frame, obj: removed.append(obj))
        detector = VehicleDetector(frozenset({'parking'}), 'unused', 0)
        add_vehicle_rois(batch, detector, lambda frame: frame, pyds)
        self.assertEqual(len(allocated), 1)
        self.assertFalse(parking.bInferDone)
        self.assertTrue(lobby.bInferDone)
        self.assertEqual((allocated[0].rect_params.width, allocated[0].rect_params.height), (1920, 1080))
        person = NS(unique_component_id=1, class_id=0)
        car = NS(unique_component_id=2, class_id=0, parent=allocated[0])
        wrong = NS(unique_component_id=2, class_id=1)
        parking.obj_meta_list = nodes([allocated[0], person, car, wrong])
        prepare_vehicle_tracks(batch, detector, lambda frame: frame, pyds)
        self.assertIsNone(car.parent)
        self.assertEqual(car.class_id, VEHICLE_TRACKER_CLASS)
        self.assertEqual(person.class_id, 0)
        self.assertIn(allocated[0], removed)
        self.assertIn(wrong, removed)
        self.assertNotIn(person, removed)

    def test_detached_vehicle_serializes_outline_as_point_arrays_or_null(self):
        import json
        obj = NS(object_id=7, class_id=100, confidence=.9, tracker_confidence=.7,
                 rect_params=NS(left=192, top=108, width=384, height=216))
        outline = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.5))
        vehicle = vehicle_from_object('parking', '2026-01-01T00:00:00Z', obj, 2, outline=outline)
        frame = FrameMetadata(0, 'parking', 1, vehicle.timestamp, None, True, (), vehicles=(vehicle,)).to_dict()
        self.assertEqual(json.loads(json.dumps(frame))['vehicles'][0]['outline'], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.5]])
        self.assertIsNone(vehicle_from_object('parking', '2026-01-01T00:00:00Z', obj, 2).to_dict()['outline'])

    def test_detached_metadata_retains_native_class_and_lossless_id(self):
        obj = NS(object_id=(1 << 63) + 3, class_id=100, confidence=.9, tracker_confidence=.7,
                 rect_params=NS(left=192, top=108, width=384, height=216))
        vehicle = vehicle_from_object('parking', '2026-01-01T00:00:00Z', obj, 0)
        frame = FrameMetadata(0, 'parking', 1, vehicle.timestamp, None, True, (), vehicles=(vehicle,),
                              vehicle_detection_enabled=True, vehicle_inference_done=True).to_dict()
        obj.rect_params.left = 0
        self.assertEqual(frame['vehicles'][0]['bbox']['x'], 192)
        self.assertEqual(frame['vehicles'][0]['class_id'], 0)
        message = detection_message(frame, 'a' * 32)
        self.assertEqual(message['persons'], [])
        self.assertEqual(message['vehicles'][0]['trackId'], str((1 << 63) + 3))
        self.assertEqual(message['vehicles'][0]['bbox']['x'], .1)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def tracked(object_id, left=800.0, top=500.0, width=300.0, height=200.0, mask=None):
    return NS(object_id=object_id, mask_params=mask or FakeMask(),
              rect_params=NS(left=left, top=top, width=width, height=height))


def ring_on(obj, cells=2):
    """What the seg src probe leaves on a det vehicle: the ring of car_grid() on its box."""
    ring = ring_mask(car_grid(), cells).astype(np.float32).ravel()
    obj.mask_params = FakeMask(75, 50, ring)
    return obj


class OutlineRuntimeTests(unittest.TestCase):
    """A (seg every VEHICLE_SEG_INTERVAL s, outlines reused per track) and E (automatic suspension)."""
    def setUp(self):
        self.clock = FakeClock()
        self.logs = []
        self.segmenter = VehicleSegmenter(frozenset({'p1', 'p2'}), 'unused', frozenset({2, 5, 7}), interval=2.0)
        self.detector = VehicleDetector(frozenset({'p1', 'p2'}), 'unused', 2, frozenset({2, 5, 7}))
        self.runtime = OutlineRuntime(self.segmenter, self.detector, clock=self.clock, log=self.logs.append)

    def rois(self, *cameras, frame_num=1):
        frames = [NS(camera_id=camera, frame_num=frame_num, bInferDone=True, obj_meta_list=None) for camera in cameras]
        added = {camera: [] for camera in cameras}
        def acquire(batch):
            return NS(rect_params=NS())
        def add(frame, obj, parent):
            added[frame.camera_id].append(obj.class_id)
        pyds = NS(NvDsFrameMeta=NS(cast=lambda v: v), nvds_acquire_obj_meta_from_pool=acquire,
                  nvds_add_obj_meta_to_frame=add)
        add_vehicle_rois(NS(frame_meta_list=nodes(frames)), self.detector, lambda f: f, pyds, self.runtime)
        return added

    def test_seg_roi_goes_to_one_due_camera_per_batch_and_repeats_after_the_interval(self):
        self.assertEqual(SEG_ROI_CLASS, 1)
        self.assertEqual(self.rois('p1', 'p2', 'lobby'), {'p1': [0, SEG_ROI_CLASS], 'p2': [0], 'lobby': []})
        self.assertEqual(self.rois('p1', 'p2'), {'p1': [0], 'p2': [0, SEG_ROI_CLASS]})  # staggered
        self.assertEqual(self.rois('p1', 'p2'), {'p1': [0], 'p2': [0]})
        self.clock.now = 1.9
        self.assertEqual(self.rois('p1', 'p2'), {'p1': [0], 'p2': [0]})
        self.clock.now = 2.0
        self.assertEqual(self.rois('p1', 'p2'), {'p1': [0, SEG_ROI_CLASS], 'p2': [0]})
        self.assertEqual(self.rois('p1', 'p2'), {'p1': [0], 'p2': [0, SEG_ROI_CLASS]})
        self.assertEqual(self.runtime.snapshot()['seg_frames'], 4)

    def test_without_runtime_only_the_detector_roi_is_added(self):
        frame = NS(camera_id='p1', frame_num=1, bInferDone=True, obj_meta_list=None)
        added = []
        pyds = NS(NvDsFrameMeta=NS(cast=lambda v: v), nvds_acquire_obj_meta_from_pool=lambda b: NS(rect_params=NS()),
                  nvds_add_obj_meta_to_frame=lambda f, obj, parent: added.append((obj.unique_component_id, obj.class_id)))
        add_vehicle_rois(NS(frame_meta_list=nodes([frame])), self.detector, lambda f: f, pyds)
        self.assertEqual(added, [(VEHICLE_ROI_ID, 0)])

    def test_masks_attach_only_on_frames_that_got_a_seg_roi(self):
        self.rois('p1', frame_num=7)
        due_det, other_det = det(800, 500), det(800, 500)
        due = NS(camera_id='p1', frame_num=7, obj_meta_list=nodes([due_det, seg(800, 500)]))
        skipped = NS(camera_id='p1', frame_num=8, obj_meta_list=nodes([other_det, seg(800, 500)]))
        self.assertEqual(self.runtime.attach(NS(frame_meta_list=nodes([skipped, due])), lambda f: f, PYDS), 0)
        self.assertIsNotNone(due_det.mask_params.allocated)
        self.assertIsNone(other_det.mask_params.allocated)
        stats = self.runtime.snapshot()
        self.assertEqual((stats['seg_objects'], stats['seg_mask_floats'], stats['paired'], stats['attached']),
                         (1, 160 * 160, 1, 1))
        for key in ('scan_ms_per_s', 'mask_read_ms_per_s', 'mask_write_ms_per_s', 'probe_ms_per_s', 'attach_max_ms'):
            self.assertGreaterEqual(stats[key], 0, key)
        # The pending seg frame is consumed once.
        self.assertEqual(self.runtime.attach(NS(frame_meta_list=nodes([due])), lambda f: f, PYDS), 0)
        self.assertEqual(self.runtime.snapshot()['seg_objects'], 0)

    def test_seg_frame_outline_is_cached_and_reattached_on_later_frames(self):
        car = ring_on(tracked(5))
        expected = outline_from_grid(ring_mask(car_grid()), 800.0, 500.0, 300.0, 200.0)
        self.assertEqual(self.runtime.track_outline(car, 'p1'), expected)
        later = tracked(5, left=820.0, top=510.0, width=330.0, height=220.0)
        self.clock.now = 3.0
        outline = self.runtime.track_outline(later, 'p1')
        self.assertEqual(outline, outline_from_grid(ring_mask(car_grid()), 820.0, 510.0, 330.0, 220.0))
        mask = later.mask_params
        self.assertEqual((mask.width, mask.height, mask.threshold), (75, 50, .5))
        np.testing.assert_array_equal(mask.allocated, ring_mask(car_grid(), 2).astype(np.float32).ravel())
        # Another camera's track with the same id and an untracked object get nothing.
        self.assertIsNone(self.runtime.track_outline(tracked(5), 'p2'))
        untracked = ring_on(tracked((1 << 64) - 1))
        self.assertIsNotNone(self.runtime.track_outline(untracked, 'p1'))
        self.assertIsNone(self.runtime.track_outline(tracked((1 << 64) - 1), 'p1'))
        stats = self.runtime.snapshot()
        self.assertEqual((stats['reattached'], stats['cache_tracks']), (1, 1))

    def test_cached_outline_expires_after_three_intervals(self):
        self.runtime.track_outline(ring_on(tracked(5)), 'p1')
        self.clock.now = 6.0
        self.assertIsNotNone(self.runtime.track_outline(tracked(5), 'p1'))
        self.clock.now = 6.1
        self.assertIsNone(self.runtime.track_outline(tracked(5), 'p1'))
        self.assertEqual(self.runtime.snapshot()['cache_tracks'], 0)

    def test_broken_mask_gives_no_outline_and_never_raises(self):
        broken = ring_on(tracked(5))
        broken.mask_params.width = 74
        self.assertIsNone(self.runtime.track_outline(broken, 'p1'))
        self.assertIsNone(self.runtime.track_outline(NS(object_id=6), 'p1'))
        self.assertGreaterEqual(self.runtime.snapshot()['errors'], 1)

    def over_budget_window(self):
        # 1.1 s of seg Python work in a 5 s window = 220 ms/s, above the 200 ms/s guard.
        self.runtime.record_cost(1.1)
        self.clock.now += 5.0
        self.rois('p1', 'p2')

    def test_three_windows_over_200_ms_per_s_suspend_seg_rois(self):
        self.runtime.track_outline(ring_on(tracked(5)), 'p1')
        self.over_budget_window()
        self.over_budget_window()
        self.assertEqual(self.logs, [])
        self.assertFalse(self.runtime.suspended)
        self.over_budget_window()  # 15 s over budget
        self.assertTrue(self.runtime.suspended)
        self.assertEqual(len(self.logs), 1)
        self.assertTrue(self.logs[0].startswith('vehicle_segmentation_suspended {'), self.logs[0])
        self.assertIn('"retry_after_s":600', self.logs[0])
        self.clock.now += 10
        self.assertEqual(self.rois('p1', 'p2'), {'p1': [0], 'p2': [0]})  # no seg ROI while suspended
        self.assertIsNone(self.runtime.track_outline(tracked(5), 'p1'))  # cached outlines dropped too
        stats = self.runtime.snapshot()
        self.assertEqual((stats['state'], stats['trips'], stats['cache_tracks']), ('suspended', 1, 0))

    def test_a_window_under_budget_resets_the_count(self):
        self.over_budget_window()
        self.over_budget_window()
        self.runtime.record_cost(1.0)  # exactly 200 ms/s is allowed
        self.clock.now += 5.0
        self.rois('p1')
        self.over_budget_window()
        self.over_budget_window()
        self.assertFalse(self.runtime.suspended)

    def test_suspension_retries_with_doubling_backoff_capped_at_an_hour(self):
        retries = []
        for _ in range(5):
            for _ in range(3):
                self.over_budget_window()
            self.assertTrue(self.runtime.suspended)
            retry = int(self.logs[-1].split('"retry_after_s":')[1].split(',')[0].rstrip('}'))
            retries.append(retry)
            self.clock.now += retry - 1
            self.assertEqual(self.rois('p1')['p1'], [0])
            self.clock.now += 1
            self.assertEqual(self.rois('p1')['p1'], [0, SEG_ROI_CLASS])
            self.assertFalse(self.runtime.suspended)
            self.assertTrue(self.logs[-1].startswith('vehicle_segmentation_resumed {'), self.logs[-1])
        self.assertEqual(retries, [600, 1200, 2400, 3600, 3600])


if __name__ == '__main__':
    unittest.main()
