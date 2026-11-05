"""CPU-only tests for private configuration and detached metadata."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

from camera_config import CameraConfigError, load_cameras
from person_metadata import (FrameMetadata, MetadataStore, UNTRACKED_OBJECT_ID,
                             person_from_object, utc_timestamp)
from preview_diagnostics import InferenceDiagnostics


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'cameras.yaml'
        self.env = self.root / '.env'
        self.config.write_text('''cameras:
  - {id: first, floor: 1, name: First, rtsp_env: FIRST, enabled: false}
  - {id: second, floor: 2, name: Second, rtsp_env: SECOND}
  - {id: third, floor: 3, name: Third, rtsp_env: THIRD}
''')

    def load(self, **kwargs):
        return load_cameras(self.config, self.env, environ=kwargs.pop('environ', {}), **kwargs)

    def test_env_parsing_mapping_and_environment_priority(self):
        self.env.write_text("SECOND='rtsp://user:p$word@camera.example/video'\nTHIRD=rtsp://third.example/video\n")
        cameras = self.load(environ={'THIRD': 'rtsp://override.example/video'})
        self.assertEqual([(c.source_id, c.camera_id) for c in cameras], [(0, 'second'), (1, 'third')])
        self.assertIn('p$word', cameras[0].rtsp_url)
        self.assertEqual(cameras[1].rtsp_url, 'rtsp://override.example/video')
        self.assertNotIn('rtsp://', repr(cameras))
        self.assertNotIn('rtsp', json.dumps(cameras[0].public_info()))

    def test_missing_env_names_specific_camera(self):
        with self.assertRaisesRegex(CameraConfigError, 'id=second.*SECOND'):
            self.load()
        cameras = self.load(environ={'SECOND': 'rtsp://second.example/v', 'THIRD': 'rtsp://third.example/v'})
        self.assertEqual(len(cameras), 2)
        with self.assertRaisesRegex(CameraConfigError, 'id=second.*SECOND'):
            self.load(environ={'SECOND': ''})

    def test_cli_bypasses_files_and_bad_env(self):
        self.config.unlink()
        cameras = self.load(cli_inputs=['rtsp://first.example/v', 'rtsp://second.example/v'])
        self.assertEqual([c.camera_id for c in cameras], ['cli_0', 'cli_1'])
        with self.assertRaisesRegex(CameraConfigError, 'source=0') as caught:
            self.load(cli_inputs=['rtsp://secret:password@camera.example:bad/v'])
        self.assertNotIn('password', str(caught.exception))

    def test_invalid_structure_duplicate_and_private_url(self):
        for document in ('cameras: [', 'cameras: wrong',
                         'cameras: [{id: one, floor: true, name: First, rtsp_env: ONE}]',
                         'cameras: [{id: one, floor: 1, name: First, rtsp_env: "rtsp://private-secret/v"}]',
                         'cameras: [{id: one, floor: 1, name: First, rtsp_env: ONE, rtsp: "rtsp://private-secret/v"}]',
                         'cameras: [{id: one, floor: 1, name: First, rtsp_env: ONE, enabled: false}, {id: one}]',
                         'cameras: []'):
            with self.subTest(document=document):
                self.config.write_text(document)
                with self.assertRaises(CameraConfigError) as caught:
                    self.load()
                self.assertNotIn('private-secret', str(caught.exception))

    def test_basement_floor_and_disabled_camera_without_secret(self):
        self.config.write_text("cameras: [{id: basement, floor: B1, name: Basement, rtsp_env: FIRST, enabled: false}, "
                               "{id: active, floor: B2, name: Active, rtsp_env: SECOND}]")
        cameras = self.load(environ={'SECOND': 'rtsp://camera.example/v'})
        self.assertEqual([(c.source_id, c.camera_id, c.floor) for c in cameras], [(0, 'active', 'B2')])

    def test_invalid_uri_names_camera_without_printing_value(self):
        private = 'rtsp://private-secret@camera.example:wrong/v'
        with self.assertRaisesRegex(CameraConfigError, 'id=second') as caught:
            self.load(environ={'SECOND': private})
        self.assertNotIn('private-secret', str(caught.exception))


class MetadataTests(unittest.TestCase):
    def object(self, track=17, confidence=.91):
        return NS(object_id=track, class_id=0, confidence=confidence, tracker_confidence=.8,
                  rect_params=NS(left=302, top=140, width=120, height=310))

    def test_metadata_detached_json_and_latest_empty_frame(self):
        obj = self.object()
        timestamp = utc_timestamp()
        person = person_from_object('second', timestamp, obj)
        obj.rect_params.left = 999
        record = person.to_dict()
        self.assertEqual(record['bbox']['x'], 302)
        self.assertEqual(record['class'], 'person')
        self.assertEqual(record['track_id'], 17)
        self.assertEqual(record['confidence'], .91)
        self.assertTrue(timestamp.endswith('+00:00'))
        frame = FrameMetadata(0, 'second', 10, timestamp, 1000, True, (person,))
        json.dumps(frame.to_dict(), allow_nan=False)
        store = MetadataStore()
        store.put(frame)
        store.put(FrameMetadata(0, 'second', 11, timestamp, 2000, True, ()))
        self.assertEqual(store.snapshot()[0].persons, ())
        self.assertEqual(len(store.snapshot()), 1)

    def test_untracked_and_unavailable_confidence(self):
        for confidence in (-.1, float('nan'), float('inf')):
            obj = self.object(UNTRACKED_OBJECT_ID, confidence)
            obj.tracker_confidence = -.1
            person = person_from_object('second', utc_timestamp(), obj)
            self.assertIsNone(person.track_id)
            self.assertIsNone(person.confidence)
            self.assertIsNone(person.tracker_confidence)
            json.dumps(person.to_dict(), allow_nan=False)

    def test_inference_counts_include_transient_detection_and_reset(self):
        diagnostics = InferenceDiagnostics()
        diagnostics.observe(True, [.91, .5], 0)
        diagnostics.observe(False, [], 100)
        diagnostics.observe(True, [], 200)
        result = diagnostics.snapshot()
        self.assertEqual(result['inferred_frames'], 2)
        self.assertEqual(result['non_inferred_frames'], 1)
        self.assertEqual(result['vehicle_detections'], 2)
        self.assertEqual(result['confidence_min'], .5)
        self.assertEqual(result['confidence_max'], .91)
        self.assertEqual(diagnostics.snapshot()['vehicle_detections'], 0)


class ProbeIntegrationTests(unittest.TestCase):
    """Execute the production tracker probe with synthetic linked PyDS metadata."""
    def probe_scenario(self, segmenter=None, mask=None):
        """Run the real on_batch on camera_0 (no detector) and camera_1 (one tracked car)."""
        import ast
        import threading
        import time
        from vehicle_detection import vehicle_outline
        from camera_config import Camera
        from camera_runtime import CameraRuntime, CameraRuntimeManager
        from preview_mjpeg import FrameStore
        from person_metadata import vehicle_from_object
        from vehicle_detection import VEHICLE_GIE_ID, VEHICLE_TRACKER_CLASS
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'preview.py').read_text())
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_shared_pipeline')
        functions = [node for node in run.body if isinstance(node, ast.FunctionDef)
                     and node.name in {'camera_for_frame', 'on_batch'}]
        functions += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'detach_vehicle']
        store = MetadataStore()
        frames = [NS(source_id=i, pad_index=i, frame_num=10, buf_pts=100, bInferDone=True,
                     obj_meta_list=None) for i in range(2)]
        # Leftover non-vehicle metadata (e.g. a former person detector) is removed, never published.
        stray = MetadataTests().object()
        stray.unique_component_id = 1
        color = NS(set=lambda *args: None)
        car = MetadataTests().object()
        car.unique_component_id, car.class_id = VEHICLE_GIE_ID, VEHICLE_TRACKER_CLASS
        car.rect_params.border_color = color
        car.text_params = NS()
        if mask is not None:
            car.mask_params = mask
        frames[1].obj_meta_list = NS(data=stray, next=NS(data=car, next=None))
        batch = NS(frame_meta_list=NS(data=frames[0], next=NS(data=frames[1], next=None)))
        failures, removed = [], []
        cameras = [Camera(i + 7, f'camera_{i}', i, f'Camera {i}', 'rtsp://camera.example/v') for i in range(2)]
        runtimes = {c.source_id: CameraRuntime(c.camera_id) for c in cameras}
        manager = CameraRuntimeManager(cameras, runtimes, {c.source_id: FrameStore() for c in cameras}, store)
        for runtime in runtimes.values():
            for _ in range(3):
                runtime.begin_attempt()
        for slot in range(2):
            manager.observe_decoded(slot, 100)
        namespace = {
            'Gst': NS(PadProbeReturn=NS(OK='OK'), CLOCK_TIME_NONE=(1 << 64) - 1),
            'pyds': NS(gst_buffer_get_nvds_batch_meta=lambda key: batch,
                       NvDsFrameMeta=NS(cast=lambda value: value),
                       NvDsObjectMeta=NS(cast=lambda value: value),
                       nvds_remove_obj_meta_from_frame=lambda frame, obj: removed.append(obj)),
            'cameras_by_source': {i: Camera(i + 7, f'camera_{i}', i, f'Camera {i}', 'rtsp://camera.example/v')
                                  for i in range(2)},
            'utc_timestamp': utc_timestamp, 'vehicle_from_object': vehicle_from_object,
            'FrameMetadata': FrameMetadata, 'metadata_store': store,
            'VEHICLE_GIE_ID': VEHICLE_GIE_ID, 'VEHICLE_TRACKER_CLASS': VEHICLE_TRACKER_CLASS,
            'counts': [0, 0], 'vehicle_counts': [None, None], 'vehicles_max': [0, 0],
            'counts_lock': threading.Lock(), 'metadata_samples': {}, 'args': NS(diagnostics=True),
            'pipeline_stop': threading.Event(), 'manager': manager,
            'vehicle_detector': NS(cameras={'camera_1'}, car_class_id=0),
            'fail': lambda *args: failures.append(args),
            'segmenter': segmenter, 'vehicle_outline': vehicle_outline, 'time': time,
            'mask_stats': {'cameras': {}}, 'costs': [],
        }
        namespace['record_mask_cost'] = lambda seconds, errors=0: namespace['costs'].append((seconds, errors))
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'preview.py', 'exec'), namespace)
        namespace['on_batch'](None, NS(get_buffer=lambda: 100))
        return NS(namespace=namespace, store=store, car=car, failures=failures, removed=removed, stray=stray,
                  frames=frames)

    def test_segmentation_outline_reaches_metadata_and_thins_the_box(self):
        import numpy as np
        grid = np.zeros((10, 20), bool)
        grid[2:8, 3:17] = True
        data = grid.astype(np.float32).ravel()
        mask = NS(size=data.nbytes, width=20, height=10, threshold=.5, get_mask_array=lambda: data)
        segmenter = NS(cameras=frozenset({'camera_1'}), box_width=0)
        run = self.probe_scenario(segmenter, mask)
        self.assertEqual(run.failures, [])
        vehicle = run.store.snapshot()[8].vehicles[0]
        self.assertIsNotNone(vehicle.outline)
        self.assertLessEqual(len(vehicle.outline), 32)
        self.assertEqual(run.car.rect_params.border_width, 0)
        self.assertEqual(run.car.text_params.display_text, 'Vehicle')
        self.assertEqual(run.namespace['mask_stats']['cameras'], {'camera_1': (1, 1)})
        self.assertEqual(len(run.namespace['costs']), 1)
        mask.size = 0  # this car got no mask from the seg nvinfer
        run = self.probe_scenario(segmenter, mask)
        self.assertIsNone(run.store.snapshot()[8].vehicles[0].outline)
        self.assertEqual(run.car.rect_params.border_width, 3)
        self.assertEqual(run.namespace['mask_stats']['cameras'], {'camera_1': (1, 0)})

    def test_multi_source_probe_keeps_osd_and_detaches_metadata(self):
        run = self.probe_scenario()
        namespace, store, car, failures, removed, stray, frames = (
            run.namespace, run.store, run.car, run.failures, run.removed, run.stray, run.frames)
        self.assertEqual(failures, [])
        self.assertIsNone(store.snapshot()[8].vehicles[0].outline)  # segmentation off
        self.assertEqual(car.rect_params.border_width, 3)
        self.assertEqual((namespace['costs'], namespace['mask_stats']), ([], {'cameras': {}}))
        self.assertEqual(namespace['counts'], [1, 1])
        self.assertEqual(namespace['vehicle_counts'], [None, 1])
        self.assertEqual(removed, [stray])
        self.assertEqual(store.snapshot()[8].persons, ())
        self.assertEqual(store.snapshot()[8].generation, 3)
        self.assertEqual(len(store.snapshot()[8].vehicles), 1)
        self.assertEqual(store.snapshot()[8].vehicles[0].camera_id, 'camera_1')
        self.assertEqual(store.snapshot()[8].vehicles[0].class_id, 0)
        self.assertEqual(car.text_params.display_text, 'Vehicle')
        self.assertIsNotNone(store.snapshot()[8].vehicles[0].track_id)
        self.assertTrue(store.snapshot()[8].vehicle_inference_done)
        self.assertTrue(store.snapshot()[8].inference_done)
        # Non-parking camera: no detector, so no inference is reported for it.
        self.assertFalse(store.snapshot()[7].vehicle_detection_enabled)
        self.assertFalse(store.snapshot()[7].inference_done)
        self.assertEqual(store.snapshot()[7].vehicles, ())
        self.assertEqual(namespace['metadata_samples'][1].vehicles[0].camera_id, 'camera_1')
        frames[1].pad_index = 0
        with self.assertRaises(RuntimeError):
            namespace['camera_for_frame'](frames[1])
        frames[1].pad_index = 1
        frames[1].obj_meta_list = None
        namespace['on_batch'](None, NS(get_buffer=lambda: 100))
        self.assertEqual(store.snapshot()[8].vehicles, ())


if __name__ == '__main__':
    unittest.main()
