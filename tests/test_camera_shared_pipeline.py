"""Execute shared pipeline construction and camera lifecycle without GPU imports."""
import ast
import contextlib
import io
from pathlib import Path
import threading
import time
from types import SimpleNamespace as NS
import unittest

from camera_config import Camera, load_cameras
from camera_runtime import CameraRuntime, CameraRuntimeManager
from person_metadata import FrameMetadata, MetadataStore
from preview_mjpeg import FrameStore
from preview_timing import FrameTiming
from startup_timing import StartupTiming
from pgie_cache import configure_pgie, prepare_engine_cache, PRECISIONS
from unittest.mock import patch
from test_camera_runtime import load_preview

ROOT = Path(__file__).resolve().parents[1]


class SharedTests(unittest.TestCase):
    def manager(self, count=2, ids=None, recovery_enabled=True):
        self.now = 0.0
        cameras = [Camera((ids or list(range(count)))[i], f'camera_{i}', 1, f'Camera {i}',
                          'rtsp://private.example/video') for i in range(count)]
        runtimes = {c.source_id: CameraRuntime(c.camera_id, clock=lambda: self.now) for c in cameras}
        stores = {c.source_id: FrameStore() for c in cameras}
        manager = CameraRuntimeManager(cameras, runtimes, stores, MetadataStore(), recovery_enabled=recovery_enabled)
        manager.begin()
        manager.arm_watchdogs()
        return manager

    def test_mapping_reverse_and_metadata_slots(self):
        manager = self.manager(ids=[7, 9])
        self.assertEqual(manager.slots_by_camera, {'camera_0': 0, 'camera_1': 1})
        self.assertEqual(manager.slots_by_source, {7: 0, 9: 1})
        self.assertEqual(manager.camera_for_frame(1, 1).source_id, 9)
        with self.assertRaises(RuntimeError):
            manager.camera_for_frame(1, 0)
        with self.assertRaises(RuntimeError):
            manager.camera_for_frame(7, 7)

    def test_disabled_excluded_from_batch_and_mapping(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'cameras.yaml'
            config.write_text('cameras: [{id: disabled, floor: 1, name: D, rtsp_env: D, enabled: false}, '
                              '{id: enabled, floor: 1, name: E, rtsp_env: E}]')
            cameras = load_cameras(config, Path(directory) / 'missing.env', environ={'E': 'rtsp://private.example/video'})
        manager = CameraRuntimeManager(cameras, {0: CameraRuntime('enabled')}, {0: FrameStore()}, MetadataStore())
        self.assertEqual(manager.batch_size, 1)
        self.assertEqual(manager.slots_by_camera, {'enabled': 0})

    def test_camera_fault_keeps_healthy_generation_and_clears_only_failed_data(self):
        manager = self.manager()
        for slot in (0, 1):
            manager.observe_decoded(slot, 100)
            manager.publish(slot, 100, lambda slot=slot: manager.stores[slot].put(b'jpeg'), output=True)
            manager.metadata_store.put(NS(source_id=slot))
        before = manager.runtimes[1].snapshot()
        manager.fail_source(0, 'rtsp_error')
        self.assertEqual(manager.runtimes[1].snapshot(), before)
        self.assertIsNone(manager.stores[0].jpeg)
        self.assertEqual(manager.stores[1].jpeg, b'jpeg')
        self.assertEqual(set(manager.metadata_store.snapshot()), {1})
        self.assertFalse(manager.publish(0, 100, lambda: self.fail('stale write')))
        self.now = 1
        manager.tick()
        self.assertEqual(manager.runtimes[0].snapshot().generation, 2)
        self.assertEqual(manager.runtimes[0].snapshot().state, 'reconnecting')
        self.assertEqual(manager.runtimes[1].snapshot().generation, 1)
        self.assertFalse(manager.publish(0, 100, lambda: self.fail('old PTS')))
        for i in range(30):
            self.now += .1
            manager.observe_decoded(0, 200 + i)
            manager.publish(0, 200 + i, lambda: manager.stores[0].put(b'new'), output=True)
        self.assertEqual(manager.runtimes[0].snapshot().state, 'online')
        self.assertEqual(manager.runtimes[0].snapshot().reconnect_count, 1)
        self.assertEqual(manager.stores[0].jpeg, b'new')
        self.assertFalse(manager.stores[0].closed)

    def test_watchdog_is_camera_local_and_pts_memory_bounded(self):
        manager = self.manager()
        self.now = 31
        manager.observe_decoded(1, 1)
        manager.publish(1, 1, lambda: None, output=True)
        manager.tick()
        self.assertEqual(manager.runtimes[0].snapshot().state, 'offline')
        self.assertEqual(manager.runtimes[1].snapshot().generation, 1)
        self.assertEqual(manager.runtimes[0].snapshot().last_error, 'no_frames')
        for pts in range(1000):
            manager.observe_decoded(1, pts)
        self.assertLessEqual(len(manager._frames[1]), 512)
        self.assertFalse(manager.observe_decoded(1, None))

    def test_reused_pts_cannot_publish_metadata_from_old_generation(self):
        manager = self.manager()
        manager.observe_decoded(0, 100)
        manager.fail_source(0, 'rtsp_error')
        self.now = 1
        manager.tick()
        manager.observe_decoded(0, 100)
        self.assertFalse(manager.publish(0, 100, lambda: self.fail('old metadata'), generation=1))
        self.assertTrue(manager.publish(0, 100, lambda: None, generation=2))

    def test_native_source_properties_without_child_configuration(self):
        class Child:
            def __init__(self, factory):
                self.factory, self.props, self.callbacks = factory, {}, {}
            def get_factory(self):
                return NS(get_name=lambda: self.factory)
            def get_name(self):
                return self.factory
            def set_property(self, name, value):
                self.props[name] = value
            def get_property(self, name):
                return self.props[name]
            def find_property(self, name):
                return name
            def connect(self, name, callback):
                self.callbacks[name] = callback

        class Bin(Child):
            def iterate_elements(self):
                return NS(next=lambda: ('DONE', None))

        decoder = Bin('nvurisrcbin')
        created, failures = [], []
        def create(index, uri, fail, source_factory, **options):
            created.append(source_factory)
            return NS(get_by_name=lambda name: decoder)
        namespace = load_preview({'create_preview_source_bin'}, {
            'create_source_bin': create,
            'Gst': NS(Element=Child, ChildProxy=Bin, Bin=Bin, IteratorResult=NS(OK='OK', RESYNC='RESYNC')),
            'GstRtsp': NS(RTSPLowerTrans=NS(TCP=4))})
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            namespace['create_preview_source_bin'](0, 'private-value', 1000, failures.append)
        self.assertEqual(created, ['nvurisrcbin'])
        self.assertEqual(decoder.props['rtsp-reconnect-attempts'], -1)
        self.assertEqual(decoder.props['rtsp-reconnect-interval'], 10)
        self.assertEqual(decoder.props['select-rtp-protocol'], 4)
        self.assertTrue(decoder.props['async-handling'])
        self.assertTrue(decoder.props['disable-audio'])
        self.assertTrue(decoder.props['drop-on-latency'])
        self.assertTrue(decoder.props['low-latency-mode'])
        self.assertEqual(decoder.props['latency'], 1000)
        self.assertEqual(decoder.props['num-extra-surfaces'], 4)
        self.assertNotIn('child-added', decoder.callbacks)
        observations = []
        namespace['create_preview_source_bin'](0, 'private-value', 1000, failures.append,
                                              observer=lambda kind, child: observations.append((kind, child)))
        # Existing/reconnected children are observed without overriding native configuration.
        for factory in ('rtspsrc', 'nvv4l2decoder'):
            child = Child(factory)
            decoder.callbacks['child-added'](decoder, child, factory)
            self.assertEqual(observations[-1], ('child', child))
            self.assertEqual(child.props, {})
            self.assertEqual(child.callbacks, {})
        self.assertEqual(failures, [])
        self.assertNotIn('private-value', captured.getvalue())

    def test_output_branch_cannot_hold_preroll_and_jpeg_failure_keeps_flow(self):
        elements, received, failures = {}, [], []
        class Element:
            def __init__(self, factory, name):
                self.props, self.callbacks = {}, {}
                elements[name] = self
            def set_property(self, name, value):
                self.props[name] = value
            def connect(self, name, callback):
                self.callbacks[name] = callback
            def add(self, element):
                pass
            def add_pad(self, pad):
                return True
            def get_static_pad(self, name):
                return name
            def link(self, element):
                return True
            def set_state(self, state):
                pass
        namespace = load_preview({'build_output'}, {
            'Gst': NS(Bin=NS(new=lambda name: Element('bin', name)),
                      Caps=NS(from_string=lambda value: value),
                      GhostPad=NS(new=lambda *args: 'pad'), State=NS(NULL='NULL'),
                      FlowReturn=NS(OK='OK', EOS='EOS'), MapFlags=NS(READ=1)),
            'make_element': Element})
        store = NS(put_frame=lambda data, pts: received.append((data, pts)))
        namespace['build_output'](0, NS(jpeg_quality=80), store, failures.append)
        self.assertFalse(elements['osd-0'].props['display-mask'])
        namespace['build_output'](1, NS(jpeg_quality=80), store, failures.append, display_mask=True)
        self.assertTrue(elements['osd-1'].props['display-mask'])
        self.assertEqual(elements['queue-0'].props['leaky'], 2)
        self.assertEqual(elements['queue-0'].props['max-size-buffers'], 1)
        self.assertFalse(elements['jpeg-sink-0'].props['async'])
        self.assertFalse(elements['jpeg-sink-0'].props['sync'])
        buffer = NS(pts=100, map=lambda flags: (True, NS(data=b'jpeg')), unmap=lambda info: None)
        appsink = NS(emit=lambda signal: NS(get_buffer=lambda: buffer))
        callback = elements['jpeg-sink-0'].callbacks['new-sample']
        self.assertEqual(callback(appsink), 'OK')
        self.assertEqual(received, [(b'jpeg', 100)])
        buffer.map = lambda flags: (False, None)
        self.assertEqual(callback(appsink), 'OK')
        self.assertEqual(failures, ['JPEG output failed; details hidden'])

    def silence(self, manager, slot, decoded_first=False):
        """Two frameless attempts for one slot (no_frames at 30 s and 61 s) while the other decodes.

        Stops at 62 s: the second backoff (2 s) has not elapsed, so no tick has acted on the
        due slot yet; the caller advances past 63 s.

        decoded_first: the slot decodes for 1 s first (no_frames at 11 s, then at 42 s and 74 s);
        stops at 74.5 s, before the third backoff (5 s) ends. Rebuild needs a frame since start.
        """
        other = 1 - slot

        def decode(decoding):
            self.now = round(self.now + .1, 1)
            pts = int(self.now * 1000)
            for s in decoding:
                manager.observe_decoded(s, pts)
                if decoded_first:  # keep output flowing so the long timeline stays online
                    manager.publish(s, pts, lambda: None, output=True)
            manager.tick()

        if decoded_first:
            for _ in range(10):
                decode((slot, other))
        for _ in range(735 if decoded_first else 620):
            decode((other,))
        self.assertEqual(manager.runtimes[other].snapshot().state, 'online')
        status = manager.runtimes[slot].snapshot()
        self.assertEqual((status.state, status.last_error, status.generation),
                         ('offline', 'no_frames', 3 if decoded_first else 2))

    def test_tick_returns_only_due_slots_and_leaves_them_offline(self):
        manager = self.manager()
        self.silence(manager, 0)
        self.now += 5
        self.assertEqual(manager.tick(), [0])
        self.assertEqual(manager.tick(), [0])  # still offline until the caller resets it
        self.assertEqual(manager.runtimes[0].snapshot().generation, 2)
        self.assertEqual(manager.recovered(0), 3)
        self.assertEqual(manager.runtimes[0].snapshot().recovery_count, 1)
        self.assertEqual(manager.tick(), [])
        manager.recovery_failed(0)
        self.assertEqual(manager.runtimes[0].snapshot().last_error, 'pipeline_error')

    def test_manager_recovery_mode_follows_each_slot(self):
        manager = self.manager()
        self.silence(manager, 0, decoded_first=True)
        self.now += 5
        self.assertEqual([manager.recovery_mode(slot) for slot in (0, 1)], ['reset', 'reset'])
        self.assertEqual(manager.tick(), [0])
        manager.recovered(0)
        self.assertEqual([manager.recovery_mode(slot) for slot in (0, 1)], ['rebuild', 'reset'])

    def test_abandoned_slot_gets_only_ordinary_retries(self):
        manager = self.manager()
        self.silence(manager, 0)
        self.now += 5
        manager.abandon_recovery(0)
        self.assertEqual(manager.tick(), [])
        status = manager.runtimes[0].snapshot()
        self.assertEqual((status.state, status.generation, status.recovery_count), ('reconnecting', 3, 0))

    def test_tick_without_recovery_keeps_observation_only_retries(self):
        manager = self.manager(recovery_enabled=False)
        self.silence(manager, 0)
        self.now += 5
        self.assertEqual(manager.tick(), [])
        status = manager.runtimes[0].snapshot()
        self.assertEqual((status.state, status.generation, status.recovery_count), ('reconnecting', 3, 0))

    def test_streams_schema_and_http_response_preserved(self):
        import json
        from http.server import BaseHTTPRequestHandler
        from urllib.parse import urlsplit
        manager = self.manager()
        tree = ast.parse((ROOT / 'preview.py').read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'stream_snapshot')
        namespace = {'args': NS(cameras=list(manager.cameras_by_slot.values())),
                     'runtime_statuses': manager.runtimes, 'runtime_session': 'a' * 32}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'preview.py', 'exec'), namespace)
        snapshot = namespace['stream_snapshot']()
        self.assertEqual(set(snapshot[0]), {'id', 'camera_id', 'floor', 'name', 'format', 'status', 'url',
                                            'runtime', 'runtime_session', 'metadata_path'})
        self.assertEqual([s['url'] for s in snapshot], ['/mjpeg/source0', '/mjpeg/source1'])
        self.assertEqual(set(snapshot[0]['runtime']), {'camera_id', 'state', 'fps', 'last_frame_at',
                         'last_frame_age', 'last_error', 'reconnect_count', 'generation', 'output_frame_age', 'error_reason',
                         'recovery_count'})
        handler_class = load_preview({'PreviewHandler'}, {'BaseHTTPRequestHandler': BaseHTTPRequestHandler,
                                                        'urlsplit': urlsplit, 'json': json})['PreviewHandler']
        handler = handler_class.__new__(handler_class)
        handler.path = '/streams.json'
        handler.server = NS(mjpeg_streams={}, stream_snapshot=namespace['stream_snapshot'])
        handler.wfile = io.BytesIO()
        handler.send_response = lambda code: self.assertEqual(code, 200)
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        handler.do_GET()
        self.assertEqual(json.loads(handler.wfile.getvalue()), snapshot)
        self.assertNotIn('private', handler.wfile.getvalue().decode())

    def test_constructs_one_gpu_chain_with_n_mux_and_demux_pads(self):
        for count in (1, 2, 4, 8, 25):
            with self.subTest(count=count):
                self.construct(count)

    def test_vehicle_option_adds_one_camera_selected_detector_before_same_tracker(self):
        self.construct(4, vehicle_enabled=True)

    def test_segmentation_off_keeps_the_vehicle_pipeline_unchanged(self):
        run = self.construct(3, vehicle_enabled=True)
        self.assertNotIn('vehicle-segmentation', run['elements'])
        self.assertEqual(run['display_masks'], {0: False, 1: False, 2: False})
        self.assertEqual(self.probe_steps(run, 'vehicle-inference', 'src'), ['observe_deployment', 'prepare_tracks'])
        self.assertNotIn('vehicle_segmentation', run['captured'])

    def test_segmentation_inserts_seg_after_the_detector_and_moves_track_preparation(self):
        run = self.construct(3, vehicle_enabled=True, segmenter_enabled=True)
        self.assertEqual(run['elements']['vehicle-segmentation'].props['batch-size'], 2)
        for link in (('vehicle-inference', 'src', 'vehicle-segmentation', 'sink'),
                     ('vehicle-segmentation', 'src', 'vehicle-tracker', 'sink')):
            self.assertIn(link, run['links'])
        self.assertNotIn(('vehicle-inference', 'src', 'vehicle-tracker', 'sink'), run['links'])
        self.assertEqual(self.probe_steps(run, 'vehicle-inference', 'sink'), ['add_rois'])
        self.assertEqual(self.probe_steps(run, 'vehicle-inference', 'src'), ['observe_deployment'])
        self.assertEqual(self.probe_steps(run, 'vehicle-segmentation', 'src'), ['attach_masks', 'prepare_tracks'])
        self.assertEqual(run['display_masks'], {0: True, 1: True, 2: False})  # camera_2 is not a parking camera
        self.assertIn('vehicle_segmentation graft=unknown', run['captured'])

    def test_segmentation_requires_the_vehicle_detector(self):
        run = self.construct(2, segmenter_enabled=True)
        self.assertNotIn('vehicle-segmentation', run['elements'])
        self.assertEqual(run['display_masks'], {0: False, 1: False})

    def probe_steps(self, run, owner, pad):
        steps = [args[2] for args in run['callbacks'][(owner, pad)] if args[1].__name__ == 'vehicle_probe']
        self.assertEqual(len(steps), 1)
        return [step.__name__ for step in steps[0]]

    def test_detached_vehicle_box_is_thin_with_an_outline_and_text_stays(self):
        from vehicle_detection import vehicle_outline
        from person_metadata import vehicle_from_object
        namespace = load_preview({'detach_vehicle'}, {'vehicle_from_object': vehicle_from_object,
                                                     'vehicle_outline': vehicle_outline})
        colors = []
        def tracked():
            return NS(object_id=5, class_id=100, confidence=.9, tracker_confidence=.8,
                      rect_params=NS(left=10.0, top=20.0, width=30.0, height=40.0, border_width=3,
                                     border_color=NS(set=lambda *rgba: colors.append(rgba))),
                      text_params=NS(display_text=None))
        outline = ((10.0, 20.0), (40.0, 20.0), (40.0, 60.0), (10.0, 60.0))
        for width in (0, 1, 3):
            obj = tracked()
            vehicle = namespace['detach_vehicle'](obj, 'camera_0', 'now', 2, outline, width)
            self.assertEqual((obj.rect_params.border_width, obj.text_params.display_text), (width, 'Vehicle'))
            self.assertEqual((vehicle.outline, vehicle.class_id, vehicle.track_id), (outline, 2, 5))
        obj = tracked()
        self.assertIsNone(namespace['detach_vehicle'](obj, 'camera_0', 'now', 2, None, 0).outline)
        self.assertEqual(obj.rect_params.border_width, 3)  # no outline: the usual thick box
        self.assertEqual(set(colors), {(1.0, 0.65, 0.0, 1.0)})

    def test_diagnostics_installs_mux_and_gpu_stage_probes_without_source_state_waits(self):
        with patch('startup_timing.gpu_memory', return_value=None):
            self.construct(2, diagnostics=True)
            self.construct(3, diagnostics=True, vehicle_enabled=True)

    def test_demux_branch_drops_eos_so_a_rebuilt_source_can_resume_output(self):
        run = self.construct(2)
        for index in range(2):
            downstream = [args for args in run['callbacks'][('stream-demuxer', f'src_{index}')] if args[0] == 2]
            self.assertEqual(len(downstream), 1)  # one EVENT_DOWNSTREAM probe per demux src pad
            _mask, probe, *extra = downstream[0]
            self.assertEqual(probe(None, NS(get_event=lambda: NS(type='EOS')), *extra), 'DROP')
            self.assertEqual(probe(None, NS(get_event=lambda: NS(type='CAPS')), *extra), 'OK')
            self.assertEqual(probe(None, NS(get_event=lambda: None), *extra), 'OK')
        # Source EOS keeps failing the camera before the mux, as before.
        _mask, probe, *extra = [args for args in run['callbacks'][('source-bin-0', 'src')] if args[0] == 2][0]
        self.assertEqual(probe(None, NS(get_event=lambda: NS(type='EOS')), *extra), 'DROP')
        self.assertEqual(run['manager'].runtimes[0].snapshot().last_error, 'source_eos')

    def test_control_loop_resets_only_the_due_source_bin_in_place(self):
        run = self.construct(2, reset={'slot': 0, 'ticks': 2})
        source, other = run['elements']['source-bin-0'], run['elements']['source-bin-1']
        self.assertEqual((source.states, source.synced, other.states, other.synced), (['NULL'], 1, [], 0))
        self.assertEqual(self.rebuild_ops(run), [  # flush_stop: a kicked (flushing) mux pad must accept the reset stream
            ('set_state', 'source-bin-0', 'NULL'), ('event', 'stream-muxer', 'sink_0', ('flush_stop', False)),
            ('sync', 'source-bin-0')])
        self.assertEqual(run['retargets'], [])  # the plugin kept its pad, so the ghost link stays
        status = run['manager'].runtimes[0].snapshot()
        self.assertEqual((status.state, status.generation, status.recovery_count), ('reconnecting', 3, 1))
        self.assertIn('source_reset {"camera_id":"camera_0","source_id":0,"generation":3,"recovery_count":1}',
                      run['captured'])
        self.assertEqual(run['pops'], 2)

    def test_control_loop_clears_ghost_target_only_when_plugin_dropped_its_pad(self):
        run = self.construct(2, reset={'slot': 1, 'ticks': 2, 'pad_parent': None})
        self.assertEqual(run['retargets'], [('source-bin-1', None)])
        self.assertEqual(run['elements']['source-bin-1'].states, ['NULL'])

    def test_control_loop_continues_when_source_reset_fails(self):
        run = self.construct(2, reset={'slot': 0, 'ticks': 3, 'result': 'FAILURE'})
        source = run['elements']['source-bin-0']
        self.assertEqual((source.states, source.synced), (['NULL'], 0))
        status = run['manager'].runtimes[0].snapshot()
        # The next tick resumes ordinary observation attempts; the failed reset only deferred recovery.
        self.assertEqual((status.state, status.last_error, status.generation, status.recovery_count),
                         ('reconnecting', 'pipeline_error', 3, 0))
        self.assertIn('source_reset_failed {"camera_id":"camera_0","source_id":0,"stage":"null"}', run['captured'])
        self.assertEqual(run['pops'], 3)
        self.assertEqual(run['elements']['preview-detection'].states, ['PLAYING', 'NULL'])

    REBUILD = ('set_state', 'get_state', 'event', 'unlink', 'release', 'remove', 'create', 'add', 'request', 'link', 'sync')

    def rebuild_ops(self, run, start=0):
        """Recovery operations after the pipeline went PLAYING, in order."""
        ops = run['ops']
        playing = ops.index(('set_state', 'preview-detection', 'PLAYING'))
        stopped = ops.index(('set_state', 'preview-detection', 'NULL'))
        return [op for op in ops[playing + 1:stopped] if op[0] in self.REBUILD][start:]

    def test_control_loop_rebuilds_the_source_on_its_second_recovery(self):
        # 5 s per tick: reset at the first due tick, no_frames 30 s later, rebuild once 120 s have passed.
        run = self.construct(2, reset={'slot': 0, 'ticks': 36, 'step': 5, 'decoded_first': True})
        bins = [e for e in run['all_elements'] if e.name == 'source-bin-0']
        self.assertEqual(len(bins), 2)  # startup bin, rebuilt bin
        old, new = bins
        self.assertEqual(self.rebuild_ops(run), [
            ('set_state', 'source-bin-0', 'NULL'), ('event', 'stream-muxer', 'sink_0', ('flush_stop', False)),
            ('sync', 'source-bin-0'),  # recovery 1: in-place reset
            ('set_state', 'source-bin-0', 'NULL'),  # recovery 2: rebuild
            ('event', 'stream-muxer', 'sink_0', ('flush_stop', False)),
            ('unlink', 'source-bin-0', 'stream-muxer', 'sink_0'),
            ('release', 'stream-muxer', 'sink_0'),
            ('remove', 'preview-detection', 'source-bin-0'),
            ('create', 0),
            ('add', 'preview-detection', 'source-bin-0'),
            ('request', 'stream-muxer', 'sink_0'),
            ('link', 'source-bin-0', 'sink_0'),
            ('sync', 'source-bin-0')])
        self.assertEqual((old.states, old.synced, new.states, new.synced), (['NULL', 'NULL'], 1, [], 1))
        self.assertEqual(run['elements']['source-bin-1'].states, [])
        self.assertEqual(run['probes'].count(('source-bin-0', 'src')), run['probes'].count(('source-bin-1', 'src')) * 2)
        self.assertEqual([r for r in run['released'] if r == ('stream-muxer', 'sink_0')], [('stream-muxer', 'sink_0')] * 2)
        self.assertEqual(run['released'].count(('stream-muxer', 'sink_1')), 1)
        status = run['manager'].runtimes[0].snapshot()
        self.assertEqual((status.state, status.recovery_count), ('reconnecting', 2))
        self.assertEqual(run['manager'].runtimes[1].snapshot().state, 'degraded')  # 0.2 fps in the loop, never failed
        lines = run['captured'].splitlines()
        self.assertEqual(sum(l.startswith('source_reset ') for l in lines), 1)
        self.assertIn('source_rebuild {"camera_id":"camera_0","source_id":0,"generation":%d,"recovery_count":2}'
                      % status.generation, lines)

    def test_control_loop_retries_rebuild_from_creation_after_a_request_failure(self):
        # The second sink_0 request (first rebuild) fails; the next rebuild 600 s later starts at creation.
        run = self.construct(2, reset={'slot': 0, 'ticks': 170, 'step': 5, 'decoded_first': True, 'fail_request': ('sink_0', 2)})
        ops = self.rebuild_ops(run, start=3)
        self.assertEqual(ops[:9], [
            ('set_state', 'source-bin-0', 'NULL'), ('event', 'stream-muxer', 'sink_0', ('flush_stop', False)),
            ('unlink', 'source-bin-0', 'stream-muxer', 'sink_0'), ('release', 'stream-muxer', 'sink_0'),
            ('remove', 'preview-detection', 'source-bin-0'), ('create', 0), ('add', 'preview-detection', 'source-bin-0'),
            ('request', 'stream-muxer', 'sink_0'),
            ('remove', 'preview-detection', 'source-bin-0')])  # the unlinked new bin leaves the pipeline
        self.assertEqual(ops[9:], [('create', 0), ('add', 'preview-detection', 'source-bin-0'),
                                   ('request', 'stream-muxer', 'sink_0'), ('link', 'source-bin-0', 'sink_0'),
                                   ('sync', 'source-bin-0')])
        lines = run['captured'].splitlines()
        self.assertIn('source_rebuild_failed {"camera_id":"camera_0","source_id":0,"stage":"request"}', lines)
        self.assertEqual(sum(l.startswith('source_rebuild ') for l in lines), 1)
        self.assertEqual(run['released'].count(('stream-muxer', 'sink_0')), 2)
        self.assertEqual(run['manager'].runtimes[0].snapshot().recovery_count, 2)  # one reset, one rebuild

    def test_control_loop_gives_up_a_rebuild_whose_bin_never_reaches_null(self):
        run = self.construct(2, reset={'slot': 0, 'ticks': 36, 'step': 5, 'decoded_first': True, 'result': 'ASYNC',
                                       'get_state': ('ASYNC', 'READY', 'NULL')})
        self.assertEqual(self.rebuild_ops(run)[3:], [('set_state', 'source-bin-0', 'NULL'),
                                                     ('get_state', 'source-bin-0', 5000)])
        self.assertIn('source_rebuild_failed {"camera_id":"camera_0","source_id":0,"stage":"null"}',
                      run['captured'].splitlines())
        self.assertEqual(run['released'].count(('stream-muxer', 'sink_0')), 1)

    def test_control_loop_survives_an_exception_inside_a_rebuild(self):
        run = self.construct(2, reset={'slot': 0, 'ticks': 36, 'step': 5, 'decoded_first': True, 'raise_release': True})
        self.assertEqual(run['pops'], 36)  # the loop kept polling after the failed rebuild
        self.assertEqual(self.rebuild_ops(run)[3:], [
            ('set_state', 'source-bin-0', 'NULL'), ('event', 'stream-muxer', 'sink_0', ('flush_stop', False)),
            ('unlink', 'source-bin-0', 'stream-muxer', 'sink_0'), ('release', 'stream-muxer', 'sink_0')])
        lines = run['captured'].splitlines()
        self.assertIn('source_recovery_error {"camera_id":"camera_0","source_id":0,"mode":"rebuild"}', lines)
        self.assertFalse(any(l.startswith('source_rebuild') for l in lines))
        self.assertNotIn('private', run['captured'])
        status = run['manager'].runtimes[0].snapshot()
        self.assertEqual((status.last_error, status.recovery_count), ('pipeline_error', 1))
        self.assertEqual(run['elements']['preview-detection'].states, ['PLAYING', 'NULL'])
        self.assertEqual(run['elements']['source-bin-1'].states, [])

    def test_control_loop_keeps_resetting_in_place_when_rebuild_is_disabled(self):
        run = self.construct(2, reset={'slot': 0, 'ticks': 36, 'step': 5, 'decoded_first': True}, rebuild_enabled=False)
        self.assertEqual(self.rebuild_ops(run), [('set_state', 'source-bin-0', 'NULL'),
                                                 ('event', 'stream-muxer', 'sink_0', ('flush_stop', False)),
                                                 ('sync', 'source-bin-0')] * 2)
        self.assertEqual(run['manager'].recovery_mode(0), 'rebuild')
        self.assertEqual(sum(l.startswith('source_reset ') for l in run['captured'].splitlines()), 2)
        self.assertNotIn('source_rebuild', run['captured'])

    def test_recovery_watchdog_constants_match_the_design(self):
        tree = ast.parse((ROOT / 'preview.py').read_text())
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '').startswith('RECOVERY_')}
        self.assertEqual(values, {'RECOVERY_HUNG_AFTER': 30, 'RECOVERY_KICK_WAIT': 10, 'RECOVERY_JOIN_TIMEOUT': 5})

    def test_hung_reset_is_kicked_and_resumes_while_other_cameras_keep_updating(self):
        # 5 s per tick: reset starts at tick 1 and hangs in NULL; hung at +30 s (tick 7), released at tick 8.
        run = self.construct(2, reset={'slot': 0, 'ticks': 10, 'step': 5, 'hang': 8, 'other_stops_at': 2})
        self.assertEqual(self.rebuild_ops(run), [
            ('set_state', 'source-bin-0', 'NULL'),
            ('event', 'stream-muxer', 'sink_0', ('flush_start',)),  # kick from the control loop
            ('event', 'stream-muxer', 'sink_0', ('flush_stop', False)), ('sync', 'source-bin-0')])
        self.assertEqual(run['dumps'], [{'all_threads': True}])
        lines = run['captured'].splitlines()
        recovery = [l for l in lines if l.startswith('source_')]
        self.assertEqual(recovery, [
            'source_recovery_hung {"camera_id":"camera_0","source_id":0,"mode":"reset","seconds":30.0}',
            'source_reset {"camera_id":"camera_0","source_id":0,"generation":3,"recovery_count":1}',
            'source_recovery_unblocked {"camera_id":"camera_0","source_id":0,"mode":"reset","seconds":35.0}'])
        self.assertIn('offline', run['other_states'][:6])  # the other camera went offline while the reset hung
        self.assertEqual(run['pops'], 10)
        self.assertEqual(len(run['workers']), 1)

    def test_abandoned_reset_leaves_the_slot_to_ordinary_retries_and_frees_other_recoveries(self):
        run = self.construct(2, reset={'slot': 0, 'ticks': 45, 'step': 5, 'hang': None, 'other_stops_at': 2})
        lines = [l for l in run['captured'].splitlines() if l.startswith('source_')]
        self.assertEqual(lines[:2], [
            'source_recovery_hung {"camera_id":"camera_0","source_id":0,"mode":"reset","seconds":30.0}',
            'source_recovery_abandoned {"camera_id":"camera_0","source_id":0,"mode":"reset","seconds":40.0}'])
        self.assertTrue(lines[2].startswith('source_reset {"camera_id":"camera_1","source_id":1,'))
        self.assertEqual(len(lines), 3)
        self.assertEqual(run['elements']['source-bin-0'].states, ['NULL'])  # never recovered again
        status = run['manager'].runtimes[0].snapshot()
        self.assertEqual(status.recovery_count, 0)
        self.assertGreater(status.generation, 3)  # ordinary observation attempts went on
        self.assertEqual(run['manager'].runtimes[1].snapshot().recovery_count, 1)
        self.assertTrue(run['workers'][0].is_alive())  # still stuck; abandoned, not joined
        self.assertEqual(run['pops'], 45)
        self.assertNotIn('private', run['captured'])

    def test_shutdown_waits_only_briefly_for_a_hung_recovery(self):
        run = self.construct(2, reset={'slot': 0, 'ticks': 8, 'step': 5, 'hang': None})
        self.assertIn('source_recovery_hung', run['captured'])
        self.assertNotIn('source_recovery_abandoned', run['captured'])
        self.assertTrue(run['workers'][0].is_alive())
        self.assertEqual(run['elements']['preview-detection'].states, ['PLAYING', 'NULL'])

    def construct(self, count, diagnostics=False, vehicle_enabled=False, reset=None, rebuild_enabled=True,
                  segmenter_enabled=False):
        manager = self.manager(count)
        test = self  # the fake bus below runs inside Element methods, where self is the element
        elements, released, links, probes, retargets, callbacks, ops = [], [], [], [], [], {}, []
        display_masks = {}
        pops = [0]
        requests = {}
        workers, dumps, other_states = [], [], []
        shutdown = threading.Event()
        hang = None
        if reset is not None and 'hang' in reset:
            # source-bin-0's first NULL blocks until pop `hang` (None: never), like a stuck native call.
            hang = {'entered': threading.Event(), 'release': threading.Event(), 'release_at': reset['hang']}

            def release_hung_worker():
                hang['release'].set()
                with contextlib.redirect_stdout(io.StringIO()):
                    settle()
            self.addCleanup(release_hung_worker)

        class Thread(threading.Thread):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                workers.append(self)

        def settle():
            """Recovery workers finish before test time moves on, except one parked in a hung NULL."""
            for worker in workers:
                while worker.is_alive() and not (hang and hang['entered'].is_set() and not hang['release'].is_set()):
                    worker.join(.01)
        if reset is None:
            shutdown.set()
        else:
            self.silence(manager, reset['slot'], decoded_first=reset.get('decoded_first', False))
            self.now += 5
            other = 1 - reset['slot']

        class Pad:
            def __init__(self, owner, name):
                self.owner, self.name = owner, name
            def add_probe(self, *args):
                probes.append((self.owner, self.name))
                callbacks.setdefault((self.owner, self.name), []).append(args)
            def link(self, pad):
                links.append((self.owner, self.name, pad.owner, pad.name))
                ops.append(('link', self.owner, pad.name))
                return 'OK'
            def unlink(self, pad):
                ops.append(('unlink', self.owner, pad.owner, pad.name))
                return True
            def send_event(self, event):
                ops.append(('event', self.owner, self.name, event))
                return True
            def get_target(self):
                return NS(get_parent=lambda: (reset or {}).get('pad_parent', 'decoder'))
            def set_target(self, target):
                retargets.append((self.owner, target))
                return True

        class Element:
            def __init__(self, factory, name):
                self.factory, self.name, self.props, self.requests, self.states = factory, name, {}, [], []
                elements.append(self)
            def set_property(self, name, value):
                self.props[name] = value
            def get_property(self, name):
                return self.props[name]
            def get_name(self):
                return self.name
            def get_static_pad(self, name):
                return Pad(self.name, name)
            def request_pad_simple(self, name):
                self.requests.append(name)
                ops.append(('request', self.name, name))
                requests[name] = requests.get(name, 0) + 1
                if reset is not None and requests[name] == reset.get('fail_request', (None, 0))[1] \
                        and name == reset['fail_request'][0]:
                    return None
                return Pad(self.name, name)
            def release_request_pad(self, pad):
                ops.append(('release', self.name, pad.name))
                if reset is not None and reset.get('raise_release') and not any(op[0] == 'raised' for op in ops):
                    ops.append(('raised', self.name, pad.name))  # once: shutdown releases still work
                    raise RuntimeError('private plugin failure text')
                released.append((self.name, pad.name))
            def add(self, element):
                ops.append(('add', self.name, element.name))
            def remove(self, element):
                ops.append(('remove', self.name, element.name))
                return True
            def get_state(self, timeout):
                ops.append(('get_state', self.name, timeout))
                return (reset or {}).get('get_state', ('SUCCESS', 'NULL', 'VOID_PENDING'))
            def link(self, element):
                links.append((self.name, 'src', element.name, 'sink'))
                return True
            def get_by_name(self, name):
                return self
            def get_bus(self):
                def timed_pop_filtered(*args):
                    pops[0] += 1
                    if hang and pops[0] == hang['release_at']:
                        hang['release'].set()
                    settle()
                    if reset is not None and reset.get('step'):
                        # Time passes between ticks; the healthy camera keeps decoding.
                        test.now = round(test.now + reset['step'], 1)
                        if pops[0] < reset.get('other_stops_at', float('inf')):
                            pts = int(test.now * 1000)
                            manager.observe_decoded(other, pts)
                            manager.publish(other, pts, lambda: None, output=True)
                        other_states.append(manager.runtimes[other].snapshot().state)
                    if reset is not None and pops[0] >= reset['ticks']:
                        shutdown.set()
                    return None
                return NS(set_sync_handler=lambda *args: None, timed_pop_filtered=timed_pop_filtered)
            def set_state(self, state):
                self.states.append(state)
                ops.append(('set_state', self.name, state))
                if self.factory == 'source' and reset is not None:
                    if hang and self.name == 'source-bin-0' and state == 'NULL' and not hang['entered'].is_set():
                        hang['entered'].set()
                        hang['release'].wait(10)
                    return reset.get('result', 'SUCCESS')
                return 'SUCCESS'
            synced = 0
            def sync_state_with_parent(self):
                self.synced += 1
                ops.append(('sync', self.name))
                return True

        namespace = {'Gst': NS(Pipeline=NS(new=lambda name: Element('pipeline', name)),
                         Caps=NS(from_string=lambda caps: caps), PadProbeType=NS(BUFFER=1, EVENT_DOWNSTREAM=2),
                         PadLinkReturn=NS(OK='OK'), State=NS(PLAYING='PLAYING', NULL='NULL'), MSECOND=1,
                         EventType=NS(EOS='EOS'), PadProbeReturn=NS(OK='OK', DROP='DROP'), SECOND=1000,
                         Event=NS(new_flush_stop=lambda reset_time: ('flush_stop', reset_time),
                                  new_flush_start=lambda: ('flush_start',)),
                         StateChangeReturn=NS(FAILURE='FAILURE', ASYNC='ASYNC', SUCCESS='SUCCESS'),
                         MessageType=NS(ERROR=1, WARNING=2, EOS=4,
                                                ELEMENT=8, STATE_CHANGED=16), debug_set_active=lambda *args: None,
                         init=lambda *args: None), 'os': NS(environ={}),
                     'threading': NS(Event=threading.Event, Lock=threading.Lock, Thread=Thread),
                     'time': NS(monotonic=lambda: test.now, perf_counter=time.perf_counter),
                     'json': __import__('json'), 'contextlib': contextlib,
                     'faulthandler': NS(dump_traceback=lambda **kwargs: dumps.append(kwargs)),
                     'RECOVERY_HUNG_AFTER': 30, 'RECOVERY_KICK_WAIT': 10, 'RECOVERY_JOIN_TIMEOUT': .2,
                     'configure_pgie': configure_pgie,
                     'make_element': Element, 'make_tracker': lambda: Element('nvtracker', 'vehicle-tracker'),
                     'create_preview_source_bin': lambda index, *args, **kwargs: (
                         ops.append(('create', index)) or Element('source', f'source-bin-{index}')),
                     'build_output': lambda index, *args, display_mask=False: (
                         display_masks.__setitem__(index, display_mask) or Element('output', f'mjpeg-source-{index}'))}
        namespace.update(FrameTiming=FrameTiming, source_observer=lambda *args: lambda *args: None,
                         bus_observer=lambda *args: lambda *args: None,
                         InferenceDiagnostics=lambda: NS(), TRACKER_CONFIG='unused',
                         log_settings=lambda *args: None)
        from vehicle_detection import add_vehicle_rois, prepare_vehicle_tracks, attach_vehicle_masks, graft_status
        namespace.update(add_vehicle_rois=add_vehicle_rois, prepare_vehicle_tracks=prepare_vehicle_tracks,
                         attach_vehicle_masks=attach_vehicle_masks, graft_status=graft_status)
        namespace = load_preview({'run_shared_pipeline', 'CameraFrameSink'}, namespace)
        args = NS(cameras=list(manager.cameras_by_slot.values()), input=['private'] * count,
                  diagnostics=diagnostics, mux_live_source=1, rtsp_latency=1000, rtsp_drop_on_latency=False,
                  camera_offline_after=10, source_recovery_rebuild=rebuild_enabled)
        if diagnostics:
            args.startup_timing = StartupTiming(args.cameras, 0, 'test')
        captured = io.StringIO()
        import tempfile
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(captured):
            config = Path(directory) / 'config.txt'
            engine = str(Path(directory) / 'cached.engine')
            config.write_text(f'[property]\nmodel-engine-file={engine}\n')
            if vehicle_enabled:
                args.vehicle_detector = NS(cameras=frozenset({'camera_0', 'camera_1'}), config_path=str(config),
                                           car_class_id=0, vehicle_class_ids=frozenset({0}))
            if segmenter_enabled:
                seg_config = Path(directory) / 'seg.txt'
                seg_config.write_text(f'[property]\nmodel-engine-file={Path(directory) / "seg.engine"}\n')
                args.vehicle_segmenter = NS(cameras=frozenset({'camera_0', 'camera_1'}), config_path=str(seg_config),
                                            vehicle_class_ids=frozenset({0}), det_weights_sha256=None, box_width=1)
            error = namespace['run_shared_pipeline'](args, manager.metadata_store, manager.stores, manager, shutdown)
        self.assertIsNone(error)
        pipeline = next(e for e in elements if e.factory == 'pipeline')
        self.assertEqual(pipeline.states, ['PLAYING', 'NULL'])
        if reset is None:
            self.assertTrue(all(not e.states for e in elements if e.factory == 'source'))
        if diagnostics:
            for owner, pad in (('stream-muxer', 'src'), ('vehicle-tracker', 'sink'), ('vehicle-tracker', 'src')):
                self.assertIn((owner, pad), probes)
            for index in range(count):
                self.assertIn(('stream-muxer', f'sink_{index}'), probes)
                self.assertIn(('stream-demuxer', f'src_{index}'), probes)
        # PeopleNet is gone: the only nvinfer is the shared vehicle detector, if enabled.
        for factory in ('pipeline', 'nvstreammux', 'nvinfer', 'nvtracker', 'nvstreamdemux'):
            expected = int(vehicle_enabled) * (1 + int(segmenter_enabled)) if factory == 'nvinfer' else 1
            self.assertEqual(sum(e.factory == factory for e in elements), expected)
        self.assertFalse(any(e.name == 'primary-inference' for e in elements))
        mux = next(e for e in elements if e.factory == 'nvstreammux')
        demux = next(e for e in elements if e.factory == 'nvstreamdemux')
        self.assertEqual(mux.props['batch-size'], count)
        if vehicle_enabled:
            vehicle = next(e for e in elements if e.name == 'vehicle-inference')
            self.assertEqual(vehicle.props['model-engine-file'], engine)
            self.assertEqual(vehicle.props['batch-size'], 2)
            self.assertIn(('vehicle-inference', 'sink'), probes)
            self.assertIn(('vehicle-inference', 'src'), probes)
            self.assertIn(('stream-muxer', 'src', 'vehicle-inference', 'sink'), links)
            if not segmenter_enabled:
                self.assertIn(('vehicle-inference', 'src', 'vehicle-tracker', 'sink'), links)
            self.assertIn(f'vehicle cameras=2 vehicle batch-size=2', captured.getvalue())
        else:
            self.assertIn(('stream-muxer', 'src', 'vehicle-tracker', 'sink'), links)
            self.assertIn('vehicle cameras=0 vehicle batch-size=0', captured.getvalue())
        self.assertEqual(mux.props['batched-push-timeout'], 33333)
        self.assertFalse(mux.props['sync-inputs'])
        self.assertEqual(mux.requests[:count], [f'sink_{i}' for i in range(count)])  # rebuilds re-request later
        self.assertEqual(demux.requests, [f'src_{i}' for i in range(count)])
        if reset is None:
            self.assertEqual(len(released), count * 2)
        self.assertFalse(any(e.factory in ('queue', 'nvvideoconvert', 'capsfilter') for e in elements))
        self.assertEqual((mux.props['width'], mux.props['height']), (1920, 1080))
        for i in range(count):
            self.assertIn((f'source-bin-{i}', 'src', 'stream-muxer', f'sink_{i}'), links)
            self.assertIn(('stream-demuxer', f'src_{i}', f'mjpeg-source-{i}', 'sink'), links)
        self.assertIn(f'sources={count} streammux batch-size={count} vehicle', captured.getvalue())
        self.assertNotIn('private', captured.getvalue())
        return {'elements': {e.name: e for e in elements}, 'captured': captured.getvalue(), 'manager': manager,
                'retargets': retargets, 'pops': pops[0], 'callbacks': callbacks, 'ops': ops,
                'released': released, 'all_elements': elements, 'probes': probes, 'workers': workers,
                'dumps': dumps, 'other_states': other_states, 'links': links, 'display_masks': display_masks}

    def test_bus_source_error_and_eos_do_not_stop_shared_pipeline(self):
        tree = ast.parse((ROOT / 'preview.py').read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'on_message')
        calls = []
        namespace = {'Gst': NS(MessageType=NS(ERROR=1, WARNING=2, EOS=3, ELEMENT=4, STATE_CHANGED=5)),
                     'message_scope': lambda message: (message.source, 'rtsp_error' if message.source != 'all' else 'pipeline_error'),
                     'stop': lambda *args: calls.append(args),
                     'classify_bus_error': lambda *args, **kwargs: {'reason': 'rtsp_error', 'domain': 'unknown', 'code': None},
                     'manager': NS(record_error_reason=lambda *args: False),
                     'args': NS(diagnostics=False)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'preview.py', 'exec'), namespace)
        for kind in (1, 3, 4):
            namespace['on_message'](None, NS(type=kind, source='1',
                get_structure=lambda: NS(get_name=lambda: 'GstRTSPSrcTimeout')))
        self.assertEqual(calls, [('rtsp_error', True, '1'), ('source_eos', True, '1'), ('rtsp_timeout', True, '1')])
        namespace['on_message'](None, NS(type=1, source='all'))
        self.assertEqual(calls[-1], ('pipeline_error', True, 'all'))

    def test_bus_warning_names_the_validated_factory_in_diagnostics(self):
        from rtsp_diagnostics import classify_bus_error, safe_factory
        tree = ast.parse((ROOT / 'preview.py').read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'on_message')
        namespace = {'Gst': NS(MessageType=NS(ERROR=1, WARNING=2, EOS=3, ELEMENT=4, STATE_CHANGED=5)),
                     'message_scope': lambda message: (message.source, 'rtsp_error' if message.source != 'all' else 'pipeline_error'),
                     'stop': lambda *args: self.fail('warnings never stop anything'),
                     'classify_bus_error': classify_bus_error, 'safe_factory': safe_factory,
                     'manager': NS(record_error_reason=lambda *args: False),
                     'args': NS(diagnostics=True)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'preview.py', 'exec'), namespace)
        error = NS(domain='gst-core-error-quark', code=10, message='private')
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            for source, name in (('all', 'nvstreammux'), ('1', 'rtspsrc'), ('all', 'Bad Name')):
                namespace['on_message'](None, NS(type=2, source=source, parse_warning=lambda: (error, 'private'),
                                                 src=NS(get_factory=lambda: NS(get_name=lambda: name))))
        self.assertEqual(captured.getvalue().splitlines(),
                         ['pipeline_source=all: GStreamer WARNING factory=nvstreammux; details hidden',
                          'pipeline_source=1: GStreamer WARNING factory=rtspsrc; details hidden',
                          'pipeline_source=all: GStreamer WARNING factory=other; details hidden'])
        self.assertNotIn('private', captured.getvalue())

    def test_bus_ancestor_maps_source_and_output_errors_separately(self):
        tree = ast.parse((ROOT / 'preview.py').read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'message_scope')
        namespace = {'args': NS(input=[None, None])}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'preview.py', 'exec'), namespace)
        def element(name, parent=None):
            return NS(get_name=lambda: name, get_parent=lambda: parent)
        # A bin already removed from the pipeline (no parent) still resolves by name: a late
        # message from the old bin during a rebuild fails that slot instead of the whole pipeline.
        self.assertEqual(namespace['message_scope'](NS(src=element('rtsp', element('source-bin-1')))),
                         ('1', 'rtsp_error'))
        self.assertEqual(namespace['message_scope'](NS(src=element('encoder', element('mjpeg-source-1')))),
                         ('1', 'pipeline_error'))
        self.assertEqual(namespace['message_scope'](NS(src=element('vehicle-tracker'))),
                         ('all', 'pipeline_error'))


if __name__ == '__main__':
    unittest.main()
