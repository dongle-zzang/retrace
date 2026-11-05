"""CPU tests for the runtime-owned vehicle model transaction (fake nvinfer element/clock)."""
import configparser
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import vehicle_deployment as vd

VERIFY = vd.SETTLE_SECONDS + vd.MEASURE_SECONDS + 1  # load -> settle -> shortest measurement window
EXPIRE = vd.SETTLE_SECONDS + vd.VERIFY_SECONDS + 2  # load -> longest window -> verdict


class FakeElement:
    def __init__(self):
        self.properties, self.callback = [], None

    def connect(self, name, callback):
        assert name == 'model-updated'
        self.callback = callback

    def set_property(self, name, value):
        self.properties.append((name, value))


class FakeBuilder:
    """Stands in for tools/build_vehicle_engine.py: writes infer.txt, engine and build.json."""
    def __init__(self, cmd, **_kwargs):
        args = dict(zip(cmd[2::2], cmd[3::2]))
        self.job, self.base, self.returncode = Path(args['--job']), args['--base'], None
        self.terminated = False
        FakeBuilder.last = self

    def complete(self, code=0):
        config = configparser.ConfigParser(interpolation=None)
        config.read(self.base)
        engine = self.job.parent / 'engines' / 'pgie' / f'{self.job.name}.engine'
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b'engine ' + self.job.name.encode())
        config['property']['onnx-file'] = str(self.job / 'best.onnx')
        config['property']['model-engine-file'] = str(engine)
        with (self.job / 'infer.txt').open('w') as stream:
            config.write(stream, space_around_delimiters=False)
        vd.write_json(self.job / 'build.json', {'engine': vd.relative(engine), 'sha256': vd.digest(engine)})
        self.returncode = code

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated, self.returncode = True, -15

    def wait(self, timeout=None):
        return self.returncode

    kill = terminate


class ControllerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.logs = []
        for target in (patch.object(vd, 'ROOT', self.root), patch.object(vd.subprocess, 'Popen', FakeBuilder),
                       patch.object(vd, 'print', create=True, new=lambda line, **_: self.logs.append(line))):
            target.start()
            self.addCleanup(target.stop)
        self.now, self.wall = [1000.0], [5000.0]
        (self.root / 'labels.txt').write_text('person\ncar\ntruck\n')
        (self.root / 'parser.so').write_bytes(b'parser')
        (self.root / 'prod.onnx').write_bytes(b'prod')
        (self.root / 'prod.engine').write_bytes(b'prod engine')
        self.base = self.root / 'vehicle.txt'
        self.base.write_text(f'[property]\nonnx-file={self.root}/prod.onnx\nmodel-engine-file={self.root}/prod.engine\n'
                             f'labelfile-path={self.root}/labels.txt\ncustom-lib-path={self.root}/parser.so\n'
                             'batch-size=2\nmaintain-aspect-ratio=1\n\n[class-attrs-all]\npre-cluster-threshold=0.25\n')
        self.directory = self.root / '.cache' / 'vehicle-deployments'
        self.detector = NS(cameras=frozenset({'p1', 'p2'}), config_path=str(self.base), vehicle_class_ids=frozenset({1, 2}))
        # Two vehicle cameras and six unrelated CCTV cameras on the same pipeline.
        self.fps = {'p1': 10.0, 'p2': 10.0, **{c: 15.0 for c in ('lobby', 'hall', 'yard', 'gate', 'roof', 'dock')}}
        self.states = {}
        cameras = {i: NS(camera_id=c, source_id=i) for i, c in enumerate(self.fps)}
        self.manager = NS(cameras_by_slot=cameras, runtimes={
            i: NS(snapshot=lambda c=c: NS(state=self.states.get(c, 'online' if self.fps[c] else 'offline'),
                                         fps=self.fps[c]))
            for i, c in enumerate(self.fps)})
        self.frame = 0

    def controller(self, element=None, prepare=None, segmenter=None):
        self.element = element or FakeElement()
        self.configured = []
        return vd.DeploymentController(self.element, self.detector, self.manager, self.directory,
                                       clock=lambda: self.now[0], wall=lambda: self.wall[0],
                                       configure=lambda e, path, batch: self.configured.append((path, batch)),
                                       prepare=prepare, segmenter=segmenter)

    def advance(self, controller, seconds, detections=1, step=1.0, cameras=('p1', 'p2'), inferred=None):
        """Each camera delivers frames at its FPS; inferred overrides the inferred frames per step."""
        for _ in range(int(seconds / step)):
            self.now[0] += step
            self.wall[0] += step
            for camera in cameras:
                count = round(self.fps[camera] * step) if camera not in (inferred or {}) else inferred[camera]
                for index in range(count):
                    self.frame += 1
                    controller.observe(camera, self.frame, True, detections if index == 0 else 0)
            self.lease(controller)
            controller.tick()

    def lease(self, controller):
        if controller.pending and not getattr(self, 'cancelled', False):
            path = self.directory / controller.pending['id'] / 'request.json'
            vd.write_json(path, {**vd.read_json(path), 'lease_at': self.wall[0]})

    def request(self, controller, action='deploy', model='m1', job='11111111-1111-4111-8111-111111111111', **manifest):
        status = vd.read_json(self.directory / 'status.json')
        job_dir = self.directory / job
        if action == 'deploy':
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / 'best.onnx').write_bytes(b'candidate ' + job.encode())
            vd.write_json(job_dir / 'manifest.json', {'model_id': model, 'contract': status['contract'],
                                                      'sha256': vd.digest(job_dir / 'best.onnx'), **manifest})
        vd.write_json(job_dir / 'request.json', {'id': job, 'model_id': model, 'action': action,
                                                 'session': status['session'], 'contract': status['contract'],
                                                 'lease_at': self.wall[0]})
        return job_dir

    def loaded(self, controller, code=0):
        name, path = controller.element.properties[-1]
        self.assertEqual(name, 'config-file-path')
        self.element.callback(self.element, code, path)
        return path

    def deploy(self, controller, model='m1', job='11111111-1111-4111-8111-111111111111', load=True, **manifest):
        self.advance(controller, vd.BASELINE_SECONDS)
        job_dir = self.request(controller, model=model, job=job, **manifest)
        controller.tick()
        self.assertEqual(controller.state['phase'], 'building')
        FakeBuilder.last.complete()
        controller.tick()
        self.assertEqual(controller.state['phase'], 'applying')
        if load:
            self.assertEqual(self.loaded(controller), str(job_dir / 'infer.txt'))
        return job_dir

    def test_deploy_hot_updates_config_verifies_inference_and_commits(self):
        controller = self.controller()
        public = vd.public_status(self.directory, lambda: self.wall[0])
        self.assertTrue(public['available'] and public['runtime_online'])
        self.assertEqual(public['active'], vd.ORIGINAL)
        self.assertNotIn('contract', public)
        job = self.deploy(controller)
        self.advance(controller, 5)
        self.assertEqual(controller.state['phase'], 'verifying')
        self.advance(controller, VERIFY - 5)
        result = vd.read_json(job / 'result.json')
        self.assertEqual((result['phase'], result['active']), ('completed', 'm1'))
        self.assertGreaterEqual(result['evidence']['p1']['frames'], vd.MIN_FRAMES)
        committed = vd.read_json(self.directory / 'committed.json')
        self.assertEqual((committed['active']['model_id'], committed['previous']['model_id']), ('m1', vd.ORIGINAL))
        self.assertEqual(committed['active']['config'], vd.relative(job / 'infer.txt'))
        self.assertEqual(vd.public_status(self.directory, lambda: self.wall[0])['previous'], vd.ORIGINAL)
        # Startup restores the committed engine before PLAYING through the normal config path.
        restarted = self.controller(prepare=lambda props, _cache: props.update(
            {'model-engine-file': committed['active'] and str(self.root / committed['active']['engine'])}))
        self.assertEqual(self.configured, [(str(job / 'infer.txt'), 2)])
        self.assertEqual((restarted.active['model_id'], restarted.previous['model_id']), ('m1', vd.ORIGINAL))

    def test_startup_falls_back_to_original_when_runtime_or_contract_changed(self):
        controller = self.controller()
        self.deploy(controller)
        self.advance(controller, VERIFY)
        restarted = self.controller(prepare=lambda props, _cache: props.update({'model-engine-file': '/other.engine'}))
        self.assertEqual(self.configured, [])
        self.assertEqual(restarted.active['model_id'], vd.ORIGINAL)
        self.assertIn('runtime_changed', restarted.state['error'])
        self.base.write_text(self.base.read_text().replace('batch-size=2', 'batch-size=3'))
        restarted = self.controller(prepare=lambda props, _cache: None)
        self.assertEqual((self.configured, restarted.active['model_id']), ([], vd.ORIGINAL))
        self.assertIn('contract_changed', restarted.state['error'])

    def test_load_failure_restores_active_model(self):
        controller = self.controller()
        job = self.deploy(controller)
        self.element.callback(self.element, 1, str(job / 'infer.txt'))
        controller.tick()
        self.assertEqual(controller.state['phase'], 'rolling_back')
        self.assertEqual(self.loaded(controller), str(self.base))
        self.advance(controller, VERIFY)
        result = vd.read_json(job / 'result.json')
        self.assertEqual((result['phase'], result['error'], result['active']), ('rolled_back', 'model_load_failed', vd.ORIGINAL))
        self.assertEqual(result['failures'][0]['checks'], [{'check': 'model_load_failed', 'status': 1}])
        self.assertFalse((self.directory / 'committed.json').exists())

    def test_no_detections_or_fps_drop_rolls_back(self):
        controller = self.controller()
        job = self.deploy(controller)
        self.advance(controller, vd.LOAD_SECONDS + 2, detections=0)
        self.assertEqual(controller.state['phase'], 'rolling_back')
        self.loaded(controller)
        self.advance(controller, VERIFY)
        self.assertEqual(vd.read_json(job / 'result.json')['error'], 'verification_failed')
        job = self.deploy(controller, job='22222222-2222-4222-8222-222222222222')
        for camera in ('lobby', 'hall', 'yard', 'gate', 'roof', 'dock'):
            self.fps[camera] = 12.0  # the new engine slows the whole shared pipeline (80 %)
        self.advance(controller, vd.LOAD_SECONDS + 2)
        self.assertEqual(controller.state['phase'], 'rolling_back')
        self.assertFalse((self.directory / 'committed.json').exists())
        check = controller.state['failures'][0]['checks'][0]
        self.assertEqual((check['check'], check['fps'], check['baseline_fps']), ('service_fps_below_baseline', 72.0, 90.0))
        self.assertEqual(len(check['cameras']), 6)

    def test_unrelated_camera_fps_drop_is_recorded_but_does_not_fail_deployment(self):
        controller = self.controller()
        self.advance(controller, vd.BASELINE_SECONDS)
        job = self.request(controller)
        controller.tick()
        self.fps['p1'] = 8.0  # stable change during the minutes-long build: baseline is re-taken at apply
        self.advance(controller, vd.BASELINE_SECONDS)
        FakeBuilder.last.complete()
        controller.tick()
        self.loaded(controller)
        self.fps['lobby'] = 6.0  # one unrelated RTSP source stays low (40 %)
        self.states['hall'] = 'reconnecting'  # another flaps briefly, before the measurement window
        self.advance(controller, 10)
        del self.states['hall']
        self.advance(controller, VERIFY - 10)
        result = vd.read_json(job / 'result.json')
        self.assertEqual((result['phase'], result['failures'], result['checks']), ('completed', [], []))
        self.assertEqual(result['evidence']['p1']['baseline_fps'], 8.0)
        self.assertEqual(result['service']['cameras']['lobby'], {'fps': 6.0, 'baseline_fps': 15.0, 'state': 'online'})
        self.assertEqual((result['service']['fps'], result['service']['baseline_fps']), (81.0, 90.0))

    def test_single_camera_fps_drop_is_advisory_while_the_shared_pipeline_keeps_its_rate(self):
        controller = self.controller()
        job = self.deploy(controller)
        self.fps['p1'] = 7.0  # 70 % of its steady median; other cameras and total throughput (17/20) hold
        self.advance(controller, VERIFY)
        result = vd.read_json(job / 'result.json')
        self.assertEqual((result['phase'], result['failures']), ('completed', []))
        self.assertEqual(result['checks'], [{'check': 'fps_below_baseline', 'camera': 'p1', 'fps': 7.0,
                                             'baseline_fps': 10.0, 'baseline_low': 10.0, 'inference_fps': 7.0,
                                             'pipeline_slow': False, 'blocking': False}])

    def test_vehicle_camera_fps_drop_with_throughput_loss_rolls_back_and_rollback_is_verified_by_inference(self):
        controller = self.controller()
        job = self.deploy(controller)
        self.fps['p1'] = 5.0  # 50 %: inference throughput 15/20 falls below 85 % as well
        self.advance(controller, EXPIRE)
        self.assertEqual(controller.state['phase'], 'rolling_back')
        failure = controller.state['failures'][0]
        self.assertEqual((failure['code'], failure['phase'], failure['model_id']), ('verification_failed', 'verifying', 'm1'))
        self.assertTrue(failure['load_event'] and failure['model_loaded'])
        self.assertEqual(failure['checks'], [
            {'check': 'fps_below_baseline', 'camera': 'p1', 'fps': 5.0, 'baseline_fps': 10.0, 'baseline_low': 10.0,
             'inference_fps': 5.0, 'pipeline_slow': True, 'blocking': True},
            {'check': 'vehicle_inference_below_baseline', 'fps': 15.0, 'baseline_fps': 20.0, 'blocking': True}])
        self.assertEqual((failure['evidence']['p1']['fps'], failure['evidence']['p1']['state']), (5.0, 'online'))
        self.assertGreaterEqual(failure['evidence']['p1']['frames'], vd.MIN_FRAMES)
        # Rollback: the original model loads and infers; FPS is reported but cannot block recovery.
        self.assertEqual(self.loaded(controller), str(self.base))
        self.advance(controller, VERIFY)
        result = vd.read_json(job / 'result.json')
        self.assertEqual((result['phase'], result['error'], result['active']), ('rolled_back', 'verification_failed', vd.ORIGINAL))
        self.assertEqual([f['code'] for f in result['failures']], ['verification_failed'])
        self.assertEqual([(c['check'], c['blocking']) for c in result['checks']],
                         [('fps_below_baseline', False), ('vehicle_inference_below_baseline', False)])
        self.assertTrue(vd.public_status(self.directory, lambda: self.wall[0])['verified'])
        # Still 5 fps with the previous model: the source changed, not the model (operator evidence only).
        self.assertEqual(result['diagnosis']['cameras'], {'p1': {
            'baseline_fps': 10.0, 'baseline_range': [10.0, 10.0], 'candidate_fps': 5.0, 'rollback_fps': 5.0,
            'verdict': 'camera_source_suspected'}})
        self.assertTrue(any(line.startswith('vehicle_deployment_diagnosis ') for line in self.logs))

    def vary(self, controller, seconds, camera, values):
        """Camera whose RTSP rate swings on its own (one value per second, cycled)."""
        for second in range(seconds):
            self.fps[camera] = values[second % len(values)]
            self.advance(controller, 1)

    def test_baseline_is_a_window_median_not_one_rtsp_burst(self):
        # Incident replay: a 20 fps burst at sampling time made a ~10 fps gate camera fail at 10 fps.
        controller = self.controller()
        self.advance(controller, vd.BASELINE_SECONDS)
        job = self.request(controller)
        controller.tick()
        self.advance(controller, vd.BASELINE_SECONDS)
        self.fps['p1'] = 20.0
        self.advance(controller, 2)
        self.fps['p1'] = 10.0
        FakeBuilder.last.complete()
        controller.tick()
        self.loaded(controller)
        self.advance(controller, VERIFY)
        result = vd.read_json(job / 'result.json')
        self.assertEqual((result['phase'], result['checks']), ('completed', []))
        self.assertEqual((result['evidence']['p1']['baseline_fps'], result['evidence']['p1']['baseline_range']),
                         (10.0, [10.0, 10.0]))

    def test_camera_own_variation_is_advisory_when_inference_keeps_pace(self):
        controller = self.controller()
        self.vary(controller, vd.BASELINE_SECONDS, 'p1', (7.0, 10.0, 13.0))  # median 10, p10 7
        job = self.request(controller)
        controller.tick()
        self.vary(controller, vd.BASELINE_SECONDS, 'p1', (7.0, 10.0, 13.0))
        FakeBuilder.last.complete()
        controller.tick()
        self.loaded(controller)
        self.fps['p1'] = 8.0  # 80 % of the median, but inside the camera's own pre-switch range
        self.advance(controller, VERIFY)
        result = vd.read_json(job / 'result.json')
        self.assertEqual(result['phase'], 'completed')
        self.assertEqual(result['checks'], [{'check': 'camera_fps_variation', 'camera': 'p1', 'fps': 8.0,
                                             'baseline_fps': 10.0, 'baseline_low': 7.0, 'inference_fps': 8.0,
                                             'blocking': False}])
        self.assertEqual(result['evidence']['p1']['baseline_range'], [7.0, 13.0])

    def test_varying_camera_still_fails_when_inference_falls_behind_its_frames(self):
        controller = self.controller()
        self.vary(controller, vd.BASELINE_SECONDS, 'p1', (7.0, 10.0, 13.0))
        job = self.request(controller)
        controller.tick()
        self.vary(controller, vd.BASELINE_SECONDS, 'p1', (7.0, 10.0, 13.0))
        FakeBuilder.last.complete()
        controller.tick()
        self.loaded(controller)
        self.fps['p1'] = 8.0
        self.advance(controller, EXPIRE, inferred={'p1': 3})
        failure = controller.state['failures'][0]
        self.assertEqual([(c['check'], c['blocking']) for c in failure['checks']],
                         [('fps_below_baseline', True), ('vehicle_inference_below_baseline', True)])
        self.assertEqual(failure['checks'][0]['inference_fps'], 3.0)
        # Recovered with the previous model: the candidate is the suspect.
        self.fps['p1'] = 10.0
        self.loaded(controller)
        self.advance(controller, VERIFY)
        result = vd.read_json(job / 'result.json')
        self.assertEqual(result['phase'], 'rolled_back')
        self.assertEqual(result['diagnosis']['cameras']['p1']['verdict'], 'model_suspected')

    def test_large_vehicle_throughput_drop_blocks_even_inside_each_camera_range(self):
        controller = self.controller()
        for _ in range(2):  # request window, then the build
            for second in range(vd.BASELINE_SECONDS):
                self.fps['p1'] = self.fps['p2'] = (5.0, 10.0, 15.0)[second % 3]
                self.advance(controller, 1)
            if not controller.pending:
                job = self.request(controller)
                controller.tick()
        FakeBuilder.last.complete()
        controller.tick()
        self.loaded(controller)
        self.fps['p1'] = self.fps['p2'] = 6.0  # each within its own p10 (5), together 60 % of the throughput
        self.advance(controller, EXPIRE)
        failure = controller.state['failures'][0]
        self.assertEqual([(c['check'], c['blocking']) for c in failure['checks']],
                         [('camera_fps_variation', False), ('camera_fps_variation', False),
                          ('vehicle_inference_below_baseline', True)])
        self.assertEqual(controller.state['phase'], 'rolling_back')
        self.assertFalse((self.directory / 'committed.json').exists())

    def test_camera_offline_before_deployment_is_excluded_and_reported(self):
        controller = self.controller()
        self.fps['p2'] = 0.0  # RTSP down before the request
        job = self.deploy(controller)
        self.advance(controller, VERIFY)
        result = vd.read_json(job / 'result.json')
        self.assertEqual(result['phase'], 'completed')
        self.assertEqual(list(result['evidence']), ['p1'])
        self.assertEqual(result['checks'], [{'check': 'camera_offline_before_deployment', 'camera': 'p2',
                                             'state': 'offline', 'blocking': False}])

    def test_camera_going_offline_during_deployment_blocks_but_not_recovery(self):
        controller = self.controller()
        job = self.deploy(controller)
        self.fps['p2'] = 0.0  # drops after the switch: the model cannot be proven on it
        self.fps['lobby'] = 0.0  # an unrelated source drops too
        self.advance(controller, EXPIRE)
        failure = controller.state['failures'][0]
        went = [c for c in failure['checks'] if c['check'] == 'camera_went_offline']
        self.assertEqual(went[0] | {'offline_samples': 0}, {'check': 'camera_went_offline', 'camera': 'p2',
                                                            'state': 'offline', 'baseline_state': 'online',
                                                            'offline_samples': 0, 'blocking': True})
        self.assertGreater(went[0]['offline_samples'], 0)
        self.assertEqual(went[1], {'check': 'camera_went_offline', 'cameras': ['lobby'], 'blocking': False})
        self.assertEqual(failure['service']['baseline_fps'], 75.0)  # the dropped source is compared apart
        # Restoring the previous model is proven on p1; p2 still being offline cannot block recovery.
        self.loaded(controller)
        self.advance(controller, VERIFY)
        result = vd.read_json(job / 'result.json')
        self.assertEqual((result['phase'], result['error']), ('rolled_back', 'verification_failed'))
        self.assertFalse(any(c['blocking'] for c in result['checks']))
        self.assertTrue(vd.public_status(self.directory, lambda: self.wall[0])['verified'])

    def test_stalled_inference_fails_verification_and_failed_rollback_keeps_both_causes(self):
        controller = self.controller()
        job = self.deploy(controller)
        self.advance(controller, vd.LOAD_SECONDS + 2, cameras=('p2',))  # p1 stops producing inferred frames
        failure = controller.state['failures'][0]
        self.assertEqual(failure['code'], 'verification_failed')
        self.assertEqual([(c['check'], c.get('camera')) for c in failure['checks']],
                         [('inference_stalled', 'p1'), ('vehicle_inference_below_baseline', None)])
        self.assertGreater(failure['checks'][0]['inference_age'], vd.INFERENCE_STALE_SECONDS)
        self.loaded(controller)
        self.advance(controller, vd.LOAD_SECONDS + 2, cameras=('p2',))
        result = vd.read_json(job / 'result.json')
        self.assertEqual((result['phase'], result['error']), ('rollback_failed', 'rollback_verification_failed'))
        self.assertEqual([(f['code'], f['phase']) for f in result['failures']],
                         [('verification_failed', 'verifying'), ('rollback_verification_failed', 'rolling_back')])
        self.assertEqual(result['failures'][1]['checks'][0]['check'], 'inference_stalled')
        logged = [json.loads(line.split(' ', 1)[1]) for line in self.logs
                  if line.startswith('vehicle_deployment_failure ')]
        self.assertEqual([(entry['operation'], entry['code']) for entry in logged],
                         [(job.name, 'verification_failed'), (job.name, 'rollback_verification_failed')])
        self.assertFalse(vd.public_status(self.directory, lambda: self.wall[0])['verified'])

    def test_stalled_inference_recovers_after_rollback(self):
        controller = self.controller()
        job = self.deploy(controller)
        self.advance(controller, vd.LOAD_SECONDS + 2, cameras=('p2',))
        self.loaded(controller)
        self.advance(controller, VERIFY)  # the original model infers on both cameras again
        result = vd.read_json(job / 'result.json')
        self.assertEqual((result['phase'], result['error']), ('rolled_back', 'verification_failed'))
        self.assertEqual(result['checks'], [])
        self.assertEqual(result['failures'][0]['checks'][0]['check'], 'inference_stalled')

    def test_missing_load_events_time_out_with_model_not_loaded(self):
        controller = self.controller()
        job = self.deploy(controller, load=False)
        self.advance(controller, vd.LOAD_SECONDS + 2)
        self.assertEqual(controller.state['phase'], 'rolling_back')
        failure = controller.state['failures'][0]
        self.assertEqual((failure['code'], failure['phase'], failure['load_event'], failure['model_loaded']),
                         ('verification_failed', 'applying', False, False))
        self.assertEqual(failure['checks'], [{'check': 'model_not_loaded'}])
        self.assertGreater(failure['elapsed_seconds'], vd.LOAD_SECONDS)
        self.advance(controller, vd.LOAD_SECONDS + 2)  # the rollback load is never confirmed either
        result = vd.read_json(job / 'result.json')
        self.assertEqual((result['phase'], result['error']), ('rollback_failed', 'rollback_verification_failed'))
        self.assertEqual([f['checks'] for f in result['failures']], [[{'check': 'model_not_loaded'}]] * 2)

    def test_late_load_still_gets_the_full_measurement_window(self):
        controller = self.controller()
        job = self.deploy(controller, load=False)
        self.advance(controller, vd.LOAD_SECONDS - 5)
        self.loaded(controller)
        self.advance(controller, vd.SETTLE_SECONDS + vd.MEASURE_SECONDS - 1)
        self.assertEqual(controller.state['phase'], 'verifying')  # never judged on a partial window
        self.advance(controller, 2)
        self.assertEqual(vd.read_json(job / 'result.json')['phase'], 'completed')

    def test_cancel_during_build_and_verification(self):
        controller = self.controller()
        self.advance(controller, vd.BASELINE_SECONDS)
        job = self.request(controller)
        controller.tick()
        builder = FakeBuilder.last
        self.wall[0] += vd.LEASE_SECONDS + 1  # worker lost: lease expired
        self.cancelled = True
        controller.tick()
        self.assertTrue(builder.terminated)
        self.assertEqual(vd.read_json(job / 'result.json')['error'], 'build_cancelled_or_timeout')
        self.cancelled = False
        job = self.deploy(controller, job='33333333-3333-4333-8333-333333333333')
        controller.tick()  # load observed -> verifying
        vd.write_json(job / 'request.json', {**vd.read_json(job / 'request.json'), 'cancel': True})
        self.cancelled = True
        controller.tick()
        self.assertEqual(controller.state['phase'], 'rolling_back')
        self.loaded(controller)
        self.advance(controller, VERIFY)  # recovery is not interrupted by cancellation
        self.assertEqual(vd.read_json(job / 'result.json')['phase'], 'rolled_back')

    def test_stale_or_invalid_requests_fail_without_touching_the_model(self):
        controller = self.controller()
        self.advance(controller, 1)
        job = self.request(controller)
        controller.tick()  # Preview just started: cameras are live but the window is too short
        self.assertEqual(vd.read_json(job / 'result.json')['error'], 'baseline_not_ready')
        self.advance(controller, vd.BASELINE_SECONDS)
        job = self.request(controller, job='12121212-1212-4121-8121-121212121212')
        manifest = vd.read_json(job / 'manifest.json')
        vd.write_json(job / 'manifest.json', {**manifest, 'contract': 'other'})
        controller.tick()
        self.assertEqual(vd.read_json(job / 'result.json')['error'], 'artifact_contract_mismatch')
        job = self.request(controller, action='rollback', job='44444444-4444-4444-8444-444444444444')
        controller.tick()
        self.assertEqual(vd.read_json(job / 'result.json')['error'], 'no_previous_model')
        self.fps.update(p1=0.0, p2=0.0)
        job = self.request(controller, job='55555555-5555-4555-8555-555555555555')
        controller.tick()
        self.assertEqual(vd.read_json(job / 'result.json')['error'], 'no_live_inference')
        self.assertEqual(self.element.properties, [])

    def test_manual_rollback_switches_to_previous_and_swaps_journal(self):
        controller = self.controller()
        self.deploy(controller)
        self.advance(controller, VERIFY)
        job = self.request(controller, action='rollback', job='66666666-6666-4666-8666-666666666666')
        controller.tick()
        self.assertEqual(self.loaded(controller), str(self.base))
        self.advance(controller, VERIFY)
        self.assertEqual(vd.read_json(job / 'result.json')['active'], vd.ORIGINAL)
        committed = vd.read_json(self.directory / 'committed.json')
        self.assertEqual((committed['active']['model_id'], committed['previous']['model_id']), (vd.ORIGINAL, 'm1'))
        restarted = self.controller(prepare=lambda props, _cache: None)
        self.assertEqual(self.configured, [])  # original config already applied by preview.py
        self.assertEqual((restarted.active['model_id'], restarted.previous['model_id']), (vd.ORIGINAL, 'm1'))

    def test_single_class_model_deploys_with_class_map_and_rolls_back(self):
        controller = self.controller()
        single = {'class_mode': 'single', 'class_map': {'0': 1}, 'source_names': ['item']}
        # Load failure of the 1-class engine: the COCO80 original is restored automatically.
        job = self.deploy(controller, model='s1', **single)
        self.element.callback(self.element, 1, str(job / 'infer.txt'))
        controller.tick()
        self.assertEqual(self.loaded(controller), str(self.base))
        self.advance(controller, VERIFY)
        self.assertEqual(vd.read_json(job / 'result.json')['phase'], 'rolled_back')
        self.assertIsNone(vd.public_status(self.directory, lambda: self.wall[0])['active_class_map'])
        # Remapped detections (runtime car id) satisfy verification; the map is committed with the model.
        job = self.deploy(controller, model='s1', job='88888888-8888-4888-8888-888888888888', **single)
        self.advance(controller, VERIFY)
        self.assertEqual(vd.read_json(job / 'result.json')['phase'], 'completed')
        committed = vd.read_json(self.directory / 'committed.json')
        self.assertEqual((committed['active']['class_mode'], committed['active']['class_map']), ('single', {'0': 1}))
        self.assertEqual(vd.public_status(self.directory, lambda: self.wall[0])['active_class_map'], {'0': 1})
        # Manual rollback returns to the COCO80 original without a class map.
        job = self.request(controller, action='rollback', job='99999999-9999-4999-8999-999999999999')
        controller.tick()
        self.assertEqual(self.loaded(controller), str(self.base))
        self.advance(controller, VERIFY)
        status = vd.public_status(self.directory, lambda: self.wall[0])
        self.assertEqual((status['active'], status['previous'], status['active_class_map']), (vd.ORIGINAL, 's1', None))

    def test_failed_recovery_blocks_new_operations_and_is_unverified(self):
        controller = self.controller()
        job = self.deploy(controller)
        self.element.callback(self.element, 1, str(job / 'infer.txt'))
        controller.tick()
        self.element.callback(self.element, 1, str(self.base))
        controller.tick()
        self.assertEqual(vd.read_json(job / 'result.json')['phase'], 'rollback_failed')
        self.assertFalse(vd.public_status(self.directory, lambda: self.wall[0])['verified'])
        later = self.request(controller, job='77777777-7777-4777-8777-777777777777')
        controller.tick()
        self.assertFalse((later / 'result.json').exists())

    def test_fault_never_raises_and_restores_after_switch(self):
        controller = self.controller()
        self.deploy(controller)
        controller.tick()
        controller.fault()
        self.assertEqual(controller.state['phase'], 'rolling_back')
        broken = self.controller()
        broken.pending = {'id': 'x'}
        broken.state['phase'] = 'applying'
        with patch.object(broken, 'rollback', side_effect=OSError):
            broken.fault()
        self.assertEqual(broken.state['phase'], 'rollback_failed')

    def test_segmentation_graft_follows_the_active_detector_weights(self):
        public = lambda: vd.public_status(self.directory, lambda: self.wall[0])
        self.controller()
        self.assertIsNone(public()['segmentation_graft'])  # segmentation off
        controller = self.controller(segmenter=NS(det_weights_sha256='a' * 64))
        self.assertEqual(public()['segmentation_graft'], 'unknown')  # original model has no weights manifest
        self.deploy(controller, weights_sha256='a' * 64)
        self.advance(controller, VERIFY)
        self.assertEqual((controller.state['phase'], public()['segmentation_graft']), ('completed', 'match'))
        self.deploy(controller, model='m2', job='22222222-2222-4222-8222-222222222222', weights_sha256='b' * 64)
        self.advance(controller, VERIFY)
        self.assertEqual((controller.state['phase'], public()['segmentation_graft']), ('completed', 'stale'))

    def test_observe_batch_counts_only_selected_vehicle_detections(self):
        controller = self.controller()
        def nodes(values):
            node = None
            for value in reversed(values):
                node = NS(data=value, next=node)
            return node
        objects = [NS(unique_component_id=2, class_id=1), NS(unique_component_id=2, class_id=0),
                   NS(unique_component_id=90, class_id=0)]
        frames = [NS(source_id=0, frame_num=7, bInferDone=True, obj_meta_list=nodes(objects)),
                  NS(source_id=2, frame_num=7, bInferDone=True, obj_meta_list=nodes(objects))]
        pyds = NS(NvDsFrameMeta=NS(cast=lambda v: v), NvDsObjectMeta=NS(cast=lambda v: v))
        camera = {0: NS(camera_id='p1'), 2: NS(camera_id='lobby')}
        controller.observe_batch(NS(frame_meta_list=nodes(frames)), lambda f: camera[f.source_id], pyds)
        self.assertEqual(controller.frames, {'p1': (self.now[0], 7, 1, 1)})


if __name__ == '__main__':
    unittest.main()
