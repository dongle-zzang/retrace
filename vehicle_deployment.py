"""Local deployment mailbox and runtime-owned vehicle model transaction (no GI/HTTP imports).

The host training worker and Preview share the repository bind mount. Only the
worker writes requests; the runtime owns engine build, activation, inference
verification and rollback. Hot updates use nvinfer's config-file-path OTA path:
an engine-only update would not carry the custom YOLO parser or letterboxing.
"""
from collections import deque
import configparser
import hashlib
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import time
from uuid import uuid4

from vehicle_detection import VEHICLE_GIE_ID, graft_status, metadata_nodes

ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / '.cache' / 'vehicle-deployments'
ORIGINAL = 'production-original'
TERMINAL = {'completed', 'failed', 'rolled_back', 'rollback_failed'}
LEASE_SECONDS = 25
BUILD_SECONDS = 1800
LOAD_SECONDS = 90  # switch -> nvinfer model-updated
SETTLE_SECONDS = 20  # engine warm-up after the load; excluded from the measurement
# CameraRuntime FPS is a 5 s window, so one reading is an RTSP burst/jitter sample, not a rate.
# Baseline and verification both use the median of 1 s samples over a window.
SAMPLE_SECONDS = 1
BASELINE_SECONDS = 60  # pre-switch window
BASELINE_MIN_SAMPLES = 10
MEASURE_SECONDS = 30  # shortest post-settle window that may pass
VERIFY_SECONDS = 60  # longest post-settle window before the transaction fails
HISTORY_SAMPLES = 2 * (BASELINE_SECONDS + LOAD_SECONDS + SETTLE_SECONDS + VERIFY_SECONDS)
MIN_FRAMES = 5
FPS_RATIO = .85
INFERENCE_STALE_SECONDS = 5
# Restoring the committed model is itself the protection action, so FPS/throughput and a source that
# drops only inform a rollback; inference on the remaining cameras still has to be proven.
ROLLBACK_ADVISORY = frozenset({'fps_below_baseline', 'service_fps_below_baseline', 'vehicle_inference_below_baseline',
                               'camera_went_offline'})
LIVE = ('online', 'degraded')
PUBLIC = ('active', 'previous', 'operation', 'phase', 'error', 'evidence', 'service', 'checks', 'failures', 'verified',
          'active_class_map', 'diagnosis', 'segmentation_graft')
REPORT = ('operation', 'phase', 'error', 'evidence', 'service', 'checks', 'failures', 'diagnosis')


def window_stats(samples):
    """One camera over a window: FPS spread while live and the inferred-frame rate."""
    live = sorted(fps for _t, state, fps, _n in samples if state in LIVE)
    counted = [(t, n) for t, _state, _fps, n in samples if n is not None]
    value = {'samples': len(samples), 'live_samples': len(live), 'median': None, 'p10': None, 'p90': None,
             'inference_fps': None}
    if live:
        value.update(median=round(statistics.median(live), 2), p10=live[int(.1 * len(live))],
                     p90=live[min(len(live) - 1, int(.9 * len(live)))])
    if len(counted) >= 2 and counted[-1][0] > counted[0][0]:
        value['inference_fps'] = round((counted[-1][1] - counted[0][1]) / (counted[-1][0] - counted[0][0]), 2)
    return value


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def local_path(relative):
    if Path(relative).is_absolute():
        raise ValueError('path_outside_repository')
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('path_outside_repository')
    return path


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def public_status(directory=DIRECTORY, wall=time.time):
    state = read_json(Path(directory) / 'status.json', {})
    fresh = wall() - state.get('updated_at', 0) < 20
    return {k: state.get(k) for k in PUBLIC} | {
        'available': bool(fresh and state.get('enabled')), 'runtime_online': fresh, 'session': state.get('session')}


def configuration(path):
    config = configparser.ConfigParser(interpolation=None)
    with Path(path).open(encoding='utf-8') as source:
        config.read_file(source)
    return config


def contract(config):
    """Pre/postprocessing, parser binary, class ordering and batch must all match."""
    props = config['property']
    value = {section: dict(config[section]) for section in config.sections()}
    for key in ('model-engine-file', 'onnx-file', 'labelfile-path', 'custom-lib-path'):
        value['property'].pop(key, None)
    value['labels'] = Path(props['labelfile-path']).read_text().splitlines()
    value['parser_sha256'] = digest(props['custom-lib-path'])
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def candidate_config(base_config, onnx, cache_dir, prepare=None):
    """Production config with only the model replaced; engine path keyed by build inputs/runtime."""
    if prepare is None:
        from pgie_cache import prepare_engine_cache as prepare
    config = configuration(base_config)
    config['property']['onnx-file'] = str(Path(onnx).resolve())
    config['property'].pop('model-engine-file', None)
    prepare(config['property'], Path(cache_dir))
    return config


class DeploymentController:
    """tick runs on Preview's existing control loop; probes only record observations."""
    def __init__(self, element, detector, manager, directory=DIRECTORY, clock=time.monotonic, wall=time.time,
                 configure=None, prepare=None, segmenter=None):
        if configure is None:
            from pgie_cache import configure_pgie as configure
        self.element, self.detector, self.manager = element, detector, manager
        self.directory, self.clock, self.wall = Path(directory), clock, wall
        self.configure, self.prepare = configure, prepare
        self.segmenter = segmenter  # optional seg nvinfer, grafted from some det weights (see graft_status)
        self.config = configuration(detector.config_path)
        self.signature = contract(self.config)
        self.session = uuid4().hex
        self.events = queue.SimpleQueue()
        self.frames = {}  # camera -> (monotonic, frame number, inferred count, detections)
        self.history = {}  # camera -> deque of (monotonic, state, fps, inferred count | None)
        self.sampled_at = float('-inf')
        self.pending = self.builder = self.build_log = self.target = None
        self.last_write = 0
        self.enabled = True
        self.state = {'phase': 'idle', 'operation': None, 'error': None, 'evidence': None, 'service': None,
                      'checks': None, 'failures': [], 'verified': True, 'diagnosis': None}
        props = self.config['property']
        self.original = {'model_id': ORIGINAL, 'config': relative(detector.config_path),
                         'engine': relative(props['model-engine-file']), 'contract': self.signature}
        self.active, self.previous = self.original, None
        journal = read_json(self.directory / 'committed.json', {})
        if journal.get('active') and journal['active'].get('model_id') != ORIGINAL:
            try:
                self.check_entry(journal['active'], rebuild_key=True)
                # Before PLAYING: startup never loads an uncommitted candidate.
                self.configure(element, str(local_path(journal['active']['config'])), len(detector.cameras))
                self.active = journal['active']
            except (OSError, ValueError, KeyError) as error:
                # Camera count, parser or TensorRT changed: keep the original model and say so.
                self.state['error'] = 'committed_model_skipped: ' + str(error)[:80]
        previous = journal.get('previous')
        if previous:
            self.previous = self.original if previous.get('model_id') == ORIGINAL else previous
        if self.previous and self.previous['model_id'] == self.active['model_id']:
            self.previous = None
        try:
            element.connect('model-updated', self.model_updated)
        except (TypeError, RuntimeError, AttributeError):
            self.enabled = False
            self.state['error'] = 'hot_update_unavailable'
        self.publish()

    # ----- helpers ----------------------------------------------------------------------
    def check_entry(self, entry, rebuild_key=False):
        """A committed/previous model must still fit the running detector exactly."""
        if entry['model_id'] == ORIGINAL:
            return
        config = configuration(local_path(entry['config']))
        if entry.get('contract') != self.signature or contract(config) != self.signature:
            raise ValueError('contract_changed')
        engine = local_path(entry['engine'])
        if Path(config['property']['model-engine-file']).resolve() != engine:
            raise ValueError('engine_path_mismatch')
        if not engine.is_file() or digest(engine) != entry.get('sha256'):
            raise ValueError('engine_changed')
        if rebuild_key:
            # The engine name embeds GPU/TensorRT/DeepStream identity; a mismatch means it may not load.
            fresh = candidate_config(local_path(entry['config']), config['property']['onnx-file'],
                                     engine.parent.parent, self.prepare)
            if Path(fresh['property']['model-engine-file']).resolve() != engine:
                raise ValueError('runtime_changed')

    def model_updated(self, _element, status, path):
        self.events.put((int(status), str(path)))

    def observe(self, camera, number, inferred, detections):
        # Streaming thread, before tracker class remapping. Bounded by camera count, no I/O.
        old = self.frames.get(camera, (0, None, 0, 0))
        if number != old[1]:
            self.frames[camera] = (self.clock(), number, old[2] + int(inferred), old[3] + detections)

    def observe_batch(self, batch, camera_for_frame, pyds):
        """Vehicle nvinfer src probe: count inferred frames/detections per selected camera."""
        for frame in metadata_nodes(batch.frame_meta_list, pyds.NvDsFrameMeta.cast):
            camera = camera_for_frame(frame).camera_id
            if camera in self.detector.cameras:
                detections = sum(1 for obj in metadata_nodes(frame.obj_meta_list, pyds.NvDsObjectMeta.cast)
                                 if obj.unique_component_id == VEHICLE_GIE_ID
                                 and obj.class_id in self.detector.vehicle_class_ids)
                self.observe(camera, int(frame.frame_num), bool(frame.bInferDone), detections)

    def publish(self):
        props = self.config['property']
        self.state.update(active=self.active['model_id'], previous=self.previous and self.previous['model_id'],
                          active_class_map=self.active.get('class_map'),
                          segmentation_graft=self.segmenter and graft_status(self.segmenter,
                                                                             local_path(self.active['config'])),
                          enabled=self.enabled, session=self.session, updated_at=self.wall(),
                          # Private export contract for the worker; the Backend never forwards it.
                          contract=self.signature, onnx=relative(props['onnx-file']),
                          labels=Path(props['labelfile-path']).read_text().splitlines())
        write_json(self.directory / 'status.json', self.state)
        self.last_write = self.clock()

    def finish(self, phase, error=None):
        self.state.update(phase=phase, error=error, verified=phase != 'rollback_failed')
        if self.pending and self.state['diagnosis']:
            print('vehicle_deployment_diagnosis ' + json.dumps(
                {'operation': self.state['operation'], **self.state['diagnosis']}, separators=(',', ':')), flush=True)
        if self.pending:
            write_json(self.directory / self.pending['id'] / 'result.json',
                       {k: self.state.get(k) for k in REPORT} | {'active': self.active['model_id']})
        self.pending = None
        self.publish()

    def stop_builder(self):
        if self.builder and self.builder.poll() is None:
            self.builder.terminate()
            try:
                self.builder.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.builder.kill()
                self.builder.wait(timeout=3)
        if self.build_log:
            self.build_log.close()
        self.builder = self.build_log = None

    def switch(self, target, phase):
        while not self.events.empty():
            self.events.get()
        self.target, self.started, self.loaded, self.loaded_at = target, self.clock(), False, None
        self.load_status = None
        self.state.update(phase=phase, evidence=None, service=None, checks=None)
        self.element.set_property('config-file-path', str(local_path(target['config'])))
        self.publish()

    def rollback(self, error):
        self.state['error'] = error
        self.switch(self.active, 'rolling_back')

    def snapshots(self):
        return {camera.camera_id: self.manager.runtimes[camera.source_id].snapshot()
                for camera in self.manager.cameras_by_slot.values()}

    def record(self):
        """1 s FPS/state/inference samples, kept continuously so a baseline never waits."""
        now = self.clock()
        if now - self.sampled_at < SAMPLE_SECONDS:
            return
        self.sampled_at = now
        for camera, snap in self.snapshots().items():
            frames = self.frames.get(camera)
            self.history.setdefault(camera, deque(maxlen=HISTORY_SAMPLES)).append(
                (now, snap.state, snap.fps, None if frames is None else frames[2]))

    def window(self, camera, start):
        return window_stats([s for s in self.history.get(camera, ()) if s[0] >= start])

    def sample(self):
        """Baseline just before a model switch (a build can take many minutes): the median of the last
        BASELINE_SECONDS per camera, with its p10-p90 spread as the camera's own normal variation."""
        now, snaps = self.clock(), self.snapshots()
        self.record()
        self.reference = {camera: self.window(camera, now - BASELINE_SECONDS) | {'state': snap.state}
                          for camera, snap in snaps.items()}
        self.rates = {camera: value['median'] for camera, value in self.reference.items()
                      if value['state'] in LIVE and value['live_samples'] >= BASELINE_MIN_SAMPLES and value['median']}
        self.online = set(self.rates) & self.detector.cameras
        # Offline before the switch: excluded from verification and reported, never counted as a failure.
        self.offline_before = {camera: snaps[camera].state if camera in snaps else 'unknown'
                               for camera in sorted(self.detector.cameras - self.online)}
        if not self.online:
            live = any(snaps[c].state in LIVE for c in self.detector.cameras if c in snaps)
            raise ValueError('baseline_not_ready' if live else 'no_live_inference')
        if any(now - self.frames.get(c, (0,))[0] > INFERENCE_STALE_SECONDS for c in self.online):
            raise ValueError('no_live_inference')

    def fail_check(self, code, phase):
        """Freeze the failed checks with camera values; a later rollback must not overwrite them."""
        now = self.clock()
        if self.load_status not in (None, 0):
            checks = [{'check': 'model_load_failed', 'status': self.load_status}]
        elif not self.loaded:
            checks = [{'check': 'model_not_loaded'}]
        else:
            checks = self.state['checks'] or []
        record = {'code': code, 'phase': phase, 'model_id': self.target['model_id'],
                  'load_event': self.load_status is not None, 'model_loaded': self.loaded,
                  'elapsed_seconds': round(now - self.started, 1),
                  'loaded_after_seconds': None if self.loaded_at is None else round(self.loaded_at - self.started, 1),
                  'checks': checks, 'evidence': self.state['evidence'], 'service': self.state['service']}
        self.state['failures'] = [*self.state['failures'], record][-4:]
        # Container log: camera IDs are public; RTSP addresses never reach this controller.
        print('vehicle_deployment_failure ' + json.dumps({'operation': self.state['operation'], **record},
                                                         separators=(',', ':')), flush=True)

    # ----- transaction ------------------------------------------------------------------
    def start(self, request):
        self.pending = request
        self.state.update(operation=request['id'], phase='validating', error=None, evidence=None, service=None,
                          checks=None, failures=[], diagnosis=None)
        if request.get('session') != self.session or request.get('contract') != self.signature:
            raise ValueError('runtime_changed')
        if self.wall() - request.get('lease_at', 0) > LEASE_SECONDS:
            raise ValueError('worker_lost')
        self.sample()
        if request.get('action') == 'rollback':
            if not self.previous:
                raise ValueError('no_previous_model')
            self.check_entry(self.previous)
            self.switch(self.previous, 'applying')
            return
        manifest = read_json(self.directory / request['id'] / 'manifest.json', {})
        if manifest.get('contract') != self.signature or manifest.get('model_id') != request.get('model_id'):
            raise ValueError('artifact_contract_mismatch')
        # Hashing, staging and TensorRT building run in a separate native process.
        job = self.directory / request['id']
        self.build_log = (job / 'build.log').open('wb')
        self.builder = subprocess.Popen(
            [sys.executable, str(ROOT / 'tools' / 'build_vehicle_engine.py'), '--base', self.detector.config_path,
             '--job', str(job), '--engines', str(self.directory / 'engines')],
            stdout=self.build_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        self.started = self.clock()
        self.state['phase'] = 'building'
        self.publish()

    def built(self):
        job = self.directory / self.pending['id']
        result = read_json(job / 'build.json', {})
        manifest = read_json(job / 'manifest.json', {})
        config = job / 'infer.txt'
        candidate = configuration(config)
        engine = Path(candidate['property']['model-engine-file']).resolve()
        if (contract(candidate) != self.signature or result.get('engine') != relative(engine)
                or not engine.is_file() or result.get('sha256') != digest(engine)):
            raise ValueError('engine_build_invalid')
        return {'model_id': self.pending['model_id'], 'contract': self.signature, 'config': relative(config),
                'engine': relative(engine), 'sha256': result['sha256'], 'onnx_sha256': result.get('onnx_sha256'),
                # Model class id -> running detector class id (1-class car head remapped inside the ONNX).
                'class_mode': manifest.get('class_mode', 'vehicle3'), 'class_map': manifest.get('class_map')}

    def assess(self, phase):
        """Window medians against the pre-switch baseline; vehicle cameras verify the model, the other
        cameras guard the shared pipeline as a whole.

        The model runs in one shared mux/nvinfer batch, so an engine that slows the pipeline lowers the
        FPS of every camera and the vehicle inference throughput. One RTSP source that varies on its own
        does not: a single vehicle camera below 85 % of its median is advisory while both pipeline-wide
        checks (other cameras' total FPS, vehicle inference throughput) hold, and blocks only when one of
        them fails too. Inside its own pre-switch p10-p90 range with inference keeping pace it is reported
        as the camera's own variation.
        """
        now, snaps, evidence, checks = self.clock(), self.snapshots(), {}, []
        start = self.loaded_at + SETTLE_SECONDS
        for camera, state in self.offline_before.items():
            checks.append({'check': 'camera_offline_before_deployment', 'camera': camera, 'state': state,
                           'blocking': False})
        for camera in sorted(self.online):
            current = self.frames.get(camera, (0, None, 0, 0))
            old = self.baseline.get(camera, (0, None, 0, 0))
            base, measured, snap = self.reference[camera], self.window(camera, start), snaps.get(camera)
            item = evidence[camera] = {
                'frames': current[2] - old[2], 'detections': current[3] - old[3],
                'fps': measured['median'] or 0.0, 'baseline_fps': self.rates[camera],
                'fps_range': [measured['p10'], measured['p90']], 'baseline_range': [base['p10'], base['p90']],
                'inference_fps': measured['inference_fps'], 'baseline_inference_fps': base['inference_fps'],
                'state': snap.state if snap else 'unknown', 'baseline_state': base['state'],
                'offline_samples': measured['samples'] - measured['live_samples'],
                'inference_age': None if current[1] is None else round(now - current[0], 1)}
            if item['state'] not in LIVE:
                checks.append({'check': 'camera_went_offline', 'camera': camera, 'state': item['state'],
                               'baseline_state': base['state'], 'offline_samples': item['offline_samples']})
                if phase == 'rolling_back':
                    continue  # recovery is proven on the cameras that still deliver frames
            elif item['fps'] < FPS_RATIO * item['baseline_fps']:
                keeps_pace = (item['inference_fps'] or 0) >= FPS_RATIO * item['fps']
                own_range = base['p10'] is not None and item['fps'] >= base['p10']
                checks.append({'check': 'camera_fps_variation' if keeps_pace and own_range else 'fps_below_baseline',
                               'camera': camera, 'fps': item['fps'], 'baseline_fps': item['baseline_fps'],
                               'baseline_low': base['p10'], 'inference_fps': item['inference_fps']})
                if keeps_pace and own_range:
                    checks[-1]['blocking'] = False
            if item['inference_age'] is None or item['inference_age'] > INFERENCE_STALE_SECONDS:
                checks.append({'check': 'inference_stalled', 'camera': camera, 'inference_age': item['inference_age']})
            elif item['frames'] < MIN_FRAMES:
                checks.append({'check': 'insufficient_inference_frames', 'camera': camera, 'frames': item['frames']})
        if not sum(item['detections'] for item in evidence.values()):
            checks.append({'check': 'no_vehicle_detections'})
        # Vehicle inference throughput: frames the model actually processed, regardless of the cause.
        inferred = round(sum(item['inference_fps'] or 0 for item in evidence.values()), 2)
        inferred_base = round(sum(item['baseline_inference_fps'] or 0 for item in evidence.values()), 2)
        if inferred_base and inferred < FPS_RATIO * inferred_base:
            checks.append({'check': 'vehicle_inference_below_baseline', 'fps': inferred, 'baseline_fps': inferred_base})
        cameras, offline = {}, []
        for camera in sorted(set(self.rates) - self.online):
            snap = snaps.get(camera)
            live = snap is not None and snap.state in LIVE
            cameras[camera] = {'fps': (self.window(camera, start)['median'] or 0.0) if live else 0.0,
                               'baseline_fps': self.rates[camera], 'state': snap.state if snap else 'unknown'}
            if not live:
                offline.append(camera)
        if offline:
            # An unrelated source that drops is reported apart from the pipeline-wide FPS comparison.
            checks.append({'check': 'camera_went_offline', 'cameras': offline, 'blocking': False})
        total = round(sum(item['fps'] for c, item in cameras.items() if c not in offline), 2)
        base = round(sum(item['baseline_fps'] for c, item in cameras.items() if c not in offline), 2)
        if base and total < FPS_RATIO * base:
            checks.append({'check': 'service_fps_below_baseline', 'fps': total, 'baseline_fps': base,
                           'cameras': [c for c, item in cameras.items()
                                       if c not in offline and item['fps'] < FPS_RATIO * item['baseline_fps']]})
        pipeline_slow = any(c['check'] in ('service_fps_below_baseline', 'vehicle_inference_below_baseline') for c in checks)
        for item in checks:
            if item['check'] == 'fps_below_baseline':
                # 2026-10-19: B1 RTSP sources swing 8-24 fps on their own while the shared batch keeps its rate.
                item['pipeline_slow'] = pipeline_slow
                item.setdefault('blocking', pipeline_slow and phase != 'rolling_back')
            item.setdefault('blocking', not (phase == 'rolling_back' and item['check'] in ROLLBACK_ADVISORY))
        self.state.update(evidence=evidence, checks=checks,
                          service={'fps': total, 'baseline_fps': base, 'cameras': cameras,
                                   'vehicle_inference_fps': inferred, 'baseline_vehicle_inference_fps': inferred_base})
        return not any(item['blocking'] for item in checks)

    def diagnose(self):
        """After a rollback: did the slow cameras recover with the previous model (the candidate slowed
        them) or stay slow (the RTSP source changed)? Advisory evidence for the operator only."""
        failure = next((f for f in reversed(self.state['failures']) if f['phase'] != 'rolling_back'), None)
        restored = self.state['evidence'] or {}
        if not failure or not failure.get('evidence') or not restored:
            return None
        cameras = {}
        for camera, before in failure['evidence'].items():
            after = restored.get(camera)
            if after is None or before['fps'] >= FPS_RATIO * before['baseline_fps']:
                continue
            if after['state'] not in LIVE:
                verdict = 'inconclusive'
            elif after['fps'] >= FPS_RATIO * before['baseline_fps']:
                verdict = 'model_suspected'
            else:
                verdict = 'camera_source_suspected'
            cameras[camera] = {'baseline_fps': before['baseline_fps'], 'baseline_range': before.get('baseline_range'),
                               'candidate_fps': before['fps'], 'rollback_fps': after['fps'], 'verdict': verdict}
        service = failure.get('service') or {}
        return {'cameras': cameras, 'service': {
            'baseline_fps': service.get('baseline_fps'), 'candidate_fps': service.get('fps'),
            'rollback_fps': (self.state['service'] or {}).get('fps')}}

    def tick(self):
        self.record()
        if self.pending:
            request = read_json(self.directory / self.pending['id'] / 'request.json', {})
            cancelled = bool(request.get('cancel')) or self.wall() - request.get('lease_at', 0) > LEASE_SECONDS
            phase = self.state['phase']
            if phase == 'building':
                if cancelled or self.clock() - self.started > BUILD_SECONDS:
                    self.stop_builder()
                    self.finish('failed', 'build_cancelled_or_timeout')
                elif self.builder.poll() is not None:
                    code = self.builder.returncode
                    self.stop_builder()
                    if code:
                        self.finish('failed', 'engine_build_failed')
                    else:
                        try:
                            target = self.built()
                        except (OSError, ValueError, KeyError):
                            self.finish('failed', 'engine_build_invalid')
                        else:
                            try:
                                self.sample()
                            except ValueError as error:
                                self.finish('failed', str(error))
                            else:
                                self.switch(target, 'applying')
            elif phase in ('applying', 'verifying', 'rolling_back'):
                self.verify(phase, cancelled)
        elif self.enabled and self.state['phase'] != 'rollback_failed':
            self.poll_requests()
        if self.clock() - self.last_write >= 2:
            self.publish()

    def verify(self, phase, cancelled):
        while not self.events.empty():
            code, path = self.events.get()
            if Path(path).resolve() != local_path(self.target['config']):
                continue
            self.load_status = code
            if code != 0:
                # nvinfer keeps the previous context when a new one fails to load.
                if phase == 'rolling_back':
                    self.fail_check('rollback_load_failed', phase)
                    self.finish('rollback_failed', 'rollback_load_failed')
                else:
                    self.fail_check('model_load_failed', phase)
                    self.rollback('model_load_failed')
                return
            self.loaded, self.loaded_at, self.baseline = True, self.clock(), dict(self.frames)
            if phase == 'applying':
                self.state['phase'] = phase = 'verifying'
        now = self.clock()
        # Load must be confirmed within LOAD_SECONDS; a confirmed load always gets the full window.
        deadline = (self.loaded_at + SETTLE_SECONDS + VERIFY_SECONDS) if self.loaded else self.started + LOAD_SECONDS
        if cancelled and phase == 'verifying':
            # Cancellation never interrupts recovery; an in-flight load is reverted once observed.
            self.rollback('deployment_cancelled')
        elif self.loaded and now - self.loaded_at >= SETTLE_SECONDS + MEASURE_SECONDS and self.assess(phase):
            if phase == 'rolling_back':
                self.state['diagnosis'] = self.diagnose()
                self.finish('rolled_back', self.state['error'])
            else:
                previous, active = self.active, self.target
                write_json(self.directory / 'committed.json', {'active': active, 'previous': previous})
                self.previous, self.active = previous, active
                self.finish('completed')
        elif now > deadline:
            if phase == 'rolling_back':
                self.fail_check('rollback_verification_failed', phase)
                self.state['diagnosis'] = self.diagnose()
                self.finish('rollback_failed', 'rollback_verification_failed')
            else:
                self.fail_check('verification_failed', phase)
                self.rollback('verification_failed')

    def poll_requests(self):
        for file in sorted(self.directory.glob('*/request.json')):
            if (file.parent / 'result.json').exists():
                continue
            request = read_json(file, {})
            if request.get('id') != file.parent.name:
                continue
            if request.get('cancel'):
                self.pending = request
                self.state.update(operation=request['id'], evidence=None, service=None, checks=None, failures=[],
                                  diagnosis=None)
                self.finish('failed', 'deployment_cancelled')
                break
            try:
                self.start(request)
            except (OSError, ValueError, KeyError) as error:
                self.stop_builder()
                code = str(error) if str(error).replace('_', '').isalpha() else 'deployment_preflight_failed'
                self.finish('failed', code)
            break

    def fault(self, error='deployment_control_failed'):
        """Control-loop exception: never propagate into RTSP/pipeline handling."""
        try:
            self.stop_builder()
            if self.pending and self.state['phase'] in ('applying', 'verifying'):
                self.rollback(error)
            elif self.pending and self.state['phase'] != 'rolling_back':
                self.finish('failed', error)
        except Exception:
            self.enabled = False
            self.state.update(phase='rollback_failed', error=error, verified=False)

    def close(self):
        self.stop_builder()
        self.enabled = False
        self.publish()
