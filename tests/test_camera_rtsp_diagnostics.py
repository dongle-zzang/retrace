"""CPU-only classifier/probe confidentiality, selection and runtime regression."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from camera_runtime import CameraRuntime
from rtsp_diagnostics import classify_error, classify_bus_error, uri_shape
from tools.check_camera_rtsp import camera_input, diagnose, probe, main

SECRET = 'rtsp://test-user:test-password@192.0.2.50/profile/profile1'


class ClassifierTests(unittest.TestCase):
    def test_structured_and_recognized_reasons_no_raw_values(self):
        cases = [('gst-resource-error-quark', 15, '', 'rtsp_auth_failed'),
                 ('gst-resource-error-quark', 3, '', 'rtsp_not_found'),
                 ('gst-stream-error-quark', 7, '', 'decoder_error'),
                 ('gst-stream-error-quark', 6, '', 'decoder_error'),
                 ('gst-resource-error-quark', 5, 'Failed to connect to server', 'rtsp_connection_failed'),
                 ('gst-resource-error-quark', 9, 'Operation timed out', 'rtsp_timeout'),
                 ('gst-resource-error-quark', 9, 'Unauthorized (401)', 'rtsp_auth_failed'),
                 ('gst-resource-error-quark', 9, 'Not Found (404)', 'rtsp_not_found'),
                 ('gst-resource-error-quark', 5, '', 'rtsp_error')]
        for domain, code, text, expected in cases:
            with self.subTest(reason=expected):
                result = classify_error(domain, code, SECRET, text + ' ' + SECRET)
                self.assertEqual(result['reason'], expected)
                for value in ('rtsp://', 'test-user', 'test-password', '192.0.2.50'):
                    self.assertNotIn(value, json.dumps(result))

    def test_uri_words_do_not_create_timeout_or_auth_false_positives(self):
        self.assertEqual(classify_error('gst-resource-error-quark', 5,
            'rtsp://user:timeout@192.0.2.50/Unauthorized(401)')['reason'], 'rtsp_error')
        self.assertEqual(classify_error('gst-resource-error-quark', 3, SECRET,
                                       factory='nvv4l2decoder')['reason'], 'decoder_error')
        result = classify_error(SECRET, SECRET, SECRET)
        self.assertEqual(result, {'reason': 'rtsp_error', 'domain': 'unknown', 'code': None})

    def test_native_warning_uses_same_classifier(self):
        error = NS(domain='gst-resource-error-quark', code=15, message=SECRET)
        message = NS(parse_warning=lambda: (error, SECRET),
                     src=NS(get_factory=lambda: NS(get_name=lambda: 'nvurisrcbin')))
        self.assertEqual(classify_bus_error(message, warning=True)['reason'], 'rtsp_auth_failed')
        self.assertEqual(classify_bus_error(message, warning=True)['factory'], 'nvurisrcbin')

    def test_bus_factory_is_a_validated_plugin_name_or_other(self):
        error = NS(domain='gst-stream-error-quark', code=7, message=SECRET)
        def message(name):
            return NS(parse_error=lambda: (error, SECRET), parse_warning=lambda: (error, SECRET),
                      src=None if name is None else NS(get_factory=lambda: NS(get_name=lambda: name)))
        self.assertEqual(classify_bus_error(message('nvstreammux'))['factory'], 'nvstreammux')
        self.assertEqual(classify_bus_error(message('nvv4l2decoder'))['reason'], 'decoder_error')
        for name in (None, 'Nv Stream/Mux', SECRET, 'a' * 33, ''):
            with self.subTest(name=name):
                result = classify_bus_error(message(name))
                self.assertEqual(result['factory'], 'other')
                self.assertNotIn('192.0.2.50', json.dumps(result))
        self.assertEqual(classify_bus_error(NS(parse_error=lambda: 1 / 0, src=None)),
                         {'reason': 'rtsp_error', 'domain': 'unknown', 'code': None, 'factory': 'other'})

    def test_runtime_detail_does_not_change_backend_error_or_healthy_camera(self):
        first, second = CameraRuntime('first'), CameraRuntime('second')
        generation = first.begin_attempt()
        second.begin_attempt()
        before = second.snapshot()
        self.assertTrue(first.record_error_reason('rtsp_auth_failed', generation))
        first.fail('rtsp_error', generation)
        self.assertEqual(first.snapshot().last_error, 'rtsp_error')
        self.assertEqual(first.snapshot().error_reason, 'rtsp_auth_failed')
        self.assertFalse(first.record_error_reason(SECRET, generation))
        self.assertFalse(first.record_error_reason('rtsp_error', generation))
        self.assertEqual(second.snapshot(), before)
        self.assertNotIn(SECRET, json.dumps(first.snapshot().to_dict()))


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config, self.env_file = root / 'cameras.yaml', root / '.env'
        self.config.write_text('cameras: [{id: baseline, floor: 1, name: B, rtsp_env: BASE}, '
                               '{id: target, floor: 1, name: T, rtsp_env: TARGET}, '
                               '{id: missing, floor: 1, name: M, rtsp_env: MISSING}]')
        self.env_file.write_text('BASE=' + SECRET.replace('profile1', 'profile2') + '\nTARGET=' + SECRET + '\n')

    def test_selects_only_one_secret_and_matches_startup_environment_precedence(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(camera_input('target', self.config, self.env_file), SECRET)
            calls = []
            result = diagnose('target', self.config, self.env_file,
                              prober=lambda uri, *args: calls.append(uri) or {'rtsp_connection': 'ok', 'codec': 'H264'})
        self.assertEqual(calls, [SECRET])
        self.assertNotIn(SECRET, json.dumps(result))
        override = SECRET.replace('profile1', 'profile2')
        self.assertEqual(camera_input('target', self.config, self.env_file, {'TARGET': override}), override)

    def test_two_camera_comparison_is_sequential_and_only_shapes_are_public(self):
        calls = []
        def prober(uri, *args):
            calls.append(uri)
            return {'rtsp_connection': 'ok', 'codec': 'H264', 'decoder_created': True}
        with patch.dict(os.environ, {}, clear=True):
            result = diagnose('target', self.config, self.env_file, compare_to='baseline', prober=prober)
        self.assertEqual(calls, [SECRET.replace('profile1', 'profile2'), SECRET])
        self.assertTrue(result['comparison']['profile_differs'])
        self.assertFalse(result['comparison']['codec_differs'])
        self.assertEqual(result['uri_shape']['path_depth'], 2)
        for value in ('test-user', 'test-password', '192.0.2.50', 'rtsp://'):
            self.assertNotIn(value, json.dumps(result))

    def test_probe_passes_uri_by_stdin_not_argv_and_suppresses_native_stderr(self):
        def runner(argv, **kwargs):
            self.assertNotIn(SECRET, ' '.join(argv))
            self.assertEqual(json.loads(kwargs['input'])['uri'], SECRET)
            self.assertEqual(kwargs['stderr'], subprocess.DEVNULL)
            self.assertEqual(kwargs['timeout'], 20)
            return NS(stdout=b'{"rtsp_connection":"ok","codec":"H264"}')
        self.assertEqual(probe(SECRET, runner=runner)['rtsp_connection'], 'ok')

    def test_worker_native_stdout_stderr_are_suppressed_even_when_containing_secrets(self):
        # This is a real child process and fd writes, without GI/RTSP/GPU.
        script = '''
import os
from tools import check_camera_rtsp as tool
def native_probe(uri, timeout, latency, emit):
    os.write(1, uri.encode())
    os.write(2, uri.encode())
    emit({'rtsp_connection': 'ok', 'codec': 'H264'})
tool.gst_probe = native_probe
tool.worker()
'''
        response = subprocess.run([sys.executable, '-c', script], input=json.dumps(
            {'uri': SECRET, 'timeout': 1, 'latency': 1000}).encode(), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=5, check=True)
        self.assertEqual(json.loads(response.stdout), {'rtsp_connection': 'ok', 'codec': 'H264'})
        self.assertEqual(response.stderr, b'')
        self.assertNotIn(SECRET.encode(), response.stdout)

    def test_teardown_timeout_retains_already_emitted_safe_result(self):
        def runner(*args, **kwargs):
            raise subprocess.TimeoutExpired('safe', 20, output=b'{"rtsp_connection":"ok"}')
        self.assertEqual(probe(SECRET, runner=runner)['rtsp_connection'], 'ok')

    def test_cli_private_config_errors_are_hidden(self):
        output = io.StringIO()
        with patch('sys.argv', ['check_camera_rtsp.py', '--camera-id', 'missing', '--cameras', str(self.config),
                                '--env-file', str(self.env_file)]), contextlib.redirect_stdout(output):
            self.assertEqual(main(), 2)
        self.assertEqual(output.getvalue().strip(), 'rtsp_connection=failed reason=rtsp_config_error')


if __name__ == '__main__':
    unittest.main()
