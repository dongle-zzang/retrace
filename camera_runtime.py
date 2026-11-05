"""Per-camera frame health and retry policy; no GStreamer/GPU dependencies."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from collections import OrderedDict, deque
import math
import threading
import time
from rtsp_diagnostics import RTSP_REASONS

# Error values are codes owned by the application, never upstream error text.
ERRORS = frozenset({'rtsp_error', 'rtsp_timeout', 'source_eos', 'pipeline_error',
                    'pipeline_setup_failed', 'no_frames', 'output_stalled', 'rtsp_config_error'})
BACKOFF = (1, 2, 5, 10, 30)
# Minimum spacing between in-place source bin resets; keeps retrying at the cap.
RECOVERY_BACKOFF = (120, 300, 600)


@dataclass(frozen=True)
class CameraStatus:
    camera_id: str
    state: str = 'connecting'
    fps: float = 0.0
    last_frame_at: str | None = None
    last_frame_age: float | None = None
    last_error: str | None = None
    reconnect_count: int = 0
    generation: int = 0
    output_frame_age: float | None = None
    error_reason: str | None = None
    recovery_count: int = 0

    def to_dict(self):
        return asdict(self)


class CameraRuntime:
    """Frame observations are accepted only for the current running attempt.

    decoded frames determine health; output progress is checked separately to
    detect a stalled inference/JPEG chain. FPS uses a bounded five-second window.
    Recovery requires continuous frames, not merely a successful RTSP handshake.

    A source that stays silent (consecutive attempts without any frame) becomes
    due for an in-place source bin reset owned by the preview control loop; this
    class only decides when and how often, never touching GStreamer.
    """
    def __init__(self, camera_id, degraded_after=3.0, offline_after=10.0,
                 connect_timeout=30.0, min_fps=1.0, recovery_seconds=2.0,
                 clock=time.monotonic, wall_clock=lambda: datetime.now(timezone.utc)):
        values = (degraded_after, offline_after, connect_timeout, min_fps, recovery_seconds)
        if not all(math.isfinite(value) and value > 0 for value in values) or degraded_after >= offline_after:
            raise ValueError('Invalid camera health thresholds')
        self.clock, self.wall_clock = clock, wall_clock
        self.degraded_after, self.offline_after = degraded_after, offline_after
        self.connect_timeout, self.min_fps, self.recovery_seconds = connect_timeout, min_fps, recovery_seconds
        self._lock = threading.Lock()
        self.attempt_stop = threading.Event()
        self._status = CameraStatus(camera_id)
        self._frames = deque(maxlen=2048)
        self._started = clock()
        self._last_frame = self._last_output = self._continuous_since = None
        self._ever_frame = self._ever_output = None
        self._running = False
        self._watchdog_armed = False
        self._failures = 0
        self._retry_at = self._started
        self._stable_reset = False
        self._frameless_attempts = 0
        self._recoveries = 0
        self._recovery_not_before = self._started

    def begin_attempt(self, watchdog=True):
        with self._lock:
            return self._begin_attempt(watchdog)

    def _begin_attempt(self, watchdog):
            retry = self._status.generation > 0
            now = self.clock()
            self._started = now
            self._frames.clear()
            self._last_frame = self._last_output = self._continuous_since = None
            self.attempt_stop.clear()
            self._running = True
            self._watchdog_armed = watchdog
            self._stable_reset = False
            self._status = replace(self._status, state='reconnecting' if retry else 'connecting',
                                   fps=0.0, generation=self._status.generation + 1,
                                   reconnect_count=self._status.reconnect_count + int(retry))
            return self._status.generation

    def arm_watchdog(self, generation):
        """Begin RTSP timeout after synchronous model/plugin initialization."""
        with self._lock:
            if self._running and generation == self._status.generation:
                self._started = self.clock()
                self._watchdog_armed = True

    def observe_frame(self, generation):
        with self._lock:
            if not self._running or generation != self._status.generation:
                return
            now = self.clock()
            if self._last_frame is None or now - self._last_frame >= self.degraded_after:
                self._continuous_since = now
            self._last_frame = self._ever_frame = now
            self._frames.append(now)
            self._frameless_attempts = 0
            self._status = replace(self._status, last_frame_at=self.wall_clock().isoformat(timespec='milliseconds'))
            self._update(now)

    def observe_output(self, generation):
        with self._lock:
            if self._running and generation == self._status.generation:
                self._last_output = self._ever_output = self.clock()

    def publish(self, generation, put, output=False):
        """Atomically reject writes from failed/previous attempts.

        put must only update an in-memory store; never perform I/O or reenter
        CameraRuntime. This orders invalidation before any subsequent writes.
        """
        with self._lock:
            if not self._running or generation != self._status.generation or self.attempt_stop.is_set():
                return False
            if output:
                self._last_output = self._ever_output = self.clock()
            put()
            return True

    def _fps(self, now):
        while self._frames and now - self._frames[0] > 5:
            self._frames.popleft()
        if len(self._frames) < 2:
            return 0.0
        return (len(self._frames) - 1) / max(now - self._frames[0], 1e-9)

    def _update(self, now):
        fps = self._fps(now) if self._running else 0.0
        state = self._status.state
        if self._running and self._last_frame is not None:
            age = now - self._last_frame
            output_age = now - (self._last_output if self._last_output is not None else self._started)
            if age >= self.degraded_after or (fps < self.min_fps and now - self._continuous_since >= self.recovery_seconds) or output_age >= self.offline_after:
                state = 'degraded'
            elif now - self._continuous_since >= self.recovery_seconds:
                state = 'online'
                # Flapping sources retain their accumulated backoff until stable.
                if not self._stable_reset and now - self._continuous_since >= 30:
                    self._failures = 0
                    self._recoveries = 0
                    self._stable_reset = True
        self._status = replace(self._status, state=state, fps=round(fps, 2))

    def _fail(self, error):
        self._running = False
        self.attempt_stop.set()
        self._failures += 1
        if self._last_frame is None:
            self._frameless_attempts += 1
        self._retry_at = self.clock() + BACKOFF[min(self._failures - 1, len(BACKOFF) - 1)]
        self._status = replace(self._status, state='offline', fps=0.0,
                               last_error=error if error in ERRORS else 'pipeline_error')

    def check_health(self):
        """Expire a faulty attempt atomically and return its safe fault code."""
        with self._lock:
            now = self.clock()
            self._update(now)
            if not self._running or not self._watchdog_armed:
                return None
            error = None
            if self._last_frame is None:
                if now - self._started >= self.connect_timeout:
                    error = 'no_frames'
            elif now - self._last_frame >= self.offline_after:
                error = 'no_frames'
            elif now - (self._last_output if self._last_output is not None else self._started) >= self.offline_after:
                error = 'output_stalled'
            if error:
                self._fail(error)
            return error

    def fail(self, error, generation):
        with self._lock:
            if self._running and generation == self._status.generation:
                self._fail(error)

    def record_error_reason(self, reason, generation):
        """Keep Backend's coarse last_error; add a whitelisted preview diagnosis."""
        with self._lock:
            if generation != self._status.generation or reason not in RTSP_REASONS:
                return False
            if reason == 'rtsp_error' and self._status.error_reason is not None:
                return False
            changed = self._status.error_reason != reason
            self._status = replace(self._status, error_reason=reason)
            return changed

    def retry_delay(self):
        with self._lock:
            return max(0.0, self._retry_at - self.clock())

    def recovery_due(self):
        """Two consecutive attempts without a single frame and the spacing has elapsed."""
        with self._lock:
            return (not self._running and self._frameless_attempts >= 2
                    and self.clock() >= self._recovery_not_before)

    def recovery_mode(self):
        """'reset' for the first recovery since the last stable period, 'rebuild' afterwards."""
        with self._lock:
            # shortcut: 첫 caps부터 거부된 스트림은 재생성하지 않음, 재시작 후에도 안 뜨는 카메라가 계속 멈춰 있으면 재검토
            return 'reset' if self._recoveries == 0 or self._ever_frame is None else 'rebuild'

    def _defer_recovery(self):
        self._recoveries += 1
        self._recovery_not_before = self.clock() + RECOVERY_BACKOFF[min(self._recoveries - 1, len(RECOVERY_BACKOFF) - 1)]

    def begin_recovery(self):
        """Start the attempt that follows an in-place source reset; returns its generation."""
        with self._lock:
            self._defer_recovery()
            self._status = replace(self._status, recovery_count=self._status.recovery_count + 1)
            return self._begin_attempt(watchdog=True)

    def recovery_failed(self):
        """The reset itself failed: space the next one out; ordinary attempts continue."""
        with self._lock:
            self._defer_recovery()
            self._status = replace(self._status, state='offline', fps=0.0, last_error='pipeline_error')

    def snapshot(self):
        with self._lock:
            now = self.clock()
            self._update(now)
            return replace(self._status,
                           last_frame_age=None if self._ever_frame is None else round(now - self._ever_frame, 2),
                           output_frame_age=None if self._ever_output is None else round(now - self._ever_output, 2))


class CameraRuntimeManager:
    """Camera health and generation routing, independent of GPU lifecycle.

    Fixed mux/demux slots are assigned at startup. Native source bins reconnect
    themselves; retries here are observation epochs, not RTSP handshake counts.
    PTS associates decoded, tracked and encoded frames with their original epoch
    so queued buffers cannot republish old metadata/JPEG after a health failure.
    """
    def __init__(self, cameras, runtimes, stores, metadata_store, recovery_enabled=True):
        self.cameras_by_slot = dict(enumerate(cameras))
        self.slots_by_camera = {camera.camera_id: slot for slot, camera in self.cameras_by_slot.items()}
        self.slots_by_source = {camera.source_id: slot for slot, camera in self.cameras_by_slot.items()}
        if (not cameras or len(self.slots_by_camera) != len(cameras)
                or len(self.slots_by_source) != len(cameras)):
            raise ValueError('Camera identities must be nonempty and unique')
        self.runtimes, self.stores, self.metadata_store = runtimes, stores, metadata_store
        self.recovery_enabled = recovery_enabled
        self._lock = threading.Lock()
        self._frames = {slot: OrderedDict() for slot in self.cameras_by_slot}

    @property
    def batch_size(self):
        return len(self.cameras_by_slot)

    def camera_for_frame(self, source_id, pad_index):
        if source_id != pad_index or source_id not in self.cameras_by_slot:
            raise RuntimeError('DeepStream source_id/pad_index mapping mismatch')
        return self.cameras_by_slot[source_id]

    def begin(self):
        for runtime in self.runtimes.values():
            runtime.begin_attempt(watchdog=False)

    def arm_watchdogs(self):
        for runtime in self.runtimes.values():
            runtime.arm_watchdog(runtime.snapshot().generation)

    def _clear(self, slot):
        source_id = self.cameras_by_slot[slot].source_id
        self._frames[slot].clear()
        self.stores[source_id].clear()
        self.metadata_store.clear(source_id)

    def fail_source(self, slot, error):
        with self._lock:
            runtime = self.runtimes[self.cameras_by_slot[slot].source_id]
            runtime.fail(error, runtime.snapshot().generation)
            self._clear(slot)

    def record_error_reason(self, slot, reason):
        with self._lock:
            runtime = self.runtimes[self.cameras_by_slot[slot].source_id]
            return runtime.record_error_reason(reason, runtime.snapshot().generation)

    def tick(self):
        """Expire faulty attempts, restart due ones; return slots due for a source reset.

        Returned slots are left offline: the caller resets the source bin outside
        this lock and then calls recovered() or recovery_failed() for that slot.
        """
        due = []
        with self._lock:
            for slot, camera in self.cameras_by_slot.items():
                runtime = self.runtimes[camera.source_id]
                if runtime.check_health():
                    self._clear(slot)
                if runtime.snapshot().state == 'offline' and runtime.retry_delay() == 0:
                    if self.recovery_enabled and runtime.recovery_due():
                        due.append(slot)
                    else:
                        # No element creation or state change: nvurisrcbin owns RTSP retry.
                        runtime.begin_attempt()
        return due

    def recovery_mode(self, slot):
        return self.runtimes[self.cameras_by_slot[slot].source_id].recovery_mode()

    def recovered(self, slot):
        return self.runtimes[self.cameras_by_slot[slot].source_id].begin_recovery()

    def recovery_failed(self, slot):
        self.runtimes[self.cameras_by_slot[slot].source_id].recovery_failed()

    def observe_decoded(self, slot, pts):
        with self._lock:
            runtime = self.runtimes[self.cameras_by_slot[slot].source_id]
            status = runtime.snapshot()
            if runtime.attempt_stop.is_set() or pts is None:
                return False
            runtime.observe_frame(status.generation)
            frames = self._frames[slot]
            frames[pts] = status.generation
            frames.move_to_end(pts)
            while len(frames) > 512:
                frames.popitem(last=False)
            return True

    def frame_generation(self, slot, pts):
        with self._lock:
            return self._frames[slot].get(pts)

    def publish(self, slot, pts, put, output=False, generation=None):
        with self._lock:
            frame_generation = self._frames[slot].get(pts)
            if frame_generation is None or (generation is not None and generation != frame_generation):
                return False
            runtime = self.runtimes[self.cameras_by_slot[slot].source_id]
            return runtime.publish(frame_generation, put, output=output)
