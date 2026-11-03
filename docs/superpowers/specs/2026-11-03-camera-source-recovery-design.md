# 카메라 source bin 복구 설계 (2026-11-03)

상태: 구현됨 (2026-11-03, 서버 체크아웃, 브랜치 `claude/camera-source-recovery`). CPU 테스트 통과, GPU 검증(§6)과 배포는 대기.
운영 설명 본문은 [docs/operations.md](../../operations.md) "카메라 runtime lifecycle"의 "source 복구 정책"으로 옮겼다.

구현 메모(설계와 다른 점):

- §4-2 2단계: ghost pad target을 무조건 비우지 않는다. NULL 전이 후 target pad가 부모 element를 잃은 경우(plugin이
  pad-removed 없이 pad를 버린 경우)에만 `set_target(None)`한다. GStreamer는 부모가 있는 pad를 다시 add할 수 없으므로
  NULL을 거쳐도 유지되는 pad는 기존 링크로 데이터가 흐르고, 교체되는 pad는 기존 `pad-removed`(target 비움) →
  `pad-added`(재연결) 순서로 처리된다. 무조건 비우면 pad가 유지되는 경우 링크가 끊겨 복구되지 않는다.
- §4-1 `recovery_failed()`도 `_recoveries`를 올려 간격이 120 → 300 → 600초로 늘어난다(실패하는 재설정을 2분마다
  반복하지 않기 위해). `recovery_count`는 성공한 재설정만 센다.
- 재설정 실패 시 `source_reset_failed {camera_id, source_id, stage}`(stage는 null/sync)를 항상 출력한다.
관련 코드: `preview.py`, `camera_runtime.py`, `app.py`. 관련 문서: [docs/operations.md](../../operations.md) "카메라 runtime lifecycle".

## 1. 배경과 근거

### 증상

카메라는 살아 있는데 웹에서 일부 카메라만 영상이 안 뜨고, `retrace` 컨테이너를 재시작해야 돌아온다.
2026-11-03 기준 21대 중 9대가 비정상이었다(기동 후 한 번도 안 뜬 3대 + 운영 중 끊긴 뒤 영구 offline 6대).

### 원인 (서버 96시간 로그와 코드로 확정)

- 앱은 카메라별 RTSP 재연결을 DeepStream 7.0 `nvurisrcbin`의 내부 재연결에만 맡긴다. `CameraRuntimeManager.tick()`은
  offline 카메라의 backoff가 끝나면 `begin_attempt()`로 관측 generation만 올리고 GStreamer element는 건드리지 않는다.
  디코더 리셋이나 source bin 재설정 코드는 어디에도 없다.
- 끊긴 6대 모두 장애 이후 세대 주기가 정확히 60초(connect_timeout 30 + 최대 backoff 30)였다. bus 메시지 없이
  앱 watchdog(`no_frames`)만 돌고 있었고, source bin은 프레임도 메시지도 내지 않는 **침묵 상태**였다.
- 침묵이 뜻하는 것: `nvurisrcbin`의 watchdog은 rtspsrc 뒤(depay) 버퍼 수신만 보므로, RTSP 세션은 살아 있는데
  디코더 이후가 멈추면 native 재연결이 다시 동작하지 않는다.
- 로그로 확정된 두 경로:
  1. **디코더 경고 후 침묵** (b1_parking_gate): 마지막 프레임과 동시에 WARNING stream/DECODE, 이후 4일간 메시지 0건.
  2. **not-negotiated 후 침묵** (3f_auditorium_door, 6f_office_inner, b1_parking_overview, 6f_lobby): 프레임 정지와 동시에
     WARNING core not-negotiated, 25~30초 뒤 `GstRTSPSrcTimeout` 1회, 그 뒤 1~3일간 메시지 0건.
- 배제된 가설: 재연결 시 decoder pad 미연결(`source_link_error` 0건), PTS 라우팅 불일치(`output_stalled` + 작은
  `last_frame_age` 조합 없음).
- `nvurisrcbin` 7.0 구현 소스는 컨테이너에 없어 재연결 내부 동작은 Not verified다. 호스트의 DS 6.1 참조 구현
  (`deepstream_source_bin.c`의 `reset_source_pipeline`)은 bin 전체를 NULL → PLAYING으로 내렸다 올려 디코더까지 함께
  리셋하고 외부 pad를 건드리지 않는다.

같은 DECODE 경고를 받고도 복구된 카메라(5f_office_inner, b1_lobby)가 있으므로 경고 자체가 아니라 "그 뒤의 침묵"이
문제다. 따라서 트리거는 오류 종류가 아니라 **프레임 없는 세대의 연속 횟수**로 둔다.

## 2. 목표와 비목표

목표

- 침묵 상태의 카메라를 프로세스 재시작 없이 그 카메라만 자동 복구한다.
- 다른 카메라, 공유 inference/tracker, HTTP/WS 전송에 영향을 주지 않는다.
- 복구 동작이 `/streams.json`과 로그로 관측 가능하다.
- 문제가 생기면 코드 변경 없이 끌 수 있다.

비목표

- 기동 후 한 번도 안 뜬 3대(3f_terrace_4k, 5f_server_room, 7f_rooftop)의 원래 원인 규명. 복구 로직은 이들에도
  동작하지만 살아난다고 보장하지 않는다. `tools/check_camera_rtsp.py`로 별도 진단한다.
- `nvurisrcbin` 내부 동작 확인, source bin 제거/재생성, 새 nvstreammux 도입.
- 디코더가 PTS 없는 프레임을 내는지 구분하는 진단 카운터(필요하면 후속).

## 3. 접근 선택

| 접근 | 내용 | 판단 |
| --- | --- | --- |
| A. source bin 제자리 재설정 | `source-bin-N`을 NULL → PLAYING(부모와 동기화). mux sink pad, ghost pad, 프로브 유지 | **채택**. deepstream-app `reset_source_pipeline`과 같은 방식이라 legacy nvstreammux와의 조합이 검증돼 있다 |
| B. source bin 제거 후 재생성 | bin 제거, mux request pad release/재요청, 재연결 | PLAYING 중 mux request pad 조작이 추가 위험. A로 충분 |
| C. 프로세스 재시작 | 멈춘 카메라가 있으면 watchdog이 종료하고 compose가 재시작 | 전체 카메라가 1~2분 끊기고(engine 로드) 운영 중 자동 재시작 금지 원칙과 충돌 |

## 4. 설계

### 4-1. 복구 정책 (`camera_runtime.py`, GStreamer 의존 없음)

`CameraRuntime`에 추가한다.

- `_frameless_attempts`: 프레임을 한 장도 못 본 채 끝난 **연속** 시도 수. `_fail()`에서 그 시도의 `_last_frame`이
  `None`이면 +1, `observe_frame()`에서 0으로 초기화.
- `_recoveries`: 마지막 안정 수신 이후 수행한 복구 횟수. 30초 이상 연속 수신으로 backoff를 초기화하는 기존
  `_stable_reset` 시점에 0으로 초기화한다.
- `_recovery_not_before`: 다음 복구를 허용하는 monotonic 시각. 초기값은 시작 시각.
- `RECOVERY_BACKOFF = (120, 300, 600)`: 복구 간 최소 간격(초). `_recoveries`가 늘수록 커지고 600초에서 고정된다.
  살아나지 않는 카메라도 10분마다 계속 시도한다(포기하지 않음).
- `recovery_due()`: `not _running and _frameless_attempts >= 2 and clock() >= _recovery_not_before`.
- `begin_recovery()`: `_recoveries += 1`, `_recovery_not_before = clock() + RECOVERY_BACKOFF[min(_recoveries - 1, 2)]`,
  상태 `recovery_count`를 +1 한 뒤 `begin_attempt()`를 호출해 새 generation을 반환한다.
- `recovery_failed()`: `_recovery_not_before`만 위와 같이 미루고 `last_error='pipeline_error'`를 기록한다. 일반 시도는
  다음 tick의 `begin_attempt()`로 계속된다.
- `CameraStatus`에 `recovery_count: int = 0` 추가. `to_dict()`로 `/streams.json` runtime과 `camera_status` 로그에 실린다.

시간 흐름(기본 임계값 기준). 운영 중이던 카메라의 마지막 프레임을 T라 하면:

| 시각 | 사건 | `_frameless_attempts` |
| --- | --- | --- |
| T+10 | `no_frames` (이 시도에는 프레임이 있었음) | 0 |
| T+11 | 시도 2 시작 (backoff 1초) | 0 |
| T+41 | connect_timeout 30초 → `no_frames` | 1 |
| T+43 | 시도 3 시작 (backoff 2초) | 1 |
| T+73 | `no_frames` | 2 → 복구 조건 충족 |
| T+78 | backoff 5초 종료, tick이 복구 대상으로 반환 → 재설정 → 시도 4 시작, `recovery_count=1` | |

기동 후 한 번도 안 뜬 카메라는 PLAYING 후 약 63초에 첫 복구 대상이 된다. nvurisrcbin의 10초 간격 재연결이
먼저 여러 번 기회를 갖는다.

`CameraRuntimeManager`에 추가한다.

- 생성자 인자 `recovery_enabled=True`.
- `tick()`이 복구 대상 slot 목록을 반환한다. 기존 분기 `state == 'offline' and retry_delay() == 0`에서
  `recovery_enabled and runtime.recovery_due()`이면 `begin_attempt()`를 부르지 않고 slot을 목록에 넣는다.
  나머지 동작은 현재와 같다. `recovery_enabled=False`이면 항상 빈 목록을 반환하고 현재 동작을 유지한다.
- `recovered(slot)`: 해당 runtime의 `begin_recovery()`를 호출하고 generation을 반환한다.
- `recovery_failed(slot)`: 해당 runtime의 `recovery_failed()`를 호출한다.
- manager lock은 tick 안에서만 잡는다. GStreamer 상태 전환은 lock 밖, preview 제어 루프에서 수행한다.

### 4-2. GStreamer 재설정 (`preview.py` `run_shared_pipeline`)

- source bin 생성 루프에서 `source_bins[index] = source_bin`으로 참조를 보관한다.
- 제어 루프(bus polling과 `manager.tick()`을 돌리는 메인 스레드)에서 `due = manager.tick()` 후 `due`가 비어 있지
  않으면 **첫 slot 하나만** 재설정한다. 나머지는 offline 상태로 남아 다음 tick에 다시 반환된다(직렬화).
- `reset_source(index)` 절차:
  1. `source_bin.set_state(Gst.State.NULL)`. 반환이 `FAILURE`면 `manager.recovery_failed(index)` 후 종료.
  2. `source_bin.get_static_pad('src').set_target(None)`. `nvurisrcbin`이 NULL 전이에서 src pad를 제거하지 않는
     경우에도 `app.py` `on_pad_added`의 "multiple video tracks" 분기를 피하기 위해서다. 같은 pad 객체가 다시
     추가되면 기존 `set_target`이 다시 연결하고, 새 pad면 정상 경로로 연결된다.
  3. `source_bin.sync_state_with_parent()`. `nvurisrcbin`은 `async-handling=True`라 즉시 반환한다. `False`면
     `manager.recovery_failed(index)` 후 종료.
  4. `generation = manager.recovered(index)`.
  5. 로그 한 줄: `source_reset {"camera_id":..,"source_id":..,"generation":..,"recovery_count":..}`. diagnostics 여부와
     무관하게 항상 출력한다(드물고 중요한 사건). 주소·원문은 포함하지 않는다.
- 재설정은 `pipeline_stop`/`shutdown`이 set된 뒤에는 수행하지 않는다. tick 자체가 PLAYING 요청 후 루프에서만 돌므로
  startup 중 재설정은 없다.
- CLI 옵션 `--source-recovery` / `--no-source-recovery` (`argparse.BooleanOptionalAction`, 기본 켬). 값은
  `CameraRuntimeManager(recovery_enabled=...)`로 전달한다. compose command는 바꾸지 않는다.

기존 메커니즘이 처리하는 부분(새 코드 불필요):

- 실패 시 `_clear(slot)`이 PTS→generation 맵, FrameStore, metadata를 비웠으므로 mux/tracker/JPEG branch에 남은 옛
  버퍼는 `frame_generation()`이 `None`을 반환해 폐기된다.
- ghost pad에 붙은 `observe_decoded`/`source_event`/timing 프로브는 pad 객체가 유지되므로 그대로 동작한다.
- 재설정 중 도착하는 옛 bus 메시지는 `fail()`이 `_running=False`인 카메라에 no-op이다. `recovered()` 이후 도착한
  늦은 메시지는 새 시도를 한 번 더 실패시킬 수 있으나 다음 tick에서 재시도된다.
- mux sink pad는 request/release 없이 유지된다. 새 stream의 STREAM_START/CAPS/SEGMENT 이벤트는 deepstream-app이
  legacy mux와 같은 방식으로 처리한다.

### 4-3. 공개 계약

- `/streams.json` runtime 객체와 `camera_status` 로그에 `recovery_count`가 추가된다(additive). backend
  `ingest.normalize_status()`와 프론트는 특정 키만 읽으므로 영향 없다. `tests/test_camera_shared_pipeline.py`의
  `test_streams_schema_and_http_response_preserved` 키 집합을 갱신한다.
- `reconnect_count`/`generation`의 의미(Python 관측 세대)는 바뀌지 않는다. 복구로 시작한 시도도 generation을 올린다.
- `last_error`/`error_reason` 코드 집합은 바꾸지 않는다. backend 허용 목록을 수정하지 않는다.

## 5. 위험과 대응

| 위험 | 대응 |
| --- | --- |
| `not-negotiated` 경로가 카메라 파라미터 변경 때문이면 재설정 후에도 legacy mux가 새 caps를 거부할 수 있다 | 재설정 후 같은 시도에서 `error_reason=source_caps_error`가 기록되면 로그에서 식별 가능하다. 간격이 600초까지 늘어나 부담은 작다. 이 패턴이 실제로 관측되면 source bin 안에 `nvvideoconvert`+capsfilter를 두거나 새 nvstreammux를 검토하는 후속 과제로 넘긴다(현재 설계는 입력 converter 없이 mux 직결) |
| `set_state(NULL)`이 rtspsrc teardown 동안 수 초 블록해 제어 루프의 bus polling과 다른 카메라 tick이 지연된다 | tick당 1대만 재설정한다. HTTP/WS와 GPU 스트리밍 스레드는 별도라 다른 카메라 영상은 유지된다. deepstream-app도 같은 스레드 모델이다. native hang은 격리하지 않는다(기존 한계와 동일) |
| `nvurisrcbin` 7.0이 NULL 전이에서 내부 child를 제대로 정리하지 않을 수 있다 | 바깥 `source-bin-N`을 통째로 NULL로 내리므로 child 전체가 NULL이 된다. 그래도 복구되지 않으면 `recovery_count`만 늘고 상태는 offline으로 남아 로그로 드러난다 |
| 일시적 네트워크 흔들림에 디코더까지 리셋 | 프레임 없는 세대 2회(마지막 프레임 후 약 73초) 뒤에만 개입한다. nvurisrcbin의 10초 재연결이 먼저 동작한다 |
| 복구 루프가 GPU/NVDEC 자원을 반복 소모 | 간격 120→300→600초, 카메라당 직렬화. 안정 수신 30초 후 카운터 초기화 |
| 차량 모델 배포(`vehicle_deployment.py`) FPS 검증 중 재설정 | 해당 카메라는 이미 offline이라 기준선에 없다. 추가 처리 없음 |

## 6. 검증

### CPU 테스트 (서버 컨테이너 밖에서도 실행 가능)

`tests/test_camera_runtime.py`

- 프레임 없는 시도 2회 연속 후 `recovery_due()`가 참, 1회 후는 거짓.
- 프레임이 있었던 시도의 실패는 산입하지 않는다(위 시간 흐름 표 재현).
- `begin_recovery()`가 generation/reconnect_count/recovery_count를 올리고 간격이 120 → 300 → 600 → 600으로 늘어난다.
- 30초 안정 수신 후 `_recoveries`와 `_frameless_attempts`가 초기화된다.
- `recovery_failed()`가 간격을 미루고 `last_error='pipeline_error'`를 남긴다.
- `to_dict()`에 `recovery_count`가 있고 비밀값이 없다.

`tests/test_camera_shared_pipeline.py`

- `tick()`이 조건 충족 slot만 반환하고 그 slot에 `begin_attempt()`를 호출하지 않는다. `recovery_enabled=False`면 빈 목록과 기존 동작.
- fake Gst로 `run_shared_pipeline` 루프를 한 바퀴 돌려 복구 대상 `source-bin-N`에만 `set_state('NULL')` → `set_target(None)` → `sync_state_with_parent()`가 호출되고 다른 source bin은 상태 변화가 없음을 확인한다. 기존 `construct()`의 "source는 상태 변화 없음" 단언은 복구 조건이 없을 때 유지된다.
- `set_state`가 `FAILURE`를 반환하면 `recovery_failed`가 호출되고 pipeline은 계속 돈다.
- 스키마 테스트 키 집합에 `recovery_count` 추가.
- 출력에 `private`가 없다.

`tests/test_rtsp_inputs.py`: `--no-source-recovery` 파서 계약.

실행: `python3 -m unittest discover -s tests -p 'test_camera_*.py'`, `python3 -m unittest discover -s tests -p 'test_rtsp_inputs.py'`.

### GPU 서버 검증 (수동, 재시작 가능한 시간에)

1. 배포 후 `docker compose logs -t --since=10m retrace | rg 'DeepStream shared pipeline|source_reset'`로 정상 기동 확인.
2. 정상 카메라 1대의 RTSP를 3분 이상 차단(카메라 쪽 네트워크 또는 방화벽)한 뒤 복구한다. `/streams.json`에서 그 카메라가 재시작 없이 online으로 돌아오고 `recovery_count`가 1 이상인지, `source_reset` 로그가 1건인지 확인한다.
3. 재설정 전후 `stats source=... fps=` 줄에서 다른 카메라 fps가 유지되는지 확인한다.
4. 기동 후 안 뜬 3대는 `recovery_count`가 600초 간격으로 증가하며 다른 카메라에 영향이 없는지 본다.
5. 며칠 운영 후 `source_reset` 로그와 `camera_status`로 자연 발생한 침묵이 복구됐는지 확인한다. 재설정 후 `source_caps_error`가 반복되는 카메라가 있으면 §5의 후속 과제로 기록한다.

## 7. 문서 갱신

- `docs/operations.md` "카메라 runtime lifecycle": "native RTSP reconnect가 멈추거나 output chain이 영구적으로 실패하면 운영자가 재시작해야 한다" 문단을 복구 정책(조건, 간격, 플래그, `recovery_count`, `source_reset` 로그)으로 바꾼다. "source bin이나 mux pad를 동적으로 remove/re-add하지 않는다"는 "제거/재생성하지 않고 제자리에서 NULL → PLAYING 재설정만 한다"로 고친다.
- `docs/agent/architecture.md` `camera_runtime.py` 항목에 복구 정책 소유를 한 줄 추가한다.
- `docs/project-status.md` §5의 "영상이 나오지 않는 카메라 3대" 항목은 유지한다(별개 원인).
- 구현 완료 후 이 문서는 삭제하거나 "구현됨"으로 표시한다. 본문은 operations.md로 옮긴다.

## 8. 구현 순서 (서버 체크아웃)

1. `camera_runtime.py`: `CameraStatus.recovery_count`, `RECOVERY_BACKOFF`, `_frameless_attempts`/`_recoveries`/`_recovery_not_before`, `recovery_due()`, `begin_recovery()`, `recovery_failed()`, manager `recovery_enabled`/`tick()` 반환/`recovered()`/`recovery_failed()`. 테스트 먼저 작성.
2. `preview.py`: `--source-recovery` 옵션, `source_bins` 보관, `reset_source()`, 제어 루프 연결, `source_reset` 로그. fake Gst 테스트 먼저 작성.
3. 스키마 테스트 키 갱신, 파서 테스트.
4. 문서 갱신(§7).
5. 서버 배포는 사용자 확인 후 사용자가 직접 수행한다. GPU 검증(§6)은 재시작 가능한 시간에 한다.
