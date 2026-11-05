# 카메라 source 2단계 복구(mux pad 재생성) 설계 (2026-11-04)

상태: 구현됨 (2026-11-04, 서버 체크아웃). CPU 테스트 통과, 배포와 GPU 검증(§6)은 대기.
후속: 복구가 제어 루프를 멈춘 사고와 감시는 [3단계 복구 감시 설계](2026-11-05-camera-recovery-watchdog-design.md).

구현 메모(설계와 다른 점):

- §4-2 `attach_source()`는 `create_preview_source_bin` 예외를 감싸지 않는다(기동 실패 로그의 예외 type 이름을 유지).
  request/link 실패만 `SourceAttachError(stage)`로 올리므로 기동 중 그 두 실패의 setup 로그 type은
  `RuntimeError`에서 `SourceAttachError`로 바뀐다. `rebuild_source()`는 그 외 예외를 stage `create`로 기록한다.
- §4-2 재생성 중 request/link/sync 실패 뒤에는 부분적으로 붙은 새 bin과 요청된 pad를 `detach_source()`로 바로
  정리해 slot을 비운다(설계대로 다음 복구는 생성부터 시작). 분리 순서(flush-stop → unlink → release → remove)는
  2단계와 같은 함수다.
- 제어 루프는 `source_bins[slot]`이 비어 있으면 `recovery_mode`와 무관하게 재생성한다(비어 있는 slot은 재설정할 bin이 없다).
- §4-4 `factory` 검증은 `rtsp_diagnostics.safe_factory()`로 두고 `classify_bus_error()`와 WARNING 로그가 함께 쓴다.
  WARNING 줄 형식은 `pipeline_source=X: GStreamer WARNING factory=<값>; details hidden`이다.
- §6 fake Gst 테스트는 bus pop마다 5초씩 시계를 진행해 재설정(복구 1) → 120초 뒤 재생성(복구 2) 순서를 실제
  정책 코드로 만든다. 요청 실패 테스트는 두 번째 `sink_0` 요청만 None으로 만들어 600초 뒤 생성부터 재시도함을 본다.
전제: [2026-11-03 1단계 설계](2026-11-03-camera-source-recovery-design.md)와 그 구현(main 0ef8320: `camera_runtime.py`
`recovery_due/begin_recovery/recovery_failed`, `preview.py` `reset_source()`). 이 문서는 그 위에 한 단계를 더 얹는다.
관련 코드: `preview.py`, `camera_runtime.py`, `app.py`, `rtsp_diagnostics.py`. 관련 문서: [docs/operations.md](../../operations.md) "source 복구 정책".

## 1. 배경과 근거 (운영 1일 결과, 2026-11-03 06:18 기동 ~ 11-04)

1단계(source bin 제자리 NULL → PLAYING 재설정)는 두 경로 중 하나만 복구했다.

| 경로 | 카메라 | 결과 |
| --- | --- | --- |
| RTSP timeout 후 침묵 | 1f_lobby, 6f_lobby | 재설정 0.2초 뒤 `NVMM video pad linked`, online 복귀 |
| not-negotiated 후 침묵 | b1_parking_overview(16시간, 재설정 94회), 7f_lobby(재설정 6회) | 매번 같은 실패 패턴 |

실패 패턴(두 카메라 동일): `source_reset` 직후 100ms 안에 `pipeline_source=all` WARNING 2~5건(공유 element: mux/
tracker/nvinfer/demux 중 하나) → 0.5~20초 뒤 `pipeline_source=N` WARNING 1건 → 프레임 없음. 10분 간격으로 반복.

같은 시각 진단 도구(`tools/check_camera_rtsp.py --source-mode nvurisrcbin`)는 b1_parking_overview를 **새** nvurisrcbin과
**새** nvstreammux로 1920×1080 NV12로 협상하고 프레임을 받았다(`direct_mux_accepts_caps: true`). 기동 시 협상값(30fps)과
같다. 즉 카메라는 정상이고, 운영 pipeline과 진단 도구의 차이는 **not-negotiated를 한 번 겪은 기존 mux sink pad**뿐이다.

결론: 1단계는 source bin만 새로 만들고 mux의 `sink_N` pad는 그대로 두므로, pad에 남은 협상 상태(caps 또는 per-pad
stream 상태)가 풀리지 않는다. 프로세스 재시작이 통하는 이유는 mux가 처음부터 협상하기 때문이다. 정확히 어느 element가
거부하는지(nvstreammux인지 tracker인지)와 무엇이 바뀌었는지(진단 도구는 프레임레이트·interlace 필드를 보여주지 않음)는
Not verified다. 설계는 이에 의존하지 않는다: pad를 새로 받으면 둘 다 초기화된다.

## 2. 목표와 비목표

목표

- 1단계 재설정으로 돌아오지 않는 카메라를 프로세스 재시작 없이 복구한다.
- 1단계는 그대로 유지한다(timeout 경로에 유효하고 가볍다).
- 어느 element가 거부하는지 로그로 확정할 수 있게 한다.
- 문제가 생기면 코드 변경 없이 2단계만 끌 수 있다.

비목표

- 새 nvstreammux(`USE_NEW_NVSTREAMMUX`) 전환, source bin 안 converter/capsfilter 추가, mux 재생성.
- nvurisrcbin 내부 동작 확인.
- 기동 후 안 뜨는 3대(3f_terrace_4k 인증 거절, 5f_server_room·7f_rooftop TCP 도달 불가)의 원인 처리.

## 3. 접근 선택

| 접근 | 내용 | 판단 |
| --- | --- | --- |
| A. mux sink pad release/재요청 + source bin 재생성 | NVIDIA `runtime_source_add_delete` 샘플이 legacy nvstreammux에서 쓰는 순서 | **채택**. pad 상태와 caps가 함께 초기화되고 slot/source_id/demux pad는 유지 |
| B. source bin 안 nvvideoconvert + capsfilter로 caps 고정 | 해상도/포맷 변경만 흡수 | 프레임레이트·interlace 변경은 못 막고 21대에 변환기 추가. 거부 원인이 caps가 아니면 효과 없음 |
| C. 새 nvstreammux | 런타임 해상도 변경 지원 | 속성·batch 동작이 다른 pipeline 전체 변경 |

## 4. 설계

### 4-1. 정책 (`camera_runtime.py`)

- `CameraRuntime.recovery_mode()`: `_recoveries == 0`이면 `'reset'`, 아니면 `'rebuild'`. `_recoveries`는 1단계 구현대로
  성공·실패한 복구마다 늘고 30초 안정 수신 후 0이 된다. 즉 마지막 안정 이후 **첫 복구는 재설정, 두 번째부터는 재생성**이다.
  b1_parking_overview 같은 경우 장애 후 약 73초에 재설정, 그로부터 120초 뒤(약 3분 20초)에 재생성으로 넘어간다.
- 간격(120 → 300 → 600초), `recovery_due()`, `begin_recovery()`, `recovery_failed()`, `recovery_count`는 그대로다.
  `recovery_count`는 성공한 재설정과 재생성을 함께 센다.
- `CameraRuntimeManager.recovery_mode(slot)`를 추가한다. `tick()`의 반환(due slot 목록)은 바꾸지 않는다.

### 4-2. 재생성 절차 (`preview.py` `run_shared_pipeline`)

생성 루프에서 source 하나를 붙이는 코드를 `attach_source(index)`로 분리해 기동과 재생성이 같은 코드를 쓴다. 이 함수는
`create_preview_source_bin` 호출, src ghost pad에 `observe_decoded`/`source_event`/timing 프로브 부착, `pipeline.add`,
`mux.request_pad_simple(f"sink_{index}")`, 링크까지 맡고 `source_bins[index]`, `mux_pads[index]`를 갱신한다. 기동 전용
`startup_probe`는 다시 붙여도 무해하다(첫 버퍼에서 제거되고 `mark`는 중복을 무시).

제어 루프: `due = manager.tick()` 뒤 `manager.recovery_mode(slot)`이 `'rebuild'`이고 `args.source_recovery_rebuild`가
참이면 `rebuild_source(slot)`, 아니면 기존 `reset_source(slot)`. tick당 1대.

`rebuild_source(index)` (NVIDIA `runtime_source_add_delete`의 `stop_release_source` + `add_sources` 순서):

1. **NULL**: `source_bin.set_state(Gst.State.NULL)`. `ASYNC`면 `get_state(timeout=5초)`로 기다린다. `FAILURE`거나 5초 안에
   NULL이 되지 않으면 `failed("null")`로 종료한다(bin은 그대로 둔다. 다음 복구 때 다시 시도).
2. **분리**: `mux_pad.send_event(Gst.Event.new_flush_stop(False))`(샘플과 동일, pad의 flushing 상태 해제) →
   `ghost_pad.unlink(mux_pad)` → `mux.release_request_pad(mux_pad)` → `pipeline.remove(source_bin)`.
   `requested_pads`에서 이 pad 항목을 제거하고 `source_bins[index] = mux_pads[index] = None`으로 둔다(종료 시 이중 release 방지).
3. **생성**: `attach_source(index)`. 새 bin은 같은 이름 `source-bin-{index}`를 쓴다(옛 bin을 pipeline에서 제거한 뒤라
   이름 충돌 없음). `create_preview_source_bin`이 예외를 내면 `failed("create")`, `request_pad_simple`이 None이면
   `failed("request")`, 링크 실패면 `failed("link")`.
4. **PLAYING**: `source_bin.sync_state_with_parent()`. 거짓이면 `failed("sync")`.
5. `generation = manager.recovered(index)` → 로그 `source_rebuild {"camera_id","source_id","generation","recovery_count"}`.

`failed(stage)`: `manager.recovery_failed(index)` + 로그 `source_rebuild_failed {"camera_id","source_id","stage"}`. 2단계
이후 실패(`create`/`request`/`link`/`sync`)로 slot에 bin이 없는 상태가 되면 `rebuild_source`는 다음 호출에서 1·2단계를
건너뛰고 3단계부터 다시 시도한다(`source_bins[index] is None`로 판단). bin이 없는 동안 그 카메라는 offline이고 다른
카메라에는 영향이 없다.

기존 메커니즘이 처리하는 부분:

- 옛 bin이 1단계 NULL 전이 중 올린 bus 메시지는 pipeline bus에 이미 들어와 있고, `message_scope`는 부모를 거슬러
  올라가며 **이름**으로 `source-bin-{index}`를 찾으므로 제거된(부모 없는) bin이어도 해당 index로 해석돼 `fail_source`
  no-op이 된다. 제거 이후의 메시지는 pipeline bus에 오지 않는다. 이 동작을 테스트로 고정한다(§6).
- 실패 시 `_clear(slot)`이 PTS→generation 맵을 비우므로 옛 bin의 잔여 버퍼는 폐기된다.
- 새 bin의 프로브는 `attach_source`가 다시 붙인다. 옛 ghost pad의 프로브는 pad와 함께 사라진다.

### 4-3. 출력 branch 보호 (demux `src_N`)

mux pad를 release하면 legacy nvstreammux가 하류에 `GST_NVEVENT_PAD_DELETED`(와 stream EOS)를 보내고, nvstreamdemux가
그 source의 `src_N`으로 EOS를 내보낼 수 있다(Not verified). JPEG branch(queue → osd → nvjpegenc → appsink)가 EOS를
받으면 이후 버퍼를 거부해 재생성 후에도 `output_stalled`가 영구화된다. 예방으로 demux `src_N` pad에
`EVENT_DOWNSTREAM` 프로브를 두고 EOS만 DROP한다(기존 `source_event`가 source bin src pad에서 하는 것과 같은 패턴).
다른 이벤트는 통과시킨다.

### 4-4. 진단 보강 (`rtsp_diagnostics.py`, `preview.py`)

- `classify_bus_error()`가 반환하는 dict에 `factory` 필드를 추가한다. 값은 `message.src.get_factory().get_name()`을
  `^[a-z0-9_]{1,32}$`로 검사한 것, 아니면 `'other'`. 원문 메시지는 여전히 출력하지 않는다.
- diagnostics 모드의 `pipeline_source=X: GStreamer WARNING; details hidden` 줄에 `factory=<값>`을 덧붙인다. ERROR 쪽
  `source_rtsp` JSON에도 `factory`가 실린다. `pipeline_source=all` 경고가 nvstreammux에서 오는지 tracker에서 오는지 이
  줄로 확정한다.

### 4-5. 설정과 계약

- CLI `--source-recovery-rebuild` / `--no-source-recovery-rebuild` (`BooleanOptionalAction`, 기본 켬).
  `--no-source-recovery`면 둘 다 꺼진다. compose command는 바꾸지 않는다.
- `/streams.json` runtime, `camera_status`의 필드 집합은 바꾸지 않는다(`recovery_count`가 두 단계를 함께 센다). 로그
  `source_rebuild`/`source_rebuild_failed`가 새로 생긴다. backend/프론트 변경 없음.

## 5. 위험과 대응

| 위험 | 대응 |
| --- | --- |
| PLAYING 중 legacy mux request pad release/재요청 | NVIDIA runtime_source_add_delete 샘플이 지원하는 경로. tick당 1대, 두 번째 복구부터만. `--no-source-recovery-rebuild`로 즉시 1단계만으로 복귀 |
| 옛 bin 메시지가 `all`로 해석돼 공유 pipeline 종료 | `message_scope`가 이름으로 해석하므로 발생하지 않음. CPU 테스트로 고정 |
| demux `src_N` EOS로 JPEG branch 영구 정지 | §4-3 EOS DROP 프로브. GPU 검증에서 재생성 후 `output_frame_age`가 작은지 확인 |
| tracker/nvinfer per-stream 상태 | mux pad delete/add 이벤트로 정리되는 것이 샘플의 전제. 재생성 후 주차장 카메라(b1_parking_overview 포함)의 `vehicles` 수가 `stats` 로그에 정상 출력되는지 GPU 확인 |
| `set_state(NULL)`이 끝나지 않음 | 5초 timeout 후 `failed("null")`. 제어 루프 최대 5초 정지. HTTP/WS/GPU 스레드는 별도 |
| 도달 불가 카메라 3대가 10분마다 재생성 | element 생성·pad 요청 비용은 작다. 수일 운영 후 메모리/GPU 메모리 추이를 `nvidia-smi`와 컨테이너 메모리로 확인. 늘어나면 `--no-source-recovery-rebuild`로 끄고 보고 |
| `attach_source` 분리 중 기동 경로 회귀 | 기존 `construct()` 테스트(1/2/4/8/25대, diagnostics, vehicle)가 그대로 통과해야 한다 |

## 6. 검증

### CPU 테스트

`tests/test_camera_runtime.py`

- `recovery_mode()`: 처음 `'reset'`, `begin_recovery()` 또는 `recovery_failed()` 뒤 `'rebuild'`, 30초 안정 수신 후 다시 `'reset'`.

`tests/test_camera_shared_pipeline.py` (fake Gst 확장: `Pad.unlink/send_event`, `Element.remove/get_state`, `Gst.Event.new_flush_stop`)

- 두 번째 복구에서 due slot의 옛 bin에만 `set_state('NULL')` → mux pad `send_event(flush_stop)` → `unlink` →
  `release_request_pad(mux, sink_N)` → `pipeline.remove(old)` → `create_preview_source_bin(index, 같은 uri)` 재호출 →
  새 bin src pad에 `observe_decoded`/`source_event` 프로브 → `request_pad_simple('sink_N')` 재요청 → 링크 →
  `sync_state_with_parent()` 순서가 일어나고 다른 bin은 무상태 변화. `source_rebuild` 로그 1건.
- 종료 시 옛 mux pad는 다시 release되지 않고 새 pad는 release된다(release 목록에 `sink_N`이 정확히 2번).
- `request_pad_simple`이 None이면 `source_rebuild_failed stage=request`, slot에 bin 없음, 다음 복구에서 생성부터 재시도.
- `message_scope`: 부모가 None인 `source-bin-1` 아래 element의 메시지가 `('1', 'rtsp_error')`로 해석된다.
- demux `src_N`의 EOS 프로브가 EOS를 DROP하고 다른 이벤트는 OK를 반환한다.
- 기존 `construct()` 단언 전부 유지.

`tests/test_camera_rtsp_diagnostics.py`: `factory` 필드(허용 문자 검사, 미지 값은 `other`). 기존 단언의 dict 비교는 필드 추가에 맞게 갱신.

`tests/test_rtsp_inputs.py`: `--no-source-recovery-rebuild` 파서 계약.

실행: `python3 -m unittest discover -s tests -p 'test_camera_*.py'`, `python3 -m unittest discover -s tests -p 'test_rtsp_inputs.py'`.

### GPU 서버 검증 (수동)

1. 배포(restart) 후 `docker compose logs -t --since=10m retrace | grep -E 'DeepStream shared pipeline|source_re'`로 정상 기동 확인.
2. 자연 발생 대기: not-negotiated 경로가 하루 한두 번 생긴다. 발생 시 기대 순서는 `source_reset`(복구 1) → 120초 뒤
   `source_rebuild`(복구 2) → `NVMM video pad linked` → online. `/streams.json`에서 그 카메라의 `recovery_count` 2,
   `output_frame_age` 작음, 주차장 카메라면 `stats` 로그 `vehicles` 값 정상.
3. 재생성 전후 다른 카메라 `stats ... fps=`가 유지되는지 확인.
4. 재생성 직후 `pipeline_source=all: GStreamer WARNING factory=...` 줄로 거부 element를 기록해 §1의 Not verified를 갱신한다.
5. 수일 운영 후 컨테이너 메모리와 `nvidia-smi` 사용량 추이 확인(도달 불가 3대의 10분 주기 재생성 영향).

## 7. 문서 갱신

- `docs/operations.md` "source 복구 정책": 두 번째 복구부터 재생성, `source_rebuild`/`source_rebuild_failed` 로그,
  `--no-source-recovery-rebuild`, demux EOS 보호, 경고 로그의 `factory`. "source bin이나 mux pad를 제거/재생성하지 않으며"
  문장을 실제 동작으로 고친다.
- `docs/agent/architecture.md` `preview.py` 항목에 `attach_source`/재생성 한 줄.
- `docs/agent/testing.md` 표의 카메라 runtime/reconnect 줄에 재생성 테스트 언급.
- 2026-11-03 설계 문서 상단에 이 문서 링크 한 줄.

## 8. 구현 순서 (서버 체크아웃, 테스트 먼저)

1. `camera_runtime.py`: `recovery_mode()`, manager `recovery_mode(slot)` + 테스트.
2. `rtsp_diagnostics.py`: `factory` 필드 + 테스트. `preview.py` WARNING 로그에 `factory`.
3. `preview.py`: `attach_source()` 분리(기존 `construct()` 테스트 통과 확인) → demux `src_N` EOS DROP 프로브 + 테스트 →
   `rebuild_source()` + 제어 루프 분기 + `--source-recovery-rebuild` + fake Gst 테스트(§6).
4. 문서 갱신(§7).
5. 배포는 사용자 확인 후 재시작 가능한 시간에. GPU 검증(§6)은 자연 발생을 기다린다.
