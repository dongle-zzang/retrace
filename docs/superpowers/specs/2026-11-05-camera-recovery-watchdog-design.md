# 카메라 복구 감시(watchdog) 설계 (2026-11-05)

상태: 구현됨 (2026-11-05). CPU 테스트 통과, 배포와 GPU 검증(§6)은 대기.

구현 메모(설계와 다른 점):

- §4-2 작업 스레드가 끝났거나 포기한 tick에는 새 복구를 시작하지 않고 다음 tick으로 넘긴다. 같은 tick의 `due`는
  작업이 끝나기 전에 계산됐을 수 있어(방금 복구한 slot이 아직 due로 남음) 같은 카메라를 곧바로 다시 복구할 수 있기 때문이다.
- mode는 감시 쪽(제어 루프 스레드)에서 정해 `recover_source(index, mode)`로 넘긴다. 로그의 mode와 실제 실행이 같다.
- kick의 `send_event`는 예외를 삼킨다. kick 실패가 제어 루프를 끝내면 이 설계의 목표에 어긋난다.
- 종료 join 시간은 상수 `RECOVERY_JOIN_TIMEOUT = 5`. 세 로그의 `seconds`는 모두 작업 시작 기준(hung 30, abandoned 40 전후).
- §4-3 flush_stop은 NULL이 FAILURE가 아닐 때만 보낸다(실패하면 기존대로 `source_reset_failed`로 끝나고, 다음 복구가
  재생성이면 `detach_source()`가 보낸다).
- 테스트 harness는 `time`을 fake `monotonic`(test.now)으로, `threading.Thread`를 기록용 subclass로, `faulthandler`를
  기록용 fake로 주입하고 bus pop마다 작업 스레드 완료를 기다린다(멈춘 NULL 제외). SIGUSR1 등록은 테스트가 없다.
전제: [1단계 재설정](2026-11-03-camera-source-recovery-design.md), [2단계 재생성](2026-11-04-camera-source-rebuild-design.md),
예외 격리(6461596), 미연결 카메라 재설정만(eb5e699). 관련 코드: `preview.py` `recover_source`/`reset_source`/`rebuild_source`와
제어 루프, `camera_runtime.py`.

## 1. 배경과 근거 (2026-11-04~05 운영)

- 11-05 01:18 UTC 확인 시 `camera_status` 로그가 11-04 07:12:20 이후 한 줄도 없었다(정상은 1분에 21대 × 12회 = 252줄).
  `/streams.json`도 그 시각에 굳어 있었다. b1_parking_gate는 `degraded`인 채 `last_frame_age` 64222초였는데, 프레임이 10초만
  끊겨도 `offline`으로 바꾸는 `check_health`가 17.8시간 동안 한 번도 돌지 않았다는 뜻이다. 도달 불가 3대의
  `recovery_count`도 8에서 멈췄다. 영상 스레드는 별도라 나머지 카메라 영상은 계속 나왔고, 그래서 "잘 뜬다"로 보였다.
- 멈춘 시각에 돌던 복구는 6f_lobby의 **1단계 재설정** 하나다. 근거: 마지막 프레임 07:11:04, 세 번째 시도 실패 07:12:17,
  backoff 5초 뒤 07:12:22 재설정 시작. 도달 불가 3대의 재생성은 07:06:48에 끝났고 다음은 07:16:48이었다. 6f_lobby의
  `source_reset` 줄은 없다. 즉 `reset_source()`가 `source_reset` 로그에 이르기 전, `set_state(NULL)` 또는
  `sync_state_with_parent()`에서 돌아오지 않았다.
- 정확한 native 호출은 Not verified다. 호스트 py-spy가 컨테이너 Python을 읽지 못했다. 유력한 메커니즘은 이렇다. 정상
  동작하던 스트림이 침묵하면, 그 source의 스트리밍 스레드가 mux sink pad 안에서 하류를 기다리며 막혀 있다. `set_state(NULL)`은
  pad 비활성화 때 그 스레드의 stream lock을 기다리므로 교착된다. "침묵"의 정체가 이 막힌 push일 수 있다.
  한 번도 협상되지 않은 도달 불가 3대는 막힌 push가 없어 같은 호출을 수백 번 해도 멈추지 않았다.
- 이 위험은 1단계(0ef8320, 11-03)부터 있었다. 1단계는 6f_lobby와 1f_lobby를 실제로 살렸지만, 같은 경로에서 한 번 멈출 때의
  대가가 모든 카메라의 상태 갱신과 복구가 영구 정지하는 것이었다.

## 2. 목표와 비목표

목표

- 복구 중 native 호출이 멈춰도 제어 루프(상태 갱신, offline 전환, `camera_status`, 다른 카메라 복구, 차량 모델 배포 tick)는
  계속 돈다.
- 멈춤을 로그로 즉시 드러내고, 어느 Python 줄에서 멈췄는지 자동으로 남긴다.
- 멈춘 호출을 한 번 풀어 보고, 풀리면 정상 복구로 이어간다. 안 풀리면 그 카메라만 복구 대상에서 빼고 운영자에게 알린다.

비목표

- 멈춘 native 호출의 강제 중단(GStreamer에는 그런 API가 없다), 멈춘 bin 강제 제거, 프로세스 자동 재시작.
- 복구 간격·조건·재설정/재생성 절차 자체의 변경(아래 §4-3 flush_stop 한 줄만 추가).
- `/streams.json` 스키마 변경, backend/프론트 변경.

## 3. 접근 선택

| 접근 | 내용 | 판단 |
| --- | --- | --- |
| A. 복구를 작업 스레드에서 실행 + 제어 루프가 감시 | 멈추면 그 스레드만 갇히고 루프는 계속. 시간 초과 시 로그·스택 덤프·kick·포기 | **채택**. 기존 복구 함수는 그대로 두고 호출 위치만 바꾼다 |
| B. `set_state` 대신 비동기 상태 전환 후 timeout | 복구 함수마다 상태 대기 로직 재작성 | GStreamer의 동기 대기(pad 비활성화)는 비동기로 바꿔도 같은 스레드에서 막힌다. 효과 없음 |
| C. 루프 정지를 감지하면 프로세스 종료 → compose가 재시작 | 21대 전부 1~2분 끊김 | 운영 중 자동 재시작 금지 원칙과 충돌. 한 카메라 때문에 전체를 끊음 |

## 4. 설계

### 4-1. 정책 (`camera_runtime.py`)

- `CameraRuntime.abandon_recovery()`: `_recovery_abandoned = True`. `recovery_due()`는 이 플래그가 참이면 거짓을 반환한다.
  그동안 `tick()`은 그 카메라에 일반 관측 재시도(`begin_attempt`)만 한다(1단계 이전 동작, 무해).
- `begin_recovery()`와 `recovery_failed()`는 플래그를 지운다. 포기한 작업 스레드가 나중에 스스로 끝나면 그 카메라는 다시
  복구 대상이 된다.
- `CameraRuntimeManager.abandon_recovery(slot)` 위임 메서드.
- `CameraStatus` 필드는 추가하지 않는다. 포기 상태는 로그(§4-4)와 멈춘 `recovery_count`로 드러난다.

### 4-2. 작업 스레드와 감시 (`preview.py` `run_shared_pipeline`)

상수: `RECOVERY_HUNG_AFTER = 30`(초), `RECOVERY_KICK_WAIT = 10`(초). CLI 옵션은 만들지 않는다.

제어 루프는 `recover_source(due[0])`를 직접 부르지 않고 `supervise_recovery(due)`를 부른다. 상태는 지역 변수 하나
`inflight`(없으면 None, 있으면 slot, mode, thread, 시작 시각, kick 시각)다.

`supervise_recovery(due)` (매 tick):

1. `inflight`가 있고 스레드가 끝났으면: kick을 보냈던 경우 `source_recovery_unblocked {camera_id, source_id, mode, seconds}`를
   출력하고 `inflight = None`. kick이 없었으면 조용히 `None`(복구 함수가 이미 자기 로그를 남겼다).
2. `inflight`가 있고 아직 돌고 있으면:
   - 시작 후 `RECOVERY_HUNG_AFTER`가 지났고 kick 전이면: `source_recovery_hung {camera_id, source_id, mode, seconds}` 출력 →
     `faulthandler.dump_traceback(all_threads=True)`로 모든 Python 스레드의 현재 줄을 stderr에 남김(파일·함수·줄 번호만,
     지역 변수 없음) → **kick**: `mux_pads[slot]`이 있으면 `send_event(Gst.Event.new_flush_start())`. kick 시각 기록.
   - kick 후 `RECOVERY_KICK_WAIT`가 지났으면: `source_recovery_abandoned {camera_id, source_id, mode, seconds}` 출력 →
     `manager.abandon_recovery(slot)` → `inflight = None`. 스레드는 버린다(daemon). 다른 카메라 복구는 다음 tick부터 가능하다.
   - 그 외에는 아무것도 하지 않는다. 새 복구도 시작하지 않는다(한 번에 하나).
3. `inflight`가 없고 `due`가 있고 종료 중이 아니면: `due[0]`의 mode를 지금처럼 정하고, `recover_source(slot)`를 실행하는
   daemon 스레드 `source-recovery-{slot}`을 시작해 `inflight`에 기록한다. `recover_source`의 예외 격리는 그대로 스레드 안에서
   동작한다.

kick의 의도: 막힌 스트리밍 스레드가 mux sink pad에서 FLUSHING을 받고 빠져나오면 stream lock이 풀려 `set_state(NULL)`이
돌아온다. 실제로 legacy nvstreammux가 flush_start로 깨어나는지는 Not verified이고, 이 로그(`unblocked` 대 `abandoned`)가
검증 수단이다. kick은 멈췄을 때만 보내므로 정상 경로에는 영향이 없다.

동시성: 감시와 작업 스레드가 공유하는 Python 상태는 `source_bins`/`mux_pads`(slot 키 dict), `requested_pads`(list
append/remove)와 manager다. manager와 runtime은 자체 lock이 있다. dict 키별 대입과 list append/remove는 GIL 아래 원자적이다.
포기된 스레드가 나중에 끝나면서 다른 slot의 작업 스레드와 겹칠 수 있다. 서로 다른 bin이고 GStreamer의 `add`/`remove`/
`request_pad`는 스레드 안전이므로 허용한다.

### 4-3. NULL 뒤 flush_stop (`reset_source`)

`reset_source()`에서 `set_state(NULL)`이 돌아온 직후, `mux_pads[index]`에 `send_event(Gst.Event.new_flush_stop(False))`를
보낸다. kick으로 flushing이 된 mux pad를 재사용하려면 반드시 필요하다. kick이 없던 경우에도 NVIDIA `runtime_source_add_delete`
샘플이 NULL 뒤에 같은 이벤트를 보내므로 무해하다. 재설정된 새 스트림의 segment도 초기화된다. `rebuild_source()`는
`detach_source()`가 이미 보낸다.

### 4-4. 진단

- 로그 세 종류를 항상 출력한다(diagnostics 여부 무관): `source_recovery_hung`, `source_recovery_unblocked`,
  `source_recovery_abandoned`. 주소·원문 없음.
- hung 시 `faulthandler.dump_traceback(all_threads=True)`. 다음 발생 때 멈춘 Python 줄(예: `reset_source`의 `set_state`)이
  확정된다.
- `run()`에서 `faulthandler.register(signal.SIGUSR1, all_threads=True)`. 루프가 다른 이유로 멈춘 것 같을 때 운영자가
  `sudo docker compose kill -s SIGUSR1 retrace`로 재시작 전에 스택을 남길 수 있다. compose `init: true`의 init이 신호를
  자식에게 전달한다.

### 4-5. 종료

`finally`에서 pipeline을 NULL로 내리기 전에 `inflight` 스레드가 있으면 최대 5초 `join`한다. 포기된 스레드가 bin을 잡고
있으면 pipeline NULL 전환이 같은 bin에서 막힐 수 있다. 그때는 docker가 grace period(기본 10초) 뒤 강제 종료한다.
재시작 결과는 같으므로 허용하고 operations.md에 적는다.

### 4-6. 계약

`/streams.json`, `camera_status`, `/ws`, backend는 바꾸지 않는다. 새 로그 세 줄과 SIGUSR1 처리만 추가된다.

## 5. 위험과 대응

| 위험 | 대응 |
| --- | --- |
| kick(flush_start)이 legacy mux에서 다른 카메라 batch에 영향 | 멈췄을 때만 보낸다. 그 경우의 대안은 루프 영구 정지라 손해가 없다. 발생 시 `stats` fps로 확인 |
| 포기된 bin이 계속 자원을 잡음 | 그 카메라만 offline이다. `source_recovery_abandoned` 로그를 보고 운영자가 재시작한다 |
| 종료가 강제 종료로 끝남 | §4-5. 허용 |
| 스레드 전환으로 기존 테스트의 순서 단언이 흔들림 | 테스트에서 작업 스레드를 즉시 join하도록 시작 함수를 주입하거나, 각 tick에서 완료를 기다린다(§6). 기존 재설정·재생성 순서 단언은 그대로 유지해야 한다 |
| faulthandler 출력에 민감 정보 | 파일 경로·함수 이름·줄 번호만 나오고 지역 변수·인자는 나오지 않는다. 경로는 컨테이너 내부(`/workspace/ReTrace/...`) |

## 6. 검증

### CPU 테스트

`tests/test_camera_runtime.py`

- `abandon_recovery()` 후 조건이 맞아도 `recovery_due()`가 거짓. `begin_recovery()` 또는 `recovery_failed()` 후 다시 참이 될 수 있음.

`tests/test_camera_shared_pipeline.py` (fake Gst, 테스트 시계 주입)

- 정상 경로: 기존 재설정·재생성·실패·예외 테스트가 그대로 통과한다(작업 스레드 완료를 기다린 뒤 단언).
- `reset_source`가 NULL 뒤 mux pad에 `flush_stop`을 보낸다.
- 멈춤 후 풀림: fake source의 `set_state('NULL')`이 `threading.Event`를 기다리게 한다. 시계를 30초 넘기면
  `source_recovery_hung` 한 줄, mux pad에 `flush_start` 이벤트, traceback 덤프가 나온다. 그 사이 다른 slot의 상태 갱신이
  계속된다(예: 다른 카메라 offline 전환이 일어남). Event를 set하면 `flush_stop` → sync → `source_reset` →
  `source_recovery_unblocked` 순서.
- 멈춤 후 포기: Event를 set하지 않고 시계를 10초 더 넘기면 `source_recovery_abandoned`, 그 slot은 이후 `recover`되지 않고
  `tick()`이 일반 재시도만 한다. 다른 slot은 다음 tick에 복구가 시작된다.
- 종료: 포기된 스레드가 살아 있어도 `run_shared_pipeline`이 반환한다(fake pipeline NULL은 막히지 않음, join timeout 확인).
- 출력에 `private`가 없다.

실행: `python3 -m unittest discover -s tests -p 'test_camera_*.py'`.

### GPU 서버 검증 (자연 발생)

1. 배포(restart) 후 `sudo docker compose logs -t --since=1m retrace | grep -c camera_status`가 252 전후.
2. 정상 카메라의 침묵 → 재설정이 다시 멈추면 기대 순서: `source_recovery_hung` → traceback 덤프 → `source_recovery_unblocked`
   (kick 효과 있음) 또는 `source_recovery_abandoned`(효과 없음). 어느 쪽이든 1번의 252줄이 유지돼야 한다.
3. traceback에서 멈춘 줄을 이 문서 §1의 Not verified에 기록한다. `abandoned`라면 다음 과제(멈추기 전에 막힌 push를 먼저
   푸는 순서, 예: NULL 전에 flush_start)를 그 줄을 근거로 정한다.

### 운영 일일 확인 (배포 후에도 유지)

```bash
sudo docker compose logs -t --since=1m retrace | grep -c camera_status
sudo docker compose logs -t --since=24h retrace | grep -E 'source_recovery_(hung|unblocked|abandoned)'
```

첫 값이 0이면 루프 정지(재시작), `abandoned`가 있으면 그 카메라는 재시작 전까지 자동 복구되지 않는다.

## 7. 문서 갱신

- `docs/operations.md` "source 복구 정책": 복구가 작업 스레드에서 돌고 30초/10초 감시, 로그 세 종류, kick, 포기 의미,
  SIGUSR1 스택 덤프, 종료 시 강제 종료 가능성, 위 일일 확인 명령.
- `docs/project-status.md` §5 운영 대응표의 "웹에서만 안 보일 때" 행에 루프 생존 확인(252줄)과 `abandoned` 확인 추가.
- 11-04 설계 문서 상단에 이 문서 링크 한 줄.

## 8. 구현 순서 (서버 체크아웃, 테스트 먼저)

1. `camera_runtime.py`: `abandon_recovery()`, `recovery_due` 조건, `begin_recovery`/`recovery_failed`의 플래그 해제, manager 위임 + 테스트.
2. `preview.py`: `reset_source`의 NULL 뒤 `flush_stop` + 테스트.
3. `preview.py`: `supervise_recovery`와 작업 스레드, 상수, 로그 세 종류, hung 시 traceback 덤프와 kick, 종료 join + fake Gst
   테스트(정상·풀림·포기·종료). 기존 테스트가 그대로 통과하는지 먼저 확인.
4. `preview.py` `run()`: `faulthandler.register(signal.SIGUSR1, all_threads=True)`.
5. 문서 갱신(§7).
6. 배포는 사용자 확인 후 재시작 가능한 시간에. 그 전까지 운영자는 일일 확인 명령으로 루프 생존을 본다.
