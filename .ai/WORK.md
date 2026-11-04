# Current work (server / pipeline / backend)

- Active task: 카메라 source bin 자동 복구 (일부 카메라가 침묵 상태로 영구 offline, 재시작만 통하는 문제)
- Status: 1단계(source bin 제자리 재설정) 구현 완료, **main 0ef8320에 반영·운영 적용됨(2026-11-03)**. 하루 운영 결과 두 경로 중 한 경로만 복구된다(아래 "운영 결과"). **2단계 복구(mux sink pad release/재요청 + source bin 재생성) 설계 문서가 main에 추가되면 구현할 예정**. 작업 브랜치와 옛 worktree(`retrace-dataset-369ed49`, `retrace-model-deploy`)는 정리했다.
- 운영 결과 (2026-11-04, 1일, 사용자 보고):
  - 복구 성공: RTSP timeout 후 침묵 경로 — 6f_lobby, 1f_lobby. 재설정(`source_reset`) 후 프레임이 돌아왔다.
  - 복구 실패: not-negotiated 후 침묵 경로 — b1_parking_overview, 7f_lobby. 재설정 직후 공유 element 경고 4건 → source 경고 1건이 반복되고 프레임은 돌아오지 않는다. bin 전체 NULL → PLAYING만으로는 기존 mux sink pad/caps 협상 상태가 풀리지 않는 것으로 보인다(원인은 Not verified; 경고 원문은 기록하지 않음).
  - 판단: 설계 문서 §5 후속 과제(재설정 후 caps 오류 반복) 조건이 충족됐다. 다음 단계는 mux sink pad를 release하고 다시 request하면서 source bin을 재생성하는 2단계 복구다. 1단계 재설정은 유지한다(timeout 경로에 유효).
- Changed (commit 0ef8320): `camera_runtime.py`(`CameraStatus.recovery_count`, `RECOVERY_BACKOFF`, `recovery_due/begin_recovery/recovery_failed`, manager `recovery_enabled`·`tick()`이 due slot 반환·`recovered/recovery_failed`), `preview.py`(`--source-recovery` 기본 켬, `source_bins` 보관, `reset_source()` NULL → sync_state_with_parent, 제어 루프에서 tick당 1대, `source_reset`/`source_reset_failed` 로그), tests(`test_camera_runtime.py` RecoveryPolicyTests 6, `test_camera_shared_pipeline.py` tick 2 + fake Gst 제어 루프 3 + 스키마 키, `test_rtsp_inputs.py` 파서 1), docs(operations.md "source 복구 정책", architecture.md, testing.md, 설계 문서 상태·구현 메모).
- 설계와 다른 점(설계 문서 "구현 메모" 참조): ghost pad target은 plugin이 pad-removed 없이 pad를 버린 경우에만 비운다(무조건 비우면 pad가 유지되는 경우 링크가 끊김). `recovery_failed()`도 간격을 120→300→600으로 늘린다.
- Verified (CPU, `/tmp/retrace-shared-venv/bin/python` 3.11): `test_camera_*.py` 75 OK, `test_rtsp_inputs.py` 6 OK, 회귀 `test_preview_realtime` 11, `test_vehicle_deployment` 24, `test_canonical_startup` 3, `test_preview_api_writes` 15 OK. `git diff --check` 통과.
- NOT verified: not-negotiated 경로에서 재설정이 실패하는 정확한 지점(nvurisrcbin 내부 caps 상태인지, legacy nvstreammux sink pad의 협상 상태인지), 재설정이 다른 카메라 fps에 준 영향(사용자 보고에 없음).
- Next action: 2단계 복구 설계 문서(`docs/superpowers/specs/`에 추가 예정, mux sink pad release/재요청 + source bin 재생성)가 main에 들어오면 그 문서 순서대로 테스트 먼저 작성 → 구현 → CPU 테스트(`/tmp/retrace-shared-venv/bin/python -m unittest discover -s tests -p 'test_camera_*.py'`). 그때까지 운영 로그에서 `source_reset` 뒤 경고 반복 패턴을 `docker compose logs -t | rg 'source_reset|source_caps_error'`로 계속 수집한다.
- Blockers: 2단계 설계 문서 미작성. `nvurisrcbin` 7.0 내부 동작은 Not verified이며 설계는 이에 의존하지 않는다.

## 원인 조사 결과 (2026-11-03, 서버 96시간 로그)

- 끊긴 카메라 6대 모두 장애 후 세대 주기가 60초로 bus 메시지 없이 앱 watchdog만 돌았다(source bin 침묵). 경로는 두 가지: 디코더 WARNING 후 침묵(b1_parking_gate), not-negotiated WARNING → RTSP timeout 1회 → 침묵(3f_auditorium_door, 6f_office_inner, b1_parking_overview, 6f_lobby). 상세와 배제된 가설은 설계 문서 §1.
- 서버 측 조사 기록(2026-11-03, 서버 체크아웃 로컬 사본에서 합침):
  - 런타임 스냅샷 05:26 UTC (컨테이너 기동 10-30 06:13 UTC): 21 enabled 중 online 12 / 비정상 9. 기동 후 한 번도 안 뜬 3대(3f_terrace_4k resource/9, 5f_server_room·7f_rooftop resource/7 rtsp_timeout)는 2026-10-06 조사의 기존 baseline. 운영 중 끊긴 6대: b1_parking_gate(10-30 07:48), 3f_auditorium_door(10-31 07:04), 6f_office_inner(11-01 23:50), b1_parking_overview(11-02 04:19), 6f_lobby(11-02 05:45), 7f_lobby(11-03 00:17). 6대 모두 last_frame_age ÷ reconnect_count ≈ 60초.
  - 가설 판정: decoder pad 미연결(2) 배제 — `NVMM video pad linked` 기동 시 18건 뿐, `source_failure`/`source_link_error` 0건. PTS 라우팅(3) 배제. 디코더 오류 후 침묵(1) — b1_parking_gate. not-negotiated → GstRTSPSrcTimeout 1회 → 침묵(4 변형) — 3f_auditorium_door, 6f_office_inner, b1_parking_overview, 6f_lobby. 7f_lobby는 11-02 00:41 nvv4l2decoder ERROR(stream/11) 69분 반복 후 01:50 자연 복구, 최종 장애는 메시지 없이 프레임만 멈춰 1·4 구분 불가. 같은 DECODE WARNING 후 복구된 카메라(5f_office_inner, b1_lobby) 있음.
  - 기동 후 안 뜬 3대도 95시간 동안 WARNING 각 1건뿐 → nvurisrcbin 재시도는 bus에 메시지를 남기지 않거나 첫 실패 후 멈춘다(구분 불가, Not verified).
  - 로그 해석 시 주의: 단일 source ERROR는 reason이 바뀔 때만 `source_rtsp`를 찍고 rtsp_error는 기존 reason을 덮지 않아 조용히 지나간다. last_error=rtsp_timeout은 GstRTSPSrcTimeout element 메시지 경로뿐이다. 다음 수집은 `docker compose logs -t`로 timestamp를 포함한다.
  - DeepStream 소스: 컨테이너(DS 7.0)에는 `sources/gst-plugins/gst-nvmultiurisrcbin/gstdsnvurisrcbin.h` 헤더만 있음. `gst-inspect-1.0 nvurisrcbin`: `rtsp-reconnect-interval` 기본 0, `rtsp-reconnect-attempts` 기본 -1, `message-forward` 기본 false, action은 start-sr/stop-sr뿐(외부 reset action 없음). 호스트 `/opt/nvidia/deepstream`은 6.1.0이며 deepstream-app 참조 `deepstream_source_bin.c`의 `watch_source_status`/`reset_source_pipeline`(bin 전체 NULL→parent sync, parser에 stream reset event, 전역 3초 reset 간격, NULL 실패 시 포기)을 읽었다.

## Important context

- 이 파일은 2026-10-28에 작업 일지(2026-10-19까지)를 현재 상태로 정리한 것이다. 서버 체크아웃의 로컬 사본은 2026-10-21까지 기록돼 있을 수 있다(Not verified). 서버에서 pull할 때 로컬 사본과 비교한 뒤 덮어쓴다. 과거 완료 기록은 git log와 [docs/project-status.md](../docs/project-status.md) §4에 있다.
- 서버 쪽 미해결 항목(상세·근거는 project-status §5):
  - 자동 라벨링(PR #9, 2f3a9af) 서버 배포와 실제 GPU/DB/프록시 검증. 배포 후 이미지 한 장으로 먼저 확인.
  - DeepStream 모델 배포(OTA + 롤백, `VEHICLE_DEPLOYMENT_ENABLED`) 실환경 검증. 이전 실행 결과는 `rollback_failed` / `verification_failed`. FPS 검증 개선이 서버에 적용됐는지 Not verified.
  - 업데이트된 워커로 Train All 재실행(기존 job f4818a65는 standard 모드 결과).
  - 서버 `main`이 최신인지, `frontend-dist/`가 현재 빌드와 같은지 확인.
  - 영상이 나오지 않는 카메라 3대 원인 확인(ID는 project-status §5).
  - 모델 삭제 API 없음. 차량 검출 판정 결과·임계값은 브라우저에만 저장됨(서버 저장은 별도 설계 필요).
