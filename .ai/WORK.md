# Current work (server / pipeline / backend)

- Active task: 카메라 source bin 자동 복구 (일부 카메라가 침묵 상태로 영구 offline, 재시작만 통하는 문제)
- Status: 설계 승인·문서화 완료 (2026-11-03, PC 체크아웃). 구현 미착수.
- Remaining work: [docs/superpowers/specs/2026-11-03-camera-source-recovery-design.md](../docs/superpowers/specs/2026-11-03-camera-source-recovery-design.md) §8 순서대로 서버 체크아웃에서 구현(`camera_runtime.py` 정책 → `preview.py` 재설정 → 테스트 → operations.md 갱신).
- Verification pending: CPU 테스트(`test_camera_*.py`, `test_rtsp_inputs.py`) 추가·통과, GPU 서버 수동 검증(설계 문서 §6). 배포는 사용자 확인 후.
- Blockers: none. `nvurisrcbin` 7.0 내부 동작은 Not verified이며 설계는 이에 의존하지 않는다.
- Next action: 서버 에이전트가 설계 문서를 읽고 §8 1단계(camera_runtime 정책 + 테스트)부터 구현한다.

## 원인 조사 결과 (2026-11-03, 서버 96시간 로그)

- 끊긴 카메라 6대 모두 장애 후 세대 주기가 60초로 bus 메시지 없이 앱 watchdog만 돌았다(source bin 침묵). 경로는 두 가지: 디코더 WARNING 후 침묵(b1_parking_gate), not-negotiated WARNING → RTSP timeout 1회 → 침묵(3f_auditorium_door, 6f_office_inner, b1_parking_overview, 6f_lobby). 상세와 배제된 가설은 설계 문서 §1.
- 서버 체크아웃의 `.ai/WORK.md` 로컬 사본에 서버 에이전트의 조사 기록("Current RTSP/output investigation" 절)이 있을 수 있다. pull 전에 비교해 이 절과 합친다.

## Important context

- 이 파일은 2026-10-28에 작업 일지(2026-10-19까지)를 현재 상태로 정리한 것이다. 서버 체크아웃의 로컬 사본은 2026-10-21까지 기록돼 있을 수 있다(Not verified). 서버에서 pull할 때 로컬 사본과 비교한 뒤 덮어쓴다. 과거 완료 기록은 git log와 [docs/project-status.md](../docs/project-status.md) §4에 있다.
- 서버 쪽 미해결 항목(상세·근거는 project-status §5):
  - 자동 라벨링(PR #9, 2f3a9af) 서버 배포와 실제 GPU/DB/프록시 검증. 배포 후 이미지 한 장으로 먼저 확인.
  - DeepStream 모델 배포(OTA + 롤백, `VEHICLE_DEPLOYMENT_ENABLED`) 실환경 검증. 이전 실행 결과는 `rollback_failed` / `verification_failed`. FPS 검증 개선이 서버에 적용됐는지 Not verified.
  - 업데이트된 워커로 Train All 재실행(기존 job f4818a65는 standard 모드 결과).
  - 서버 `main`이 최신인지, `frontend-dist/`가 현재 빌드와 같은지 확인.
  - 영상이 나오지 않는 카메라 3대 원인 확인(ID는 project-status §5).
  - 모델 삭제 API 없음. 차량 검출 판정 결과·임계값은 브라우저에만 저장됨(서버 저장은 별도 설계 필요).
