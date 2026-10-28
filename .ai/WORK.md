# Current work (server / pipeline / backend)

- Active task: none
- Status: idle (2026-10-28, PC 체크아웃 기준)
- Remaining work: none in progress. 미해결 항목은 아래 Important context.
- Verification pending: none
- Blockers: none
- Next action: record the next requested task here before implementation.

## Important context

- 이 파일은 2026-10-28에 작업 일지(2026-10-19까지)를 현재 상태로 정리한 것이다. 서버 체크아웃의 로컬 사본은 2026-10-21까지 기록돼 있을 수 있다(Not verified). 서버에서 pull할 때 로컬 사본과 비교한 뒤 덮어쓴다. 과거 완료 기록은 git log와 [docs/project-status.md](../docs/project-status.md) §4에 있다.
- 서버 쪽 미해결 항목(상세·근거는 project-status §5):
  - 자동 라벨링(PR #9, 2f3a9af) 서버 배포와 실제 GPU/DB/프록시 검증. 배포 후 이미지 한 장으로 먼저 확인.
  - DeepStream 모델 배포(OTA + 롤백, `VEHICLE_DEPLOYMENT_ENABLED`) 실환경 검증. 이전 실행 결과는 `rollback_failed` / `verification_failed`. FPS 검증 개선이 서버에 적용됐는지 Not verified.
  - 업데이트된 워커로 Train All 재실행(기존 job f4818a65는 standard 모드 결과).
  - 서버 `main`이 최신인지, `frontend-dist/`가 현재 빌드와 같은지 확인.
  - 영상이 나오지 않는 카메라 3대 원인 확인(ID는 project-status §5).
  - 모델 삭제 API 없음. 차량 검출 판정 결과·임계값은 브라우저에만 저장됨(서버 저장은 별도 설계 필요).
