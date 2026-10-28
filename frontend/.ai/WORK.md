# Frontend current work

- Active task: none
- Status: idle (2026-10-28). `main`에 차량 검출 주차 판정(PR #6)과 자동 라벨링(PR #9)까지 병합됨.
- Remaining work: none in progress
- Verification pending: 서버 `frontend-dist/`가 현재 `main` 빌드와 같은지(Not verified). 자동 라벨링 UI는 실서버·실제 GPU에서 미검증. 차량 검출 판정 임계값은 실영상으로 조정하지 않음.
- Blockers: none
- Next action: record the next requested task here before implementation.

## Important context

- 2026-10-28에 2026-10-07~10-19 작업 일지를 정리했다. 과거 기록은 git log와 루트 [docs/project-status.md](../../docs/project-status.md) §4·§5.
- 배포는 `npm run generate` 후 `.output/public/` 내용물을 서버 `frontend-dist/`에 복사한다(index.html은 마지막에). 절차는 [README](../README.md#정적-배포).
- 알려진 데이터 문제(프론트 범위 밖): 모델이 P202 주차면을 검출하지 못함, 일부 색상 baseline이 차량이 있을 때 등록됐을 가능성(project-status §5).
