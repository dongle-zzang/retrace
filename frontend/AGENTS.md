# 프론트엔드 에이전트 작업 지침

`frontend/`(ReTrace 제품 대시보드, Nuxt 4 CSR) 작업에 적용한다. 서버 영역(루트의 Python/DeepStream, `backend/`, `web/`, 루트 `docs/`, 루트 `.ai/`)은 별도로 관리하며 프론트 작업에서 수정하지 않는다.

## 작업 시작

1. `git status`로 기존 변경을 확인한다. dirty 상태에서 자동 pull/rebase/reset을 하지 않는다.
2. [.ai/WORK.md](.ai/WORK.md)(프론트 현재 상태)와 [.ai/MEMORY.md](.ai/MEMORY.md)(프론트 장기 제약의 이유)를 읽는다(Claude Code는 `CLAUDE.md` import로 이미 로드됨). 프로젝트 결정 사항·이유와 미해결 이슈는 루트 [docs/project-status.md](../docs/project-status.md)에 있다. 루트 `.ai/`는 서버 작업 상태이며 프론트 작업 상태를 기록하지 않는다.
3. 구조와 연결 방식은 [README.md](README.md)를 따른다. 서버 API/WebSocket 계약의 원문은 루트 `docs/preview-realtime.md`와 참조 구현 `web/preview.js`이며 읽기만 한다.
4. 실제 코드를 확인한 뒤 작업한다. 파일, API, 명령, 환경 변수를 추측하지 않는다.

## 경계

- 요청과 무관한 수정, 정리, refactoring으로 범위를 넓히지 않는다. 새 dependency는 명확한 필요가 있을 때만 추가한다.
- 실제 서버 주소, credential, RTSP 주소를 코드·문서·커밋에 넣지 않는다. 서버 origin은 Git에서 제외된 `.env`의 `NUXT_PUBLIC_BACKEND_BASE_URL`에만 둔다.
- 코드는 같은 origin의 상대 경로 `/api`, `/ws`만 사용한다(이유는 MEMORY).
- shadcn-vue 컴포넌트(`app/components/ui/`)는 공용이다. 화면별 스타일은 사용하는 쪽에서 class로 덮어쓴다.
- `.vue`는 Composition API + `<script setup lang="ts">`로만 쓴다(Options API 없음). 상태 관리 라이브러리 없이 `app/composables/`의 composable을 쓴다.
- 새 UI 문구는 영어 원문을 키로 `t('...')`에 쓰고 한국어를 `app/lib/i18n.ko.ts`에 추가한다. `tests/i18n.test.cjs`가 누락을 검사한다.
- 모달 위에 뜨는 확인 다이얼로그는 `z-[60]`과 `overlayClass` prop으로 겹침 순서를 맞춘다(`AppConfirmDialog.vue`).

## 검증

```powershell
npm test
npm run typecheck
npm run build
```

정적 배포 산출물은 `npm run generate` 후 `.output/public/`의 내용물이며 서버의 `frontend-dist/`에 올린다(`frontend/` 아님). 존재하지 않는 lint/format 명령을 만들어내지 않는다. 화면 변경은 가능하면 실제 브라우저(또는 headless Chrome 스크린샷)로 확인하고, 확인하지 못한 항목은 보고한다.

## Git

- 커밋 전 `git diff --cached --name-status`로 `frontend/` 밖의 경로가 섞이지 않았는지 확인한다.
- 커밋에는 `frontend/` 경로만 포함한다. `.env`, `.nuxt/`, `.output/`, `dist/`, `node_modules/` 등 로컬 산출물은 넣지 않는다.
- commit/push는 사용자가 요청할 때만 한다. push 전에 `git fetch origin`으로 원격 `main`과의 차이를 확인한다. 원격이 단순히 앞서 있으면 rebase 후 push하고, 히스토리가 재생성되어 갈라져 있으면 rebase하지 않고 사용자와 동기화 방법을 정한다(2026-10-23 사례는 루트 `.ai/MEMORY.md`).

## 인수인계

- 의미 있는 작업 시작, 주요 단계 완료, 중요한 제약 발견, 미완성 종료 시 `.ai/WORK.md`를 갱신한다. 완료된 작업은 세부 기록을 누적하지 않고 정리한다.
- 반복적으로 중요한 비자명한 제약은 이유와 함께 `.ai/MEMORY.md`에 둔다. README에 충분히 설명된 내용은 중복하지 않는다.
