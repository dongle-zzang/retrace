# Project memory

반복적으로 중요한 비자명한 제약과 그 이유만 둔다. 프론트 전용 제약은 [frontend/.ai/MEMORY.md](../frontend/.ai/MEMORY.md)에 있다.

## 2026-10-23에 저장소 히스토리를 재생성했다

- 이유: 과거 커밋에 남은 AI 문서와 서버 경로를 제거하려고 `main`을 정리된 62커밋으로 다시 만들었다(그 끝이 06956b5). 그 전 히스토리는 폐기됐다.
- 변경 시 확인: 재생성 전 체크아웃(PC, 노트북, 서버)은 `git fetch origin` 후 `git reset origin/main`(hard 아님)으로 맞추고, 원격에 있는 파일은 `git checkout origin/main -- .`로 갱신한다. rebase나 merge를 하면 폐기된 옛 커밋이 다시 올라간다. `reset --hard`는 ignore 상태였던 로컬 문서를 지우므로 쓰지 않는다.
- 잘못 바꾸면: 옛 커밋이 복원되어 지웠던 AI 문서·서버 경로가 공개 히스토리에 다시 남는다.

## AI 지침·인수인계 문서는 Git으로 공유한다 (2026-10-28)

- 이유: 재생성 때는 공개 저장소 노출을 피하려고 `.ai/`, `AGENTS.md`, `CLAUDE.md`, `docs/agent/`를 gitignore했지만, 그러면 PC·노트북·서버·다른 에이전트 사이에 작업 맥락이 전달되지 않았다. 노출 문제는 서버 경로·사용자명·주소를 문서에 쓰지 않는 규칙으로 대신한다. `.claude/`(도구 설정)와 `CLAUDE.local.md`(개인 지침)만 로컬 전용이다.
- 변경 시 확인: 커밋 전 `git grep -nE '/home/|rtsp://|[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+'`로 내부 값을 검사한다. 다른 장비에 ignore 상태로 남아 있던 로컬 사본(특히 서버의 `.ai/WORK.md`)은 pull 전에 비교한다.
- 잘못 바꾸면: 다시 gitignore하면 문서를 고쳐도 다른 장비에 전달되지 않고, 한 장비의 사본만 최신이 된다.

## 루트 `.gitignore`의 `*.ts`와 `/datasets/`는 프론트 소스를 가릴 수 있다

- 이유: `*.ts`는 MPEG-TS 녹화 파일, `/datasets/`는 비공개 데이터셋 이미지를 무시하려는 규칙이다. `frontend/app/pages/datasets/`와 새 TypeScript 소스가 같은 이름을 쓰므로 `frontend/.gitignore`의 `!*.ts`와 `/datasets/`의 앞 슬래시(루트 앵커)로 풀어 두었다.
- 변경 시 확인: 루트 `.gitignore`를 고친 뒤 `git status --short -uall`과 `git check-ignore -v`로 `frontend/` 아래 새 파일이 무시되지 않는지 본다. 상세와 Tailwind 영향은 [frontend/.ai/MEMORY.md](../frontend/.ai/MEMORY.md).
- 잘못 바꾸면: 새 프론트 파일이 커밋에서 조용히 빠진다.
