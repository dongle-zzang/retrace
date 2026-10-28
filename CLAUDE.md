# Claude Code project instructions

@AGENTS.md
@.ai/WORK.md

## Claude Code

- `.ai/WORK.md`는 위 import로 세션 시작 시 이미 로드된다. 작업 시작 절차에서 다시 읽지 않아도 된다.
- 다른 PC나 다른 에이전트에서도 필요한 프로젝트 지식은 auto memory(이 PC의 `~/.claude/projects/` 아래에만 저장되고 Git으로 공유되지 않음)가 아니라 `.ai/MEMORY.md`에 기록한다.
- `docs/agent/*`는 import하지 않는다. `AGENTS.md`의 라우터에 따라 필요한 문서만 읽는다.
- 배포, DB/schema 변경, 운영 모델·서비스 교체, 대량 삭제, 원격 Git 변경(push, 브랜치 삭제) 전에는 사용자 확인을 받는다.
