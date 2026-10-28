# 배포와 로컬 환경

## 저장소에서 확인되는 구성

- `compose.yml`은 GPU `retrace`, CPU `backend`, PostgreSQL 서비스와 DB named volume을 정의한다. retrace는 backend/postgres의 기동에 의존하지 않는다.
- 루트 Dockerfile은 DeepStream 7.0 기반에 PyDS 1.1.11을 설치하고 Compose와 같은 preview 실행 경로를 사용한다. 소스는 Compose bind mount로 제공하며 루트 이미지에 앱 소스를 COPY하는 방식이 아니다.
- Backend Dockerfile은 Python 3.12 slim에 앱/공통 카메라 로더를 COPY하고 비-root 사용자, Uvicorn worker 하나로 실행한다. YAML은 read-only mount다.
- retrace는 컨테이너 내부 40225, backend는 8000을 사용한다. 40225에서 CSR 빌드, MJPEG, `/ws`(JPEG·metadata), `/api/*` 프록시를 제공한다. GET/HEAD와 기본 차단의 주차 CRUD 접근 제어는 [Backend README](../../backend/README.md#preview-주차-crud-프록시)를 따른다. Backend 호스트 공개는 기본 loopback이며 retrace 호스트 binding은 ignored override에서 정한다. 별도 reverse proxy 서비스와 CI/CD workflow는 없다.
- 별도 개발 PC 접근은 기존 loopback 바인딩을 유지하는 SSH 터널 + Mac Nuxt 개발 프록시를 우선한다. `/api`는 터널의 Backend로 전달하고 Preview `/ws`는 기존 연결을 유지한다. 사설 인터페이스 직접 접근 및 `BACKEND_CORS_ORIGINS`는 별도 선택 사항이다. 환경 변수/접근 절차는 [backend README](../../backend/README.md#별도-개발-pc에서-접근)를 따른다. 서버 Compose에는 frontend service가 없다.

- CSR 산출물은 기본 `frontend-dist/`에 업로드하며 `FRONTEND_DIST_DIR`은 컨테이너 내부 경로다. `PREVIEW_BACKEND_URL`은 내부 Backend origin이다. 빌드가 없으면 `/`는 404이고 `web/`의 영상 확인 페이지는 `/diagnostics`에서만 제공한다. 업로드·최초 컨테이너 재생성·HTTP 확인 절차는 [frontend deployment](frontend-deployment.md)를 따른다. retrace healthcheck와 `tools/check_preview.py`는 `/streams.json`을 사용한다.

## 실행 원문과 변경 판단

Preview 영상과 `/ws`는 모두 TCP 40225 하나로 제공하므로 기존 bridge 네트워크와 ignored override의 port binding으로 충분하다.
WebRTC용 host network·UDP 설정은 제거했다. 기본 Compose와 private override의 port/GPU 예약/DB volume은 보존한다.

환경 준비, `.env` 복사, 모델/engine 준비, Compose build/up, 컨테이너 교체·복구 주의사항은 [루트 README](../../README.md)의 실행 섹션을 따른다. Backend/PostgreSQL 실행, DB 확인, 단일 worker를 유지하는 이유는 [backend README](../../backend/README.md)의 “실행” 및 “CPU 테스트”에 있다. 현재 호스트의 실제 배포/영상 수신 상태는 문서 작성에서 검증하지 않았다.

`docker compose build`, `docker compose up -d`, `docker compose ps`는 존재하는 기본 운영 명령이다. 이미지/command/mount/ports 변경은 컨테이너 재생성이 필요한지 확인한다. 기존 동작 중 서비스 교체는 이번 코드 변경의 범위와 운영 영향을 확인한 뒤 수행한다.

모델·labels·TensorRT cache와 GPU/NVIDIA runtime은 장비마다 준비해야 한다. Git clone/pull만으로 실행 환경이 완성되지 않는다. 모델/engine 생성과 재사용 조건을 변경할 때 README의 모델 준비 절차를 확인한다.

## 공유 문서와 private 설정

Git 공유 대상은 `AGENTS.md`, `CLAUDE.md`, `.ai/`, `docs/agent/`다(2026-10-23 재생성 때 gitignore했다가 2026-10-28 다시 공유로 결정, 이유는 루트 `.ai/MEMORY.md`). `.claude/`와 `CLAUDE.local.md`는 로컬 전용이다. 실제 `.env`, host binding override, 모델·engine·캐시·출력 및 도구별 로컬 설정은 공유 대상과 구분한다.

환경 변수 목록과 역할은 `.env.example`, Compose, `settings.py` / `camera_config.py`에서 확인한다. Backend/PostgreSQL에는 명시된 DB 변수만 전달한다. `docker compose config` / inspect / logs에는 private 값이 포함될 수 있으므로 원문을 공유 문서나 인수인계에 복사하지 않는다. 실행 경로 확인이 필요하면 README의 `tools/check_container_startup.py` 요약 절차를 사용한다.

데이터셋은 Backend의 새 `dataset_data` named volume(`/app/dataset-storage`)과 additive 테이블을 쓴다.
`DATASET_CAPTURE_ENABLED` 기본은 0이며 1로 바꾸면 Preview 재생성과 native GPU 확인이 필요하다.
기존 PostgreSQL volume·모델/config를 보존하고 DB/파일 볼륨을 함께 백업한다.
운영 적용과 검증 순서는 [AI 데이터셋](../datasets.md#실행과-아직-필요한-검증)을 따른다.

웹 fine-tuning은 Backend에 `TRAINING_*` 변수(토큰이 비면 비활성)를 전달하고, 학습 자체는 컨테이너가 아닌 host의
`training/.venv`에서 `worker.py`가 loopback Backend(:8000)에 붙어 실행한다. DeepStream(`retrace`)은 멈추거나 재시작하지 않으며
자원 부족 시 학습이 대기·중지한다. 토큰 설정·worker 실행·보호 기준은 [웹 fine-tuning](../training.md)을 따른다.
