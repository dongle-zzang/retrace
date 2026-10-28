# Frontend 개발 경계

제품 Frontend는 별도 로컬 PC에서 CSR로 개발·빌드하고 정적 산출물을 GPU 서버에 업로드한다.
`preview.py`의 기존 40225 listener가 `preview_web.py`를 통해 빌드와 같은 origin의 API·영상을 제공한다.
서버에 Node runtime이나 별도 frontend 서비스는 필요 없다. 업로드·Nuxt 설정·적용 절차의 원문은
[frontend deployment](frontend-deployment.md)다.

빌드 디렉터리는 `FRONTEND_DIST_DIR`로 지정하며 기본값은 저장소의 `frontend-dist/`다(Git 무시). `frontend/`는 제품 프론트 소스이며 제공하지 않는다.
`index.html`이 있으면 `/`와 JS/CSS/이미지 등 빌드 파일을 제공한다. 확장자 없는 HTML navigation은
SPA index로 fallback하며 누락된 asset, 숨김 경로, 상위 경로 및 외부 symlink는 404다.
정적 제공 범위는 전용 빌드 디렉터리이며 저장소 전체가 아니다.
빌드가 없으면 `/`와 `/index.html`은 404이며, `web/index.html`·`web/preview.js`의 영상 확인 페이지는
`/diagnostics`에서만 제공한다. 개발 PC의 dev server가 40225를 API base로 쓸 때 브라우저로 `/`에 접속해도 확인 페이지가 노출되지 않게 하기 위함이다.

확인 페이지(`web/preview.js`)는 `/streams.json`으로 목록을 받고 WebSocket 하나로 **모든 카메라**의 JPEG와
`camera_status`를 받아 `<img>`에 표시한다. 브라우저 origin당 HTTP 연결 6개 제한이 있어 카메라별
`/mjpeg/sourceN`으로는 6대를 넘길 수 없기 때문이다. JPEG에 서버 OSD bbox(주차장 카메라 차량, `Vehicle` 라벨만)가 그려져 있으므로
별도 overlay는 없다. 카메라마다 decode는 한 장씩만 하고 대기 중 프레임은 최신으로 덮어쓰며, 표시가 바뀐 뒤
이전 object URL을 해제한다. socket이 닫히면 2초 후 재접속한다. `/diagnostics`는 build가 있어도
확인 페이지를 제공한다.
연동 API와 동기화 한계는 [실시간 Preview](../preview-realtime.md)를 따른다.
Backend는 기존 Preview JSON을 polling하며 영상을 relay하지 않는다.

배포한 Frontend는 `/api/*`를 같은 40225 origin으로 호출한다. HTTP request thread에서
`PREVIEW_BACKEND_URL`의 Backend로 GET/HEAD를 전달한다. 쓰기는 주차면 CRUD 경로의
POST/PATCH/DELETE 및 데이터셋의 제한된 POST/PUT/DELETE를 허용하며, 40225에 접근 가능한 클라이언트면 별도 IP/Origin 허용 목록 없이 사용한다.
method/query/JSON body/status를 유지한다. CORS/preflight는 제공하지 않으므로 같은 origin 또는 Nuxt 개발 프록시를 쓴다.
설정·제한·적용은 [Backend README](../../backend/README.md#preview-주차-crud-프록시)를 따른다.
Backend 장애/timeout은 502이며 frontend 정적 파일 및 MJPEG 제공과 별도로 처리한다.
Backend API의 `preview_path`(`/mjpeg/sourceN`)는 단일 카메라 img용이고, 여러 카메라는 `metadata_path`(`/ws`)를 같은 origin에서 사용한다.
API/metadata 계약은 [backend](backend.md), 제한된 인터페이스 바인딩과
브라우저 CORS 설정은 [deployment](deployment.md) 및 해당 문서가 연결한
backend README를 따른다. 실제 주소/credential은 공유 문서에 기록하지 않는다.

Preview HTTP와 API readiness의 CPU 검증 명령은 [testing](testing.md)에 있다.

제품 Nuxt 구현은 [frontend/README.md](../../frontend/README.md)를 따른다. 제품 프론트 작업 규칙과 장기 제약은 [frontend/AGENTS.md](../../frontend/AGENTS.md)와 [frontend/.ai/MEMORY.md](../../frontend/.ai/MEMORY.md)를 따른다.

색상 기반 주차의 logical space/zone API와 `parking.status_updated` 통합 상태는 [색상 주차 프론트 연결](../color-parking.md#프론트엔드-연결)을 따른다. 기존 차량 주차 계약과 다르며 초기/재접속 시 REST status로 최신 점수를 조회한다.

`/datasets` 수집/관리와 `/datasets/images/:imageId` 박스 편집은 기존 Nuxt 프론트에 통합한다.
API 프록시는 데이터셋 JPEG/ZIP만 큰 응답을 스트리밍한다. 계약·원본 프레임 경계와 운영 검증은
[AI 데이터셋](../datasets.md), 화면 구현은 frontend README를 따른다.
