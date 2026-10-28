# CSR 프론트 배포

사용자는 `http://<서버IP>:40225/` 하나로 프론트에 접속한다.

| 경로 | 처리 |
| --- | --- |
| `/`, 프론트 경로, JS/CSS/이미지 | 업로드한 CSR 정적 빌드 (빌드가 없으면 404) |
| `/api/*` | Python HTTP request thread → 내부 Backend; GET/HEAD 및 접근 제어를 설정한 주차 CRUD |
| `/ws` | WebSocket 하나로 여러 카메라의 JPEG(binary)·metadata·status |
| `/diagnostics`, `/preview.js` | 진단 페이지 |
| `/mjpeg/sourceN` | 카메라별 MJPEG (화면당 6대 이하) |
| `/streams.json`, `/metadata.json` | 기존 Preview JSON |

이 경로들은 프론트 라우터보다 우선하며, 프론트에서 별도 Backend/Preview 주소를
지정하지 않는다. 카메라 한두 대는 `<img src=preview_path>`로 표시할 수 있다. 브라우저는 같은 origin에
HTTP 연결을 6개까지만 열므로, 여러 카메라 화면은 `metadata_path`(`/ws`)에서 `subscribe(video: true)`로
JPEG를 받아 표시한다. 여러 카메라를 쓰는 제품 build는 이 방식으로 갱신해야 한다. 연동 방법은 [실시간 Preview](../preview-realtime.md)를 따른다. 같은 origin이므로 배포된 프론트에 별도 CORS 설정은 필요 없다.
개발 PC의 dev server가 직접 Backend에 접근하는 경우에는 기존 CORS 안내를 따른다.

## 개발 PC에서 빌드

제품 프론트 소스는 이 저장소의 `frontend/`(Nuxt 4, `ssr: false`)이며 개발 PC에서 빌드한다.
루트 package.json은 Codex 도구용이므로 루트에서 npm build를 실행하지 않는다.
빌드 명령과 환경 변수는 [frontend/README.md](../../frontend/README.md#정적-배포)를 따른다.

`frontend/`에서 `npm run generate`를 실행한다. `.output/public/` **내용물**이 정적 산출물이다.
서버에는 `.output/server/`나 Node dependency를 올릴 필요 없다.
이는 [Nuxt 공식 CSR 배포 안내](https://nuxt.com/docs/3.x/getting-started/deployment#client-side-only-rendering)에 따른다.
Nuxt server routes는 정적 빌드에서 실행되지 않으므로 데이터 API는 ReTrace의 `/api/*`를 사용한다.
사이트는 `/`에 배포하므로 asset base도 루트 경로를 사용한다.

## 서버에 업로드

호스트의 저장소 체크아웃(`~/retrace`) 아래 `frontend-dist/`에 산출물 내용물을 복사한다.
`frontend-dist/index.html`과 `frontend-dist/_nuxt/` 또는 해당 프레임워크 asset 폴더가
바로 위치해야 한다. `frontend-dist/public/index.html`처럼 한 단계 더 중첩하지 않는다.
이 디렉터리는 Git과 Docker build에서 제외하며 기존 저장소 bind mount(`.:/workspace/ReTrace`)를 통해 제공한다.
`frontend/`는 Git이 관리하는 제품 프론트 소스이므로 빌드를 그곳에 올리지 않는다.

```bash
cd ~/retrace
mkdir -p frontend-dist
# /path/to/frontend-build는 업로드한 실제 정적 산출물 디렉터리로 바꾼다.
cp -a /path/to/frontend-build/. frontend-dist/
```

기본 환경은 Compose가 다음 값을 주입한다. `.env`를 수정하지 않아도 기본값이 적용된다.

```dotenv
FRONTEND_DIST_DIR=/workspace/ReTrace/frontend-dist
PREVIEW_BACKEND_URL=http://backend:8000
```

`FRONTEND_DIST_DIR`은 **컨테이너 안에서 보이는 경로**다. 외부 호스트 디렉터리를 쓰려면
private override에 해당 디렉터리의 read-only mount도 추가해야 한다.
`PREVIEW_BACKEND_URL`은 브라우저 주소가 아니라 Python 서버가 호출하는 내부 origin이다.
credentials나 `/api` suffix를 포함하지 않는다. Docker 없이 호스트 Python에서 실행할 때는
기본 Backend origin이 `http://127.0.0.1:8000`이다. 필요하면 환경 변수로 바꾼다.

## 처음 적용

새 HTTP handler와 Compose 환경을 적용하려면 retrace 컨테이너를 재생성한다.
이때 영상 pipeline도 다시 시작된다. 기존 private 포트 binding을 사용한다.
Backend와 PostgreSQL이 이미 실행 중이면 두 번째 명령만 실행한다.
소스 bind mount를 사용하므로 이번 Python 변경에는 이미지 rebuild가 필요 없다.

```bash
docker compose up -d backend
docker compose up -d --no-deps --force-recreate retrace
```

이후 프론트 파일을 갱신할 때는 동일 폴더의 산출물을 교체하고 브라우저를 새로고침한다.
빌드 파일은 매 요청에서 읽으므로 파일 교체만으로 retrace를 재시작할 필요는 없다.
최초 JS/asset 교체가 끝난 뒤 index.html을 교체하면 업로드 중 잘못된 asset 참조를 줄일 수 있다.
프론트 빌드가 없으면 `/`는 404다. 영상 확인 페이지는 빌드 유무와 관계없이 `/diagnostics`에서만 제공한다.

## 확인

브라우저에서 `http://<서버IP>:40225/`와 프론트 하위 경로를 새로고침한다.
Network에서 JS/CSS, `/api/cameras`, `/mjpeg/sourceN`이 같은 origin인지 확인한다.
컨테이너 안에서는 아래 명령으로 HTTP routing을 확인할 수 있다.

```bash
docker compose exec -T retrace python3 - <<'PY'
from urllib.request import urlopen
for path in ('/', '/api/health', '/api/cameras', '/streams.json'):
    with urlopen('http://127.0.0.1:40225' + path, timeout=5) as response:
        print(path, response.status, response.headers.get('Content-Type'))
PY
```

Backend health는 DB/Preview 준비 상태에 따라 기존 API의 503을 반환할 수 있다.
502는 Backend 연결 실패/timeout 또는 허용되지 않은 redirect/과대 응답이며 원문 예외를 공개하지 않는다.
주차 CRUD 경로만 POST/PATCH/DELETE를 전달한다(별도 IP/Origin 허용 목록 없음).
쓰기 경로·body 제한 및 retrace 적용 절차는
[Backend README](../../backend/README.md#preview-주차-crud-프록시)를 따른다.
