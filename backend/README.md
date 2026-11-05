# ReTrace Backend

FastAPI API와 PostgreSQL은 영상 처리와 별도 process/service로 실행한다.
Backend에는 GPU·GI·PyDS 또는 RTSP 비밀값이 필요하지 않다. Nuxt/Re-ID는 포함하지 않는다.

## 구성과 데이터 흐름

- 기존 루트: `preview.py`, `camera_config.py`, `camera_runtime.py`,
  `person_metadata.py`, `configs/`, `tests/`, `tools/`.
- `preview.py`는 HTTP 서버와 하나의 공유 batch pipeline을 실행한다.
  MJPEG는 `/mjpeg/source{source_id}`, 여러 카메라의 JPEG·metadata·status는 `/ws` 하나이며
  40225 포트를 유지한다.
- `/streams.json`은 배열이다. 각 항목은 `camera_id`, `floor`, `name`, `id`
  (source_id), `format`, `status` (기존 UI 호환), `url` (상대 MJPEG 경로),
  `metadata_path` (`/ws`), `runtime`, `runtime_session`을 포함한다. Backend는 runtime의 state를 사용한다.
- `CameraRuntime`은 카메라별 상태를 메모리에 보관한다. `MetadataStore`는
  source별 최신 immutable FrameMetadata만 보관한다. VehicleMetadata는
  camera_id/timestamp/track_id/class_id/class/confidence/tracker_confidence/bbox이다.
- 새 `/metadata.json`은 `{runtime_session, frames: [...]}`이다.
  frame은 source_id/camera_id/generation/timestamp/frame_number/좌표 크기/vehicles
  등을 포함한다. `persons`는 PeopleNet 제거 후 항상 빈 배열이며 Backend는 생략도 허용한다. 실패·이전 generation의 frame은 제외한다. JSON snapshot만
  제공하며 worker에서 네트워크나 DB I/O를 수행하지 않는다.
- Backend 단일 background thread가 2초 간격으로 두 JSON endpoint를 순서대로
  polling한다. HTTP timeout은 1.5초, 응답 크기 제한은 8 MiB다.
  status와 metadata의 session/generation/source_id를 대조하여 재연결 경합을 처리한다.
- push는 inference 쪽 전송/실패 관리가 필요하므로 선택하지 않았다. polling은
  추론 callback이 Backend/PostgreSQL을 호출하거나 기다리지 않아 장애 격리가 단순하다.
  retrace에는 backend/postgres의 depends_on을 추가하지 않았다.
- Backend가 preview에 접근하지 못하면 offline/fps=0으로 최신 행을 갱신하고
  현재 bbox를 비운다. 통신 장애는 카메라 자체 고장과 구별하는
  `preview_unreachable` 코드다. 상태 관찰이 10초 이상 오래되어도 조회 시 offline이다.
  DB 장애는 API 503, polling은 자동 재시도하며 외부 예외 내용을 로그에 쓰지 않는다.

## YAML과 DB

`camera_config.py`의 공통 YAML validation은 비밀값을 읽지 않는다.
`load_public_cameras()`는 camera_id/floor/name/enabled/source_id만 반환한다.
Backend 시작 시 YAML -> DB로 동기화하며 비활성 카메라도 유지한다.
삭제된 카메라는 과거 track 참조 보존을 위해 DB에서 disabled 처리한다.
활성 카메라 순서로 source_id를 부여하며 비활성 카메라는 source_id가 없다.
설정 변경 후 retrace와 backend를 함께 재시작한다. API 설정 수정 기능은 없다.

초기 테이블 생성은 SQLAlchemy `create_all`을 사용한다. 지금은 migration이
필요한 기존 Backend schema가 없으므로 Alembic을 추가하지 않았다. 향후 schema
변경은 migration이 필요하며 `create_all`은 기존 테이블 구조를 수정하지 않는다.
주차 기능은 기존 세 테이블을 수정하지 않고 `parking_spaces`, `parking_events` 테이블 및 index/check
constraint를 추가한다. 기존 DB에서도 Backend 초기화의 `create_all`로 추가되며
재실행은 멱등적이다. 기존 카메라·과거 사람 track·PostgreSQL volume은 보존한다.
향후 주차면 테이블의 컬럼 변경에는 명시적 migration이 필요하다.
PostgreSQL 데이터는 `postgres_data` named volume에 유지한다.
DB URL은 SQLAlchemy URL 객체로 구성하여 비밀번호의 특수문자를 지원한다.
DB statement/lock/connect timeout과 parameter masking을 적용한다.

## 저장 정책

- `cameras`: 공개 설정만 저장. RTSP URL/rtsp_env/계정은 컬럼 자체가 없다.
- `camera_status`: 카메라당 최신 상태 한 행을 UPDATE한다. state history는
  이번 단계에서 소비하는 기능이 없어 만들지 않았다. 동일 상태를 반복하거나
  state가 변경되어도 행 수는 증가하지 않는다.
- 실시간 bbox/confidence: Backend의 `LiveStore`에 camera별 최신 frame을 교체한다.
  빈 frame은 기존 차량을 제거하며 frame timestamp TTL은 10초다. DB에는 bbox가 없다.
- `person_tracks`: PeopleNet 제거(2026-10-13) 후 **동결된 과거 기록**이다. 새 행을 쓰지 않고
  retention 삭제도 하지 않으며 `/api/cameras/{id}/tracks`의 `recent`로만 조회한다(`current`는 빈 배열).
  아래는 제거 전 동작이다. polling에서 관찰된 tracked person만
  `(camera_id, runtime_session, generation, track_id)` 단위로 INSERT/UPDATE했다.
  first_seen_at/last_seen_at/max_confidence만 저장한다. 같은 frame 재조회는 UPDATE도
  생략한다. 매 frame 저장이 아니며 polling에서 놓친 짧은 track은 수집되지 않는다.
- track_id는 uint64이므로 DB/API에서 10진 문자열로 표현한다. PostgreSQL signed
  BIGINT overflow와 JavaScript 정밀도 손실을 피한다.
- (제거 전) track 요약은 기본 최근 7일만 유지하며 60초마다 오래된 last_seen_at 행을 삭제했다. `parking_events`의 7일 retention은 유지한다.
  DB 장애 중에는 정리가 지연될 수 있다. 이 단계에서는 Re-ID 영구 이동 이력을
  보장하지 않는다. last_seen_at은 마지막 관찰 시각이며 track 종료를 뜻하지 않는다.
- timestamp는 기존 pipeline 처리 시각이다. 카메라 capture/NTP 시각은 아니다.

## API

| Method / path | 응답 |
| --- | --- |
| GET /api/health | DB readiness와 preview 연결 상태; DB 장애 503 |
| GET /api/cameras | 공개 카메라 목록과 최신 status, 상대 preview_path |
| GET /api/cameras/{camera_id} | 공개 카메라 상세; 알 수 없는 ID 404 |
| GET /api/cameras/{camera_id}/status | 최신 runtime 상태 및 observed_at/stale |
| GET /api/cameras/{camera_id}/tracks?limit=100 | current bbox와 recent DB 요약; limit 1~500 |
| GET /api/cameras/{camera_id}/vehicles | 최신 차량 pixel bbox, 문자열 track ID, 선택적 `outline`, detector 가용성 |
| GET /api/status | 전체 카메라 상태 목록과 state별 counts |
| POST /api/cameras/{camera_id}/parking-spaces | 주차면 생성, 201 |
| GET /api/cameras/{camera_id}/parking-spaces | 카메라별 목록, 생성 시각/ID 순 |
| GET /api/cameras/{camera_id}/parking-spaces/{space_id} | 주차면 상세 |
| PATCH /api/cameras/{camera_id}/parking-spaces/{space_id} | 이름/Polygon 부분 수정 |
| DELETE /api/cameras/{camera_id}/parking-spaces/{space_id} | 삭제, 204 (본문 없음) |
| GET /api/cameras/{camera_id}/parking-summary | total / occupied / empty / unknown 개수 |

Pydantic 응답 schema는 `app/schemas.py`, OpenAPI는 `/openapi.json`에 있다.

### 차량 윤곽(outline)

`/api/cameras/{camera_id}/vehicles`의 차량마다 `outline`이 있다(2026-11-05 추가, 다른 필드와 의미는 그대로).

- 값: `[[x, y], ...]` 또는 `null`. 좌표계는 같은 차량의 `bbox`와 같은 mux pixel(`bbox_width × bbox_height`, 지금
  1920×1080)이고 원점은 왼쪽 위, y는 아래로 증가한다. 0~1 정규화는 `bbox_width`/`bbox_height`로 나눈다.
- 닫힌 다각형: 마지막 점이 첫 점으로 이어지며 첫 점을 반복하지 않는다. 화면 기준 시계 방향, 3–32점, 소수 첫째 자리.
- 윤곽은 tracker 박스 위에 늘인 것이라 항상 `bbox` 안에 있다.
- `null`: seg를 끔(`VEHICLE_SEG_INFER_CONFIG` 비어 있음), seg가 그 차량을 찾지 못함, mask가 박스의 15% 미만, 계산
  오류, preview가 보낸 값이 형식에 맞지 않음. 형식 오류는 그 차량만 null이 되고 frame과 다른 차량은 그대로다
  (poller가 frame 하나의 검증 실패로 모든 카메라 frame을 버리기 때문). 프론트는 null이면 bbox 판정으로 대체한다.

알려진 한계(주차면 판정에 쓸 때):

- 윤곽은 mask에서 만든 다각형 하나이며, mask가 있는 행마다 가장 왼쪽과 가장 오른쪽 경계를 잇는다(최대 16행).
  같은 행에서 mask가 여러 조각이면 그 사이를 메운다. 그래서 기둥·다른 차·사람에 가려 mask가 끊긴 차량은 윤곽이
  가린 물체를 가로질러 그 영역까지 포함한다.
- 윤곽 하단(접지부): 바퀴 사이의 빈 곳은 메워져 차량 바닥 면적으로 보인다. 차량 아래쪽이 앞 물체에 가려지면 하단은
  가린 물체를 포함하거나, mask가 끝난 곳에서 끝나 실제 접지선보다 위에 있을 수 있다. 가려진 부분은 추정하지 않는다.
- seg 박스와 det 박스가 다르면 윤곽이 그만큼 늘거나 줄어든다(정차 차량은 차이가 작다). 행 사이의 오목한 모양은
  직선으로 이어진다.

Browser 영상 요청은 기존 retrace:40225로 직접 보낸다. Backend의 preview_path는
preview origin 기준 상대 경로다. `metadata_path`(`/ws`)는 여러 카메라 JPEG와 metadata를
WebSocket 하나로 받는 경로다. Backend는 영상을 relay하지 않는다. binary 형식은
[실시간 Preview](../docs/preview-realtime.md)를 따른다. Backend poller의 Preview origin은
`BACKEND_PREVIEW_URL`(기본 `http://retrace:40225`)이다.
배포한 CSR 프론트는 40225에서 정적 제공하며 API도 같은 origin의 `/api/*`로 호출한다.
Preview HTTP request thread가 내부 Backend로 전달하므로 브라우저에 8000 주소를
지정할 필요 없다. 업로드·적용 절차는 [프론트 정적 배포](../frontend/README.md#정적-배포)를 따른다.
로그인은 아직 없으므로 Backend host binding은 기본 loopback이다.

### 주차면 계약과 저장

생성 요청 예시 (상대 경로 `/api/cameras/{camera_id}/parking-spaces`):

```json
{
  "name": "A-01",
  "polygon": [
    {"x": 0.1, "y": 0.2},
    {"x": 0.8, "y": 0.2},
    {"x": 0.8, "y": 0.9},
    {"x": 0.1, "y": 0.9}
  ]
}
```

- 원점은 전체 영상 좌상단이다. x는 영상 너비, y는 높이로 나눈 0~1 정규화
  좌표다. UI의 letterbox 여백/크롭 좌표를 포함하지 않는다. 향후 DeepStream
  bbox도 동일한 전체 영상 기준으로 정규화해야 한다.
- Polygon은 3~64개의 서로 다른 점이다. 마지막에 첫 점을 반복하지 않는다.
  유한한 숫자만 허용하며 문자열/boolean/null/범위 밖 값, 면적 0, 자기 교차,
  변 접촉/되짚기는 422다. 시계/반시계 방향과 오목한 단순 Polygon을 허용한다.
  이름은 앞뒤 공백 제거 후 1~128자다.
- 응답은 `space_id`(UUID 문자열), `camera_id`, `name`, `polygon`, `occupancy`,
  `revision`, `occupancy_observed_at`, `created_at`, `updated_at`이다. 시각은 UTC
  ISO 8601이다. 생성 시 `occupancy="unknown"`, `revision=1`,
  `occupancy_observed_at=null`이다. 상태는 `occupied / empty / unknown`만 허용한다.
  `camera_id`는 `/api/cameras`가 반환하는 설정 ID 문자열이며 숫자 source_id와 다르다.
  예를 들어 생성/단건/PATCH 응답은 다음 형식이고, 목록 GET은 이 객체의 배열이다.

  ```json
  {
    "space_id": "11111111-1111-4111-8111-111111111111",
    "camera_id": "parking-a", "name": "A-01",
    "polygon": [{"x": 0.1, "y": 0.2}, {"x": 0.8, "y": 0.2},
                {"x": 0.8, "y": 0.9}, {"x": 0.1, "y": 0.9}],
    "occupancy": "unknown", "revision": 1, "occupancy_observed_at": null,
    "created_at": "2026-10-08T00:00:00Z", "updated_at": "2026-10-08T00:00:00Z"
  }
  ```

- PATCH는 `name`과 `polygon` 중 하나 이상을 받는다. 명시적 null과 알 수 없는
  필드(occupancy/camera_id/revision 포함)는 422다. 실제 Polygon 변경은 revision을
  증가시키고 점유를 unknown으로 초기화한다. 이름만 바꾸거나 동일한 Polygon을
  저장하면 점유와 revision을 유지한다. 동시 수정은 PostgreSQL 행 잠금으로
  직렬화하며 같은 필드는 나중에 잠금을 얻어 적용한 요청 값이 남는다. HTTP 도착
  순서 보장은 없다. `revision`은 Polygon/판정 세대이며 편집용 optimistic lock이
  아니다. `If-Match`/ETag/expected revision 및 충돌 409/412는 구현하지 않았다.
  편집 중 다른 사용자의 변경을 자동 거부하지 않으므로 저장 후 객체를 다시 반영한다.
  이름만 변경하는 요청 예시는 `{"name":"A-02"}`다. Polygon만 수정할 때는 생성
  요청과 동일한 `polygon` 배열만 보낸다. DELETE 성공은 204이며 JSON 본문이 없다.
- 없는 카메라/주차면, 다른 카메라에 속한 면은 404, 잘못된 UUID/요청은 422,
  DB 장애는 안전한 503이다. 비활성 카메라의 도면도 관리할 수 있다. YAML에서
  제거된 카메라는 기존 정책대로 disabled로 남고 주차면도 유지한다.
  오류 본문은 `{"detail":"..."}`다. 422는 `Invalid request`, 빈 PATCH는
  `At least one update field is required`, 404는 `Camera not found` 또는
  `Parking space not found`, DB 오류 503은 `Database unavailable`이다.
- Summary 예: `{"camera_id":"first","total":3,"occupied":1,"empty":1,"unknown":1}`.
  도면이 없으면 모두 0이다. 현재 조회는 오래된 관찰을 unknown으로 응답하고 poller는
  카메라/metadata 장애를 unknown으로 저장한다. 탐지·시간 판정·설정·이벤트 계약은
  [차량/점유 문서](../docs/parking-occupancy.md)를 따른다.
- `parking_spaces`는 Polygon(JSON), 현재 점유 상태와 시각을 PostgreSQL에 저장한다.
  Backend 재시작/카메라 동기화가 Polygon을 삭제하지 않는다. Backend 시작 시 점유는
  unknown으로 바꾸며 상태 변경 이력은 `parking_events`에 보관한다.
  기존 `sessions.begin()` transaction과 Pydantic 응답 패턴을 따른다.
- `backend/app/parking.py`는 점유 저장과 transition 생성 경계다. Backend poller는
  `occupancy.py`의 판정 결과를 Polygon 행 잠금과 같은 transaction에서 저장한다.
  `update_occupancy(session, camera_id, space_id, revision, occupancy, observed_at)`는
  별도 producer 연동 경계로 유지하며 이전 revision/중복/과거 시각을 거부한다.
  [차량/점유 문서](../docs/parking-occupancy.md)에 추가 조회 API, metadata/WebSocket,
  모델 활성화와 실영상 재생 절차가 있다.

## 실행

기존 `.env`에 `.env.example`의 Backend 항목을 추가한다. POSTGRES_PASSWORD는
고유한 로컬 비밀번호를 설정한다. 빈 비밀번호면 postgres/backend 시작에 실패한다.
Compose interpolation은 retrace 단독 실행도 가능하도록 빈 기본값을 허용한다.
Backend/PostgreSQL에는 명시적인 DB 변수만 주입하며 `.env`를 mount하지 않는다.
실제 비밀번호와 기존 RTSP 설정은 Git에 포함하지 않는다.

```bash
docker compose build
docker compose up -d
docker compose ps
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS http://127.0.0.1:8000/api/cameras
# Preview에 loopback binding이 있는 환경에서만 사용한다.
curl -fsS -o /dev/null http://127.0.0.1:40225/streams.json
docker compose exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;"'
docker compose exec -T retrace python3 tools/check_backend.py
```

BACKEND_PORT를 변경했다면 curl의 포트를 맞춘다. preview host binding은 기존
ignored compose.override.yaml을 유지한다. `check_backend.py`는 실제 API와
preview JSON 전체를 읽고 로컬 RTSP URL/host/계정/비밀번호와 DB password의
노출을 검사하며 응답 내용이나 비밀값을 출력하지 않는다.

### 별도 개발 PC에서 접근

서버에는 retrace/backend/postgres만 유지한다. Frontend/Nuxt는 별도 PC에서
개발한다. **우선 경로는 기존 loopback 바인딩 + Mac SSH 터널 + Nuxt 개발 프록시다.**
Backend host binding `${BACKEND_BIND_IP:-127.0.0.1}`, Preview의 제한된 LAN/Tailscale
바인딩, PostgreSQL 비공개 정책을 유지한다. Mac에서 다음과 같이 실행한다.

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:18000:127.0.0.1:8000 "$RETRACE_SSH_TARGET"
curl --fail http://127.0.0.1:18000/api/health
```

`RETRACE_SSH_TARGET`은 Mac의 private 환경 변수(SSH 사용자/호스트 또는 SSH config
alias)다. SSH 서버의 TCP forwarding 허용과 기존 LAN/Tailscale 접근 권한이 필요하다.
18000은 Mac의 미사용 loopback 포트로 선택하고, 8000은 실제 Backend 공개 포트와
맞춘다. 위 명령은 서버의 Docker 바인딩을 바꾸지 않는다.

Mac 개발 설정에서는 `RETRACE_API_UPSTREAM=http://127.0.0.1:18000` 같은 private
환경 변수를 정하고 Nuxt 개발 프록시의 upstream으로 연결한다. 이 변수명은 **Mac
설정용 제안이며 현재 ReTrace 서버가 자동으로 읽는 변수가 아니다.** 프론트 코드는
이 저장소 작업에서 변경하지 않는다. 프록시는 `/api/...` 경로와 query를 보존하고
GET/POST/PATCH/DELETE, JSON body, status code, 응답을 전달해야 한다.
브라우저는 Nuxt origin의 상대 `/api/...`를 사용하므로 Backend CORS 변경이 필요 없다.

```text
Mac browser /api/... → Mac Nuxt dev proxy
                    → Mac 127.0.0.1:18000 (SSH)
                    → server 127.0.0.1:8000/api/...
```

Preview 40225를 기존 API upstream으로 사용하는 경우에는 아래 “Preview 주차 CRUD
프록시”의 접근 제어를 설정한 뒤 주차면 쓰기를 사용할 수 있다.
`/ws`는 기존 Preview JPEG/metadata 연결을 유지하며 Backend 8000으로 프록시하지
않는다. 직접 cross-origin WS를 새로 연결하려면 별도의 `PREVIEW_WS_ORIGINS` 검토가
필요하다. REST용 `BACKEND_CORS_ORIGINS`는 WS 허용 목록에 영향을 주지 않는다.
WS 프록시를 사용하는 경우에도 Host/Origin 검증이 통과하는지 확인해야 한다.

실제 Mac에서 터널 연결 후 health뿐 아니라 `/api/cameras`와 배포된 주차면 GET도
확인한다. health 200만으로 주차면 코드 배포를 증명할 수 없다. SSH 정책·Mac 프록시·
브라우저 왕복은 해당 Mac에서 검증해야 한다.

#### 대안: 승인된 사설 인터페이스 직접 접근

직접 접근을 선택할 경우에만 서버의 private `.env`에서 `BACKEND_BIND_IP`를 기존
Preview에 사용 중인 LAN 또는 Tailscale 인터페이스 주소 하나로 지정할 수 있다.
`0.0.0.0` 또는 `::`를 사용하지 않는다. 실제 주소는 공유 파일에 기록하지 않는다.
이 설정은 기존 loopback binding을 선택한 인터페이스로 대체한다.
컨테이너 내부 Uvicorn의 wildcard listener와 호스트 공개 범위는 별개다.
PostgreSQL은 호스트에 공개하지 않으며 Preview의 private override는 유지한다.

브라우저 API 호출에는 서버의 private `.env`에서 `BACKEND_CORS_ORIGINS`를
쉼표로 구분한 정확한 origin 목록으로 설정한다. 예:
`http://localhost:3000,http://127.0.0.1:3000`. 두 origin은 서로 다르며 실제
개발 서버 포트를 맞춘다. 기본값은 비어 있어 cross-origin 허용 응답이 없다.
허용 method는 GET/POST/PATCH/DELETE이며 JSON Content-Type 및 OPTIONS preflight는
middleware가 처리한다.
credential 허용은 꺼져 있으며 wildcard origin은 시작 시 거부한다.
Nuxt 서버 측 호출에는 브라우저 CORS가 적용되지 않지만 네트워크 접근은 필요하다.

설정 후 `docker compose up -d --build --no-deps backend`로 Backend만 재생성한다.
Frontend service, pipeline, 모델/tracker 변경은 필요 없다. 새 Backend 초기화 시
주차면 테이블만 추가된다.
MJPEG는 Preview origin의 상대 `preview_path`를 기존 40225 주소에 붙여
`<img src>`로 표시할 수 있다. 6대를 넘게 표시하려면 `/ws`를 사용한다. Preview에는 CORS 헤더가 없으므로 fetch나
canvas 픽셀 읽기가 필요해지면 별도로 검토한다.

주차면 코드 적용은 Docker 권한이 있는 운영자가 다음 명령으로 Backend만 수행한다.
PostgreSQL named volume을 제거하거나 retrace를 재시작할 필요는 없다.

```bash
docker compose up -d --build --no-deps backend
docker compose exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT count(*) FROM parking_spaces;"'
```

적용 후 실제 카메라 ID로 CRUD 및 Mac 연결을 확인한다. 테이블 생성은 DB 사용자의
CREATE 권한이 필요하다. SQLite CPU 테스트는 실제 PostgreSQL DDL/행 잠금/네트워크
검증을 대체하지 않는다.

## Preview 주차 CRUD 프록시

Preview의 `/api` GET/HEAD는 기존대로 전달한다. 쓰기는 아래 조합만 허용한다.

| Method | 경로 |
| --- | --- |
| POST | `/api/cameras/{camera_id}/parking-spaces` |
| PATCH / DELETE | `/api/cameras/{camera_id}/parking-spaces/{space_id}` |

카메라 ID는 기존 ASCII ID 규칙, space_id는 UUID 경로 형식을 요구한다. 다른 쓰기
경로·조합은 405이며 `/ws`·정적 파일·영상에는 쓰기를 연결하지 않는다. 원본 method,
query, Content-Type, JSON bytes, Authorization/Cookie를 전달한다. Backend의
201/200/204/4xx/5xx와 body를 보존하며 204는 body와 Content-Length 없이 보낸다.
기존 redirect 차단(502)은 유지하여 내부 Location을 노출하지 않는다.

현재 로그인/사용자 인증과 별도 쓰기 허용 목록은 없다. 개인 개발 환경에서 기존
LAN/Tailscale 인터페이스로 제한된 40225 포트에 접근할 수 있는 클라이언트는 위 경로의
주차면 CRUD를 사용할 수 있다. 접근 범위는 이 포트 바인딩과 네트워크(Tailscale ACL 등)로
정해지므로 Preview를 `0.0.0.0`·공개 인터페이스에 바인딩하지 않는다. Backend 8000의
loopback 공개도 유지한다.

Preview는 CORS 헤더와 OPTIONS preflight를 제공하지 않는다(OPTIONS는 501).
브라우저는 같은 40225 origin 또는 Nuxt 개발 프록시를 통해 `/api`·`/ws`를 사용한다.
Backend CORS와 Preview WS Origin 설정은 별개다.

POST/PATCH는 Content-Length와 application/json을 요구한다. DELETE는 body가 없어도
된다. 최대 body는 64KiB, 전체 읽기 제한은 기본 3초다. 중복/잘못된 Content-Length,
Transfer-Encoding는 400, 길이 누락은 411, 초과는 413, 비 JSON은 415, Expect는 417,
읽기 시간 초과는 408이다. JSON 내용 검증은 기존 Backend가 처리한다. 연결 실패/timeout,
4MiB 초과 응답은 안전한 502이고 내부 주소·예외 원문을 반환하지 않는다.

Preview 소스만 변경했으므로 아래 재시작으로 적용한다. 새 환경 변수는 없으며
Backend 재빌드와 DB schema 작업이 필요 없다.

```bash
sudo docker compose restart retrace
```

재시작 동안 Preview 영상/WS가 잠시 끊긴다. Docker 권한이 없으면 운영자가 직접 실행하며
권한 변경·sudo 우회는 하지 않는다. 적용 후 Mac에서 주차면 CRUD와 /ws를
확인한다. 실제 Mac 연결은 CPU 테스트가 대신 검증하지 않는다.

## 운영 주차 기능 적용 순서

작업 트리의 주차 API/relay 구현을 함께 적용한다. Backend는 이미지 COPY이므로
빌드·재생성이 필요하고 Preview는 bind mount이므로 retrace 재시작이 필요하다.
Backend 교체 중 API, retrace 재시작 중 JPEG/WS가 잠시 중단된다. Backend 시작은
`create_all`로 없는 테이블만 추가하고 기존 Polygon을 유지하되 점유를 unknown으로
초기화한다. 기존 카메라 동기화·기본 7일 history retention 정책은 유지한다.
기존 테이블 ALTER/삭제, PostgreSQL 재생성·볼륨 제거는 수행하지 않는다.

아래는 Docker 권한이 없는 계정 대신 **운영자가 직접 실행하는 명령**이다. 배포 전
실행 중 backend의 DB 대상/포트 바인딩이 현재 private Compose 설정과 같은지 확인한다.
기존 parking 테이블의 schema가 현재 모델과 다르면 이 절차를 중단하고 migration을
별도로 검토한다. `create_all`은 기존 schema를 고치지 않는다.

```bash
cd ~/retrace
sudo docker compose ps --all
sudo docker compose exec -T postgres sh -c 'psql -X -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT table_name,column_name,data_type FROM information_schema.columns WHERE table_schema = '\''public'\'' AND table_name IN ('\''parking_spaces'\'','\''parking_events'\'') ORDER BY table_name,ordinal_position;"'
```

DB 백업은 저장소 밖의 새 비공개 디렉터리에 저장한다. `pg_dump` 성공, 비어 있지 않은
파일, `pg_restore --list` 성공을 모두 확인해야 다음 적용 단계로 진행한다. manifest와
dump에는 운영 데이터가 있으므로 공유하지 않는다. 복원 실행은 이 절차에 포함하지 않는다.

```bash
umask 077
RETRACE_BACKUP_DIR=$(mktemp -d "$HOME/retrace-db-backup.XXXXXX")
sudo docker compose exec -T postgres sh -c 'exec pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "$RETRACE_BACKUP_DIR/before-parking.dump"
test -s "$RETRACE_BACKUP_DIR/before-parking.dump"
sudo docker compose exec -T postgres pg_restore --list < "$RETRACE_BACKUP_DIR/before-parking.dump" > "$RETRACE_BACKUP_DIR/manifest.txt"
```

위 검사가 성공한 뒤에만 아래를 순서대로 실행한다. 중간 단계가 실패하면 중단하고
원인을 확인한다. postgres와 의존 서비스를 재생성하지 않으며 private 설정은 변경하지 않는다.

```bash
sudo docker compose build backend
sudo docker compose up -d --no-deps --no-build --force-recreate backend
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/parking
sudo docker compose restart retrace
sudo docker compose ps --all
sudo docker compose exec -T postgres sh -c 'psql -X -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT to_regclass('\''public.parking_spaces'\'') IS NOT NULL AS spaces, to_regclass('\''public.parking_events'\'') IS NOT NULL AS events;"'
```

Backend 응답이 200이 될 때까지 기다린 뒤 실제 `/api/cameras`의 ID로 주차면 목록을
조회한다. retrace가 영상 추론을 준비하는 동안 metadata 상태는 잠시 unknown일 수 있다.
Preview `/ws` 최초 연결·재연결, JPEG/people, 주차 상태를 확인한다. 격리된 카메라/도면
대상이 없으면 운영 CRUD 쓰기는 생략한다. 모델이 비활성이면 occupied/empty 판정 대신
unknown이 정상이다. 로그와 Docker inspect/config 원문에는 private 값이 포함될 수
있으므로 외부 공유 전에 마스킹한다. 차량 모델 설치/활성화는 이 적용에 포함하지 않는다.

## CPU 테스트

Python 3.11+ 환경에서 GPU 없이 실행한다.

```bash
python3.11 -m venv /tmp/retrace-backend-test
/tmp/retrace-backend-test/bin/pip install -r backend/requirements-test.txt
/tmp/retrace-backend-test/bin/python -m pytest -q backend/tests
/tmp/retrace-backend-test/bin/python -m unittest discover -s tests -p 'test_camera_*.py'
/tmp/retrace-backend-test/bin/python -m unittest discover -s tests -p 'test_rtsp_inputs.py'
/tmp/retrace-backend-test/bin/python -m unittest discover -s tests -p 'test_canonical_startup.py'
/tmp/retrace-backend-test/bin/python -m unittest discover -s tests -p 'test_preview_http.py'
/tmp/retrace-backend-test/bin/python -m unittest discover -s tests -p 'test_check_preview.py'
```

Backend 테스트는 SQLite 파일 DB와 실제 SQLAlchemy transaction을 사용하고 preview를
HTTP MockTransport로 대체한다. 실제 PostgreSQL 연결/GPU/RTSP/컨테이너 build 검증은
위 서버 명령으로 수행한다. 단일 Uvicorn worker만 사용한다. worker 증설 시 polling
중복과 메모리 snapshot 분리가 생기므로 별도 설계가 필요하다.

## 색상 기반 주차 MVP

새 `/api/parking/spaces`, `/api/parking/zones`, `/api/parking/status`와 영역별 calibrate API는 차량 모델 없이 동작한다. 논리 공간과 다중 카메라 영역을 별도 테이블에 보존하며 기존 카메라별 주차 API는 유지한다. 기존 JPEG 프레임을 재사용한다. 요청·응답, 환경 변수, WS 연결, 추가 테이블 적용 및 한계는 [색상 주차 계약](../docs/color-parking.md)을 따른다. Backend image rebuild(OpenCV/NumPy 포함)와 Preview restart가 필요하다.

## AI 차량 데이터셋

`app/dataset.py`의 router/service와 `app/dataset_models.py`가 캡처·영속 이미지 관리·픽셀 박스
저장·검수·YOLO ZIP 작업을 제공합니다. 새 `dataset_images`, `dataset_exports`는 기존 Base의
additive `create_all`로 생성하며, JPEG/ZIP은 `DATASET_STORAGE_DIR`의 `dataset_data` 볼륨에 둡니다.
기존 Preview OSD JPEG를 데이터셋으로 저장하지 않습니다. 기본 비활성인 요청 시 pre-OSD 캡처,
API·DB·좌표·분할·복구·용량 제한·단일 worker 경계와 운영 적용/미검증 사항의 원문은
[AI 데이터셋](../docs/datasets.md)입니다. `tests/test_datasets.py`로 CPU 검증합니다.

데이터셋 후속: 새 ZIP은 `0: car`만 내보냅니다. 기존 truck/bus 라벨을 자동 변경하지 않고 해당 검수 데이터가 있으면 내보내기 오류로 안내합니다. `DELETE /api/datasets/exports/{uuid}`는 building을 거부하고 기록·ZIP만 삭제하며 원본 이미지·라벨은 보존합니다. Backend rebuild와 Preview 프록시 restart 후 사용합니다.

## 웹 차량 모델 fine-tuning

`app/training.py`(router `/api/training`, `TrainingService`)와 `app/training_models.py`(additive `training_jobs`,
`training_models`)가 학습 job 생성·진행·중지·결과·모델 registry를 제공합니다. 학습은 API 요청 안에서 실행하지 않으며
host의 `training/worker.py`가 `TRAINING_WORKER_TOKEN`으로 `/api/training/worker/*`를 호출합니다(Preview proxy는 404).
시작·중지·다운로드는 `TRAINING_ADMIN_TOKEN` Bearer가 필요하고, 두 토큰이 비어 있으면 해당 기능은 503으로 꺼집니다.
데이터셋 snapshot·산출물은 `DATASET_STORAGE_DIR/training/`에 둡니다. 계약·GPU 보호 기준·운영 절차는
[웹 fine-tuning](../docs/training.md), CPU 검증은 `tests/test_training.py`입니다.
