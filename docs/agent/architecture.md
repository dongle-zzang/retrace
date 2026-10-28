# 구조와 실행 경계

Re-ID/카메라 간 동일 인물 매칭은 구현되어 있지 않다. 영상 처리와 Backend/DB는 독립 프로세스/서비스로 연결된다.

```text
RTSP → retrace (preview.py, 공유 DeepStream batch pipeline)
           ├─ 카메라별 OSD/JPEG → /mjpeg/sourceN, 또는 /ws binary(모든 카메라 연결 1개) → Frontend <img>
           ├─ 최신 metadata/status → /ws JSON
           └─ 공개 상태/metadata JSON ← backend poller → PostgreSQL
                                                   └─ FastAPI 조회 API
```

## 코드 진입점과 책임

- `preview.py`: 기본 실행부. HTTP 서버와 하나의 mux/차량 nvinfer(주차장 카메라만)/NvDCF/demux pipeline을 관리하며 공유 bus를 polling한다. enabled 카메라 수로 batch를 구성하고 source별 nvurisrcbin native RTSP reconnect와 OSD/JPEG branch를 둔다. TCP/latency/drop-on-latency/decoder surface는 nvurisrcbin 속성으로 설정하며, child-added는 선택적 진단 관측만 수행한다. drop-on-latency 기본값은 true다. source bin은 입력 queue/converter/capsfilter 없이 legacy nvstreammux에 직접 연결하며, mux가 1920×1080으로 스케일링한다.
- `app.py`: preview가 재사용하는 RTSP source bin helper만 남았다. PeopleNet CLI/PGIE 준비 함수는 2026-10-13 제거했다.
- `pgie_cache.py`: 명시적 TensorRT build identity, `.cache/vehicle/pgie` 모델 staging 및 config/GObject engine 경로 일치를 담당한다. 캐시 정책과 최초 rebuild/legacy 처리 범위는 [README의 모델 준비](../../README.md)를 따른다.
- `camera_config.py`: 공통 YAML 검증, 공개 설정 추출, private RTSP resolve. backend는 `load_public_cameras()`만 사용한다.
- `camera_runtime.py`: CameraRuntimeManager/CameraRuntime의 슬롯 매핑, 프레임 기반 상태, 관측 재연결 세대와 PTS 세대 라우팅. GPU lifecycle은 소유하지 않는다. 상태 설정과 실제 연결 대상인 YAML `enabled`는 역할이 다르다.
- `person_metadata.py`: vehicle PyDS 값의 immutable 복사와 최신 frame 저장(모듈명과 빈 `persons` 필드는 계약 호환용).
- `vehicle_detection.py`: `VEHICLE_CAMERA_IDS` 주차장 카메라의 full-frame ROI → 공유 YOLO11m secondary detector 하나(batch=주차장 카메라 수) → class 분리 후 NvDCF. enabled가 아닌 ID는 건너뛴다. 모델/config 준비는 기존 engine cache를 별도 namespace로 재사용한다.
- `vehicle_deployment.py`: `VEHICLE_DEPLOYMENT_ENABLED=1`일 때 차량 nvinfer의 config 기반 on-the-fly model update, 추론 검증, committed 모델 journal과 자동/수동 롤백을 preview control loop에서 소유한다. host worker와는 `.cache/vehicle-deployments` 파일 mailbox로만 통신하고 engine은 `tools/build_vehicle_engine.py` 별도 프로세스가 만든다. 절차·판정 기준은 [docs/training.md §4-2](../training.md).
- `parking_relay.py`: Backend 현재 점유/이벤트 API를 GPU callback 밖에서 읽어 기존 Preview `/ws` hub로 전달한다. 구조·정책·모델/검증 한계는 [차량/점유](../parking-occupancy.md)를 따른다.
- `preview_socket.py`: 같은 HTTP listener의 `/ws`. 다중 camera 구독, normalized bbox와 상태/event, 요청 시 카메라별 최신 JPEG binary 전송. 네트워크는 GPU callback 밖에서 수행한다.
- `preview_mjpeg.py`: 카메라별 최신 JPEG 저장(FrameStore)과 `/mjpeg/sourceN` multipart 전송.
- `backend/app/`: GPU를 import하지 않는 API·poller·DB 계층. 서버/API/DB 수정은 [backend](backend.md)를 읽는다.
- `preview_web.py`: 전용 디렉터리의 CSR 정적 빌드 제공과 같은 40225 origin의 `/api/*` 프록시. GET/HEAD를 유지하고 쓰기는 주차 CRUD 및 데이터셋 캡처/라벨/삭제/내보내기 경로로 제한하고 body 크기/framing/timeout을 검사한다(별도 IP/Origin allowlist·CORS 없음). API 요청은 HTTP request thread에서 처리하며 공유 GPU callback과 연결하지 않는다. Backend polling 계약은 유지한다.

## 경계를 변경할 때

preview 영상은 tracker 결과를 nvdsosd로 그린 JPEG 하나뿐이다. WebRTC/H.264 경로는 제거했다(사유는 실시간 Preview 문서).
app.py 직접 실행은 tracker/video를 포함하지 않는다. native PyDS 대신 복사된 metadata를 전달한다.
`/metadata.json`과 Backend polling은 유지하며 `/ws` normalized 좌표 계약은 별도다.
브라우저 origin당 HTTP 연결 6개 제한 때문에 여러 카메라 화면은 `/ws` 영상을 사용한다. binary 형식·backpressure·대역폭은 [실시간 Preview](../preview-realtime.md)가 원문이다.

실제 runtime lifecycle, 설정 재해석, stale frame 제거, 세대와 좌표 의미, native source reconnect/네이티브 장애의 격리 한계는 [루트 README](../../README.md)의 “카메라 runtime lifecycle”과 “구조화된 metadata와 탐지 진단”을 참조한다. Backend와 preview의 장애 격리를 위해 polling을 선택한 이유와 저장 범위는 [backend README](../../backend/README.md)의 “구성과 데이터 흐름” 및 “저장 정책”이 원문이다.

첫 영상의 source/RTSP/decoder/mux/차량 nvinfer/tracker/JPEG/HTTP 계측 위치, 기준 시각과
1·2·4·8 source 서버 비교 실험은 [startup timing](startup-timing.md)을 참조한다.

새 장비는 Git 문서만으로 모델/secret/GPU 환경까지 복구할 수 없다. 실행 환경 준비는 [deployment](deployment.md)를 따른다.

색상 주차 worker는 기존 FrameStore의 `/snapshots/{cameraId}.jpg`를 CPU Backend에서 주기적으로 조회·decode한다. RTSP/source/GPU pipeline은 추가하지 않는다. `parking_relay.py`가 새 통합 상태를 기존 `/ws`에 `parking.status_updated`로 전달한다. 상세 경계/좌표/TTL은 [색상 주차](../color-parking.md)를 따른다.

데이터셋은 opt-in `preview_capture.py`가 기존 decode 결과의 OSD RGBA 변환 앞에서 요청당 한 프레임을
JPEG로 전달하고, `backend/app/dataset.py`가 파일 볼륨/DB·라벨·ZIP을 관리한다. 기존 미리보기와
metadata polling은 유지한다. 해상도·동일 프레임 한계·저장 경계는 [AI 데이터셋](../datasets.md)을 따른다.
