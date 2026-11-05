# CCTV 실시간 Preview

## 전송 구조

`preview.py`는 입력·mux(1920×1080)·주차장 카메라 전용 YOLO11m 차량 detector·NvDCF와 native RTSP reconnect를 공유 pipeline 하나로 실행한다(PeopleNet은 2026-10-13 제거).
tracker probe는 PyDS 값을 복사하고 네트워크/DB I/O는 하지 않는다. 영상은 카메라별 OSD/JPEG branch 하나뿐이다.

```text
RTSP → nvurisrcbin(NVMM) → nvstreammux → 차량 nvinfer(주차장 카메라 ROI만) → [seg nvinfer, 선택] → NvDCF
  ├─ tracker probe → MetadataStore → /ws JSON, /metadata.json
  └─ nvstreamdemux → 카메라별 queue(leaky) → nvdsosd → nvjpegenc(1280×720) → appsink
      → FrameStore(카메라당 최신 JPEG 1장)
          ├─ /mjpeg/sourceN : multipart HTTP, 카메라당 연결 1개
          └─ /ws            : 구독한 모든 카메라 JPEG를 binary message로, 연결 1개
```

JPEG에는 서버 OSD(주차장 카메라의 차량 bbox와 `Vehicle` 라벨만)가 그려져 있다. seg(`VEHICLE_SEG_INFER_CONFIG`)를 켜면
주차장 카메라 차량에 윤곽선이 그려지고, 윤곽이 붙은 차량의 박스는 `VEHICLE_OUTLINE_BOX_WIDTH`(기본 1, 0이면 숨김)로
얇아진다. 윤곽이 없는 차량은 기존 굵은 박스다([운영](operations.md#차량-윤곽선seg-선택)). 차량 track ID는 metadata에만 있다. 같은 정보가 `/ws` detections로도 오므로
제품 화면은 JPEG만 표시하거나 별도 overlay를 그릴 수 있다.

브라우저는 HTTP/1.1에서 origin당 동시 연결을 6개로 제한한다. `/mjpeg/sourceN`은 응답을 계속 열어 두므로
한 화면에서 6개를 넘게 재생할 수 없다. `/ws`는 모든 카메라를 연결 하나로 전달하므로 이 제한이 없다.
여러 카메라를 한 화면에 표시할 때는 `/ws` 영상을 사용한다.

WebRTC/H.264 경로는 2026-10-07에 제거했다. 브라우저 ICE가 `connecting`에서 진행되지 않았고, Docker bridge
ICE candidate·host network·UDP 방화벽처럼 추가로 운영해야 할 요소가 많았기 때문이다.

## WebSocket 계약

기존 HTTP listener의 `/ws`를 upgrade한다. 기본 같은 origin만 허용한다.
Nuxt dev origin을 직접 연결할 때는 `PREVIEW_WS_ORIGINS`에 정확한 origin을 쉼표로 설정하거나
dev proxy에서 `/ws` upgrade를 전달한다. Backend의 CORS 설정과 별개다.

### Client command

```json
{"version": 1, "type": "subscribe", "cameraIds": ["camera-01", "camera-02"], "video": true, "videoFps": 10}
```

- 연결 직후에는 모든 카메라의 metadata/status만 전송하고 영상은 보내지 않는다.
- `subscribe`가 metadata 대상과 영상 요청을 함께 바꾼다. 빈 배열은 구독 해제다.
- `video`(기본 `false`)가 `true`이면 `cameraIds`의 JPEG를 보낸다. `videoFps`는 카메라당 **평균** 전송률 상한이며
  기본 10, 범위 1~30이다. 원본 FPS보다 높게 보내지 않는다. 제품 프론트와 `/diagnostics`는 30을 요청한다.
- 응답은 `{"type": "subscribed", "cameraIds": [...], "video": true, "videoFps": 10}`이다. 잘못된 값이나
  지원하지 않는 command는 `{"type": "error", "code": "command_failed"}`다.

### 영상 binary message

| offset | 크기 | 내용 |
| --- | --- | --- |
| 0 | 1 | format version, 현재 `1` |
| 1 | 1 | cameraId UTF-8 byte 길이 `N` |
| 2 | N | cameraId UTF-8 |
| 2+N | 4 | frame sequence, uint32 big-endian (카메라별 증가, wrap 가능) |
| 6+N | 나머지 | JPEG (`image/jpeg`) |

서버는 카메라별 **최신 JPEG만** 보낸다. 같은 JPEG를 다시 보내지 않는다. 전송률은 token bucket으로 제한한다:
카메라별 credit이 `videoFps`장/초로 차오르고(최대 `VIDEO_BURST_FRAMES`=8), credit이 1 미만이면 건너뛴다.
따라서 카메라가 16~22 ms 간격 burst 뒤 0.5 s 쉬는 패턴이어도 burst 8장까지는 모두 전달되고 평균만 `videoFps`
이하로 유지된다(고정 `1/videoFps` 간격 규칙은 9 fps 카메라에서도 25~50%를 버렸다, 2026-10-20 측정).
영상 구독 중에는 loop 주기가 10 ms다(20 ms는 30 fps 카메라를 ~25 fps로 양자화). 느린 client는 queue가 쌓이지 않고 프레임을 잃는다. 한 프레임 write가 5초 이상 막히면 연결을
종료한다. 카메라가 offline이 되어 JPEG가 지워지면 프레임을 보내지 않으므로, 화면은 `camera_status`로
상태를 표시한다. 텍스트 응답(`subscribed` 등)은 같은 loop의 영상보다 먼저 보낸다.

브라우저에서는 `socket.binaryType = "arraybuffer"`로 받아 header를 해석하고, JPEG를
`new Blob([bytes], {type: "image/jpeg"})` → `URL.createObjectURL` → `<img src>`로 표시한다.
카메라마다 decode 중에는 새 프레임으로 덮어쓰고 한 장씩만 decode한다. 표시가 바뀐 뒤 이전 URL은
`revokeObjectURL`로 해제한다. `web/preview.js`가 참조 구현이다.

### Metadata message

envelope는 `version`, `type`, `cameraId`, `timestamp`(UTC 처리 시각), `runtimeSession`, `generation`을 사용한다.
camera 상태와 detections에는 `sourceId`도 있다. track identity는 `(cameraId, runtimeSession, generation, trackId)`다.
uint64 trackId와 나노초 PTS는 JavaScript 정밀도 보존을 위해 **10진 문자열**로 전달한다.

```json
{
  "version": 1,
  "type": "detections",
  "cameraId": "camera-01",
  "sourceId": 0,
  "runtimeSession": "process-session",
  "generation": 1,
  "timestamp": "2026-10-07T00:00:00.123+00:00",
  "frameNumber": 100,
  "ptsNs": "1234567890",
  "inferenceDone": true,
  "persons": [],
  "vehicles": [{
    "trackId": "42",
    "bbox": {"x": 0.12, "y": 0.24, "width": 0.18, "height": 0.46},
    "confidence": 0.93,
    "trackerConfidence": 0.85
  }],
  "vehicleDetectionEnabled": true,
  "vehicleInferenceDone": true
}
```

bbox는 mux 크기로 나누고 화면 경계에 clip한 0~1 값이다. JPEG와 같은 화면 비율이다. `object-fit: contain`의
letterbox가 있으면 실제 영상 사각형에 맞춘다. `timestamp`는 camera capture/NTP 시간이 아니다. metadata는
약 10 Hz로 최신 값만 반영하며 JPEG 프레임과 1:1로 동기화되지 않는다. 프레임과 정확히 맞는 bbox가 필요하면
JPEG의 서버 OSD를 사용한다.

| type | payload/역할 |
| --- | --- |
| `detections` | vehicles, vehicleDetectionEnabled/vehicleInferenceDone; `persons`는 호환용 빈 배열; 무효화 시 배열을 비우는 `stale:true` clear |
| `camera_status` | `status`에 기존 runtime 필드 포함 |
| `track_update` | 확장 producer의 `data` (현재 producer 없음) |
| `parking_status` | 확장 producer의 `data` (예: spaces 배열) |
| `event` | 확장 producer의 `data` (예: kind=line_crossing/zone, eventId) |
| `hello` / `subscribed` / `error` | connection/control 응답. error는 안전한 code만 제공 |

차량/주차 점유는 Backend poller와 `ParkingRelay`로 연결한다. `parking_status`는 현재 compact
상태이고 `event.data.kind=parking_occupancy_changed`는 DB transition이다. 초기 연결/재구독은
cached 현재 상태를 받고, socket 누락 event는 Backend cursor API로 복구한다.
[차량/점유](parking-occupancy.md)에 payload·설정·정확도/성능 한계가 있다.
Re-ID/line/zone 계산은 구현하지 않았다. 다른 producer는
`server.realtime_hub.publish_message(camera_id, type, detached_json_data)`로 확장 메시지를
발행할 수 있다. 구독별 bounded outbox로 전달하며 네트워크 I/O는 WebSocket thread가 맡는다.
모든 이벤트를 영구 보관하거나 재생하는 전달 보장은 없다. outbox가 넘치거나 metadata write가
1초 이상 막히면 connection을 종료한다. 주차 transition만 Backend DB에 기본 7일 저장하며 다른 event의 durable 저장은 별도 정책이 필요하다.

`/metadata.json`의 pixel bbox/snake_case 계약은 유지한다. `persons`는 항상 빈 배열이며 `inference_done`은 차량 추론 여부다.
차량마다 `outline`(`[[x, y], ...]` pixel 다각형 또는 null)이 있으며 의미와 한계는
[backend README의 차량 윤곽](../backend/README.md#차량-윤곽outline)이 원문이다. `/ws` detections에는 윤곽을 넣지 않는다.
`/streams.json` 항목은 `format=mjpeg`, `url=/mjpeg/sourceN`, `metadata_path=/ws`를 제공한다.
Backend `/api/cameras`의 `preview_path`는 `/mjpeg/sourceN`, `metadata_path`는 `/ws`다.

## Vue/Nuxt와 진단 페이지

업로드한 제품 build는 자동으로 바뀌지 않는다. 여러 카메라를 표시하는 화면은 mount 이후 화면당 WebSocket
하나를 열고 표시할 cameraId로 `subscribe(video: true)`를 보낸다. 화면 이동이나 보이는 카메라 변경 시
`subscribe`를 다시 보내고, unmount에서 socket을 닫고 object URL을 해제한다. 서버 재시작으로 socket이 닫히면
재접속 후 `subscribe`를 다시 보낸다. Nuxt SSR에서는 WebSocket을 `onMounted` 이후에만 만든다.
카메라 한 개만 보는 화면은 기존 `<img src="/mjpeg/sourceN">`도 쓸 수 있다.

`/diagnostics`는 제품 build가 있어도 진단 페이지를 표시한다. `/streams.json`으로 목록을 받고 WebSocket
하나로 모든 카메라를 표시하며, 카메라 상태는 `camera_status`로 갱신한다.

## 설정과 한계

| 설정 | 기본값 |
| --- | --- |
| `--jpeg-quality` | 80 |
| `PREVIEW_WS_ORIGINS` | 빈 값, 같은 origin |
| WebSocket 연결 수 | 서버 전체 16 |
| `videoFps` | client 요청, 기본 10, 최대 30 |

JPEG 1장은 해상도/장면에 따라 수십~수백 KB다. 카메라 수 × videoFps × JPEG 크기만큼 대역폭이 필요하다.
예를 들어 21대 × 10 fps × 100 KB는 약 170 Mbps다. Tailscale이나 느린 회선에서는 `videoFps`를 낮추거나
보이는 카메라만 구독한다. 실제 JPEG 크기와 서버 CPU(WebSocket write)는 운영 카메라로 측정해야 한다.

CPU 회귀와 실제 loopback WebSocket(영상 binary 포함)은 `tests/test_preview_realtime.py`,
진단 페이지 동작은 `tests/preview_web.test.cjs`로 확인한다. 실제 브라우저 렌더링, 21대 동시 표시 대역폭,
RTSP 장애/복귀와 컨테이너 재시작 후 영상 복귀는 GPU 서버에서 확인해야 한다.

색상 기반 주차는 같은 envelope에 `type:"parking.status_updated"`, `data:{spaces:[...]}`를 추가한다. 카메라 구독과 재연결 cache를 유지하며 기존 차량 기반 `parking_status`와 별개다. score/updatedAt만의 변화는 이벤트가 아니므로 최신 수치는 REST status로 조회한다. 전체 계약은 [색상 주차](color-parking.md#프론트엔드-연결)를 따른다.
