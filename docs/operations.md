# ReTrace 운영 가이드

프로젝트 소개와 화면은 [루트 README](../README.md)를 보세요. 이 문서는 설치·실행·진단 절차를 담은 운영 문서입니다(2026-10-23에 루트 README에서 옮김).

ReTrace는 주차장 CCTV RTSP 영상에서 차량을 검출·추적하고 주차 상태를 판단하는 개인 토이 프로젝트입니다.
Python과 NVIDIA DeepStream을 사용하며, GPU 실행은 Linux 서버에서 수행합니다.
MacBook에서는 코드를 작성하고 GitHub의 `main` 브랜치로 작업을 이어갑니다.

## 현재 기능

- 여러 RTSP 입력을 하나의 공유 DeepStream pipeline에서 batch 처리
- 주차장 카메라(`VEHICLE_CAMERA_IDS`)만 YOLO11m 하나로 batch 차량 검출, 다른 카메라는 추론 없이 영상만 처리
- `preview.py`: NvDCF로 카메라별 차량 추적(track ID는 metadata/점유 판단용), 영상에는 `Vehicle` Bounding Box 표시
- source별 OSD 포함 MJPEG 미리보기: 카메라별 HTTP 또는 WebSocket 하나로 전체 카메라
- WebSocket metadata/status 및 프레임/FPS/검출 통계
- 선택적 pipeline 단계별 타이밍 진단
- `app.py`: Preview가 재사용하는 RTSP source bin helper (기존 PeopleNet 전용 CLI는 제거)

카메라 runtime 상태 감지와 자동 재연결을 지원합니다. PeopleNet 사람 검출/추적은 2026-10-13에 제거했습니다(Re-ID는 구현된 적이 없음).
Backend의 기존 `person_tracks` 기록은 삭제하지 않고 조회만 가능하며 새 사람 track은 저장하지 않습니다.
차량 Track ID는 camera_id와 재연결 generation을 함께 사용해 해석합니다. 차량/점유 구조는 [차량/점유](parking-occupancy.md)를 따릅니다.

## Pipeline

```text
N개의 enabled CCTV / RTSP
  → source별 nvurisrcbin / NVIDIA NVMM
  → 공유 nvstreammux(batch=N) → 주차장 카메라 frame에만 full-frame ROI
  → YOLO11m nvinfer(secondary, batch=주차장 카메라 수) → NvDCF nvtracker
  ├─ VehicleMetadata → 최신 snapshot → /ws JSON, /metadata.json
  └─ nvstreamdemux → source별 queue → nvvideoconvert → nvdsosd → nvvideoconvert → nvjpegenc → appsink
     → 최신 JPEG 한 장 ─┬─ /mjpeg/sourceN HTTP multipart → 브라우저 <img> (카메라당 연결 1개)
                         └─ /ws binary message → 브라우저 <img> (모든 카메라 연결 1개)
```

MJPEG는 source별 최신 프레임만 보관합니다. 느린 클라이언트는 지난 프레임을 건너뜁니다.
브라우저는 같은 주소에 HTTP 연결을 6개까지만 열기 때문에 `/mjpeg/sourceN`은 한 화면에서 6대를 넘길 수 없습니다.
여러 카메라를 한 화면에 표시할 때는 `/ws` 하나로 모든 카메라 JPEG와 metadata/status를 받습니다.
기존 pixel 좌표 `/metadata.json`과 Backend polling은 유지하며 WebSocket bbox는 0~1 좌표입니다.
구조·binary 형식·Vue/Nuxt 연동·대역폭 한계는 [실시간 Preview](preview-realtime.md)를 따릅니다.

`http://<서버 IP>:40225/`는 업로드한 CSR 프론트를 제공하며 `/api/*`는 Backend API를 호출합니다.
제품 Frontend는 로컬 PC에서 빌드한 산출물을 `frontend-dist/`에 업로드합니다(`frontend/`는 소스). 서버에 Node runtime은 필요 없습니다.
[프론트 정적 배포](../frontend/README.md#정적-배포)를 따르며 `/diagnostics`에서 기본 확인 페이지를
항상 사용할 수 있습니다. 빌드(`frontend-dist/index.html`)가 없으면 `/`는 404입니다.
기존 HLS 출력물을 사용하거나 새로운 영상 파일을 저장하지 않습니다.

## 실행 환경

- Linux, NVIDIA GPU 및 호환 드라이버
- Docker Compose와 NVIDIA Container Toolkit
- DeepStream 7.0, Python 3.10, PyDS 1.1.11, PyGObject/GStreamer
- 차량 모델(YOLO11m ONNX, DeepStream-Yolo parser); TensorRT engine은 실행 환경에서 생성

이미지는 `nvcr.io/nvidia/deepstream:7.0-triton-multiarch`를 기반으로 합니다.
모델, engine, 캐시, 영상 및 로컬 설정은 Git에 포함하지 않습니다.

## 로컬 환경 설정

프로젝트 루트에서 예제 파일을 복사합니다.

```bash
cp .env.example .env
chmod 600 .env
id -u
id -g
```

`.env`에는 실제 RTSP 주소만 입력합니다. `.env.example`은 빈 변수명 예제입니다.
`configs/cameras.yaml`에서 공개 가능한 `id`, `floor`, `name`, `rtsp_env`, `enabled`를 설정합니다.
이전 변수명 `RTSP_URL` / `RTSP_URL_2`를 그대로 지원하며 새 `CCTV_*_RTSP` 이름도 사용할 수 있습니다.
사용하지 않는 카메라는 `enabled: false`로 설정합니다. 활성 카메라에 주소가 없으면 해당 id와
환경변수명을 표시하고 종료합니다. 실제 주소나 YAML/.env 원문은 오류에 출력하지 않습니다.
카메라 id와 name에는 접속 주소를 쓰지 마세요.

Python도 프로젝트 루트 `.env`를 읽습니다. 환경변수가 `.env`보다 우선하며, 비어 있는 환경변수도
우선하므로 누락 오류가 발생합니다. 파일이 없어도 필요한 환경변수가 모두 있으면 실행됩니다.
Python 로더는 `$`를 보간하지 않습니다. Compose에서도 보간을 피하려면 값을 작은따옴표로 감싸세요.
기존 `--input` 반복 입력은 설정 파일을 우회하며 `cli_0`, `cli_1` 등의 임시 camera_id를 만듭니다.
CLI에 실제 주소를 넣으면 셸 history와 프로세스 인자에 남으므로 설정 파일 실행을 권장합니다.
Compose/inspect 출력에도 비밀 값이 포함될 수 있으므로 공유하지 마세요.
`LOCAL_UID` / `LOCAL_GID`는 필요하면 셸 환경변수로 설정합니다(기본 1000).

호스트 접속 IP는 공개 파일에 기록하지 않습니다. 신규 설치는 다음 파일을 복사하고
로컬 파일에 원하는 LAN/Tailscale 바인딩을 추가하세요. `compose.override.yaml`은 Git과 Docker build에서 제외됩니다.
기존 작업 환경의 두 바인딩은 이 로컬 파일로 옮겼습니다.

```bash
cp compose.override.example.yaml compose.override.yaml
```

## Docker Compose 실행

```bash
docker compose build
docker compose up -d --no-deps retrace
docker compose ps
docker compose logs --tail=50 retrace
```

Preview API는 `http://<서버 IP>:40225/streams.json`으로 확인합니다. Compose가 자동으로
`python3 -u preview.py --diagnostics`를 실행하므로 동일 포트에 두 번째 프로세스를 띄우지 마세요.
컨테이너의 HTTP bind는 `0.0.0.0:40225`이며 호스트 공개 범위는 로컬 override의 ports로 정합니다.
Dockerfile CMD와 Compose command는 모두 `/workspace/ReTrace/preview.py`를 실행하고
`--cameras /workspace/ReTrace/configs/cameras.yaml --env-file /workspace/ReTrace/.env`를 명시합니다.
기본 서비스에 RTSP_URL/RTSP_URL_2는 필요하지 않습니다. app.py는 source bin helper만 제공하며 직접 실행하지 않습니다.
이전 컨테이너의 command/image/mount 설정은 소스 수정만으로 바뀌지 않으므로 교체 시 재생성합니다.
원문 command/inspect/Compose config에는 비밀 값이 있을 수 있으므로 공유하지 마세요.
컨테이너 실행 경로 확인은 다음처럼 비밀 값을 제거하는 도구를 사용합니다(첫 번째/두 번째 입력 순서).

```bash
sudo docker inspect retrace retrace-retrace-run-b670cfbe85e3 | python3 tools/check_container_startup.py
```

healthy one-off가 포트를 사용 중이면 새 서비스는 같은 포트를 동시에 bind할 수 없습니다.
새 이미지를 build하고 임시 `run --rm --no-deps`로 camera_config.load_cameras만 실행해 설정을
검증한 뒤, 사용자가 교체할 시점에 one-off를 중지하고 `compose up -d --force-recreate --wait`로
서비스를 실행합니다. healthy 확인 전에는 보존 가능한 이전 컨테이너를 삭제하지 마세요.
inspect 요약의 auto_remove가 true이면 중지 시 이전 one-off가 자동 삭제되므로 docker start로
되돌릴 수 없습니다. false이면 새 서비스 실패 시 새 retrace를 중지하고 이전 컨테이너를 다시 시작할 수 있습니다.

카메라 설정 또는 `.env` 변경 후 컨테이너를 재시작하세요. 포트 바인딩 변경 시 재생성이 필요합니다.
healthcheck는 `/streams.json` HTTP 응답만 확인하므로 실제 영상 수신은
각 `/mjpeg/sourceN`과 진단 로그에서 확인합니다.

```bash
docker compose exec -T retrace python3 tools/check_preview.py
```

### 입력 설정 확인

`--cameras <파일>`과 `--env-file <파일>`로 설정 경로를 지정할 수 있습니다.
기본값은 프로젝트 루트의 `configs/cameras.yaml`, `.env`입니다.
활성 카메라 순서로 연속 source_id를 부여하며 `.env` 누락 시 활성 카메라 id와 변수명을 안내합니다.

## 차량 모델 준비

PeopleNet은 제거했습니다(`configs/pgie_peoplenet.txt`, PeopleNet 로딩/추론 코드 삭제). 로컬
`models/peoplenet/` 파일과 기존 engine은 롤백 참고용으로 남아 있을 수 있으나 더 이상 읽지 않습니다.
차량 모델 config(`VEHICLE_INFER_CONFIG`, 현재 `models/yolo11m/vehicle_infer.txt`), 주차장 카메라 선택
(`VEHICLE_CAMERA_IDS`)과 적용 절차는 [차량/점유](parking-occupancy.md)를 따릅니다.
모델 파일은 자동 다운로드하지 않습니다. 실행용 config와 TensorRT engine은 `.cache/vehicle/pgie/`에 생성합니다.
캐시 키는 모델/부속 build 파일의 SHA-256, batch, precision, 입력 차원/순서,
출력 binding/format, workspace, implicit batch, GPU/DLA 및 layer precision 설정과
GPU 종류/compute capability, TensorRT 버전, DeepStream infer 라이브러리 내용으로 결정됩니다.
threshold/NMS/class filter/interval/전처리/labels 변경은 engine을 무효화하지 않으며,
generated config는 매 실행 시 최신 runtime 설정으로 다시 씁니다. engine batch는 선택된 주차장 카메라 수입니다.

DeepStream의 기본 빌더는 로드용 `model-engine-file`과 별도로 모델 경로에서 저장 파일명을 만듭니다.
`pgie_cache.py`는 차량 cache 디렉터리의 `pgie/`(`.cache/vehicle/pgie/`)에 build key별 모델 하드링크(다른 파일시스템이면 복사본)를 두고
`<모델>_bN_gpuN_<precision>.engine`을 config와 GObject 속성 양쪽에 지정합니다.
심볼릭 링크는 nvinfer의 `realpath()`가 원본 경로로 되돌리므로 사용하지 않습니다.
native precision fallback이 발생하면 같은 build key 안의 FP16/FP32 engine만 다음 실행에서 찾습니다.
대상은 nvinfer 기본 ONNX 빌더입니다(파일 stem `peoplenet_`은 기존 cache 호환용 이름). 별도 custom engine builder는
임의 config와 출력 경로를 쓸 수 있어 이 정책으로 조용히 재사용하지 않고 오류로 알립니다.
기존 모델 옆 engine과 구 digest engine은 build/runtime 호환성을 입증할 정보가 없어 자동 이관하지 않습니다.
새 정책의 첫 실행은 engine 생성 시간이 필요할 수 있고, 이후 동일 조건에서 재사용합니다.
GPU/TensorRT 환경이 다른 장비의 engine을 그대로 복사하지 않습니다.

## 미리보기와 검증

기본 미리보기는 하나의 legacy nvstreammux, batch-size=enabled 카메라 수, 1920×1080 mux,
RTSP TCP/latency 1000ms/drop-on-latency=true, NvDCF 640×384, JPEG 1280×720/quality 80을 사용합니다.
`--port`, `--jpeg-quality`, `--rtsp-latency`, `--rtsp-drop-on-latency`,
`--mux-live-source`, `--diagnostics`로 필요한 설정을 조정합니다.

통계의 `vehicles`는 주차장 카메라의 마지막 프레임 차량 수입니다(다른 카메라는 `NA`).
RTSP 오류/EOS/프레임 timeout은 해당 카메라 상태와 공개 데이터를 무효화합니다. source bin은 native 재연결을 수행하며 공유 inference를 다시 생성하지 않습니다. Ctrl+C/SIGTERM으로 전체 정상 종료할 수 있습니다.
애플리케이션 오류/진단은 입력 주소와 GStreamer 원문을 출력하지 않고 GST 디버그를 끕니다.
외부 라이브러리나 Docker 로그도 공유 전 확인하세요.

### 카메라 runtime lifecycle

`enabled: true`는 연결 대상이라는 뜻입니다. 연결 가능 여부는 YAML을 바꾸지 않고
`camera_runtime.py`의 `CameraStatus`로 관리합니다. `floor`는 정수 또는 B1/B2 같은 지하층 표기를 지원합니다.
프레임 기반 상태는 connecting → online, 프레임 공백/낮은 FPS 시 degraded,
ERROR/EOS/timeout 시 offline, 재시도 시작 시 reconnecting입니다.
기본적으로 2초 이상 프레임이 지속 수신되고 FPS가 기준을 충족해야 online으로 전환합니다.
FPS는 최근 5초의 decoded 프레임 수신율입니다. 기본 degraded 공백 3초, offline 공백 10초,
최초 프레임 대기 30초, degraded FPS 기준 1fps입니다. FPS 기준은 카메라 설정에 맞춰 조정하세요.

```text
--camera-degraded-after 3
--camera-offline-after 10
--camera-connect-timeout 30
--camera-min-fps 1
```

GPU 모델/플러그인의 동기 초기화와 대기 시간은 RTSP 연결 timeout에서 제외합니다.
연결 시도 후 JPEG 출력이 10초 이상 멈춰도 output_stalled로 해당 카메라만 offline 처리합니다.
`CameraRuntimeManager`는 source slot ↔ 공개 source_id ↔ camera_id와 상태/세대를 관리하며
GStreamer element 생성이나 state 전환을 소유하지 않습니다. 하나의 공유 bus에서 source bin
조상을 통해 오류를 해당 카메라로 라우팅합니다. tracker 이전 source EOS는 mux에 전달하지 않아
단일 source 또는 모든 source의 EOS가 공유 pipeline을 종료하지 않습니다.
GStreamer 원문 대신 rtsp_error, rtsp_timeout, source_eos, no_frames 등의 고정 오류 코드만 저장합니다.
last_error는 마지막 장애 이력으로 복구 후에도 유지됩니다. last_frame_at은 UTC 처리 시각,
last_frame_age/output_frame_age는 monotonic 시간 기반 경과 초입니다.

재연결은 DeepStream 7.0 `nvurisrcbin`의 `rtsp-reconnect-interval`(offline 기준 초, 최소 1초),
`rtsp-reconnect-attempts=-1`과 `async-handling=true`를 사용합니다. TCP, latency 1000ms/drop-on-latency=true, decoder low-latency-mode=true(디코더 보류 약 170 ms 제거),
추가 decoder surface 4개를 nvurisrcbin 속성으로 설정합니다. 내부 child 속성을 재설정하지 않으며,
child-added는 diagnostics/probe 관측 시에만 연결합니다. 지연 초과 드롭을 끄려면
`--no-rtsp-drop-on-latency`를 사용합니다. source별 decoder는 필요하지만 inference/tracker는 각각 하나입니다.
침묵한 source bin은 먼저 제자리에서 NULL → PLAYING으로 재설정하고, 그래도 돌아오지 않으면 그 source bin과
mux sink pad만 제거하고 다시 만듭니다(아래 복구 정책). mux/tracker/demux와 다른 카메라는 그대로입니다.
주소/credential은 시작 시 읽은 값을 재사용합니다. **기존 worker와 달리 실행 중 .env 변경을 재연결 시 다시 읽지 않습니다.**
`.env` 또는 enabled 카메라 목록 변경 후 컨테이너를 재생성해 환경변수와 batch-size를 함께 갱신하세요.
`--input`으로 지정한 카메라는 명시 URI를 유지합니다.

상태 관측 재시도 backoff는 1 → 2 → 5 → 10 → 30초이며 최대 30초입니다. 관측 재시도 시작 시
reconnect_count와 generation이 증가합니다. 이는 **Python health 관측 세대 수**이며 native plugin 내부의
RTSP handshake 횟수를 정확히 나타내지 않습니다. 30초 이상 안정적인 프레임 수신 후 backoff를 초기화합니다.
장애 카메라의 stale JPEG와 metadata는 지우고 FrameStore와 HTTP multipart 연결은 유지합니다.
decoded PTS에 관측 generation을 연결해 tracker/encoder에 대기 중이던 이전 세대 출력을 버립니다
(source당 최대 512개 PTS; 유효 PTS가 없는 프레임은 공개하지 않습니다).
재연결 후 새 JPEG가 들어오면 같은 브라우저 연결로 재생됩니다. 이미 브라우저에 표시된 마지막
이미지는 새 JPEG가 올 때까지 남을 수 있으므로 /streams.json의 runtime 상태를 함께 확인하세요.

source 복구 정책: 프레임을 한 장도 받지 못한 관측 시도가 2회 연속 끝나면(마지막 프레임 후 약 73초,
기동 후 한 번도 안 뜬 카메라는 PLAYING 후 약 63초) 제어 루프가 그 카메라의 `source-bin-N`만 NULL → PLAYING으로
재설정합니다(deepstream-app `reset_source_pipeline`과 같은 방식). mux sink pad, ghost pad, probe는 유지되고
nvurisrcbin이 RTSP 연결과 디코더를 다시 만듭니다. ghost pad target은 plugin이 pad-removed 없이 pad를 버린
경우에만 비웁니다. 재설정 간격은 120 → 300 → 600초(최대, 포기하지 않음)이며 30초 이상 안정 수신 후
초기화됩니다. tick당 1대만 재설정합니다. `/streams.json` runtime과 `camera_status` 로그의 `recovery_count`가
증가하고 `source_reset {camera_id, source_id, generation, recovery_count}` 한 줄을 항상 출력합니다. 재설정
자체가 실패하면 `source_reset_failed {camera_id, source_id, stage}`를 남기고 last_error=pipeline_error로 다음
간격까지 미루며 일반 관측 재시도는 계속됩니다. `--no-source-recovery`로 끄면 기존 관측 재시도만 수행합니다.

마지막 안정 수신 이후 두 번째 복구부터는 제자리 재설정 대신 **재생성**합니다(NVIDIA
`runtime_source_add_delete` 샘플 순서): `source-bin-N`을 NULL(ASYNC면 최대 5초 대기) → mux `sink_N`에
flush-stop → unlink → `release_request_pad` → pipeline에서 제거 → 같은 이름의 bin을 새로 만들어 프로브를 다시
붙이고 `sink_N`을 다시 요청해 연결 → 부모와 동기화. not-negotiated를 겪은 mux sink pad의 협상 상태가 재설정만으로는
풀리지 않는 경우(2026-11-04 운영 결과)를 위한 것이며 slot/source_id/demux `src_N`/JPEG branch는 유지됩니다.
프로세스 시작 후 프레임을 한 번도 받지 못한 카메라는 mux pad에 풀 협상 상태가 없으므로 몇 번째 복구든 재설정만
합니다(도달 불가 카메라가 10분마다 nvurisrcbin을 만들었다 지우지 않도록). 성공 시
`source_rebuild {camera_id, source_id, generation, recovery_count}`, 실패 시 `source_rebuild_failed {camera_id,
source_id, stage}`(stage는 null/create/request/link/sync)를 출력합니다. NULL 단계 실패는 bin을 그대로 두고, 그 뒤
단계의 실패는 slot을 비워 두었다가 다음 간격에 생성부터 다시 시도합니다(그동안 그 카메라만 offline).
재설정·재생성 도중 plugin 호출이 예외를 내면 `source_recovery_error {camera_id, source_id, mode}`(mode는
reset/rebuild, 원문 없음)를 남기고 실패한 복구와 같이 다음 간격으로 미루며 제어 루프와 다른 카메라는 계속됩니다.
`recovery_count`는 재설정과 재생성을 함께 셉니다. mux pad 제거가 하류로 보낼 수 있는 EOS가 JPEG branch를 영구
정지시키지 않도록 demux `src_N`에서 EOS만 DROP합니다. `--no-source-recovery-rebuild`로 끄면 두 번째 복구부터도
제자리 재설정만 반복합니다(`--no-source-recovery`면 둘 다 꺼짐). diagnostics 모드의
`pipeline_source=X: GStreamer WARNING factory=<plugin>; details hidden` 줄은 경고를 낸 element의 factory 이름
(`[a-z0-9_]{1,32}` 외는 `other`)을 함께 적어 재생성 직후 어느 공유 element가 거부하는지 확인하는 데 씁니다.
`source_rtsp` JSON에도 같은 `factory` 필드가 실립니다.

기존 `/streams.json`에 runtime 객체를 추가합니다. 호환 status는 connected/starting을 유지하고
실제 상태는 runtime.state로 확인합니다. `run(..., runtime_statuses={})`에 전달한 dict에는
source 번호별 CameraRuntime이 등록되므로 별도 소비자는 snapshot().to_dict()로 읽을 수 있습니다.
`--diagnostics`는 5초마다 카메라별 상태/FPS/프레임 age/안전한 error/reconnect_count를 요약합니다.
기존 단계별 timing과 inference/confidence 진단도 유지합니다.

공유 pipeline의 legacy mux는 live-source=1, batched-push-timeout=33333µs(40000은 partial batch를 ~25 fps로 제한해 30 fps 카메라를 깎음),
sync-inputs=false, gpu-id=0, nvbuf-memory-type=0을 사용합니다. frame이 있는 source로 partial batch를
내보내므로 offline source를 무한히 기다리지 않습니다. 모든 source가 offline이면 내보낼 frame도 없습니다.
차량 detector config와 runtime property의 batch는 선택된 주차장 카메라 수이며 interval=0과 모델 threshold는 유지합니다.
engine cache는 batch별 fingerprint 경로를 사용하므로 주차장 카메라 수가 바뀌면 첫 engine 생성 시간이 필요하며,
그동안 pipeline이 PLAYING 전이라 모든 카메라 영상이 멈춥니다.
새 mux로 교체하지 않습니다. legacy width/height/live-source 속성이 없는 plugin 선택은 setup 오류입니다.

Demux branch는 leaky queue(최대 1 frame)와 appsink sync=false/async=false를 사용해 느린 JPEG
branch의 대기와 offline branch의 preroll을 제한합니다. 출력은 차량 OSD bbox(`Vehicle`) 및 1280×720 JPEG입니다.
하나의 batch buffer를 여러 branch가 공유하므로 네이티브 JPEG/OSD plugin 자체가 멈추는 경우까지
완전한 장애 격리를 보장하지 않습니다. 공유 inference/tracker 오류는 모든 카메라를 실패 처리하고
exit-code=1로 종료합니다. source 오류/timeout만으로 전체 restart하지 않습니다. native RTSP reconnect가
멈춘 카메라는 위 복구 정책이 source bin을 재설정하며, 재설정 후에도 `recovery_count`만 늘고 online이 되지
않거나 output chain이 영구적으로 실패하면 운영자가 진단 후 retrace를 재시작해야 합니다.
GPU OOM/드라이버 오류/native crash는 source 단위로 격리되지 않습니다.
25대 처리 가능 여부는 GPU/NVDEC/JPEG/네트워크 측정 후 판단해야 합니다.

DeepStream 7.0 reconnect 지원은 [NVIDIA의 7.0 동작 확인](https://forums.developer.nvidia.com/t/rtsp-reconnect-question/293250),
source NVMM ghost pad는 [PyDS 1.1.11 test3](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/v1.1.11/apps/deepstream-test3/deepstream_test_3.py)를 참고했습니다.
[nvinfer property 우선순위](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html)와
[legacy mux partial batch 동작](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html)은 NVIDIA 문서 기준입니다.
7.0 archive 문서는 이번 환경에서 접근하지 못했으며 설치된 plugin 속성은 서버에서 확인해야 합니다.

### GPU 서버 단계별 검증

Codex의 CPU 검증은 실제 PLAYING/RTSP 복구/GPU 처리율을 증명하지 않습니다.
`configs/cameras.yaml`에서 enabled 수를 **1 → 2 → 4 → 8** 순서로 변경하며 각 단계에서 아래를 실행하세요.
25대부터 시작하지 마세요. 정상 영상 및 아래 장애 검증을 통과한 뒤 필요하면 증가시킵니다.

```bash
# 최초 한 번: DS 7.0 native reconnect property 확인 (RTSP 값 입력 불필요)
docker compose exec -T retrace gst-inspect-1.0 nvurisrcbin
# enabled/.env 변경마다 retrace만 재생성; Backend/PostgreSQL은 유지
docker compose up -d --no-deps --force-recreate retrace
# 앱의 안전한 구조/상태 로그만 확인
docker compose logs --since=10m retrace | rg 'DeepStream shared pipeline|Shared pipeline.*ERROR|Vehicle (detector|camera)|Tracker=NvDCF|camera_status '
docker compose exec -T retrace python3 tools/check_preview.py
docker compose exec -T retrace python3 tools/check_backend.py
curl -fsS http://127.0.0.1:8000/api/cameras
curl -fsS http://127.0.0.1:8000/api/status
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory --format=csv -l 2
```

각 단계에서 `sources=N streammux batch-size=N vehicle cameras=M vehicle batch-size=M ... demux branches=N`과
`state=PLAYING` 로그를 확인하세요. Backend 호스트 포트를 변경했다면 curl 포트도 맞추세요.
브라우저 `/diagnostics`에서 **모든 source**의 MJPEG와 주차장 카메라의 차량 bbox를 확인하고
`/streams.json`의 runtime online/FPS와 Backend camera status가 일치하는지 확인합니다.
GPU memory/utilization을 관측하며 첫 engine 생성과 정상 운전을 구분하세요.
2대 이상 단계에서는 한 카메라만 네트워크를 끊었다 복구해 그 카메라의 degraded/offline/reconnecting/online,
generation 증가, stale metadata/JPEG 제거, 같은 MJPEG 연결의 복귀를 확인합니다.
정상 카메라는 FPS/MJPEG를 유지하고 generation/reconnect_count가 증가하지 않아야 합니다.
시작부터 offline인 source가 있어도 정상 source가 재생되는지, 모든 source 단절 후에도 HTTP가 응답하며
복구 후 재생되는지도 확인하세요. source EOS와 장시간 단절 복구는 실제 카메라/서버 환경에서 별도 확인합니다.

### 카메라 한 대씩 RTSP 연결 진단

공유 pipeline이 PLAYING인데 일부 카메라만 offline이면 전체 batch를 재시작하며 디버깅하지 않습니다.
정상 1대와 실패 1~2대만 `tools/check_camera_rtsp.py`로 검사합니다. 도구는 YAML/.env에서 선택한
camera_id의 입력만 읽으며 다른 카메라의 credential 누락에 영향을 받지 않습니다. 환경변수가 .env보다
우선하는 preview 시작 시 해석 규칙을 따릅니다. 명시한 camera_id는 enabled와 관계없이 검사할 수 있습니다.

```bash
docker compose exec -T retrace python3 tools/check_camera_rtsp.py --camera-id 1f_outdoor_parking
docker compose exec -T retrace python3 tools/check_camera_rtsp.py --camera-id 4f_lobby --compare-to 1f_outdoor_parking
# 필요할 때 세 번째 카메라 한 대만 추가 검사
docker compose exec -T retrace python3 tools/check_camera_rtsp.py --camera-id b1_lobby
```

`--compare-to`는 기준 카메라를 먼저 검사한 뒤 대상 카메라를 순차 검사하며 한 번에 두 연결을 열지 않습니다.
기존 공유 pipeline은 계속 동작하며 재시작/설정 변경이 필요 없습니다. RTSP 서버의 세션 제한이 있다면
추가 진단 연결이 제한될 수 있으므로 도구 결과와 기존 정상 영상 수신을 함께 해석하세요.
URI/IP/계정/path 문자열은 출력하지 않으며 scheme, path 깊이, profile1/profile2 토큰, query/credential 존재
여부만 비교합니다. codec은 SDP의 고정된 encoding-name 목록으로만 표시하고, decoder 생성/하드웨어
decoder 여부 및 마지막 RTSP 요청 단계(OPTIONS/DESCRIBE/SETUP/PLAY)를 제공합니다.
단계 정보는 마지막으로 관측한 요청이며 정확한 서버 실패 응답 위치를 항상 증명하는 것은 아닙니다.

도구는 `uridecodebin → fakesink` 한 개의 독립 검사로 실제 decoded video frame 한 장을 받아야 ok입니다.
TCP/latency 1000ms/drop-on-latency=true/추가 NVIDIA decoder surface 4개/device memory 설정을 사용하며
`--rtsp-latency`와 `--timeout`(기본 15초)으로 조정합니다. inference/mux/tracker/MJPEG와 자동 재연결은 없습니다.
nvurisrcbin 자체를 검사하는 도구는 아니며 software decoder를 선택한 성공은 NVDEC 정상의 증명이 아닙니다.
URI는 child process의 stdin으로 전달하고, native stdout/stderr는 /dev/null로 보냅니다.

JPEG software decode 성공과 shared source 연결 성공은 별도로 확인합니다. 실제 preview source helper의
`nvurisrcbin` 속성 및 `source → legacy nvstreammux`
입력 경로를 재사용하는 개발용 probe는 다음과 같습니다. PGIE/tracker는 만들지 않으며 기존 서비스 설정을
바꾸지 않습니다. 기준 카메라와 대상 카메라는 순차 검사합니다.

```bash
docker compose exec -T retrace python3 tools/check_camera_rtsp.py --source-mode nvurisrcbin --camera-id 4f_lobby --compare-to 1f_outdoor_parking
```

출력의 `decoder`는 실제 decoder factory 이름의 허용 목록이며, `caps`에는 내부 decoder sink/src,
nvurisrcbin src, 내부 converter sink/src(존재 시), mux sink/src의 안전한 media/format/크기/memory만 기록합니다.
`evidence=advertised`는 query 결과이며 실제 협상 증거가 아닙니다. `current`/`negotiated` caps를 기준으로
NVMM/SYSTEM을 판단합니다. `direct_mux_accepts_caps`도 출력 pad caps의 accept-caps 조회 결과로,
실제 프레임 수신 검증을 뜻하지 않습니다. `static_link_to_streammux=ok`는 정적 연결만
뜻하고, `link_to_streammux=ok`는 source를 직접 연결한 실제 mux 출력 프레임 수신을 뜻합니다.
이 모드의 `rtsp_connection=failed`는 mux 프레임까지 도달하지 못했다는 의미이므로 `failure_stage`를 함께
읽습니다. RTSP 실패로 단정하지 않습니다.

현재 source helper는 NVMM이 아닌 video 출력의 ghost pad 연결을 거부합니다. 이 경우 mux에
도달하기 전에 `failure_stage=source_nvmm_gate`, `reason=source_caps_error`를 기록합니다. NVMM 검사는 유지하며 입력 queue/converter/capsfilter 없이 mux에 직접 연결합니다.
해상도는 legacy mux의 width/height(1920×1080)로 맞춥니다. shared 상태의 `last_error`는 Backend의 기존 `pipeline_error`를
사용하고, 세부 `error_reason`으로 caps/link/decode 오류를 구분합니다. 지원되지 않는 입력 format은 실제 협상 caps를 확인해 진단합니다. URI/IP/credential과 raw caps/error/debug 문자열은 출력하지 않습니다.
별도 pipe에는 Python이 만든 안전한 JSON만 출력하며 native setup/teardown의 hang도 제한합니다.

RTSP bus ERROR와 nvurisrcbin이 전달한 WARNING을 아래 고정 reason으로 분류합니다.

| reason | 관측 근거 |
| --- | --- |
| rtsp_auth_failed | NOT_AUTHORIZED 또는 명확한 401/403 응답 문구; 인증/접근 거부 |
| rtsp_not_found | NOT_FOUND 또는 명확한 404 응답 문구; path/profile 확인 필요 |
| rtsp_connection_failed | 연결 거부/주소 해석 실패/네트워크 경로 오류 문구 |
| rtsp_timeout | timeout 문구, RTCP timeout event, 단일 검사 frame 대기 제한 |
| decoder_error | decoder/codec 오류 enum 또는 decoder element 오류 |
| rtsp_error | 근거가 충분하지 않은 기타 오류; OPEN_READ만으로 네트워크 실패를 단정하지 않음 |

GStreamer error domain/code와 원문을 메모리에서만 검사하고 고정 domain 이름/정수 code만 출력합니다.
공유 서비스의 `/streams.json` runtime에는 `error_reason`을 추가합니다. 기존 `last_error`는 Backend의
허용 목록과 호환되는 coarse code로 유지하며 Backend/DB는 수정하지 않습니다. `error_reason`은 최근 진단
이력으로 복구 후에도 남습니다. WARNING을 관측했다고 카메라를 강제로 fail하거나 reconnect하지 않습니다.
`--diagnostics`의 `source_rtsp` 로그와 camera_status에서 확인하며, 실행 중인 기존 preview 프로세스에는
다음 계획된 재시작 후 적용됩니다. 독립 검사 도구는 즉시 사용할 수 있습니다.
[GStreamer 오류 domain/code](https://gstreamer.freedesktop.org/documentation/gstreamer/gsterror.html) 및
[rtspsrc 요청/timeout 동작](https://gstreamer.freedesktop.org/documentation/rtsp/rtspsrc.html)을 참고합니다.

### 구조화된 metadata와 탐지 진단

`camera_config.py`는 카메라 설정과 source_id 매핑을 관리하고, `person_metadata.py`는
PyDS 객체를 immutable `VehicleMetadata` / `FrameMetadata`로 복사합니다.
`preview.run(args, metadata_store=MetadataStore())`에 저장소를 전달하면 별도 스레드에서
`snapshot()`을 읽을 수 있습니다. source별 최신 프레임 한 개만 보관하며 차량이 없는 프레임은
빈 vehicles로 갱신합니다. 기존 계약 호환을 위해 `persons`는 항상 빈 배열이며 `inference_done`은 차량 추론 여부입니다. 모든 이벤트를 보존하는 큐는 아닙니다. Backend snapshot polling과 실시간 WebSocket 전송은 각각 별도로 수행합니다.
`/streams.json`은 숫자 id/runtime/session과 공개 camera_id/floor/name을 유지하며 mode별 영상 경로를 제공합니다.

`VehicleMetadata.to_dict()`는 camera_id, timestamp, track_id, class_id(모델 원래 car class), class=vehicle,
confidence, bbox, tracker_confidence를 제공합니다. timestamp는 UTC 처리 시각입니다.
track_id 미할당 및 detector/tracker confidence 미제공 값은 null로 표현합니다.
confidence는 detector 점수이며 tracker_confidence와 별개입니다.
bbox는 JPEG 1280×720 좌표가 아닌 mux 1920×1080 좌표이며 FrameMetadata에 크기를 명시합니다.
track_id는 camera_id와 generation을 함께 해석하고 재연결/재시작 후 연속성을 가정하지 마세요.
FrameMetadata.source_id는 설정 순서의 애플리케이션 source 번호입니다. 공유 pipeline의
NvDsFrameMeta.source_id/pad_index는 mux sink_N/demux src_N 슬롯 N이며 pipeline_source_id로 기록합니다.
tracker src probe(트래킹 후, demux 전)에서 batch의 각 frame을 camera_id로 매핑합니다.
공유 NvMultiObjectTracker의 ID counter는 같은 instance의 stream 전체에 걸쳐 증가하며 useUniqueID 설정에 따라
stream별 상위 32bit가 추가될 수 있습니다([NVIDIA ID 정책](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html)).
숫자만으로 카메라 간 identity를 해석하지 않습니다. Backend는 이미 camera_id/runtime_session/generation/track_id
복합 키를 사용하므로 수정하지 않습니다. HTTP /mjpeg/sourceN과 /streams.json, /metadata.json schema는 유지합니다.

`--diagnostics`는 실제 생성된 차량 detector 설정의 interval/class별 threshold와 NvDCF preset의
수치 설정을 시작 시 출력합니다.
NvDCF sample의 `%YAML:1.0`은 OpenCV 형식의 헤더로 PyYAML이 그대로 읽을 수 없습니다.
진단용 읽기에서만 첫 줄을 메모리 안에서 비우며, `nvtracker`에는 원본 파일 경로를 그대로 전달합니다.
detector/tracker 진단 파싱 실패는 `diagnostics settings unavailable`과 안전한 traceback을 출력하고
pipeline 설정을 계속합니다. 실제 모델/tracker 설정의 유효성은 DeepStream plugin이 판단합니다.
`--diagnostics`를 켜면 Python setup 예외의 파일/함수/라인과 YAML parser의 행/열도 확인할 수 있습니다.
예외 메시지·설정 내용·source code·locals·config 파일명은 traceback에 출력하지 않습니다.
source caps의 negotiated FPS를 최초 프레임에서 한 번 표시합니다. 값이 없으면 unknown입니다.
5초마다 decoded(수신 후 디코딩된 source FPS), mux-batch(프레임이 아닌 batch FPS),
demux, encoded-mjpeg FPS/PTS/gap 통계와 차량 detector 직후 주차장 카메라별 bInferDone 기반
inference frame FPS, 추론 누락 프레임 수, 차량 검출 수/confidence 최소·최대를 출력합니다.
이는 실제 파이프라인 처리율이며 GPU 커널 시간/모델 단독 benchmark는 아닙니다.
카메라의 설정된 송신 FPS 자체는 카메라 설정에서 확인해야 합니다.
tracker 이후 metadata 예제는 카메라별 5초당 최대 5대, 최신 nonempty 프레임만 출력합니다.
NvDCF의 probationAge/minDetectorConfidence/minTrackerConfidence 등도 확인해
detector 검출과 tracker 출력 사이의 차이를 진단하세요.

Metadata 필드 의미는 [NVIDIA NvDsFrameMeta 문서](https://docs.nvidia.com/metropolis/deepstream/7.1/python-api/PYTHON_API/NvDsMeta/NvDsFrameMeta.html)와
[NvDsObjectMeta 문서](https://docs.nvidia.com/metropolis/deepstream/7.1/python-api/PYTHON_API/NvDsMeta/NvDsObjectMeta.html)를 참고합니다.

```bash
# GPU 없이 설정/metadata 및 실제 CLI parsing 검사
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -p 'test_camera_*.py'
python3 -m unittest discover -s tests -p 'test_rtsp_inputs.py'
# DeepStream 서버의 전체 회귀 검사 (기존 MJPEG HTTP/element link 검사 포함)
docker compose exec -T retrace python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m unittest discover -s tests -p 'test_preview_http.py'
python3 -m unittest discover -s tests -p 'test_check_preview.py'
# 실행 중인 MJPEG 측정 (영상 저장 없음)
docker compose exec -T retrace python3 tools/measure_preview.py \
  --seconds 60 --report outputs/mjpeg_test.json
```

## GitHub main workflow

GitHub 연결 후 각 장비에서 저장소를 복제합니다.

```bash
git clone <GITHUB_REPOSITORY_URL> ReTrace
cd ReTrace
# 작업 시작
git pull --rebase
# 코드 수정 후 변경 확인
git status
git diff
# 수정한 공개 파일을 명시적으로 추가
git add <CHANGED_SOURCE_FILES>
git commit -m "<COMMIT_MESSAGE>"
git push origin main
```

다른 장비에서도 작업 시작 전에 `git pull --rebase`를 실행합니다.
장비를 옮기기 전에 commit/push하고, pull 전에는 작업 트리를 정리합니다.
충돌이 발생하면 내용을 검토하여 해결한 뒤 `git rebase --continue`를 실행합니다.
모델과 `.env`는 각 실행 장비에서 별도로 준비합니다.
로컬 테스트 자료 `hy_test/`는 보존하되 공개 저장소에서 전체 제외합니다.

Runtime 구현 참고: [GStreamer bus](https://gstreamer.freedesktop.org/documentation/gstreamer/gstbus.html), [NVIDIA runtime source 관리](https://developer.nvidia.com/blog/managing-video-streams-in-runtime-with-the-deepstream-sdk/).

## Backend + PostgreSQL

별도 FastAPI `backend` service와 `postgres` service를 추가했습니다.
기존 `retrace`는 DeepStream/MJPEG/WebSocket(40225)를 담당하며 Backend 장애와 독립적으로
동작합니다. 구성·데이터 저장 정책·API·CPU 테스트·서버 검증 명령은
[backend/README.md](../backend/README.md)를 참고하세요.
