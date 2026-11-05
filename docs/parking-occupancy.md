# 차량 탐지와 주차 점유

`VEHICLE_CAMERA_IDS`가 비어 있으면 비활성이다. 주차장 카메라 선택과 로컬 모델을 준비한 뒤 활성화한다.
실제 실행 경로는 `preview.py`이며, RTSP source·mux·공유 NvDCF·demux·기존 MJPEG/WebSocket 영상을 유지한다.
PeopleNet 사람 검출/추적은 2026-10-13에 제거했으므로 차량 detector가 pipeline의 유일한 추론 단계다.
저장소의 WebRTC 경로는 이전 작업에서 제거됐다. 알림 발송은 포함하지 않는다.

## 주차장 카메라 선택 기준 (2026-10-13)

카메라 YAML/DB에는 유형·그룹·태그 필드가 없다. 기존 기준은 프론트의 `isParkingCamera`(카메라 이름에
`주차장` 포함, `frontend/app/lib/parking.ts`)이며, 운영 DB의 주차 zone도 정확히 같은 5대에만 있다.
선택은 기존 private `.env`의 `VEHICLE_CAMERA_IDS`로 한다(새 YAML/DB 필드 없음).

| camera_id | 이름 | 판정 |
| --- | --- | --- |
| `b1_parking_entrance`, `b1_parking_overview`, `b1_parking_gate`, `b1_parking_inner`, `1f_outdoor_parking` | 이름에 `주차장`, zone 있음 | 주차장 (5대) |
| `b1_parking_tower` | 주차타워 안; id에 `parking`이 있으나 이름에 `주차장` 없음, zone 없음 | **운영자 확인 필요** — 기본 제외 |

추가·제거: YAML에서 카메라를 빼거나 `enabled: false`로 바꾸면 해당 ID는 로그(`Vehicle camera skipped`)만 남기고
건너뛰며 나머지 주차장 카메라의 차량 탐지는 유지한다. 새 주차장 카메라는 YAML 추가 후 `VEHICLE_CAMERA_IDS`에 ID를
더하고 retrace를 재생성한다. 주차장 카메라 수가 바뀌면 engine batch가 바뀌어 첫 시작에 engine을 새로 만든다.

## 모델과 pipeline

[TrafficCamNet](https://docs.nvidia.com/tao/tao-toolkit-archive/tao-30-2202/text/model_zoo/cv_models/trafficcamnet.html)은
DetectNet_v2/ResNet18 기반이며 주 용도는 car 탐지다. 현재 운영 모델은 아래 평가에 따라 YOLO11m이다.
[DashCamNet](https://docs.nvidia.com/tao/tao-toolkit-archive/tao-40/text/model_zoo/cv_models/dashcamnet.html)은
이동 카메라에서 car 탐지를 주 용도로 하므로 고정 CCTV에서는 우선 선택하지 않았다.
YOLO 계열은 현장 학습 선택지가 있지만 별도 parser/build 검증이 필요하다. 현재 adapter는
기존 native nvinfer engine cache를 재사용하므로 custom engine builder를 받지 않는다.
TrafficCamNet의 car가 모든 트럭·이륜차를 포함한다고 가정하지 않는다. 현장 평가가 필요하다.

```text
nvurisrcbin (기존 다중 입력, 카메라당 RTSP 1개) → mux(batch=전체 카메라)
  → 주차장 카메라 frame에만 전체 영상 ROI metadata 추가
  → synchronous vehicle nvinfer (secondary detector, YOLO11m 1개, batch=주차장 카메라 수)
  → ROI 부모 제거 / 차량 class 분리 → 공유 NvDCF → demux → 기존 OSD/JPEG
                                            └─ detached vehicles metadata (persons는 항상 [])
                                                 → Backend poller → 점유/이벤트 DB
```

[nvinfer의 secondary mode](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html)는
upstream object 영역을 입력으로 받는다. 이 기능은 선택 카메라에 전체 mux 영상(1920×1080)의
synthetic ROI 한 개를 만든다. 따라서 다른 카메라는 추론 없이 영상/상태만 처리되며 차량 모델 batch 크기는
선택 카메라 수다. primary detector 없이도 secondary nvinfer가 ROI만으로 동작한다(카메라별 모델 로드 없음).
ROI의 좌상단이 (0,0)이므로 차량 output bbox는 기존 전체 영상 좌표다.

- 차량 detector ID=2, synthetic ROI ID=90 (구 PeopleNet ID=1은 더 이상 생성되지 않으며 tracker probe가 다른 object를 제거).
- ROI는 untracked이며 async classifier cache를 사용하지 않는다. 추론 interval=0,
  secondary reinfer interval=0이다. `bInferDone`을 ROI 준비에서 reset하고, detector가
  빈 결과를 포함해 처리 완료 시 설정한 값을 `vehicle_inference_done`과 `inference_done`으로 복사한다
  (주차장 외 카메라는 둘 다 false).
- 차량 child metadata의 parent를 제거한 뒤 ROI를 반환한다. labels의 car/truck/bus(있는 것만)를
  모두 내부 tracker class 100으로 바꾼다. 같은 차량이 car↔truck으로
  흔들려도 track이 끊기지 않게 하나의 class로 추적한다. NvDCF의 기존 `checkClassMatch=1`을
  확인하며 tuning은 변경하지 않는다. 외부 차량 metadata의 class_id는 모델의 car class ID다
  (truck/bus도 car ID로 보고하며 TrafficCamNet=0, COCO YOLO=2).
- `VehicleMetadata`는 native 참조 없는 immutable bbox/track ID/confidence 복사다.
  source/generation/session 라우팅과 실패 시 metadata 비우기는 기존 경계를 따른다.
- 모델/config 준비 실패는 안전한 고정 메시지를 출력하고 차량 기능을 비활성화한다.
  영상/상태는 계속 제공되며 점유는 unknown이다. native 차량 plugin의 실행/협상 오류는
  기존 공유 pipeline 오류 경계를 사용하므로 GPU 서버에서 반드시 확인해야 한다.

## 활성화와 DB

private `.env`에 아래 **이름과 역할**의 설정을 추가한다. 실제 카메라 ID와 모델 config
경로는 로컬 값이다. 서버 IP/RTSP 주소를 넣을 필요 없다.

| 설정 | 역할 / 기본값 |
| --- | --- |
| `VEHICLE_CAMERA_IDS` | 차량 탐지를 적용할 주차장 camera ID의 쉼표 목록; 빈 값이면 비활성. enabled가 아닌 ID는 건너뜀 |
| `VEHICLE_INFER_CONFIG` | 로컬 native nvinfer config 경로 (운영: `models/yolo11m/vehicle_infer.txt`) |
| `VEHICLE_DEPLOYMENT_ENABLED` | `1`이면 웹 학습 모델 배포/롤백(무중단 nvinfer 교체) 허용; 기본 꺼짐. [웹 배포](training.md) §4-2 |
| `VEHICLE_MIN_CONFIDENCE` | 차량 class detector pre-cluster-threshold override; 미설정 시 config의 class/`class-attrs-all` 값(없으면 0.4) |
| `PARKING_POLICY` | Backend 판정 설정 JSON object; 기본 `{}` |

모델은 공식 배포 파일과 해당 모델용 nvinfer config/labels를 운영자가 준비한다.
labels에 `car`가 정확히 한 개 있어야 하며 `truck`/`bus`는 선택(각 최대 한 개)이다. 모델·labels·parser·calibration 경로는
원래 config 디렉터리 기준으로 해석한다. 엔진만 있는 config는 지원하지 않는다.
ONNX 또는 TAO 모델 등 native source model이 필요하다. 다운로드는 자동 실행하지 않는다.
generated config와 엔진은 `.cache/vehicle/pgie`로 분리하고 기존 cache identity를 재사용한다.

### 준비한 로컬 모델 (models/trafficcamnet/, Git 제외)

NGC 공개 모델 `nvidia/tao/trafficcamnet:pruned_onnx_v1.0.4`의 4개 파일(`labels.txt`,
`nvinfer_config.txt`, `resnet18_trafficcamnet_pruned.onnx`, `resnet18_trafficcamnet_pruned_int8.txt`)을
NGC API의 SHA-256과 대조해 내려받았다. 다른 장비에서는 같은 버전을 받아 해시를 확인한다.

```bash
curl -fL -o resnet18_trafficcamnet_pruned.onnx \
  'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet/pruned_onnx_v1.0.4/files?redirect=true&path=resnet18_trafficcamnet_pruned.onnx'
```

labels는 `car;bicycle;person;road_sign`(car=0)이다. NGC `nvinfer_config.txt`는 `[property]`가 없는
조각이므로 같은 디렉터리에 `vehicle_infer.txt`를 둔다. ONNX 출력 tensor 이름은 파일에서 확인했다.
V100은 INT8 tensor core가 없어 FP16을 쓴다.

```ini
[property]
onnx-file=resnet18_trafficcamnet_pruned.onnx
labelfile-path=labels.txt
net-scale-factor=0.00392156862745098
offsets=0.0;0.0;0.0
infer-dims=3;544;960
model-color-format=0
maintain-aspect-ratio=0
output-tensor-meta=0
num-detected-classes=4
cluster-mode=2
network-mode=2
output-blob-names=output_bbox/BiasAdd:0;output_cov/Sigmoid:0

[class-attrs-all]
topk=20
nms-iou-threshold=0.5
pre-cluster-threshold=0.4
```

`VEHICLE_INFER_CONFIG=models/trafficcamnet/vehicle_infer.txt`는 컨테이너 working_dir 기준 상대 경로다.
`prepare_vehicle_detector`로 config 생성(car=0, 나머지 class filter, batch=선택 카메라 수)은 CPU에서
확인했으나 TensorRT engine build와 실제 탐지는 DeepStream 7.0 컨테이너에서만 검증된다.
`VEHICLE_*`는 `os.environ`에서 읽으므로 `.env` 변경은 `restart`가 아닌 retrace 재생성이 필요하다.

### YOLO11m 전환 준비 (models/yolo11m/, Git 제외)

TrafficCamNet 설정(`models/trafficcamnet/vehicle_infer.txt`)은 그대로 두고 모델별 config를 분리한다.
전환/복귀는 `VEHICLE_INFER_CONFIG`만 바꾸고 retrace를 재생성한다.

- 가중치: Ultralytics `yolo11m.pt`(assets v8.3.0 release, 오프라인 평가와 동일 파일).
- Parser: [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) commit
  `2894babce8e75c49115dbe0c7b516289ed853565`(MIT)를 `models/yolo11m/DeepStream-Yolo`에 둔다.
  README는 DeepStream 7.0 = CUDA 12.2 / TensorRT 8.6을 명시한다.
- ONNX: `utils/export_yolo11.py -w yolo11m.pt --dynamic`(opset 17, 입력 `input` [B,3,640,640],
  출력 `output` [B,8400,6] = x1,y1,x2,y2,score,class). 80개 COCO `labels.txt`도 생성된다.
  Ultralytics가 설치된 별도 Python 환경에서 실행하며 retrace 이미지에는 넣지 않는다.
- `vehicle_infer.txt`: `parse-bbox-func-name=NvDsInferParseYolo`, `custom-lib-path`, letterbox
  (`maintain-aspect-ratio=1`, `symmetric-padding=1`), FP16, NMS 0.45, threshold 0.25.
  `scaling-filter`: nvinfer 기본값 `NvBufSurfTransformInter_Default`는 GPU-Nearest다. 2026-10-08 격리 검증에서
  nearest와 GPU bilinear(1) 모두 P1 탐지 10~13%로 부족했다. GPU bilinear는 1920→640 3배 축소에서 antialias를
  하지 않으며, 같은 프레임 오프라인 비교에서 antialias 축소만 P1을 안정적으로 탐지했다. 이 값은 engine build 입력이
  아니어서 변경해도 engine을 재사용한다. 다음 후보는 3(GPU-Super, 축소 전용)이며 운영 반영 전 격리 검증이 필요하다.
  `engine-create-func-name`은 쓰지 않는다. `pgie_cache`가 custom builder를 거부하므로 engine은
  nvinfer native ONNX builder로 `.cache/vehicle/pgie`에 생성한다(실제 build는 미검증).
- 차량 class: car=2, bus=5, truck=7을 tracker class 100으로 추적한다. 나머지 77개 class는 filter.

**1) Parser library build** — GPU 없이 CPU 컴파일만 하는 일회성 컨테이너다. 운영 retrace를 건드리지 않는다.

```bash
sudo docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace/ReTrace \
  -w /workspace/ReTrace/models/yolo11m/DeepStream-Yolo retrace:deepstream7.0 \
  bash -c 'export CUDA_VER=12.2 && make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo'
ls -la models/yolo11m/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
```

이미지의 `/usr/local/cuda-12.2/bin/nvcc` 존재는 미검증이다. 없으면 build가 실패하며 운영에는 영향이 없다.

**2) 단일 카메라 engine/탐지 검증(운영 유지)** — `.cache/vehicle-validation/cameras.yaml`(b1_parking_overview만,
RTSP는 .env 변수명)으로 포트를 publish하지 않는 별도 컨테이너를 실행한다. 최초 실행은 YOLO11m FP16 engine을
빌드하므로 수 분간 GPU를 크게 사용해 운영 FPS가 떨어질 수 있고, 같은 카메라에 RTSP 세션이 하나 더 열린다.
VRAM은 PeopleNet b1 + YOLO11m engine만큼 추가된다. 생성된 batch-1 engine은 같은 GPU/TensorRT에서
운영 전환 시 재사용된다.

```bash
sudo docker compose run --rm --no-deps --name retrace-yolo-check \
  -e VEHICLE_CAMERA_IDS=b1_parking_overview -e VEHICLE_INFER_CONFIG=models/yolo11m/vehicle_infer.txt \
  retrace timeout --signal=INT --kill-after=15s 900s \
  python3 -u /workspace/ReTrace/preview.py --cameras /workspace/ReTrace/.cache/vehicle-validation/cameras.yaml \
  --env-file /workspace/ReTrace/.env --diagnostics --port 40226 \
  2>&1 | grep -E "PGIE detector|Vehicle detector|ERROR|Error|engine"
# 다른 터미널: 차량 metadata 확인
sudo docker exec retrace-yolo-check python3 -c "import json,urllib.request as u; f=json.load(u.urlopen('http://127.0.0.1:40226/metadata.json'))['frames']; print([(x['camera_id'], x['vehicle_inference_done'], [(v['class_id'], round(v['confidence'], 2)) for v in x['vehicles']]) for x in f])"
```

`Vehicle detector unavailable`가 나오면 library/model 경로 문제이며 영상은 차량 탐지 없이 계속 동작한다.

**3) 운영 전환(승인 후)** — private `.env`에서 `VEHICLE_INFER_CONFIG=models/yolo11m/vehicle_infer.txt`로 바꾸고
`sudo docker compose up -d --no-deps --no-build --force-recreate retrace`를 실행한다. 전체 Preview 영상/WS가
재시작 동안 끊긴다. 복귀는 `models/trafficcamnet/vehicle_infer.txt`로 되돌리고 같은 명령을 실행한다.
Backend 판정 로직 변경은 `sudo docker compose up -d --build --no-deps backend`로 적용하며(DB schema 변경 없음,
시작 시 상태 unknown), `validated_cameras`는 실영상 검증 후 `PARKING_POLICY`에 추가하고 backend를 재생성한다.

### 모델 평가 기록 (2026-10-08, 오프라인)

운영 파이프라인과 분리해 Preview MJPEG(OSD 포함 1280×720 → 1920×1080 확대)에서 6개 주차 카메라
174프레임(주간, 20초 간격 약 10분)을 수집하고 CPU onnxruntime으로 비교했다. TrafficCamNet은 nvinfer와
같은 nearest 960×544/RGB/DetectNet_v2 decode, YOLO11은 DeepStream-Yolo와 같은 대칭 letterbox 640을 사용했다.

| 지표 | TrafficCamNet@0.4 | YOLO11s@0.25 | YOLO11m@0.25 |
| --- | --- | --- | --- |
| 정차 차량 프레임별 탐지율(overview / inner / outdoor) | 0.00 / 0.26 / 0.00 | 0.72 / 0.88 / 0.89 | 1.00 / 0.90 / 0.93 |
| V100 PyTorch FP16 batch1 median | 운영 중(미측정) | 14.1 ms | 17.1 ms |

탐지율 분모는 어느 모델이든 30% 이상 프레임에서 탐지한 객체이므로 모든 모델이 놓친 원거리/가려진 차량은
빠져 있어 recall을 과대평가한다. 정답 라벨, 야간, 이동 차량은 없었다. 육안 검토 24장에서 YOLO 오탐은 보지 못했다.
결론: TrafficCamNet은 이 현장에 부적합하고 YOLO11m이 후보다. 운영 적용 전 확인 사항:
Ultralytics 코드/가중치 AGPL-3.0, DeepStream-Yolo(MIT) parser lib을 DS 7.0 이미지(CUDA 12.2)에서 빌드,
`engine-create-func-name` 사용 시 `pgie_cache` engine 경로 호환, `vehicle_detection.py`가 `car` 한 class만
허용하므로 car/truck/bus 매핑 변경, 탐지 누락 시 empty 오판 방지 정책.
캐시 utility의 내부 파일 stem이 `peoplenet`이어도 모델 hash/경로 namespace는 분리된다.

DB 주차면 Polygon은 기존 `/api/cameras/{camera_id}/parking-spaces` CRUD로 관리한다.
Backend poller가 transaction에서 카메라별 Polygon을 조회하며 GPU probe는 DB/API를 호출하지 않는다.
`parking_events` 테이블을 추가하며 기존 테이블 ALTER는 없다. `create_all`은 새 테이블을
추가할 수 있지만 기존 schema 변경을 수행하지 않는다. Backend 시작 시 현재 상태를 unknown으로
전환하고 Polygon을 보존한다. 기존 점유는 history에서 확인할 수 있다.

Docker 권한이 있는 운영자의 적용 명령:

```bash
docker compose up -d --build --no-deps backend
docker compose restart retrace
```

Backend image는 소스를 COPY하므로 rebuild가 필요하다. retrace는 bind mount 소스를 읽으므로
모델/설정/소스 적용에는 restart가 필요하다. 기존 바인딩/네트워크/DB volume은 변경하지 않는다.

## 좌표와 판정

Polygon은 전체 영상 좌상단 기준 0~1 좌표다. 차량 tracker bbox를 `bbox_width/height`로
나누고 화면 범위로 clip한다. 화면 여백/크롭 좌표를 Polygon에 저장하지 않는다.
하단 영역을 차량의 지면 근처 영역으로 근사하고, Polygon과 이 영역의 교차 면적을 계산한다.
오목한 단순 Polygon도 지원한다.

- `overlap = 교차 면적 / 차량 하단 영역 면적`
- `space_coverage = 교차 면적 / 주차면 면적`
- 기본적으로 하단 중앙점도 Polygon 안에 있어야 한다.
- 하나의 차량은 최대 한 면에 배정한다. 거의 같은 점수의 두 면은 unknown 근거로 처리한다.
  같은 차량의 다른 면과의 유효 겹침도 그 면의 빈자리 근거로 사용하지 않는다.
- tracked car이고 detector/tracker confidence가 모두 기준 이상이어야 정차를 확인한다.
  tracker-only/null confidence나 untracked 차량의 유효 겹침은 불확실 근거다.
- 같은 session/generation/track이 속도·기준 위치 이탈 조건을 만족하며 지속적으로 관찰돼야
  occupied가 된다. 이미지 좌표의 조건이며 실제 m/s는 아니다. 통과/느린 이동과 ID 교체는
  정차 시간을 다시 시작한다.
- **탐지 0개는 빈자리 증거가 아니다.** detector가 정상 실행돼도 주차 차량을 놓칠 수 있으므로
  `validated_cameras`에 등록한(실영상 검증을 마친) 카메라만 부재로 empty를 확정한다.
  미등록 카메라의 부재는 `absence_unverified_camera` unknown이다(기본값: 모든 카메라 미등록).
- empty 확정 조건(검증 카메라): 주차면에 유효/불확실 차량 후보가 없고 연속 부재가
  `empty_seconds` 이상이며 유효 관찰이 `min_empty_observations` 이상이어야 한다.
- occupied → empty는 보수적이다. 점유 차량(같은 track)이 움직이거나 면 밖에서 추적되면
  `departure_observed`로 기록하고 위 조건으로 empty가 된다. 출차가 관찰되지 않은 채 사라지면
  `vacancy_seconds` 동안 occupied를 유지한 뒤 `occupant_lost` unknown이 되고, 부재가
  `lost_empty_seconds`(기본 60초)까지 이어져야 `prolonged_absence` empty다. 다른 track ID로의
  교체(ID switch/통과 차량)와 detector 장애 중 점유도 같은 lost 규칙을 따른다. 같은 track이
  같은 위치에 다시 나타나면 즉시 occupied로 복귀하고 새 track은 정차 시간을 다시 채워야 한다.
- confidence가 `min_confidence` 미만인 후보가 면에 있으면 empty를 보류하고 unknown이다.
  detector threshold(YOLO11m 0.25)를 Backend `min_confidence`(0.4)보다 낮춰 이 근거를 전달한다.
- 카메라/metadata 장애, detector 미완료, stale 관찰은 unknown이다. session/generation 변경,
  Polygon revision 변경 또는 큰 관찰 공백은 판정 시간을 초기화한다(같은 revision의 점유는 lost 유지).
- assessments evidence에 `camera_validated`, `absent_seconds`, `absent_observations`,
  `occupant_lost`, `departure_observed`와 후보 점수가 남는다. reason 값:
  `stationary_vehicle`, `vehicle_moving_or_confirming`, `vacancy_grace`, `occupant_lost`,
  `departure_confirming`, `departure_observed`, `confirmed_absence`, `prolonged_absence`,
  `confirming_empty`, `absence_unverified_camera`, `ambiguous_or_low_confidence`, `detector_unavailable`.
- 동일/과거 frame 재조회는 시간을 전진시키지 않는다. DB 실패 시 trial evaluator 상태도
  폐기하므로 commit하지 못한 결과가 현재 상태로 발행되지 않는다.

`PARKING_POLICY`에서 다음 필드를 바꿀 수 있다. 숫자 값은 유한한 숫자로 검증한다.

| 필드 | 기본값 | 의미 |
| --- | --- | --- |
| footprint_height / horizontal_inset | 0.25 / 0.1 | bbox 하단 높이 비율 / 좌우 제외 비율 |
| enter_overlap / exit_overlap | 0.55 / 0.35 | occupied 진입 / 유지 겹침 기준 |
| min_space_coverage | 0.1 | 주차면 최소 교차 비율 |
| min_confidence / min_tracker_confidence | 0.4 / 0.2 | 유효 차량 신뢰도 |
| stationary_seconds | 10 | 정차 확인 시간 |
| empty_seconds / vacancy_seconds | 5 / 12 | 최초 빈자리 확인 / occupied 누락 유예 |
| uncertain_seconds | 10 | 불확실 차량 근거로 occupied를 유지하는 최대 시간 |
| max_observation_gap | 5 | 관찰 사이 최대 간격(초) |
| max_speed / max_displacement | 0.015 / 0.025 | normalized 거리/초 / 정차 기준 위치 이탈 |
| ambiguity_margin | 0.05 | 두 면의 겹침 점수 모호성 기준 |
| require_anchor | true | 하단 중앙점 포함 조건 |
| validated_cameras | [] | 부재로 empty를 확정할 수 있는 검증된 camera ID 목록 |
| min_empty_observations | 3 | empty 확정에 필요한 연속 유효 부재 관찰 수(정수) |
| lost_empty_seconds | 60 | 출차 미관찰 occupied 차량이 사라진 뒤 empty까지 필요한 부재 시간 |

예: `PARKING_POLICY='{"stationary_seconds":15,"validated_cameras":["b1_parking_overview"]}'`.
카메라별 임계값 override는 현재 없으며 전역 policy를 사용한다.

이 근사는 비스듬한 CCTV에서 차량 전체 bbox보다 지면에 가까운 영역을 평가하기 위한 것이다.
Perspective 보정·차량 segmentation·깊이 추정은 아니다. 가림/카메라 이동/조명/작은 차량 때문에
차량 자체를 놓치면 검증 카메라에서도 긴 부재 후 empty로 판정될 수 있다. 실영상 없이 정확성을 보장하지 않는다.

## 조회와 실시간 계약

| GET API | 역할 |
| --- | --- |
| `/api/cameras/{camera_id}/vehicles` | 최신 차량 pixel bbox, 문자열 track ID, 선택적 `outline`([계약](../backend/README.md#차량-윤곽outline)), detector 가용성 |
| `/api/cameras/{camera_id}/parking-spaces` | Polygon과 현재 상태; stale 관찰은 unknown으로 응답 |
| `/api/cameras/{camera_id}/parking-summary` | 현재 total / occupied / empty / unknown |
| `/api/cameras/{camera_id}/parking-assessments` | revision에 맞는 최신 판정 reason, 후보 점수(최대 20개), footprint, 지속 시간, 실제 policy |
| `/api/parking` | 전체 현재 상태의 compact snapshot와 event_cursor; Polygon은 CRUD 조회에서 제공 |
| `/api/parking/events?after=0&limit=100` | 저장된 상태 변경 이력; cursor 오름차순, limit 최대 500 |

`/api/parking`의 camera 항목은 `camera_id`, `summary`, `spaces`다. space는
`space_id`, `revision`, `occupancy`, `occupancy_observed_at`이다. 현재 조회는 관찰 TTL
(기존 stale_after=10초)을 적용한다. 장애 중 DB의 마지막 결과를 확정 상태로 계속 표시하지 않는다.
자동 판정 이유는 assessments, 과거 변경 이유는 events로 확인한다.

기존 Preview `/ws`의 `detections`에 `vehicles`, `vehicleDetectionEnabled`,
`vehicleInferenceDone`을 추가했다. `persons`는 호환용으로 항상 빈 배열이며 차량 bbox는 0~1, ID는 문자열이다.
stale clear는 vehicles를 비운다.

`ParkingRelay`가 Backend를 1초마다 읽으며 GPU callback 밖에서 기존 hub에 전달한다.
poller는 기존 2초 주기로 관찰하므로 전송은 매 추론 frame의 lossless 전달이 아니다.

현재 snapshot은 PostgreSQL `REPEATABLE READ` transaction에서 cursor와 주차면을
함께 읽는다. relay는 해당 cursor까지의 이벤트만 모아 event 오름차순 → snapshot
순서로 전달한다. snapshot 조회 이후 commit된 이벤트는 다음 snapshot까지 보류한다.
한 번에 최대 4페이지(페이지당 100)를 읽고 backlog가 남으면 같은 snapshot을 유지한
채 다음 polling에서 이어 읽는다. 완료 전에는 일부 이벤트/이전 snapshot을 공개하지 않는다.
전체 batch 검증 후 hub lock 안에서 최종 cache와 큐를 함께 갱신하므로 최초 연결·
재연결·재구독이 이벤트와 snapshot 사이의 오래된 cache를 받지 않는다. 네트워크 송신은
기존 별도 socket loop가 수행한다. 실패 시 cursor를 확정하지 않고 pending snapshot을
폐기하여 fresh snapshot으로 복구한다. 기존 느린 소비자/큐 초과 시 연결 종료 정책은 유지한다.

- `type=parking_status`: `data`에 compact camera snapshot. 최초 연결/재구독 시 cache를 재전송한다.
  전체 마지막 면 삭제는 spaces=[] 및 0 counts를 보낸다. Backend 연결 실패는 cached 상태를 unknown으로 보낸다.
- `type=event`: `data.kind=parking_occupancy_changed`, `event_id` 문자열,
  camera_id/space_id/revision/previous_occupancy/occupancy/reason/observed_at/evidence.
  과거 source identity는 evidence의 runtime_session/generation/frame_number에서 확인한다.
  외부 envelope는 전달 시점의 runtime 정보다.
- DB state와 transition은 같은 transaction으로 commit한다. 같은 상태 관찰은 event를 추가하지 않는다.
  PostgreSQL transition INSERT는 transaction advisory lock으로 commit까지 직렬화하여
  cursor보다 작은 ID가 늦게 commit되는 누락을 방지한다. 실제 PostgreSQL 동작 검증은 별도로 필요하다.
  이벤트 retention은 기존 track retention 설정(기본 7일)을 사용한다. 면 삭제 후 history는 retention까지 유지한다.
- Preview relay 최초 시작은 현재 cursor에서 시작한다. 연결된 socket만 live event를 받으며,
  브라우저 재접속 중 누락은 저장한 event cursor로 API를 조회해 복구한다. 전달은 exactly-once가
  아니므로 event_id로 중복을 제거한다. outbox 크기/느린 socket 종료 정책은 기존과 같다.
  Polygon CRUD 조회와 compact snapshot을 조합하면 프론트 코드 수정 없이 서버 계약을 검증할 수 있다.
  기존 프론트가 새 메시지를 표시하도록 자동 변경되지는 않는다.

### 프론트엔드 메시지 예시

아래는 코드 계약 예시이며 운영 캡처가 아니다. `/ws`는 Preview에 연결한다.
구독 요청은 `{"version":1,"type":"subscribe","cameraIds":["parking-a"],"video":false}`다.
JPEG도 함께 받으려면 기존 계약대로 `video:true`와 선택적 `videoFps`를 사용한다.

```json
{
  "version": 1, "type": "parking_status", "cameraId": "parking-a", "sourceId": 0,
  "runtimeSession": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "generation": 1,
  "timestamp": "2026-10-08T00:00:00Z",
  "data": {
    "camera_id": "parking-a",
    "spaces": [{"space_id": "11111111-1111-4111-8111-111111111111", "revision": 1,
                "occupancy": "empty", "occupancy_observed_at": "2026-10-08T00:00:00Z"}],
    "summary": {"total": 1, "occupied": 0, "empty": 1, "unknown": 0}
  }
}
```

동일 envelope의 `type:"event"`에는 아래 `data`가 들어간다.

```json
{
  "kind": "parking_occupancy_changed", "event_id": "12", "camera_id": "parking-a",
  "space_id": "11111111-1111-4111-8111-111111111111", "revision": 2,
  "previous_occupancy": "occupied", "occupancy": "unknown",
  "reason": "polygon_changed", "observed_at": "2026-10-08T00:01:00+00:00", "evidence": {}
}
```

`GET /api/parking`는 `{"event_cursor":"12","cameras":[...]}`이고 `cameras`의 각
항목은 `parking_status.data` 형식이다. 도면이 없는 카메라는 이 목록에서 빠진다.
`GET /api/parking/events?after=12&limit=100`은 `{"cursor":"12","events":[]}`처럼
응답하며 이벤트가 있으면 위 event data에서 `kind`를 뺀 객체가 들어간다.
WS `subscribe`에는 resume cursor가 없다. 누락 history가 필요하면 REST로 조회하되
과거 이벤트를 현재 상태 위에 다시 적용하지 않고, REST/WS 현재 snapshot으로 표시를
동기화한다. 최초 relay 조회 전에는 WS cache가 없을 수 있다.

`space_id`는 서버 생성 UUID 문자열, `camera_id`는 YAML의 안정된 문자열 ID이며
숫자 sourceId와 다르다. `revision`은 Polygon 세대다. CRUD와 concurrent edit 한계는
[Backend 계약](../backend/README.md#주차면-계약과-저장), 터널/프록시는
[Mac 접근](../backend/README.md#별도-개발-pc에서-접근)을 따른다.

## 실제 영상 검증과 성능

실영상에서 차량 GT bbox와 면별 점유 라벨을 작성하고, `/metadata.json` frame을 시간 순으로
저장해 JSONL sidecar를 만든다. raw 영상/RTSP 주소를 sidecar에 넣을 필요 없다.
각 줄은 `frame`, `runtime_session`, `spaces`(space_id/camera_id/revision/polygon),
선택 필드 `expected`(space_id → occupancy), `vehicle_labels`(normalized x/y/width/height)다.
full-frame metadata를 polling에서 수집하면 짧은 track/차량을 놓칠 수 있으므로 라벨링 시 고려한다.

```bash
/tmp/retrace-shared-venv/bin/python tools/replay_parking.py /tmp/labelled-parking.jsonl --policy /tmp/parking-policy.json
```

`--policy`는 선택 사항이다. Backend CPU dependency가 설치된 Python 3.11+ 환경에서 실행한다.
출력은 점유 confusion/accuracy와 IoU 0.5 기준 차량 TP/FP/FN/precision/recall이며 라벨이 없으면
해당 정확도는 null이다. detector confidence가 유효한 차량만 precision/recall을 평가하며 NvDCF bbox를
사용하므로 이 값은 pipeline 탐지/추적 결과의 평가다. 단일 결과 수치를 실제 주차 정확도 보장으로 해석하지 않는다.

검증 영상에는 주간/야간/비스듬한 각도/인접면/출입·통과/정차/가림/장기 누락/장애·복귀를 포함한다.
assessments의 footprint/anchor/겹침 비율과 영상을 비교해 policy를 보정한다.
활성화 전후 동일 카메라 구성에서 `/streams.json` FPS, startup timing, MJPEG/JPEG 수신률,
GPU utilization/memory와 차량 stage arrival gap을 측정한다. 주차장 외 카메라 출력도 비교한다.
detector는 주차장 카메라의 crop/inference와 공유 NvDCF 부하를 만든다. latency/VRAM 변화는 실측해야 한다.

현재 CPU 테스트는 synthetic bbox·native API fake·SQLite·loopback WebSocket이다.
실제 차량 model/engine, DeepStream 7 native parent 처리·SGIE→NvDCF 협상, PostgreSQL 행 잠금,
GPU latency/정확도·브라우저 표시 검증을 대체하지 않는다.

## 주차장 전체 확대 적용 절차 (운영자, sudo 필요)

전환 전 기준값(2026-10-13, PeopleNet b21 + YOLO11m b1, 60초 평균): GPU memory 4934 MiB, util 55%,
Backend 합계 FPS 263.5(online 18). Preview 코드는 bind mount라 이미지 rebuild가 필요 없다. DB schema 변경은 없다.
각 단계는 Preview 전체 영상/WS를 잠시 끊는다. 새 batch engine 생성 단계는 수 분간 모든 카메라 영상이 멈춘다.

1. **PeopleNet 제거, 차량 1대 유지** — `.env` 변경 없음, 기존 b1 YOLO11m engine 재사용(새 build 없음).
   `sudo docker compose restart retrace`
2. **3대** — `.env` 백업 후 `VEHICLE_CAMERA_IDS=b1_parking_overview,b1_parking_gate,b1_parking_inner`,
   `sudo docker compose up -d --no-deps --no-build --force-recreate retrace` (b3 engine 생성).
3. **5대** — `VEHICLE_CAMERA_IDS=b1_parking_entrance,b1_parking_overview,b1_parking_gate,b1_parking_inner,1f_outdoor_parking`,
   같은 recreate 명령 (b5 engine 생성). `b1_parking_tower`는 운영자 확인 후에만 추가한다.

단계마다 확인(5분 이상 관찰):

```bash
sudo docker compose logs --since=10m retrace | grep -E 'DeepStream shared pipeline|Vehicle (detector|camera)|Shared pipeline.*ERROR|PGIE detector|state=PLAYING'
# 기대: "vehicle cameras=M vehicle batch-size=M", state=PLAYING, "PGIE detector=PeopleNet" 없음
curl -s http://127.0.0.1:8000/api/cameras | python3 -c "import sys,json,urllib.request as u
for c in json.load(sys.stdin):
    if c['enabled']:
        v=json.load(u.urlopen('http://127.0.0.1:8000/api/cameras/%s/vehicles' % c['camera_id']))
        print(c['camera_id'], v['vehicle_detection_enabled'], v['vehicle_inference_done'], len(v['vehicles']))"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 5   # 60초 이상, 기준값과 비교
curl -s http://127.0.0.1:8000/api/status | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['counts'], round(sum(c['fps'] for c in d['cameras']),1))"
```

선택 카메라만 `True True n`, 나머지는 `False False 0`이어야 한다. 중단 기준: 합계 FPS가 기준 대비 10% 이상
떨어지거나, 주차장 외 카메라가 degraded/offline으로 바뀌거나, `Shared pipeline ERROR`/OOM이 나오면 이전 단계 `.env`로
되돌리고 recreate한다. 전체 롤백(PeopleNet 복귀)은 Git에서 이 변경을 되돌린 소스 + `restart`다(로컬
`models/peoplenet/`와 engine은 삭제하지 않았다). Backend의 person 저장 중단/이력 보존은 Backend rebuild 시 적용되며,
현재 소스 rebuild는 보류 중인 데이터셋 기능도 함께 배포하므로 [AI 데이터셋](datasets.md)의 백업 우선 절차를 따른다.

## 변경 파일과 CPU 검증 기록

- 차량/GPU metadata: `vehicle_detection.py`, `preview.py`, `person_metadata.py`.
- 실시간 전달: `preview_socket.py`, `parking_relay.py`.
- Backend: `backend/app/db.py`, `schemas.py`, `main.py`, `ingest.py`, `settings.py`,
  `parking.py`, `occupancy.py`; Compose에는 Backend `PARKING_POLICY` 전달만 추가했다.
- 평가 도구: `tools/replay_parking.py`.
- 테스트: `backend/tests/test_backend.py`, `test_parking.py`, `test_occupancy.py`,
  `tests/test_vehicle_detection.py`, `test_parking_relay.py`, `test_camera_metadata.py`,
  `test_camera_shared_pipeline.py`, `test_preview_realtime.py`.
- 문서: 이 문서, `backend/README.md`, `docs/preview-realtime.md`.

CPU 결과: Backend 전체 103개, 카메라 회귀 65개, 차량 adapter 4개, relay 3개,
WebSocket 8개, HTTP 4개, engine cache 10개, CLI 8개, canonical startup 3개 통과.
마지막 transition 저장 경계 변경 후 관련 Backend 64개가 추가 통과했다.
Compose 실제 JSON 전달(빈 값/기본/override), service isolation, 문서 링크와
`git diff --check`를 확인했다. Starlette/httpx의 기존 deprecation warning은 남아 있다.
실제 model/GPU/PostgreSQL/브라우저 및 성능/정확도 검증은 수행하지 못했다.
Docker 상태 조회가 sandbox 밖에서도 socket permission denied로 실패했으며
컨테이너 build/restart 또는 운영 DB 변경을 하지 않았다.
