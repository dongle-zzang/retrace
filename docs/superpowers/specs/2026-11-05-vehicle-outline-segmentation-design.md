# 주차장 카메라 차량 윤곽선(instance mask) 설계 (2026-11-05)

상태: 설계 승인(2026-11-05). A′, seg를 꺼도 `outline: null` 포함, §5 기준 승인. 검토 반영 3건: 이식 출처 기록과
오래됨 경고(§4-8), 윤곽 차량 박스 두께 설정(§4-5), 행 스캔이 조각 사이를 잇는 한계를 API 계약에 명시(§4-7).
구현 계획은 [plans/2026-11-05-vehicle-outline-segmentation.md](../plans/2026-11-05-vehicle-outline-segmentation.md).
구현됨(2026-11-05): CPU 테스트 통과, GPU 검증(§7-2)과 운영 적용은 대기. 운영 절차는 `docs/operations.md` "차량 윤곽선(seg, 선택)".

구현 메모(설계와 다른 점):

- 차량 한 대의 박스 두께·글자·복사는 `preview.detach_vehicle()`로 뺐다(기존 `on_batch` AST 테스트로 seg 켬/끔을 모두 검증).
- 차량 probe는 단계 함수 목록을 받는다: det sink `add_rois`, det src `observe_deployment`(+ seg 끔이면 `prepare_tracks`),
  seg src `attach_masks → prepare_tracks`. 진단 probe 시간은 `attach_vehicle_masks` 호출과 윤곽 계산 시간이다.
- `prepare_vehicle_segmenter()`는 예외를 내지 않는다(실패 시 고정 문구를 남기고 None). `run()`은 그냥 부른다.
- OTA 빌더의 PAUSED 직렬화를 `tools/build_vehicle_engine.serialize_engine()`으로 빼서 seg 미리 빌드 도구가 쓴다.
- 패치·export는 CPU에서 확인했다: patch가 `a8ec7ec`에 적용되고, 패치한 파서가 DS 7.0·TensorRT·CUDA 12.2 헤더로 문법
  검사를 통과하며, 패치한 export가 training venv(ultralytics 8.4.176)에서 dynamic batch ONNX(`EfficientNMSX_TRT`,
  `ROIAlignX_TRT`)를 만든다. `graft_seg.py`의 결과는 설계 측정용 이식과 텐서가 같다.

## 1. 목표와 범위

목표: 주차장 카메라(`VEHICLE_CAMERA_IDS`) 영상에서 차량 박스 대신 차량 모양을 따라가는 윤곽선이 보이게 하고,
같은 윤곽을 `/api/cameras/{camera_id}/vehicles`의 차량마다 좌표로 실어 프론트가 주차면 판정
("주차면 다각형 ∩ 차량 마스크(특히 하단 접지부)")에 쓸 수 있게 한다. 배경은 박스 판정이 비스듬한 화각에서 옆 칸까지
겹쳐 오판하는 문제다(§2-5의 테스트 이미지에서도 박스가 옆 칸 주차선을 덮고 윤곽은 차체만 따라간다).

지켜야 할 것:

- 운영 중인 미세 조정 검출 모델(이하 det)의 박스·신뢰도·클래스와 OTA 배포·롤백(`vehicle_deployment.py`)을 바꾸지 않는다.
- 새 기능은 환경 변수 하나로 켜고 끈다. 끄면 pipeline, OSD, 추론, bbox가 지금과 같다(JSON에 `outline: null`만 추가된다, §4-7).
- 기존 `bbox` 필드와 의미는 바꾸지 않는다. 하위 호환을 유지한다.
- mask 처리 오류가 영상이나 검출을 멈추지 않는다(§4-4).

비목표: 프론트 주차 화면, 프론트 판정 로직, backend 서버측 점유 판정(`occupancy.py`, 계속 bbox 기반), `/ws` detections의
윤곽 전송, seg 모델의 OTA, seg 라벨 수집·seg 미세 조정.

## 2. 조사 결과

### 2-1. DeepStream 7.0 nvinfer는 secondary(ROI) 모드에서도 instance mask를 붙인다

운영 컨테이너의 DS 7.0 소스(`/opt/nvidia/deepstream/deepstream-7.0/sources`)를 읽어 확인했다.

- `gst-nvinfer/gstnvinfer.cpp`: 출력 처리에서 `IS_DETECTOR_INSTANCE || IS_INSTANCE_SEGMENTATION_INSTANCE`이면
  primary/secondary 구분 없이 `attach_metadata_detector()`를 부른다.
- `gst-nvinfer/gstnvinfer_meta_utils.cpp` `attach_metadata_detector()`: `output_instance_mask && obj.mask`이면 mask를
  `g_malloc`으로 복사해 `obj_meta->mask_params`(data/size/threshold/width/height)에 넣는다. process mode 조건이 없다.
  secondary 모드에서는 박스에 부모(ROI) 좌표를 더하고 부모의 자식으로 붙인다. 지금 ROI는 (0, 0, 1920, 1080)이라
  좌표 변화가 없다. DS 7.0은 mask가 있는 객체의 `border_color`를 무작위 색으로 바꾼다.
- `libs/nvdsinfer/nvdsinfer_context_impl.cpp` `InstanceSegmentPostprocessor::initResource()`: `cluster-mode`가 4(NONE)가
  아니면 초기화 실패. NMS는 파서나 모델 안에서 해야 한다. `output-instance-mask`는 config 키로도 읽는다
  (`gstnvinfer_property_parser.cpp`).
- `gst-nvdsosd/gstnvdsosd.c`: `display-mask`가 켜져 있고 `mask_params.data`가 있으면 객체의 **현재 `rect_params`** 위에
  mask를 그린다(`nvll_osd_draw_segment_masks`, closed source). 플러그인 코드에는 process mode 제한이 없다.

결론: 지금 구조(process-mode=2, 합성 전체 화면 ROI)를 그대로 쓰는 두 번째 nvinfer로 instance segmentation을 돌릴 수 있다.
대안 구조는 필요 없다. 실제 동작은 GPU에서 확인한다(§7-2).

Not verified:

- nvll_osd가 GPU 모드(process-mode=1)에서 mask를 그리는지, mask를 어떤 색·투명도로 칠하는지. DS 7.0 문서는
  "display-mask: Controls instance mask drawing"만 적고 있다.
- tracker가 객체 meta를 그대로 두어 `mask_params`가 OSD까지 남는지. DS 7.0 nvtracker 문서는 "the tracker plugin just
  adds/updates the data in the existing NvDsBatchMeta (and its NvDsObjectMeta)"라고 적는다. 지금 interval=0이라
  tracker 단독 프레임은 없다.

### 2-2. pyds 1.1.11

`deepstream_python_apps` v1.1.11 `bindings/src/bindnvosd.cpp`:

- `NvOSD_MaskParams.get_mask_array()`: 복사 없이 C 버퍼를 가리키는 1차원 float32 배열(길이 size/4)을 돌려준다. 쓰기도 된다.
  배열이 버퍼 수명을 붙잡지 않으므로 probe 밖으로 들고 나가면 안 된다.
- `alloc_mask_array()`: 기존 data를 `g_free`로 해제한 뒤 `width*height*4`바이트를 `g_malloc0`으로 할당하고 같은 형태의
  view를 돌려준다. nvinfer도 `g_malloc`을 쓰므로 할당·해제 방식이 같다.
- `data` 포인터를 Python에서 대입하는 방법은 쓸 수 없다(pybind11이 `float*`를 float로 바꾼다).

### 2-3. 운영 컨테이너 Python 환경

`/usr/local/lib/python3.10/dist-packages`에 numpy 1.26.1과 pyds 1.1.11이 있고 OpenCV(cv2)는 없다. 그래서 윤곽 계산은
numpy만 쓴다. 이미지 재빌드가 필요 없다(§4-6에서 cv2와 정확도가 같음을 확인했다).

### 2-4. YOLO11-seg 파서: marcoslucianops/DeepStream-Yolo-Seg

- 라이선스 MIT. 기준 commit `a8ec7eccd1e57829c95f9ed1283176787df7f09f`(2025-10-13). README가 DS 7.0을 지원 목록에 두며,
  DS 7.0 x86 빌드는 `CUDA_VER=12.2 make -C nvdsinfer_custom_impl_Yolo_seg`다. Makefile `CUDA_ARCH`에 70(V100)이 있다.
- export: `utils/export_yolo11_seg.py`(`--dynamic`, `--simplify`, `--opset` 기본 17, `--conf-threshold` 기본 0.25,
  `--iou-threshold` 0.45, `--max-detections` 100). NMS는 ONNX 안의 custom TRT plugin `EfficientNMSX_TRT`, mask는
  `ROIAlignX_TRT`이며 두 plugin이 파서 `.so`에 들어 있다.
- config: `network-type=3`, `cluster-mode=4`, `output-instance-mask=1`, `segmentation-threshold=0.5`,
  `parse-bbox-instance-mask-func-name=NvDsInferParseYoloSeg`.
- mask: 객체마다 검출 박스 영역만 RoiAlign한 netW/4 × netH/4(640 입력이면 160×160) sigmoid 확률이다. Mask R-CNN과 같이
  박스 기준이다.
- 고칠 점 2개(원문 확인):
  1. `nvdsparseseg_Yolo.cpp` `addBBoxProposal()`(30–50행)은 폭이나 높이가 1 미만이면 `classId`·`detectionConfidence`를
     설정하지 않고 돌아오는데, 호출부(71–76행)는 그 객체를 그대로 push한다. nvinfer는
     `perClassDetectionFilterParams[obj.classIndex]`를 바로 인덱싱하므로 초기화되지 않은 값으로 범위 밖 접근이 가능하다.
     같은 프로세스의 영상 21대가 함께 죽을 수 있다. → 그런 객체는 push하지 않게 고친다.
  2. 파서는 박스를 네트워크 프레임으로 clamp하지만 export(131–134행)는 clamp 전 박스로 RoiAlign한다. 화면 가장자리에서
     잘린 차량은 mask가 박스와 어긋난다. 주차장 카메라는 가장자리 차량이 흔하다. → export에서 RoiAlign 전에 같은 범위로
     clamp한다.
- ultralytics YOLO11 가중치와 코드는 AGPL-3.0이다. 지금 운영 YOLO11m과 같은 조건이다.

### 2-5. 측정: COCO seg 모델은 회사 영상에서 차량의 절반만 찾는다

조건: 회사 테스트 split 10장(GT 차량 76대, 운영 모델을 만든 학습 run의 test split, 학습에 쓰지 않음), imgsz 640,
IoU 0.45 NMS, GT와 IoU ≥ 0.5이면 일치, CPU, ultralytics 8.4.176. 차량 클래스는 car/truck/bus.

| 모델 | conf ≥ 0.25 GT 일치 | conf ≥ 0.1 GT 일치 |
| --- | --- | --- |
| YOLO11m 검출, COCO 사전 학습 | 37/76 = 0.487 | 46/76 = 0.605 |
| YOLO11m-seg, COCO 사전 학습 | 37/76 = 0.487 | 46/76 = 0.605 |
| YOLO11s-seg, COCO 사전 학습 | 37/76 = 0.487 | 42/76 = 0.553 |
| 운영 미세 조정 검출(활성 모델 `e3b72c71`) | 69/76 = 0.908 | 74/76 = 0.974 |
| **이식 seg**(아래) | **69/76 = 0.908** | **74/76 = 0.974** |

같은 test split에서 학습 평가 기록(`metrics.json`)은 COCO 기준선 mAP50 0.663 / mAP50-95 0.400, 미세 조정 모델 0.958 / 0.723이다.

핵심 발견: 공식 YOLO11m-seg(ultralytics assets v8.4.0)의 검출 가지 텐서 649개가 YOLO11m 검출 모델과 **완전히 같다**.
seg 모델에만 mask 계수 head(`model.23.cv4`)와 proto(`model.23.proto`)가 더 있다. 그래서 COCO seg의 박스 recall은 COCO
검출 기준선과 정확히 같다. 공식 표도 YOLO11m과 YOLO11m-seg의 box mAP가 51.5로 같다.

이를 이용해 운영 모델 가중치 649개를 YOLO11m-seg에 덮어쓴 "이식 seg"를 만들어 측정했다(COCO mask 가지는 그대로 둠).

- 운영 모델 박스 80개를 80개 모두 재현했다(PyTorch 출력 박스끼리 IoU > 0.99).
- 두 모델이 함께 찾은 차량 31대에서 이식 seg mask와 COCO seg mask의 IoU: 평균 0.901, 최소 0.708. 미세 조정한 특징
  위에서도 COCO mask 가지가 동작한다.
- mask가 박스 안을 채우는 비율: 중앙값 0.595, 0.15 미만 1/87대.

### 2-6. 지금 부하 (2026-11-05)

- 최근 OTA 배포 검증 기록(10:03 UTC): 주차장 카메라 FPS 1f_outdoor_parking 22.5, b1_parking_entrance 10.5,
  b1_parking_gate 9.6, b1_parking_inner 8.2, b1_parking_overview 10.6. 차량 추론 처리량 합계 60.4 frame/s, 서비스 전체
  139.6 FPS. 프레임당 차량은 평균 2.2–4.1대다.
- `nvidia-smi`: V100 16 GB 중 4212 MiB 사용, GPU 사용률 23–28%.
- 학습 guard 기본값: 시작에 여유 VRAM 6000 MiB 이상, claim 후 GPU 사용률 85% 이하(`backend/app/settings.py`).
- 공식 표(T4, TensorRT 10, 640): YOLO11m 68.1 GFLOPs 4.7 ms, YOLO11m-seg 113.3 GFLOPs 6.3 ms, YOLO11s-seg 33.1 GFLOPs
  2.9 ms. mask mAP는 m-seg 41.5, s-seg 37.8.

## 3. 접근 선택

| 안 | 내용 | 운영 det | 윤곽 커버리지(§2-5) | 추가 GPU | 판단 |
| --- | --- | --- | --- | --- | --- |
| A | COCO YOLO11m/s-seg를 두 번째 nvinfer로 | 그대로 | 약 0.49–0.60 | m +113 / s +33 GFLOPs/frame | 차량 절반에 윤곽이 없다 |
| **A′** | **이식 seg(운영 가중치 + COCO mask 가지)를 두 번째 nvinfer로** | **그대로** | **약 0.91(det와 같음)** | **+113 GFLOPs/frame** | **선택** |
| B | det를 COCO seg로 교체 | 교체 | 검출 recall 0.908 → 0.487 | +45 | 정확도 손실로 제외 |
| B′ | det를 이식 seg로 교체(nvinfer 1개) | 교체 | 약 0.91 | +45 | 후속 후보 |

B의 정확도 손실은 §2-5 수치 그대로다(recall 0.908 → 0.487, mAP50 0.958 → 0.663). seg를 회사 영상에 미세 조정하려면 차량
mask 라벨이 필요한데 데이터셋은 박스 라벨뿐이다(`docs/datasets.md`). 이식은 이 라벨 문제를 피한다.

A′를 고른 이유: 운영 det와 OTA를 건드리지 않으면서 det가 찾은 차량 대부분에 윤곽을 붙일 수 있다. 끄면 seg element가
아예 없다. A와 코드가 같아서 config의 모델 파일만 바꾸면 COCO seg(A)로도 돌릴 수 있다.

B′는 GPU 비용이 A′의 약 40%지만 운영 det의 후처리(nvinfer NMS와 config 임계값 → ONNX 안 NMS와 export 시 고정한 임계값),
OTA export·contract, 롤백 경로를 모두 바꾼다. A′를 운영해 mask 품질과 비용을 확인한 뒤 GPU가 부족할 때만 따로 설계한다.

이식의 한계: OTA로 새 det가 배포되면 이식 seg의 박스 가지는 이전 모델 기준으로 남는다. 짝짓기는 IoU로 하므로 동작은
계속되고 커버리지만 조금 떨어질 수 있다. 새 det 배포 후 이식·export·engine 준비·재시작을 수동으로 다시 한다(§4-8).
1클래스 head 모델은 텐서 모양이 달라 이식할 수 없다(스크립트가 거부한다, 운영 모델은 80클래스 vehicle3 형식).

## 4. 설계

### 4-1. 모델 준비 (호스트, 운영 영향 없음)

1. `models/yolo11m-seg/`(새 gitignore 항목)에 DeepStream-Yolo-Seg를 commit `a8ec7ec`로 clone하고 저장소의 patch
   `tools/deepstream-yolo-seg-a8ec7ec.patch`(§2-4의 2건, 각 몇 줄)를 `git apply`로 적용한다.
2. `training/graft_seg.py`(새, training venv): `--seg yolo11m-seg.pt --det <운영 best.pt> --out-dir models/yolo11m-seg`.
   det의 모든 텐서가 seg에 같은 이름·모양으로 있는지 확인하고 아니면 거부한다. 덮어쓴 `graft.pt`와 출처
   `manifest.json`(`det_weights_sha256`, `seg_weights_sha256`)을 out-dir에 쓴다. seg config도 같은 디렉터리에 두므로
   runtime은 config 옆 `manifest.json`을 읽는다(배포 디렉터리의 `infer.txt` 옆 `manifest.json`과 같은 관례, §4-8).
   운영 best.pt는 활성 모델의 배포 디렉터리에 있다(`committed.json`의 `active.config` 옆, 그 `manifest.json`의
   `weights_sha256`과 같은지 스크립트가 출력한다). `yolo11m-seg.pt`는 ultralytics가 내려받는다. runtime은 내려받지 않는다.
3. export: `export_yolo11_seg.py -w graft.pt --dynamic --simplify --conf-threshold 0.1`. seg 임계값을 det(0.25)보다 낮춰
   det가 찾은 차량을 seg가 놓치지 않게 한다. seg 객체는 짝짓기에만 쓰고 바로 지우므로 오검출은 영향이 없다.
4. config `models/yolo11m-seg/vehicle_seg_infer.txt`: DeepStream-Yolo-Seg의 `config_infer_primary_yolo11_seg.txt`를 바탕으로
   지금 det config처럼 `engine-create-func-name`은 넣지 않는다(nvinfer native ONNX builder, `pgie_cache` 규칙). custom
   plugin은 custom-lib을 dlopen할 때 등록된다(Not verified, §7-2 G3에서 확인).
5. 파서 `.so`는 운영 컨테이너 안에서 빌드한다(§7-2 G2). 기존 DeepStream-Yolo 파서와 같은 방식이다.

### 4-2. 설정과 켜고 끄기 (`vehicle_detection.py`)

- 새 환경 변수 `VEHICLE_SEG_INFER_CONFIG`(nvinfer config 경로). 비어 있거나 없으면 seg는 꺼지고 pipeline은 지금과 같다.
- 새 `prepare_vehicle_segmenter(detector, cache_dir, environ)`는 det가 켜져 있을 때만 부른다. det와 같은 카메라 집합을
  쓰고, `prepare_vehicle_detector()`와 같은 방식(상대 경로 해석, 의존 파일 확인, label에서 car 1개·truck/bus 선택, 다운로드
  없음, engine 캐시 namespace `.cache/vehicle-seg`)으로 config를 만든다. 공통 부분은 한 helper로 뺀다. det가 만드는
  config 텍스트는 바이트 단위로 같아야 하며(OTA contract) 테스트로 고정한다.
- 강제 값: `process-mode=2`, `network-type=3`, `output-instance-mask=1`, `cluster-mode=4`, `gie-unique-id=3`
  (`VEHICLE_SEG_GIE_ID`), `operate-on-gie-id=90`, `operate-on-class-ids=0`, `batch-size=주차장 카메라 수`, `interval=0`,
  `filter-out-class-ids=차량 외 클래스`, `gpu-id=0`. 나머지 secondary 값은 det와 같다. `parse-bbox-instance-mask-func-name`과
  `custom-lib-path`가 없으면 거부한다. `VEHICLE_MIN_CONFIDENCE`는 seg에 적용하지 않는다.
- 준비가 실패하면 "Vehicle segmentation unavailable; details hidden; outlines disabled"를 남기고 seg 없이 계속한다.
  차량 검출은 영향받지 않는다.

### 4-3. pipeline과 probe 배치 (`preview.py`)

```text
mux → det nvinfer(gie 2) → seg nvinfer(gie 3, 켰을 때만) → NvDCF → demux → 카메라별 OSD/JPEG
       sink: add_vehicle_rois (그대로)
       src : deployment.observe_batch (그대로) + [seg 끔] prepare_vehicle_tracks
                     src: attach_vehicle_masks → prepare_vehicle_tracks   [seg 켬]
                                                  src: on_batch → outline 계산, metadata 게시
```

- seg는 det와 같은 ROI(gie 90)를 추론하므로 ROI는 seg 뒤에서 지운다. seg를 켜면 `prepare_vehicle_tracks`가 det src에서
  seg src로 옮겨 간다. 끄면 위치·동작이 지금과 같다.
- `prepare_vehicle_tracks`는 ROI·차량 외 det 객체에 더해 seg 객체(gie 3)도 지운다. 부모(ROI)는 자식 뒤에 지운다.
- OTA 관찰(`observe_batch`)은 det src에 그대로 두어 det의 `bInferDone`만 본다. seg도 같은 프레임에 `bInferDone=TRUE`를
  쓰지만 두 nvinfer가 같은 ROI를 매 프레임 추론하므로 공개 `vehicle_inference_done`의 의미는 달라지지 않는다(seg만 도는
  프레임은 이론상만 있다).
- seg src에 `add_timing_probe(..., "vehicle-seg-batch", "all")`을 둔다(det의 `vehicle-batch`와 처리율 비교용).

### 4-4. det ↔ seg 짝짓기와 mask 이식 (`attach_vehicle_masks`, seg src probe)

주차장 카메라 프레임마다:

1. det 차량(gie 2, 차량 클래스)과 mask가 있는 seg 차량(gie 3, `mask_params.size > 0`)을 모은다.
2. IoU 내림차순 탐욕 1:1 짝짓기, 임계값 0.5. 클래스는 보지 않는다(det 클래스를 유지).
3. 짝마다:
   - seg mask를 `get_mask_array().reshape(height, width) > threshold`로 복사한다(view는 보관하지 않음).
   - det 박스를 4 mux px 격자(`ceil(w/4) × ceil(h/4)`)로 나누고, 각 칸 중심이 seg 박스 기준 mask의 어느 칸인지 최근접으로
     찾아 채운다. seg 박스 밖은 비운다.
   - 채운 비율이 0.15 미만이면 mask를 붙이지 않는다(이식 seg에서 1/87, 신뢰할 수 없는 mask).
   - 격자에서 4-이웃이 모두 차 있는 내부를 빼 테두리(ring)만 남긴다(§4-5). 테두리의 바깥 윤곽은 채운 mask와 같아 API
     윤곽은 그대로 나온다(§4-6, 프로토타입에서 확인).
   - det 객체에 `mask_params.width/height`(격자 크기), `threshold=0.5`를 쓰고 `alloc_mask_array()`가 준 view에 1.0/0.0을 쓴다.
4. 짝이 없는 det 차량은 mask 없이 지나간다(윤곽 null, OSD는 기존 굵은 박스).

오류 격리: 이 probe의 예외는 지금 `fail()`로 가서 **pipeline 전체를 멈춘다**. 그래서 짝마다 try/except로 감싸 실패하면 그
차량의 mask만 건너뛰고 오류 수를 센다. 함수 밖으로 예외를 내지 않는다.

mask는 det 박스(tracker 전) 기준으로 만들고, tracker가 바꾼 현재 박스 위에 늘여 그리고 읽는다. 정차 차량은 차이가 작다.
API 윤곽은 API bbox(현재 박스) 안에 들어가서 두 값이 서로 맞는다.

### 4-5. 영상 표시: 테두리 mask를 nvdsosd가 그린다

| 방식 | 장점 | 단점 |
| --- | --- | --- |
| `display-mask` + 채운 mask | 코드 거의 없음, GPU가 그림 | 윤곽이 아니라 면이 칠해진다. 색·투명도 Not verified, 차량 질감을 가릴 수 있다 |
| **`display-mask` + 테두리(ring) mask** | **선처럼 보임, 프레임당 Python 그리기 없음, 선 두께가 차량 크기와 무관(1칸 = 4 mux px ≈ JPEG 2.7 px)** | **GPU 모드 동작·색 Not verified** |
| display meta 선(line) | GPU 모드에서 그려짐이 확실 | 차량당 최대 32선, display meta 하나에 16개 제한, 프레임마다 Python에서 pyds 속성 약 200회 쓰기 |

선택: 테두리 mask + `display-mask`. ring 계산은 numpy 세 줄이고 그리기는 GPU 몫이다.

- `display-mask`는 seg를 켰을 때 주차장 카메라의 OSD branch에서만 켠다. 나머지와 seg를 끈 경우는 지금처럼 `False`다.
- 윤곽이 붙은 차량의 박스 두께는 새 환경 변수 `VEHICLE_OUTLINE_BOX_WIDTH`(정수 0–3, 기본 1)로 정한다. 0이면 박스를
  숨기고 `Vehicle` 글자는 남긴다. 범위 밖이나 정수가 아니면 1을 쓰고 한 줄 남긴다(표시용 설정이라 seg를 끄지 않는다).
  윤곽이 없는 차량은 지금처럼 3이다(굵은 박스 = 윤곽 없음). 주황 테두리 색은 그대로 둔다. 2-1에 따르면 mask 색은 객체
  border 색을 따르는 것으로 보인다(Not verified). G6 확인 후 기본값을 정한다.
- 두께 0과 글자(DS 7.0 `gstnvdsosd.c` 확인): 객체마다 `rect_params`는 박스 목록에, `text_params`는 글자 목록에 따로
  복사되고, 글자는 `text_params.x_offset/y_offset`(nvinfer가 검출 시 박스 왼쪽 위 −10px로 설정)에 그려진다. 박스 두께는
  글자 위치에 쓰이지 않는다. mask도 `rect_params`의 위치·크기만 쓴다. 두께 0인 사각형을 `nvll_osd_draw_rectangles`
  (closed source)가 정말 그리지 않는지는 Not verified다(G6). GPU 모드는 테두리 투명도를 지원하지 않아(DS 7.0 문서)
  투명색으로 숨기는 방법은 쓰지 않는다.
- 대비책: GPU 확인(§7-2 G6a)에서 GPU 모드가 mask를 그리지 않거나 색을 쓸 수 없으면, 같은 윤곽 다각형으로 display meta
  선을 그리는 방식으로 바꾼다. 이때만 프레임당 그리기 비용을 다시 잰다. OSD를 CPU 모드로 바꾸는 방법은 1920×1080
  RGBA를 CPU로 그려야 해서 쓰지 않는다.

### 4-6. 윤곽 계산 (`on_batch`, tracker src)

seg를 켰고 객체에 `mask_params.size > 0`이면 mask 격자를 복사해 윤곽을 만든다.

- 알고리즘(numpy, 행 스캔): mask가 있는 행 중 16행을 고르게 고르고(처음·마지막 행 포함), 행마다 가장 왼쪽·오른쪽
  칸의 바깥 경계를 잡는다. 오른쪽 변을 위→아래, 왼쪽 변을 아래→위로 이어 최대 32점을 만든다. 첫 행은 위 경계, 마지막
  행은 아래 경계 y를 쓴다. 칸 좌표를 객체의 현재 `rect_params`로 옮겨 소수 첫째 자리로 반올림한다.
- 정확도(이식 seg, 회사 테스트 이미지 86대, 윤곽 다각형과 원래 mask의 IoU):

  | 방식 | 평균 | 하위 10% | 최소 |
  | --- | --- | --- | --- |
  | 행 스캔 16행(선택, 최대 32점) | 0.836 | 0.690 | 0.452 |
  | 행 스캔 전체 행(점 수 제한 없음) | 0.839 | 0.706 | 0.444 |
  | cv2 최대 외곽선 + approxPolyDP 32점 | 0.829 | 0.573 | 0.469 |

  행 수를 줄여도 손실이 없고 cv2와 같은 수준이다. 남는 차이는 기둥·다른 차에 가려 mask가 여러 조각인 경우다(가장 큰
  조각이 mask의 68% 이하인 차량이 10%). 행 스캔은 같은 행의 조각 사이를 잇는다. 접지부(바퀴 사이) 판정에는 오히려 맞다.
  다각형 하나로는 여러 조각을 표현할 수 없다는 한계는 문서에 적는다.
- 비용(호스트 Python 3.11, 차량 1대·프레임 1장): mask 이식 약 170 µs + 윤곽 약 100 µs. 컨테이너 Python 3.10은 약 30%
  느리게 잡는다. 평균 4대 × 60 frame/s면 초당 65–85 ms(한 코어의 7–9%)이고 두 streaming thread에 나뉜다. 차량 20대
  프레임은 batch당 약 6 ms다. 허용 기준과 측정은 §5, 넘으면 §6의 대비책을 쓴다.
- 윤곽 계산 오류는 그 차량의 `outline=None`으로 끝낸다(`on_batch` 예외는 카메라 실패가 되므로).

### 4-7. 데이터와 API 계약

- `person_metadata.VehicleMetadata`에 `outline: tuple[tuple[float, float], ...] | None = None`을 추가한다.
  `vehicle_from_object(..., outline=None)`. seg를 끄면 mask를 읽지 않고 항상 None이다.
- `/metadata.json`(preview → backend, snake_case pixel): 차량마다 `"outline": [[x, y], ...]` 또는 `null`.
- backend `schemas.VehicleIn.outline`: 유한한 `[x, y]` 3–32개 또는 null, 기본 null. 예전 preview(필드 없음)도 그대로
  받는다. 잘못된 값(비유한 수, 2점 이하, 33점 이상, 형식 오류)은 그 차량의 `outline`만 null로 만든다. poller는
  `/metadata.json`의 frame 하나라도 검증에 실패하면 모든 카메라 frame을 버리므로(`ingest.py` `poll_once`), frame 거부로
  처리하면 윤곽 하나의 오류가 전체 차량 표시와 점유 판정을 지운다(계획 작성 중 확인, 2026-11-05).
- `GET /api/cameras/{camera_id}/vehicles`: `model_dump()`를 그대로 내보내므로 차량마다 `outline`이 생긴다. 다른 필드는
  그대로다.

```json
{"camera_id": "b1_parking_overview", "bbox_width": 1920, "bbox_height": 1080,
 "vehicles": [{"track_id": "42", "class_id": 2, "class": "vehicle", "confidence": 0.91,
   "tracker_confidence": 0.8, "bbox": {"x": 812.0, "y": 430.0, "width": 300.0, "height": 210.0},
   "outline": [[1050.0, 430.0], [1104.0, 520.6], [1092.5, 640.0], [828.0, 640.0], [812.0, 541.2], [866.4, 430.0]]}]}
```

(예시는 6점이다. 실제로는 최대 32점.)

- 좌표계: bbox와 같은 mux pixel(`bbox_width × bbox_height`, 지금 1920×1080). 원점 왼쪽 위, y 아래.
- 닫힌 다각형: 마지막 점이 첫 점으로 이어진다. 첫 점을 반복하지 않는다. 화면 좌표 기준 시계 방향, 최대 32점.
- `null`: seg 꺼짐, seg가 그 차량을 찾지 못함, mask 신뢰도 낮음, 계산 오류. 프론트는 지금 bbox 판정으로 대체한다.
- `/ws` detections에는 넣지 않는다(요청 범위 밖, 나중에 추가해도 하위 호환).

알려진 한계(계약 문서에도 그대로 적는다):

- 윤곽은 mask에서 만든 다각형 하나이고, mask가 있는 행마다 가장 왼쪽과 가장 오른쪽 경계를 잇는다. 같은 행에서
  mask가 여러 조각이면 그 사이를 메운다. 기둥·다른 차·사람에 가려 차량 mask가 끊긴 경우, 윤곽이 가린 물체를
  가로질러 그 영역까지 포함한다. 테스트 이미지에서 차량 10%는 가장 큰 조각이 mask의 68% 이하였다(§4-6).
- 접지부(윤곽 하단) 판정에 쓸 때: 바퀴 사이의 빈 곳은 메워져 차량 바닥 면적으로 보인다(의도한 동작). 반대로 차량
  아래쪽이 앞 물체에 가려지면 윤곽 하단은 가린 물체를 포함하거나, mask가 가려진 부분에서 끝나 실제 접지선보다 위에
  있을 수 있다. 하단 경계는 mask가 있는 가장 아래 행이며, 가려진 부분을 추정하지 않는다.
- 윤곽은 tracker 박스(API `bbox`) 위에 늘인 것이라 항상 `bbox` 안에 있다. seg 박스와 det 박스가 다르면 그만큼 늘이거나
  줄여진다(정차 차량은 차이가 작다).
- 행을 16개만 고르므로 행 사이의 오목한 모양은 직선으로 이어진다(측정상 손실 없음, §4-6).

### 4-8. OTA 배포·롤백과의 관계

- seg는 별도 element·config·engine 캐시다. `DeploymentController`는 det element만 다루고 seg를 모른다. 롤백·`model-updated`
  흐름은 그대로다.
- OTA의 FPS 검증은 기준선과 후보를 같은 seg 부하에서 재므로 영향이 없다.
- seg config 교체는 OTA가 아니라 재시작으로 한다. 새 det 배포 후에는 §4-1을 새 best.pt로 다시 하고 §7-2 G3 → 재시작.
- `VEHICLE_SEG_INFER_CONFIG`를 비우고 재생성하면 seg가 빠진다.

이식 출처와 오래됨 경고(자동 재이식은 범위 밖):

- 기록: `graft_seg.py`가 seg config 디렉터리의 `manifest.json`에 `det_weights_sha256`(이식에 쓴 det best.pt의 sha256)을
  쓴다. `prepare_vehicle_segmenter`가 이 값을 읽어 segmenter에 둔다. 파일이 없거나 값이 없으면 None이다.
- 비교 대상(활성 det의 가중치 sha256): OTA가 켜져 있으면 `DeploymentController.active['config']` 옆 `manifest.json`의
  `weights_sha256`(배포 디렉터리에 이미 있음), 꺼져 있으면 det config 옆 `manifest.json`. 처음 모델
  (`production-original`, `models/yolo11m`)은 이 파일이 없어 비교할 수 없다.
- 상태 `graft`: `match`(같음) / `stale`(다름) / `unknown`(어느 한쪽 sha256이 없음). seg를 끄면 상태가 없다(None).
  계산은 `vehicle_detection.graft_status(segmenter, det_config_path)` 한 곳에서 하고, manifest 파일을 읽는 것뿐이다.
- 노출:
  1. 시작 로그 한 줄 `vehicle_segmentation graft=<상태>`. `stale`이면 `WARNING`과 재이식 안내 문구를 붙인다.
  2. `--diagnostics` 5초 `vehicle_masks` 줄의 `"graft"` 필드(OTA로 활성 det가 바뀌면 다음 보고부터 반영).
  3. `DeploymentController`가 쓰는 `status.json`과 `PUBLIC` 키에 `segmentation_graft`(상태 문자열 또는 null)를 더한다.
     호스트 worker heartbeat가 이 dict를 그대로 backend로 보내고 `GET /api/training/deployment`가 그대로 응답하므로
     backend 코드는 바꾸지 않는다. worker는 `vehicle_deployment` 모듈을 한 번 import해 두므로 반영하려면
     `systemctl --user restart retrace-training-worker`가 필요하다. OTA가 꺼져 있으면 `status.json`이 없어 API에는
     나오지 않는다(로그와 진단에는 나온다).

### 4-9. 진단 (`--diagnostics`, 기존 5초 보고)

5초마다 한 줄: `vehicle_masks {"cameras": {"<id>": {"vehicles": n, "with_outline": k}}, "probe_ms_per_s": x,
"probe_max_ms": y, "errors": e, "graft": "match|stale|unknown"}`. `probe_*`는 `attach_vehicle_masks` 전체와 윤곽 계산 시간을 `time.perf_counter`로 합친
값이다. seg를 끄면 이 줄은 없다.

### 4-10. engine 미리 빌드 (`tools/build_vehicle_seg_engine.py`, 새)

첫 재시작에서 seg engine을 빌드하면 그동안 영상 21대가 모두 멈춘다. `tools/build_vehicle_engine.py`처럼 별도 프로세스에서
같은 `prepare_vehicle_segmenter()`로 config를 만들고 nvinfer element 하나를 PAUSED까지 올려 engine을 만든다. 실행 중인
pipeline은 건드리지 않는다. 이 단계가 config 해석, 파서 `.so`와 plugin 로드, engine 생성까지 확인한다.

## 5. 성능: 측정 방법과 허용 기준

seg 끔(기준선)과 seg 켬을 비슷한 시간대에 각각 30분 잰다. 같은 카메라가 online이어야 한다.

| 항목 | 방법 | 허용 기준 |
| --- | --- | --- |
| 카메라별 FPS | `--diagnostics` 5초 `fps=` 로그 중앙값, `tools/measure_preview.py --seconds 600 --report ...` | 서비스 합계 ≥ 기준선의 95%, 카메라별 ≥ 90% |
| 차량 추론 처리율 | `timing ... stage=vehicle-batch`와 `vehicle-seg-batch` total 증가율 | det ≥ 기준선의 95%, seg ≥ det의 98%(밀리지 않음) |
| VRAM | 호스트 `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader` 1초 간격 | 증가 ≤ 1.5 GiB, 여유 ≥ 8 GiB |
| GPU 사용률 | 같은 명령 | 중앙값 ≤ 60%(학습 guard 85%에 여유) |
| probe 비용 | §4-9 `vehicle_masks` | `probe_ms_per_s` 중앙값 ≤ 50, `probe_max_ms` ≤ 10인 보고가 95% 이상 |
| 프로세스 CPU | 호스트 `top -b -d 5 -n 120 -p <preview python3 PID>` | 증가 ≤ 한 코어의 15%p |
| 윤곽 커버리지 | §4-9 `with_outline / vehicles` | 주차장 카메라 합계 ≥ 0.8 |
| 메모리 누수 | 24시간 RSS와 VRAM 추이 | 단조 증가 없음 |

## 6. 위험과 대응

| 위험 | 대응 |
| --- | --- |
| 파서가 1px 미만 박스로 nvinfer 범위 밖 접근 | patch 1(§2-4). patch 적용을 G1에서 `git diff`로 확인 |
| 가장자리 차량 mask 어긋남 | patch 2. GPU에서 가장자리 차량을 눈으로 확인 |
| GPU 모드에서 mask를 안 그림 / 색 문제 | §4-5 대비책(display meta 선) |
| tracker 뒤에 `mask_params`가 사라짐 | G6b에서 윤곽이 null만 나오면 확인. 그때는 seg src에서 윤곽을 계산하고 `detector_bbox_info`로 tracker 뒤 객체와 맞추는 방식으로 바꾼다 |
| mask 처리 예외가 pipeline 전체를 멈춤 | §4-4 차량 단위 try/except, 예외를 밖으로 내지 않음. CPU 테스트로 고정 |
| GPU 부하 초과 | §5로 먼저 잰다. 기준을 넘으면 B′ 설계와 COCO s-seg(커버리지 약 0.5) 중 사용자가 고른다 |
| probe 비용 초과 | 윤곽을 track별로 캐시해 0.5초에 한 번만 다시 계산(API는 2초 주기로 읽음) |
| OTA 후 이식 seg가 오래됨 | `graft=stale`을 시작 로그·진단·`/api/training/deployment`에 노출(§4-8), 커버리지 진단. 재이식은 수동 |
| seg engine 빌드로 재시작 지연 | §4-10 미리 빌드 |
| 이식 seg mask 품질이 COCO seg보다 낮은 차량 | 측정 평균 IoU 0.90. GPU에서 눈으로 확인, 필요하면 config 모델만 COCO seg로 바꿔 비교 |

## 7. 검증

### 7-1. CPU 테스트 (구현 전에 먼저 작성, `/tmp/retrace-shared-venv` 등 numpy가 있는 Python)

- `tests/test_vehicle_detection.py`
  - seg 끔: env가 없으면 `prepare_vehicle_segmenter`가 None, det 생성 config 텍스트가 지금과 같다.
  - seg config 강제 값(§4-2), 의존 파일 누락·함수 이름 누락 거부, 별도 캐시 namespace.
  - 윤곽: 평행사변형·둥근 지붕 합성 mask에서 점 ≤ 32, 모든 점이 박스 안, 다각형과 mask IoU ≥ 0.9, 시계 방향, 첫 점 반복
    없음. 빈 mask·한 행 mask는 None. 테두리 mask와 채운 mask의 윤곽이 같다.
  - `attach_vehicle_masks`(fake pyds, `get_mask_array`/`alloc_mask_array` 흉내): IoU 짝짓기, 1:1, 임계값 미만은 짝 없음,
    채움 0.15 미만 건너뜀, 격자 크기, 예외가 난 짝만 건너뛰고 나머지는 처리, 함수가 예외를 내지 않음.
  - `prepare_vehicle_tracks`가 seg 객체와 ROI를 지우고 ROI를 마지막에 지움.
  - `vehicle_from_object(..., outline)`와 `FrameMetadata.to_dict()`가 `[[x, y], ...]`/`null`로 직렬화.
  - `VEHICLE_OUTLINE_BOX_WIDTH`: 0–3 그대로, 없으면 1, 범위 밖·정수 아님은 1.
  - 이식 출처: config 옆 `manifest.json`의 `det_weights_sha256`을 읽음(없으면 None). `graft_status`가 match/stale/unknown,
    seg 끔이면 None.
- `tests/test_vehicle_deployment.py`: `status.json`과 `public_status()`에 `segmentation_graft`, OTA commit으로 활성 모델이
  바뀌면 `match` → `stale`.
- `tests/test_camera_shared_pipeline.py`(fake Gst)
  - seg 끔: element 순서·probe 위치가 지금과 같고 모든 OSD `display-mask=False`.
  - seg 켬: `vehicle-inference → vehicle-segmentation → vehicle-tracker` 연결, `prepare_vehicle_tracks`가 seg src로 이동,
    `observe_batch`는 det src에 남음, 주차장 카메라 OSD만 `display-mask=True`.
  - seg 준비 실패: 경고를 남기고 seg 없이 시작.
- `backend/tests/test_backend.py`: vehicles API가 `outline`을 그대로 내보냄, 필드 없는 frame은 null, 비유한 값·33점·2점은
  거부. `backend/tests/test_preview_contract.py`의 `/metadata.json` 계약에 outline 포함.
- `training/tests/test_graft_seg.py`: 작은 torch state dict로 이식 복사, 모양이 다른 텐서가 있으면 거부, manifest에
  두 sha256 기록.
- 회귀: `test_camera_*.py`, `test_vehicle_deployment.py`, `test_preview_realtime.py`, `test_pgie_cache.py`,
  `python -m pytest -q backend/tests`.

### 7-2. GPU 서버 검증 (사용자 실행, 결과는 `.ai/WORK.md`에 기록)

재시작 전(운영 영향은 engine 빌드 동안의 GPU 사용뿐):

- G1 호스트에서 모델 준비(§4-1): clone, `git apply`, 이식, export. `git -C models/yolo11m-seg/DeepStream-Yolo-Seg diff`로
  patch 확인.
- G2 파서 빌드:
  `sudo docker compose exec -T retrace bash -lc 'cd models/yolo11m-seg/DeepStream-Yolo-Seg && CUDA_VER=12.2 make -C nvdsinfer_custom_impl_Yolo_seg'`
- G3 engine 미리 빌드:
  `sudo docker compose exec -T -e VEHICLE_SEG_INFER_CONFIG=models/yolo11m-seg/vehicle_seg_infer.txt retrace python3 tools/build_vehicle_seg_engine.py`

재시작(사용자가 시간 결정):

- G4 seg 끔 상태로 §5 기준선 30분.
- G5 `.env`에 `VEHICLE_SEG_INFER_CONFIG=models/yolo11m-seg/vehicle_seg_infer.txt`를 넣고
  `sudo docker compose up -d --force-recreate retrace`(env_file은 재생성해야 반영). 로그에서 seg engine을 다시 빌드하지 않고
  deserialize했는지 확인.
- G6 확인: (a) 주차장 카메라 JPEG에 윤곽선이 보이고 윤곽 있는 차량의 박스가 얇은지(display-mask GPU 모드, 색).
  `VEHICLE_OUTLINE_BOX_WIDTH=0`으로 한 번 재생성해 박스가 사라지고 `Vehicle` 글자가 같은 자리에 남는지 확인
  (a′) 시작 로그 `vehicle_segmentation graft=match`, `systemctl --user restart retrace-training-worker` 뒤
  `/api/training/deployment`의 `segmentation_graft`
  (b) `curl -s http://127.0.0.1:8000/api/cameras/b1_parking_overview/vehicles`에 `outline` 배열 (c) 윤곽이 JPEG의 차량과
  맞는지, 가장자리 차량 포함 (d) `vehicle_masks` 커버리지·probe 비용 (e) §5 30분 측정 (f) 24시간 메모리.
- 되돌리기: `.env`의 `VEHICLE_SEG_INFER_CONFIG`를 비우고 `sudo docker compose up -d --force-recreate retrace`.

## 8. 문서 갱신

- `docs/preview-realtime.md`: pipeline 그림에 seg, OSD 설명(윤곽선과 얇은 박스), `/metadata.json` `outline`.
- `backend/README.md` API와 `docs/parking-occupancy.md` 조회 표: vehicles 응답의 `outline` 계약(§4-7).
- `docs/agent/architecture.md`: `vehicle_detection.py` seg 책임, 진입점 설명.
- `docs/operations.md`: `VEHICLE_SEG_INFER_CONFIG`, `VEHICLE_OUTLINE_BOX_WIDTH`, 모델 준비·되돌리기 요약.
  `training/README.md`: 이식·export 절차. `docs/training.md`: 배포 상태의 `segmentation_graft`.
- `docs/agent/testing.md`: 새 테스트 범위. `docs/project-status.md` §2 결정 표와 §5 검증 항목. `.gitignore`: `models/yolo11m-seg/`.

## 9. 구현 순서 (테스트 먼저)

1. CPU 테스트 작성 → 실패 확인(§7-1).
2. `vehicle_detection.py`: 공통 config helper 추출(기존 det 테스트 통과 확인) → `prepare_vehicle_segmenter` → 윤곽·이식 함수.
3. `person_metadata.py` outline 필드 → backend `schemas.py`.
4. `preview.py`: seg element·probe 이동·`display-mask`·박스 두께·윤곽 게시·진단.
5. `tools/build_vehicle_seg_engine.py`, `training/graft_seg.py`, patch 파일, `.gitignore`.
6. 전체 CPU 테스트, 문서 갱신, `.ai/WORK.md`에 GPU 검증 대기 항목 기록 → 커밋.
