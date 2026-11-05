# YOLO11m 차량 탐지 fine-tuning

ReTrace 웹(`/datasets`)에서 만든 YOLO ZIP으로 현재 운영 차량 모델(YOLO11m, COCO 80 class)을
고정 CCTV 시점에 맞게 fine-tuning하는 **독립 학습 환경**이다. DeepStream 이미지, Backend,
PeopleNet, 운영 `models/yolo11m/`, `.env`는 바꾸지 않는다. 학습 결과는 운영에 자동 반영되지 않는다.

```text
training/
  README.md  requirements.txt  requirements-lock.txt   # 추적됨
  check_dataset.py   # ZIP/폴더 검증 (+안전한 압축 해제)
  dataset_views.py   # 원본을 건드리지 않는 파생 view (coco80 라벨 재매핑)
  train.py           # GPU 전용 학습 (검증·VRAM 가드 포함)
  evaluate.py        # 기존 모델 vs 새 모델 비교 표/이미지
  worker.py          # 웹(/training) job 실행 worker (host, stdlib만 사용): docs/training.md
  tests/             # CPU unittest (합성 이미지, worker 통합 테스트)
  .venv/ .ultralytics/ weights/ datasets/ runs/        # Git 제외: 환경·CCTV 데이터·가중치·결과
```

## 1. 환경 (2026-10-13 서버 확인값)

| 항목 | 값 |
| --- | --- |
| OS | Ubuntu 20.04.6 LTS |
| GPU | Tesla V100-PCIE-16GB (Volta, compute capability 7.0), 1장 |
| Driver / 지원 CUDA | 550.127.05 / 12.4 |
| Python | 3.11.4 (`/usr/bin/python3.11`; 시스템 기본 3.8은 torch 2.5+ 미지원) |
| PyTorch | 2.6.0+cu124, torchvision 0.21.0+cu124 (`get_arch_list`에 `sm_70` 포함 확인) |
| Ultralytics | 8.4.176 |
| ONNX export | onnx 1.17.0, onnxslim 0.1.98 |

버전 선택 근거: driver 550의 최대 CUDA는 12.4이므로 driver와 같은 cu124 wheel을 쓴다
(PyTorch cu124 index의 최신은 2.6). PyTorch 공지에 따르면 2.11부터 CUDA 12.8/12.9 wheel에서 Volta가 빠졌다
([dev-discuss](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290)).
2.15부터는 Volta용 공식 wheel이 없다
([공지](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)).
torch를 올릴 때는 반드시 `torch.cuda.get_arch_list()`에 `sm_70`이 있는지 확인한다. 현재 설치본은
`sm_70` 포함과 GPU fp16 행렬곱 동작을 확인했다.

### 재현 (Python 3.11에 ensurepip이 없음)

```bash
cd training
python3.11 -m venv --without-pip .venv
python3 -m pip download --no-deps --only-binary=:all: --python-version 3.11 pip -d /tmp/pipwheel
W=$(ls /tmp/pipwheel/pip-*.whl) && .venv/bin/python "$W/pip" install --no-index "$W"
.venv/bin/python -m pip install -r requirements.txt        # 정확한 전체 버전은 requirements-lock.txt
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
mkdir -p weights && cp ../models/yolo11m/yolo11m.pt weights/   # 운영 모델과 동일 파일(복사본)
```

`train.py`/`evaluate.py`는 `YOLO_CONFIG_DIR=training/.ultralytics`를 쓰고 Ultralytics
analytics `sync`와 외부 logger를 끈다. weights/datasets/runs 경로도 `training/` 아래로 고정한다.
첫 학습의 AMP 검사가 `weights/yolo26n.pt`를 한 번 내려받는다(다운로드만, CCTV 데이터 업로드 없음).

## 2. 운영 GPU와의 자원 공유

운영 Preview(DeepStream, PeopleNet + YOLO11m)가 같은 GPU를 쓴다. 측정(18개 카메라 online, 각 5초 샘플):

| 상태 | GPU 메모리 최대 | GPU util 평균 | Preview 합계 FPS |
| --- | --- | --- | --- |
| 학습 없음 (60초) | 4.8 GiB | 59% | 287.6 |
| 640 / batch 4 smoke (1 epoch) | 8.5 GiB | 74% | 284.3 (−1%) |
| 960 / batch 4, 5 epoch | 12.5 GiB | 83% | 263.8 (−8%) |

torch 최대 reserved는 640/b4 약 3.3 GiB, 960/b4 약 7.2 GiB(OOM 없음). `train.py`는 시작 전
남은 VRAM이 `--min-free-mib`(기본 6000) 미만이면 시작하지 않는다. 960 학습은 Preview FPS를
약 8% 낮추므로 업무 외 시간에 돌리거나 운영자 동의를 받는다. dataloader worker 기본값은 4이다(CPU 8개 공유).
CPU fallback은 없다. CUDA가 없으면 종료한다.

## 3. 데이터셋 준비·검증

ZIP 구조는 `backend/app/dataset.py`가 만든다(루트에 `data.yaml`, `manifest.json`,
`images|labels/{train,val,test}`. 상위 `dataset/` 폴더 하나로 감싼 ZIP도 허용).
클래스는 데이터셋 전용 `0 car / 1 truck / 2 bus`이다(COCO ID 아님). 현재 exporter는 `0 car`만 내보내며(car-only),
`check_dataset.py`는 두 형식을 모두 받는다. coco80 view는 data.yaml의 이름으로 COCO ID를 찾는다(car→2). 세부 계약은 [docs/datasets.md](../docs/datasets.md).

```bash
cd training
.venv/bin/python check_dataset.py /path/to/retrace_export.zip   # → datasets/<zip 이름>/ 에 1회 해제
.venv/bin/python check_dataset.py datasets/<name>               # 이미 해제된 폴더
```

검사 항목: data.yaml class 이름, 이미지↔TXT 1:1, 5필드/좌표 범위/class 범위/2px 미만 박스,
이미지 decode, manifest의 split·sha256·box_count·counts 일치, 빈(검수된) 이미지 비율,
동일 bytes 중복, 세션이 split 하나에만 속하는지, 같은 카메라에서 30분 미만 간격인데 split이
다른 쌍(누수), train과 거의 같은 val/test 프레임(dHash ≤ 4bit), split·class·크기(COCO small
<32², medium <96², 원본 px)·카메라·세션 분포. 결과는 `<dataset>/check_report.json`.

종료 코드: 0 정상, 1 오류(학습 금지), 2 경고만 있음(예: val/test 10장 미만). 경고는 사람이 판단한다.
**val/test를 복제하거나 만들어 채우지 않는다.** val이 비면 `train.py`가 학습을 거부한다.
test가 비면 `evaluate.py`가 비교를 거부한다. 해결은 다른 날/시간대의 독립 세션을 더 수집하는 것뿐이다.

빈 TXT는 웹에서 "차량 없음"을 명시적으로 확인한 이미지에만 나온다(exporter가 강제). 그래도
라벨이 빠진 차량이 있으면 모델은 그 차량을 배경으로 배운다. 즉 미검출을 학습한다. 원거리·가려진
차량까지 모두 라벨링됐는지 검수 단계에서 확인한다.

## 4. 클래스 전략

| 전략 | 방식 | 장점 | 단점 |
| --- | --- | --- | --- |
| **coco80 (기본)** | 파생 view에서 car→2, truck→7, bus→5로 재매핑. 80-class head 유지 | 사전학습 분류 head를 그대로 이어받아 적은 데이터에 유리. 결과 ONNX/labels.txt가 운영과 같은 80 class 배치라 **ONNX 파일만 교체**하면 됨 | 다른 77개 class 출력은 학습되지 않아 drift 가능(운영은 `filter-out-class-ids`로 버림) |
| single (B) | `single_cls=True`. 모든 박스를 1 class로 | car/truck 라벨 혼동이 무의미. 운영도 3종을 한 tracker class로 합침 | 1-class head를 새로 초기화. labels.txt를 `car` 한 줄, `num-detected-classes=1`로 바꿔야 함 |
| multi (A) | 0 car/1 truck/2 bus | 종류 구분 유지 | 3-class head 새로 초기화. 웹 UI가 새 박스를 car로 만들어 truck/bus 표본이 적으면 불균형·혼동. 운영 labels/config 변경 필요 |

기본값은 coco80이다. 근거는 두 가지다. 운영 경로(`vehicle_detection.py`)는 car/truck/bus를 하나의
차량 class로 합친다. 또 현재 라벨이 주로 car다. 파이프라인 fixture(coco128을 ReTrace 형식으로 바꾼
합성 검증용, CCTV 아님)에서 1 epoch 후 vehicle mAP50은 baseline 0.486, coco80 0.493, single 0.009였다.
head 재초기화 비용을 보여 주는 결과일 뿐 최종 품질 비교는 아니다. 실제 데이터에서는 coco80과
single을 둘 다 학습해 `evaluate.py`로 비교하는 것을 권장한다. multi는 truck/bus가 각각 train 100개
이상이고 val에도 있을 때 검토한다. 어느 전략도 원본 ZIP/라벨을 바꾸지 않는다(view는 `datasets/<name>__coco80/`, 이미지 symlink).

## 5. 학습 실행

웹 UI(`/training`)에서 시작하면 host의 `worker.py`가 아래와 같은 인자로 `train.py`/`evaluate.py`를 실행한다
(진행률·중지 파일 `--progress-file`/`--stop-file`, 평가 `--predictions-json` 추가). 절차는 [웹 fine-tuning](../docs/training.md).

```bash
cd training
# 1) 짧은 smoke (1 epoch): 오류·VRAM 확인용, 결과 모델은 사용하지 않음
.venv/bin/python train.py --data datasets/<name> --smoke
# 2) 본 학습 (초기 기준값)
.venv/bin/python train.py --data datasets/<name> --epochs 50 --imgsz 640 --batch 4 --seed 42
# 비교 실험 예
.venv/bin/python train.py --data datasets/<name> --classes single
.venv/bin/python train.py --data datasets/<name> --imgsz 960 --batch 4   # 원거리 소형 차량; Preview 영향 −8%
```

학습 증강: 고정 시점이므로 rotation/shear/perspective/flipud는 0이다. 원거리 소형 차량과 조명 변화를
위해 scale 0.5, HSV, 좌우 flip, mosaic(가림·밀집)는 유지한다. patience 15로 과적합 시 조기 종료한다.
OOM이 나면 `--batch 2`로 낮춘다. 장시간 학습은 `nohup ... > runs/train.log 2>&1 &`로 실행한다.

결과는 `runs/train/<run>/`에 저장된다: `weights/best.pt`, `last.pt`, `results.csv`(epoch별 train/val loss·P·R·mAP),
`results.png`, PR/F1 curve, confusion matrix, `args.yaml`, `run_info.json`(버전·데이터 counts·경고·peak VRAM).
과적합 판단: `results.csv`에서 train loss는 계속 내려가는데 val box/cls loss가 오르거나 val mAP가 정체되는지 본다.

### 전체 학습 (Train All, `--train-all`)

데이터가 적을 때(예: 25장) Train/Val/Test 이미지를 모두 가중치 학습에 쓴다. `coco80` 전용이다(COCO 80클래스
head 유지, car는 COCO 2의 사전학습 가중치에서 시작).

```bash
.venv/bin/python train.py --data datasets/<name> --train-all --epochs <일반 학습의 best epoch> --imgsz 640 --batch 4
```

- 원본 export·분할은 바꾸지 않는다. `<name>__trainall/`에 학습용 사본(이미지 hardlink 또는 복사, label은 COCO id로
  변환, `train_all.json`에 원래 split 기록)을 만들어 학습한다.
- 독립 검증이 없으므로 `val=False`(epoch별 검증 없음), `patience=0`(조기 종료 없음, 지정 epoch 모두 실행),
  최종 가중치는 마지막 epoch(`best.pt`를 `last.pt`로 덮어씀)다. epoch 수는 같은 데이터의 일반 학습에서
  val mAP50-95가 가장 높았던 epoch를 쓰는 것이 기준이다(웹 UI가 제안).
- Ultralytics는 마지막 epoch 뒤 `val`(=학습 이미지)로 한 번 검증한다. 이 값은 학습 적합도이며 성능이 아니다.
  웹은 이 값을 표시하지 않는다. 성능은 아래 새 이미지 평가로만 잰다.
- 2026-10-15 CPU harness(yolo11n, 합성 이미지 12장, Ultralytics 8.4.176, torch 2.14 CPU — 서버 torch 2.6과 다름)로 확인: epoch 1–2 검증 생략,
  마지막 epoch만 검증, `best.pt`=`last.pt`, head nc 80, 원본 파일 sha256 불변. GPU 실학습은 미검증.

## 6. 평가와 기존 모델 비교

```bash
.venv/bin/python evaluate.py --data datasets/<name> --split test \
    --model ft=runs/train/<run>/weights/best.pt [--model single=runs/train/<run2>/weights/best.pt] [--imgsz 960]
```

- baseline은 `weights/yolo11m.pt`(운영과 동일 bytes)이다. COCO car/truck/bus를 이름으로 데이터셋 class에 매핑한다.
- 주 지표는 **class 무관 vehicle** 지표다(운영이 3종을 합치므로). mAP50, mAP50-95, best-F1 P/R을 낸다.
  운영 임계값(conf 0.25, NMS IoU 0.45)에서의 P/R/TP/FP/**FN(미검출)**, 크기별·class별 FN도 낸다.
  fine-tuned 모델은 Ultralytics `val` 결과(class별 포함)도 함께 기록한다.
- 출력은 `runs/eval/<name>_<split>_<imgsz>/`: `report.md`, `metrics.json`, `per_image.csv`,
  `images/`(모델별 패널. 초록 GT, 파랑 TP, 주황 FP, 빨강 미검출)이다.
- 비교는 test(독립 세션)로 한다. val은 `best.pt` 선택에 쓰였으므로 낙관적이다. test가 10장 미만이거나
  near-duplicate 경고가 있으면 개선을 단정하지 않는다(보고서에 caveat로 출력).
- 새 CCTV 이미지 평가: `--eval-only`는 학습 split이 빈 평가 전용 데이터셋을 허용한다. 웹은 모델이 학습한 이미지와
  같은 촬영 세션 이미지를 뺀 사본(`dataset_views.subset_view`, 모두 test)을 만들어 이 옵션으로 실행한다.
- Ultralytics 오프라인 추론이며 DeepStream(nvinfer 스케일링, TensorRT FP16, 1920×1080 mux)과 다르다.
  최종 판단은 아래 GPU 검증 절차로 한다.

## 7. 재학습

새 세션을 수집·검수한 뒤 웹에서 ZIP을 다시 만든다(이전 ZIP은 immutable). 새 이름으로
`check_dataset.py` → `train.py` → 같은 test로 `evaluate.py`를 실행한다. 이전 best.pt에서 이어 학습하려면
`--model runs/train/<run>/weights/best.pt`를 쓴다. 다만 split이 바뀐 ZIP이면 이전 모델이 새 test 이미지를
학습했을 수 있으므로 baseline과의 비교는 항상 원본 `yolo11m.pt`에서 새로 학습한 모델로 한다.

## 8. DeepStream 적용 전 확인 (자동 적용 금지, 운영자 수행)

640px의 coco80·single(car) 모델은 웹 [DeepStream에 배포]가 아래 1~5를 자동화한다(export·클래스 매핑·호환성 검사·engine 생성·
무중단 교체·추론 검증·복구, [docs/training.md §4-2](../docs/training.md)). single은 운영 config를 바꾸지 않고 ONNX에서 class 0을
운영 car ID로 바꾸므로 아래 2의 single 전용 config가 필요 없다. 아래 수동 절차는 960 등 다른 입력 크기나 배포 기능을 끈 경우에 쓴다.

운영 차량 모델: `models/yolo11m/vehicle_infer.txt`(onnx `yolo11m.onnx`, 80 labels,
DeepStream-Yolo `NvDsInferParseYolo`, `.env`의 `VEHICLE_INFER_CONFIG`). Preview 이미지 기반은
`nvcr.io/nvidia/deepstream:7.0-triton-multiarch`이다. NVIDIA DS 7.0 설치 문서 기준 x86은
CUDA 12.2 + TensorRT 8.6.1.6이다([DS 7.0 Installation](https://docs.nvidia.com/metropolis/deepstream/7.0/text/DS_Installation.html)).
TensorRT 8.6은 opset 17과 Volta를 지원한다. 다만 Docker 권한이 없어 컨테이너 안 실제 버전은 확인하지 못했다(Not verified).

1. ONNX export는 Ultralytics `model.export`가 아니라 운영과 같은 DeepStream-Yolo 스크립트로 한다.
   출력 layout `[batch, 8400, 6]`(box 4, score, class)을 parser가 기대한다.
   ```bash
   mkdir -p runs/export/<run> && cd runs/export/<run>
   cp ../../train/<run>/weights/best.pt . && \
   ../../../.venv/bin/python ../../../../models/yolo11m/DeepStream-Yolo/utils/export_yolo11.py -w best.pt --dynamic --simplify
   ```
   확인 결과: coco80 smoke 모델 export 성공. 입력 `input [batch,3,640,640]`, 출력 `output [batch,8400,6]`,
   opset 17. 생성 labels.txt는 운영 labels.txt와 byte 동일하다.
2. 전략별 config 차이:
   - coco80: 새 디렉터리(예: `models/yolo11m-ft-<date>/`)에 기존 `vehicle_infer.txt`, `labels.txt`를 복사하고
     `onnx-file`만 새 ONNX로 지정한다. `custom-lib-path`는 기존 `.so`를 상대 경로로 가리킨다.
   - single: labels.txt를 `car` 한 줄로 쓰고 `num-detected-classes=1`. `vehicle_detection.py`는 정확히 하나의
     `car` label을 요구하므로 Ultralytics 이름 `item`을 그대로 쓰면 시작 시 거부된다.
   - multi: labels.txt `car/truck/bus`, `num-detected-classes=3`.
   - `--imgsz 960`으로 학습했다면 `-s 960`으로 export한다. engine 입력·연산량이 커지므로 FPS/VRAM을 다시 측정한다.
3. engine cache는 ONNX 내용 hash로 key가 정해진다(`pgie_cache.py`). 새 ONNX는 첫 시작에 TensorRT engine을
   새로 빌드한다(수 분, V100 sm_70에서 생성). 기존 engine/모델 파일은 지우지 않는다(롤백용).
4. 적용: private `.env`의 `VEHICLE_INFER_CONFIG`만 새 config로 바꾸고 retrace만 재생성한다
   (`sudo docker compose up -d --no-deps --no-build --force-recreate retrace`). 롤백은 값을 되돌리고 재생성한다.
5. 적용 후: 로그의 engine build/deserialize, `/api/cameras/<camera>/vehicles`, 주차 점유, 전체 카메라 FPS/VRAM,
   PeopleNet 사람 검출 회귀를 적용 전과 비교한다. 기존 P1 미검출 원인(비-antialias GPU 축소)은
   모델 교체로 해결되지 않을 수 있으므로 같은 장면에서 따로 확인한다.
6. 차량 윤곽선 seg를 쓰는 경우: 새 det를 운영에 올린 뒤 `graft_seg.py`로 그 best.pt를 YOLO11m-seg에 다시 이식하고
   DeepStream-Yolo-Seg로 export한다(80클래스 vehicle3 모델만 가능). 명령은 [운영 문서](../docs/operations.md#차량-윤곽선seg-선택)
   G1–G3이다. 테스트: `.venv/bin/python -m unittest discover -s tests -p 'test_graft_seg.py'`(torch만 필요).

## 검증

```bash
cd training && .venv/bin/python -m unittest discover -s tests -v   # CPU, 합성 이미지
```

GPU smoke 기록(2026-10-13): coco128(공개 데이터, 파이프라인 검증용)으로 640/b4 1 epoch 40초, 960/b4 5 epoch 84초,
ReTrace 형식 fixture로 check → coco80/single smoke 학습 → evaluate(표·이미지) → DeepStream-Yolo export까지 확인했다.
실제 ReTrace CCTV ZIP으로의 학습·평가는 아직 수행하지 않았다(서버에 ZIP 없음, datasets API 미배포).

## 공개 금지

CCTV 이미지/ZIP, 학습 가중치(`*.pt`, `*.onnx`, `*.engine`), runs 결과(예측 이미지 포함), 서버 주소·RTSP URL·
인증 정보는 커밋하지 않는다. `.gitignore`가 `training/{datasets,weights,runs,.ultralytics}`와 `.venv`를 제외한다.
`evaluate.py` 이미지에는 CCTV 장면이 그대로 들어가므로 외부 공유 전에 확인한다.
