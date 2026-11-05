# 웹 기반 차량 모델 fine-tuning (`/training`)

ReTrace 웹에서 데이터셋 export를 골라 YOLO11m 차량 모델 fine-tuning을 시작·감시·중지하고, 결과 모델을
기존 모델과 같은 test set에서 비교한다. 학습은 **브라우저나 API 요청 안에서 돌지 않는다.** GPU 서버 host의
별도 worker 프로세스가 기존 `training/.venv`로 `train.py`/`evaluate.py`를 실행한다. 학습 스크립트·클래스
전략·GPU 측정값의 원문은 [training/README.md](../training/README.md)이다.

```text
브라우저 /training ──/api/training/*──> Preview :40225 proxy ──> Backend :8000 (FastAPI, PostgreSQL)
                                                                    ▲  job 상태·지표·로그·산출물 저장
host: training/.venv/bin/python worker.py ──loopback HTTP + worker token──┘
        └─ subprocess: train.py → evaluate.py (GPU), runs/web/<job>/
```

## 1. 안전 경계

- 동시 학습은 1개다. `training_jobs.active_slot`(진행 중 1, 종료 NULL)의 unique constraint가 DB 수준에서 막는다.
- 학습 시작·중지·모델/산출물 다운로드는 `TRAINING_ADMIN_TOKEN`(Bearer)이 필요하다. 토큰이 비어 있으면 503으로 꺼진다.
  조회 GET은 기존 `/api/*`와 같은 공개 범위다. 앱 전체 사용자 인증은 아직 없다(아래 §6).
- worker API(`/api/training/worker/*`)는 `TRAINING_WORKER_TOKEN`이 필요하고, Preview proxy는 이 경로를 404로 막는다.
  worker는 host loopback으로 Backend에 직접 붙는다. 두 토큰은 24자 이상이고 서로 달라야 한다(시작 시 검증).
- 사용자 문자열로 CLI를 만들지 않는다. worker가 job 설정을 정수·허용 값으로 다시 검증해 인자 list를 만든다.
- 업로드된 모델/코드를 실행하지 않는다. base는 host의 `training/weights/yolo11m.pt`(등록된 sha256과 비교) 또는
  이전 job의 `best.pt`(sha256 확인)만 쓴다.
- 데이터셋은 job 생성 시 export ZIP을 `dataset-storage/training/snapshots/<export_id>.zip`으로 고정(hardlink/복사 +
  sha256)한다. 이후 라벨링이나 export 삭제가 학습에 영향을 주지 않는다. worker는 sha256을 다시 확인하고
  export 구조 allowlist·symlink 금지·파일 수/크기 제한(Zip Slip 방지)으로 압축을 푼다.
- 운영 YOLO11m/DeepStream은 학습 완료로 자동 교체되지 않는다. admin이 모델을 골라 [DeepStream에 배포]를 누를 때만
  바뀌며(§4-2), 이 기능은 retrace의 `VEHICLE_DEPLOYMENT_ENABLED=1`일 때만 켜진다. 학습 데이터·이전 모델·engine은 지우지 않는다.
- DeepStream 컨테이너를 멈추거나 재시작하지 않는다. 자원이 부족하면 학습 쪽이 기다리거나 멈춘다.

## 2. GPU·운영 보호

| 시점 | 동작 | 설정 |
| --- | --- | --- |
| 시작 전 (UI·API) | VRAM 추정 `150 + 800·batch·(imgsz/640)²` MiB, 필요 여유 `max(MIN_FREE, 추정 + RESERVE)`. worker·GPU·Preview 상태 확인 | `TRAINING_MIN_FREE_MIB`(6000), `TRAINING_GPU_RESERVE_MIB`(2048) |
| claim 후 | GPU 여유 메모리·util이 기준을 넘으면 대기, 시간 초과 시 `gpu_busy` 실패 | `TRAINING_MAX_GPU_UTIL`(85), `TRAINING_GPU_WAIT_SECONDS`(1800) |
| 학습 중 (heartbeat 5초) | claim 시 online 카메라별 FPS 기준선 대비 하락, Preview 응답 없음, GPU 여유 < reserve가 연속되면 안전 중지(`production_protection`) | `TRAINING_MAX_FPS_DROP`(0.15), `TRAINING_GUARD_SAMPLES`(6), `TRAINING_REQUIRE_PREVIEW`(1) |
| worker 소실 | heartbeat가 끊긴 job은 `worker_lost`, worker 재시작 시 진행 중 job은 `worker_restarted`로 실패 처리 | `TRAINING_WORKER_STALE_SECONDS`(60) |
| 저장 | Backend volume 여유가 부족하면 산출물 업로드 507 | `TRAINING_MIN_STORAGE_MIB`(2048) |

기본값은 training/README.md §2의 측정(640/b4 torch reserved 약 3.3 GiB, 960/b4 약 7.2 GiB)에 맞췄다. 카메라 수나
운영 부하가 바뀌면 다시 측정해 `.env`로 조정한다. CUDA OOM은 `gpu_oom`으로 기록되고 오류 메시지가 batch/imgsz를 낮추도록 안내한다. 설정 화면은 현재 GPU 여유로 가능한 batch를 보여 준다.

## 3. 상태와 실패 코드

상태: `queued → running → (stopping) → completed | stopped | failed`. phase는 `queued → preparing → validating → waiting_gpu → training → evaluating → saving`(자동 라벨링 job은 `preparing → waiting_gpu → labeling → saving`).
중지는 다음 batch 경계에서 `trainer.stop`으로 끝내고(검증·저장 후 종료), `--stop-grace` 안에 끝나지 않으면 SIGTERM → SIGKILL.

주요 `error_code`: `dataset_invalid`, `dataset_too_large`, `snapshot_changed`, `no_validation`, `invalid_config`,
`base_model_missing`, `base_model_changed`, `cuda_unavailable`, `gpu_busy`, `gpu_oom`, `storage_full`, `killed`,
`training_failed`, `download_failed`, `artifact_upload_failed`, `production_protection`, `worker_lost`, `worker_restarted`,
`evaluation_failed`(새 이미지 평가 job), `autolabel_failed`(자동 라벨링 job: autolabel.py 실패 또는 일부 이미지 결과 누락). 배포 job: `deployment_unavailable`, `export_failed`, `runtime_changed`, `runtime_lost`,
`deployment_timeout`, `deployment_unverified`, `deployment_failed`/`deployment_rolled_back`/`deployment_rollback_failed`
(runtime 결과, 세부 원인은 error 문자열: `artifact_contract_mismatch`, `engine_build_failed`, `model_load_failed`,
`verification_failed`, `deployment_cancelled`, `no_live_inference`, `no_previous_model` 등).

## 4. API (`/api/training`)

| 메서드·경로 | 권한 | 용도 |
| --- | --- | --- |
| GET `/options`, `/status?imgsz&batch` | 공개 | 설정 범위·기본값, worker/GPU/Preview/진행 job·VRAM 추정 |
| GET `/datasets`, `/datasets/{export_id}/check?train_mode=standard\|all` | 공개 | 학습 가능한 export와 학습 방식별 적합성(blocked/limited/ready, 세션·split·라벨 기준). 최상위 값은 standard, `modes`에 방식별 값 |
| GET `/evaluations/plan?model_id&export_id` | 공개 | 새 이미지 평가 대상: 전체·제외(학습에 사용/같은 세션)·남은 이미지 수 |
| GET `/jobs`, `/jobs/{id}`, `/jobs/{id}/comparison` | 공개 | 목록, 진행·지표·로그, test 이미지별 GT/TP/FP/FN 비교 |
| GET `/models`, `/models/{id}`, `/models/compare?ids=` | 공개 | 모델 registry, 지표 비교 |
| POST `/auth` | admin | 토큰 확인(204) |
| POST `/jobs` | admin | 검증 후 job 생성(`train_mode`: `standard` 기본, `all`). limited 데이터셋은 `acknowledge_limits: true` 필요 |
| POST `/evaluations` | admin | 등록된 fine-tuned 모델을 새 이미지로 평가하는 job 생성(`model_id`, `export_id`, `acknowledge_limits`) |
| GET `/autolabels/plan?model_id&camera_id&include_auto_drafts` | 공개 | 자동 라벨링 대상 수(`eligible`), 제외 사유별 수(`skipped.reviewed/labeled/auto_drafts`), `max_images`, `ready`/`blocker`(§4-3) |
| POST `/autolabels` | admin | `{model_id, camera_id?, conf, include_auto_drafts}` → 자동 라벨링 job 생성(201, 학습과 같은 단일 slot). 단일 이미지 제안 form `{model_id, conf, image_ids: [1~20], apply: false}`는 DB를 바꾸지 않고 결과를 `dataset_summary.results`로 돌려준다(§4-3) |
| POST `/jobs/{id}/stop` | admin | 안전 중지 요청(배포 job은 취소 → 적용 전이면 변경 없음, 적용 후면 이전 모델 복구) |
| GET `/deployment` | 공개 | 현재 적용 모델(`active`), 롤백 대상(`previous`), runtime phase/error/추론 근거(`evidence`, `service`, `checks`, `failures`, `diagnosis`), `available`·`confirmed` |
| POST `/deployments` | admin | `{"model_id", "action": "deploy"}` 또는 `{"action": "rollback"}`. 학습 queue의 단일 slot을 함께 쓴다 |
| GET `/jobs/{id}/artifacts/{name}`, `/models/{id}/files/{best\|last}` | admin | 산출물·가중치 다운로드 |
| `/worker/*` (heartbeat, claim, dataset, weights, artifacts, finish, `jobs/{id}/images/{image_id}`, `jobs/{id}/autolabels`) | worker | host worker 전용, Preview에서 404 |

job별 산출물: `best.pt`, `last.pt`, `results.csv`, `results.png`, `args.yaml`, `run_info.json`, `train.log`,
`check_report.json`, `eval_metrics.json`, `eval_predictions.json`, `eval_report.md`. 저장 위치는 Backend volume의
`dataset-storage/training/jobs/<job_id>/`, worker 작업 폴더는 host `training/runs/web/<job_id>/`이다.
테이블은 `training_jobs`, `training_models`이며 기존 테이블처럼 `create_all`로 추가만 된다(삭제·초기화 없음).

평가는 baseline(COCO yolo11m)과 새 모델(재학습이면 base 모델 포함)을 같은 test split에서 비교한다. test 10장 이상·
test 세션 2개 이상·경고 없음일 때만 `conclusive`이며, 아니면 UI가 개선을 주장하지 않는다.

## 4-1. 전체 학습(Train All)과 새 CCTV 이미지 평가

목적은 고정 CCTV 차량 검출 향상이며 주차 판정은 학습 대상이 아니다. 데이터가 적을 때(25장: Train 20/Val 2/Test 3)
모든 이미지를 가중치 학습에 쓰고, 성능은 학습에 쓰지 않은 새 이미지로 따로 잰다.

| 항목 | 일반 학습 (`standard`) | 전체 학습 (`all`) |
| --- | --- | --- |
| 가중치 학습 | train | train + val + test (별도 학습용 사본, 원본 export·split 불변) |
| 조기 종료·best 선택 | val fitness, patience | 없음. 고정 epoch, 마지막 epoch 가중치(`best.pt` = `last.pt`) |
| 학습 중 지표 | train/val loss, val P/R/mAP | train loss만. 마지막 epoch의 학습 이미지 검증값은 저장·표시하지 않음(`metric_point`) |
| job 직후 평가 | test에서 YOLO11m과 비교 | 생략(`evaluation.status=skipped`). 모델 registry의 val/test는 `null` |
| 클래스 | vehicle3 또는 single | vehicle3만(COCO 80 head, car=COCO 2 사전학습 가중치) |

- 데이터셋 적합성은 방식별로 다르다. 전체 학습은 val이 없어도 막지 않지만 "독립 검증 없음" 제한을 항상 표시해
  확인(`acknowledge_limits`)을 받는다.
- epoch 수: 조기 종료가 없으므로 같은 export의 완료된 일반 학습에서 val mAP50-95가 최고였던 epoch를 UI가 제안한다.
- 새 이미지 평가 job(`config.kind = evaluate`)은 학습 job과 같은 큐·worker·이력·산출물·비교 화면을 쓴다. Backend는
  모델과 그 base 모델 체인이 학습에 쓴 이미지(standard: train+val, all: 전체)와 그 촬영 세션을 평가 export에서 빼고,
  남은 이미지 id만 job에 넣는다. worker는 snapshot에서 그 이미지만 test로 담은 사본을 만들어
  `evaluate.py --eval-only`로 baseline(COCO yolo11m)과 비교한다. 결과는 모델의 `metrics.new_images`(최근 10개)에 붙고
  모델을 새로 등록하지 않는다. 10장·2세션 미만이면 `conclusive=false`다.
- 전체 학습 모델에서 이어서 일반 학습을 하면 test 이미지가 이미 학습된 것이므로 `dataset_summary.test_overlap`에
  기록하고 평가를 non-conclusive로 표시한다.
- 고정 카메라는 다른 날 이미지도 비슷할 수 있다. 새 이미지는 다른 날·시간대·날씨에서 모은다.
- DeepStream 운영 모델은 바꾸지 않는다(§1과 동일).

## 4-2. DeepStream 배포와 롤백 (`VEHICLE_DEPLOYMENT_ENABLED=1`)

차량 탐지는 별도 Python YOLO 프로세스가 아니라 `preview.py` 공유 pipeline 안의 차량 nvinfer(secondary, 주차장 ROI만,
batch = 주차장 카메라 수) 하나다. 따라서 배포는 이 nvinfer의 on-the-fly model update로 하고 영상·RTSP·tracker는 재시작하지 않는다.

```text
웹 [DeepStream에 배포] → POST /deployments (deploy job, 학습과 같은 단일 slot)
host worker: best.pt(sha256) → training/deploy_export.py (운영과 같은 DeepStream-Yolo exporter, CPU) → 호환성 검사
          → .cache/vehicle-deployments/<job>/request.json (5초마다 lease 갱신, 중지 시 cancel)
retrace (vehicle_deployment.py, 기존 control loop):
  tools/build_vehicle_engine.py 별도 프로세스로 TensorRT engine 생성 → nvinfer config-file-path 교체
  → model-updated 성공 신호 → 20초 후 실시간 추론 확인 → committed.json 기록(완료) / 실패 시 이전 config로 복구
```

- 호환성(모두 맞아야 함, 아니면 운영 모델 변경 없이 실패): 640px, YOLO11m state_dict 구조 동일(1클래스는 Detect의 class 출력
  conv `cv3.*.2`만 80→1 허용), ONNX 입력 `[dynamic,3,640,640]`·출력 `[dynamic,8400,6]`과 이름·dtype이 운영 ONNX와 동일, 생성 config의
  전처리(scale/letterbox/scaling-filter)·후처리(parser `.so` hash, NMS/threshold/class 필터)·batch가 운영 config와 동일(`contract`).
- 클래스 매핑: 운영 nvinfer는 80 labels 그대로다. DeepStream-Yolo parser는 출력의 class 값을 그대로 class ID로 쓰고 그 ID로
  per-class threshold를 고르며, config의 `filter-out-class-ids`와 tracker 매핑(`vehicle_class_ids`)도 COCO ID 기준이다. 또 OTA는
  `num-detected-classes`/per-class 설정을 이전 값으로 덮어쓴다(gstnvinfer_impl.cpp). 그래서 모델 쪽을 맞춘다.
  - `vehicle3`(COCO80): class 80개·순서가 운영 labels.txt와 같아야 하며 매핑 없음(car = 2, truck 7, bus 5).
  - `single`(1클래스): Ultralytics single_cls 이름 `item` 또는 `car` 1개, 학습 export의 클래스가 car뿐이어야 한다(legacy car/truck/bus
    export로 학습한 single은 거부). export 후 ONNX 끝에 `Slice/Add/Concat`을 붙여 class 0을 운영 car ID(labels.txt에서 찾음, 현재 2)로
    바꾼다. 붙인 부분만 onnx reference로 실행해 box/score 불변·class 0→2를 확인하고, 매핑은 manifest·`committed.json`·
    `/deployment.active_class_map`에 남는다. threshold는 운영 car class 값(`class-attrs-2`)이 적용된다.
- 입력 크기: 640 외(960 등)는 자동 배포하지 않는다. nvinfer OTA는 network 해상도가 다르면 새 context를 거부하고
  (`isNewContextValid`), 960은 연산량·VRAM이 커서 FPS 재측정 없이 운영 판단을 할 수 없다. 960 학습 모델을 640으로 export하는 것도
  정확도 검증이 없어 막는다. 이 경우 training/README.md §8 수동 절차(새 config + retrace 재생성 + 측정)를 따른다.
- engine-only 교체(`model-engine-file`)는 쓰지 않는다. DeepStream이 새 context에 custom parser·letterbox 설정을 복사하지
  않기 때문이다(DS 6.1 `gstnvinfer_impl.cpp` 확인). config 파일 교체는 전체 설정을 다시 읽는다.
- FPS 기준선: 컨트롤러가 1초마다 카메라별 decoded FPS(5초 창)·상태·추론 누계를 상시 기록한다. engine 생성(수 분) 직후,
  config 교체 직전의 최근 60초(live 표본 ≥10개)에서 카메라별 **중앙값**과 p10–p90(그 카메라의 평소 변동 폭), 추론 프레임 속도를
  고정한다. 한 시점 값은 RTSP burst/jitter 표본이라(2026-10-19 게이트 카메라: 기준 20.03 → 새 모델 10.09 → 롤백 후 9.13 fps)
  기준선으로 쓰지 않는다. 자동 복구는 같은 기준선을 쓴다. 표본이 부족하면(Preview 직후) `baseline_not_ready`로 시작하지 않는다.
- 성공 판정: nvinfer `model-updated` status 0(전환 후 90초 안) → 20초 안정화 → 최소 30초 측정 구간(최대 60초까지 매 tick
  재판정, 구간 전체 중앙값)에서 모두 충족. 85% 기준은 그대로이며 비교 대상만 순간값에서 구간 중앙값으로 바뀌었다.
  - 검증 대상(기준선 시점 live 주차장 카메라)마다: 상태 online/degraded, 추론 frame ≥5, 마지막 추론 ≤5초. 차량 검출 합계 > 0.
    구간 FPS 중앙값이 기준선 중앙값의 85% 미만이면 `fps_below_baseline`으로 기록하되, 아래 두 합계 검사가 모두 통과하면
    `pipeline_slow: false`·비차단이다. 모델은 공유 batch에서 돌므로 카메라 한 대만 느려지는 것은 모델이 아닌 소스 신호이기 때문이다
    (2026-10-19 B1 카메라들은 기존 모델에서도 8–24 fps를 오갔다). 합계 검사 중 하나라도 실패하면 차단된다. 85% 미만이지만 (a) 카메라
    자신의 전환 전 p10 이상이고 (b) 추론 프레임 속도가 그 카메라 FPS의 85% 이상이면 `camera_fps_variation`(항상 비차단)으로 남긴다.
  - 차량 추론 처리량: 검증 대상 카메라의 추론 프레임/초 **합계** ≥ 기준선 합계의 85%(`vehicle_inference_below_baseline`, 차단).
    원인과 무관하게 실제 처리량이 크게 줄면 막으므로, 카메라별 저하가 여러 대에 겹치거나 모델이 느리면 배포되지 않는다.
  - 서비스 보호(그 밖의 기준선 시점 online 카메라): 구간 FPS 중앙값 **합계** ≥ 기준선 합계의 85%. 모델은 공유 mux/nvinfer batch에서
    돌기 때문에 모델이 파이프라인을 느리게 하면 모든 카메라와 합계가 함께 떨어진다. 한 RTSP 소스만 바뀌면 그렇지 않다.
  - 오프라인 구분: 기준선 시점에 이미 오프라인/표본 부족인 주차장 카메라는 검증에서 빼고 `camera_offline_before_deployment`(비차단),
    검증 중 오프라인이 된 주차장 카메라는 `camera_went_offline`(차단, `baseline_state`/`offline_samples` 포함), 기타 카메라가
    끊기면 서비스 합계에서 빼고 `camera_went_offline`(`cameras` 목록, 비차단)으로 남긴다.
  60초 측정 구간까지 충족하지 못하면 이전 모델로 자동 복구한다. 빈 주차장(검출 0)에서는 성공으로 판정하지 않으므로 차량이 있는 시간에 배포한다.
- 실패 처리: export/검사/engine 생성 실패는 운영 모델을 건드리지 않는다. 적용 후 load 실패·검증 실패·취소·worker lease
  만료(25초)는 직전 committed 모델의 config로 되돌린다. 복구 검증은 해당 config의 `model-updated` status 0과 검증 대상 카메라의
  추론(상태·frame·최근 추론·검출)으로 판정하고, FPS·처리량 항목과 복구 중 끊긴 카메라는 `blocking: false`로 기록만 한다(복구 자체가
  보호 조치이므로; 끊긴 카메라는 추론 판정에서도 빠지고 나머지 카메라에서 추론이 확인돼야 한다).
- 원인 진단(`diagnosis`): 복구 검증이 끝나면 배포 실패 때 85% 미만이던 카메라마다 전환 전 중앙값/범위 → 새 모델 → 이전 모델 FPS를
  비교한다. 이전 모델에서 85% 이상으로 회복하면 `model_suspected`, 여전히 낮으면 `camera_source_suspected`, 복구 후 오프라인이면
  `inconclusive`. 판정에는 쓰지 않는 운영자용 근거이며 retrace 로그 `vehicle_deployment_diagnosis {json}`과 /training에 표시된다.
  복구 확인까지 실패하면
  `rollback_failed`로 새 배포를 막고 UI에 미확인으로 표시한다. 이때는 `sudo docker compose restart retrace`로 재시작하면
  `committed.json`의 마지막 검증 모델(없으면 원래 운영 모델)로 시작한다.
- 재시작: retrace는 시작 시 `committed.json`의 모델을 PLAYING 전에 적용한다. 카메라 수(batch)·parser·TensorRT/GPU가 바뀌어
  계약이나 engine key가 달라지면 원래 운영 모델로 시작하고 `committed_model_skipped`를 표시한다.
- 수동 롤백: UI [이전 모델로 롤백]은 runtime의 `previous`(직전 검증 모델, 처음에는 원래 운영 모델)로 같은 검증 절차를 거친다.
- 실패 진단: `result.json`/`status.json`(→ job `run_info.deployment`, GET `/deployment`)에 `failures`(실패 순간을 고정한 기록:
  `code`, `phase`, `model_id`, `load_event`, `model_loaded`, `elapsed_seconds`, `loaded_after_seconds`, `checks`, 카메라별 `evidence`,
  `service`), 최근 판정 `checks`, 검증 대상 `evidence`(frames/detections/fps·baseline_fps(구간 중앙값)/fps_range·baseline_range
  (p10–p90)/inference_fps·baseline_inference_fps/state·baseline_state/offline_samples/inference_age), 기타 카메라와 차량 추론
  합계 `service`, 복구 후 `diagnosis`가 남는다. check 코드: `model_not_loaded`, `model_load_failed`, `camera_offline_before_deployment`,
  `camera_went_offline`, `fps_below_baseline`, `camera_fps_variation`, `inference_stalled`, `insufficient_inference_frames`,
  `no_vehicle_detections`, `vehicle_inference_below_baseline`, `service_fps_below_baseline`(이전 결과의 `settling`,
  `camera_unavailable`는 더 만들지 않는다). 같은 내용이 retrace
  로그에 `vehicle_deployment_failure {json}` 한 줄로 남고 /training 모델 카드와 배포 job 패널에 표시된다.
- 파일: `.cache/vehicle-deployments/{status.json, committed.json, engines/, <job>/(best.pt, best.onnx, labels.txt, manifest.json,
  infer.txt, build.log, build.json, request.json, result.json)}`. 로컬 산출물이며 자동 삭제하지 않는다.
- engine 생성 중에는 같은 GPU를 쓰므로 Preview FPS가 잠시 내려갈 수 있다(수 분). 학습과 동시에 실행되지 않는다.
- 차량 윤곽선 seg를 켰으면 상태에 `segmentation_graft`(`match`/`stale`/`unknown`, seg 끔이면 null)가 있다. 이식 seg의
  det 가중치(seg config 옆 `manifest.json`의 `det_weights_sha256`)와 활성 모델 배포 디렉터리 `manifest.json`의
  `weights_sha256`을 비교하며, 처음 모델은 비교할 수 없어 `unknown`이다. 새 det를 배포하면 `stale`이 되고 재이식은
  수동이다([운영](operations.md#차량-윤곽선seg-선택)). worker가 상태 모듈을 다시 읽어야 API에 보인다
  (`systemctl --user restart retrace-training-worker`).

## 4-3. 자동 라벨링 job (`config.kind = 'autolabel'`)

`/datasets`의 AI 초안 기능이다. 대상 조건·저장 규칙·클래스 매핑의 원문은 [docs/datasets.md "자동 라벨링"](datasets.md#자동-라벨링-ai-초안)이고, 여기에는 job·API 계약만 둔다.

- 생성(`POST /autolabels`): 모델은 `kind=base` 또는 `best.pt`가 있는 fine-tuned(`status=available`)만. `conf` 0.1~0.9(기본 0.35).
  대상이 0장이면 422 `No unlabeled pending images.`, GPU/worker 조건은 evaluation과 같은 `gpu_check(640, batch 1)`(오프라인 503, 그 외 409),
  slot 충돌 409. job은 `epochs=1`, `export_id`/`dataset_sha256` 없음(`null`), `name = autolabel-<모델 label>`,
  `config = {kind, imgsz: 640, conf, class_mode, camera_id, include_auto_drafts, images(대상 수), required_free_mib, estimated_training_mib}`
  (대상 id/revision 목록 `targets`는 worker claim에만 전달되고 공개 응답에서는 `images` 수로 바뀐다),
  `dataset_summary = {kind, camera_id, model_label, conf, include_auto_drafts, total, processed, labeled, boxes, no_detections, skipped_changed}`.
- `GET /autolabels/plan`의 `blocker`는 프론트 번역 키이므로 문장을 그대로 쓴다: `No auto-label model is available yet.`(모델 없음/incomplete/host에 base 가중치 없음),
  `The training worker is offline.`, `Another training job is running.`(ACTIVE 상태 job 존재), `No unlabeled pending images.`. 모델이 없어도 404가 아니라 `ready:false`다.
- `GET /jobs` 목록에서는 자동 라벨링 job을 제외한다(학습 표의 epoch/지표 칼럼 오표시 방지). `GET /jobs/{id}`와 `GET /status`의 `active_job`에는 포함되므로
  데이터셋 화면이 진행을 polling하고 학습 시작은 slot 점유로 막힌다. `progress`는 heartbeat가 아니라 결과 batch 적용(`processed/total`)으로만 갱신된다.
- worker 흐름: claim → `resolve_weights`(base sha256 확인 또는 `best.pt`) → `GET /worker/jobs/{id}/images/{image_id}`로 대상 JPEG(20 MiB 제한)를
  `runs/web/<job>/images/`에 내려받음(그 사이 삭제된 이미지는 건너뜀) → `wait_for_gpu` → `training/autolabel.py` subprocess(검증된 인자 list,
  `run_child`의 heartbeat/stop/SIGTERM 재사용, 중지는 즉시 SIGTERM) → JSON lines 결과를 최대 50장씩 `POST /worker/jobs/{id}/autolabels`
  (`{worker_id, results: [{image_id, boxes: [{x, y, width, height, score}]}]}`, 이미지당 300개)로 올리고 Backend가 한 transaction에서 이미지별
  row lock·revision 검사 후 적용해 summary를 돌려준다 → `finish`(`completed`/`stopped`/`failed` + `autolabel_failed` 등). finish는 모델 등록·best.pt·평가 검사를 하지 않는다.
  `train.log`만 산출물로 남는다.
- 단일 이미지 제안(`apply: false`, 편집 화면의 "이 이미지 자동 라벨링"): body는 `{model_id, conf, image_ids(1~20, 없는 id 404), apply: false}`이고
  일괄 form의 `camera_id`/`include_auto_drafts`와 섞이거나 `apply`가 `false`가 아니면 422다. 대상 조건을 검사하지 않아 reviewed 이미지도 된다.
  job `config.apply=false`, `targets`는 id만(revision 없음). worker는 일괄과 똑같이 추론·업로드하고, Backend는 이미지 row를 읽기만 한 채
  clip·2px 제거·300개 제한을 거친 박스를 `dataset_summary.results = [{image_id, boxes: [{x, y, width, height, score}]}]`(원본 픽셀, score 소수 3자리)에
  누적하며 `processed/labeled(박스 ≥1)/boxes/no_detections`를 갱신한다. boxes·status·revision은 불변이고, 프론트가 `GET /jobs/{id}`를 polling해
  `completed` 후 편집 중 초안에 합친다. 1장이라도 worker claim 간격(기본 `--poll 5`초)과 GPU 대기가 체감 지연의 대부분이다.
- 운영 보호: phase `labeling` 중에도 학습과 같은 heartbeat guard(FPS 하락·Preview 응답 없음·GPU reserve)가 `production_protection` 중지를 건다.

## 5. 운영 절차 (GPU 서버, 운영자 수행)

```bash
# 1) 비공개 .env에 토큰 추가 (각 24자 이상, 서로 다르게). 값은 문서·로그에 남기지 않는다.
#    TRAINING_ADMIN_TOKEN=...   TRAINING_WORKER_TOKEN=...
# 2) Backend만 재생성 (DeepStream retrace 서비스는 건드리지 않음)
sudo docker compose up -d --build --no-deps backend
# 3) Preview proxy 변경(preview_web.py) 반영: retrace 재시작은 영상이 잠시 끊기므로 운영 합의 후 수행
sudo docker compose restart retrace
# 4) 프론트 정적 산출물: frontend에서 npm run generate 후 .output/public/ 내용을 서버 frontend-dist/에 업로드
# 5) worker 시작 (host, 기존 venv 재사용, 별도 설치 없음)
cd training
RETRACE_TRAINING_WORKER_TOKEN='<TRAINING_WORKER_TOKEN 값>' nohup .venv/bin/python worker.py > runs/web-worker.log 2>&1 &
```

worker 옵션: `--backend`(기본 `http://127.0.0.1:8000`, `RETRACE_BACKEND_URL`), `--gpu 0`, `--poll 5`, `--work-dir runs/web`,
`--min-disk-mib 10000`, `--stop-grace 180`, `--dataloader-workers 4`. Backend 포트는 compose의
`BACKEND_BIND_IP`(기본 127.0.0.1):`BACKEND_PORT`(기본 8000)이므로 host worker가 loopback으로 닿는다.
상시 실행이 필요하면 systemd unit에서 `EnvironmentFile`을 root 전용 파일(권한 600)로 두고 `Restart=on-failure`로 실행한다.

첫 실행은 UI에서 smoke(epochs 1, 640/b4)로 GPU 여유와 Preview FPS를 확인한 뒤 본 학습을 한다. 실제 CCTV 데이터로
장시간 학습을 자동으로 시작하는 장치는 없다.

### 5-1. 배포 기능 활성화 (선택, 운영 합의 후)

```bash
# 1) Backend(/deployment, /deployments) 반영
sudo docker compose up -d --build --no-deps backend
# 2) 비공개 .env에 VEHICLE_DEPLOYMENT_ENABLED=1 추가 후 retrace 재생성(영상이 잠시 끊김, .env는 restart로 반영되지 않음)
sudo docker compose up -d --no-deps --no-build --force-recreate retrace
# 3) worker 재시작(새 worker.py/deployment.py/deploy_export.py 적재), 4) frontend-dist 갱신
```

활성화 직후 UI의 현재 적용 모델이 `기존 운영 모델`, 상태가 `배포 준비됨`인지 확인한다. 첫 배포는 차량이 있는 시간에 하고
`docker compose logs retrace`의 `Load new model`/engine build 로그, 전체 카메라 FPS, `/api/cameras/<camera>/vehicles`, 주차 점유를
배포 전과 비교한다. 이어서 수동 롤백도 한 번 확인한다. 1클래스(car) 모델은 COCO80 확인 후 따로 배포해 `/api/training/deployment`의
`active_class_map`이 `{"0": 2}`인지, 같은 카메라에서 차량 검출·점유가 유지되는지 보고 롤백까지 확인한다. 아직 실제 컨테이너에서 확인하지 않은
항목: DS 7.0 `config-file-path` OTA와 `model-updated` 경로 인자, 컨테이너 안 engine builder 프로세스, TensorRT 8.6의 remap 노드 parse,
검증 기준(load 90초/안정화 20초/측정 30–60초, 기준선 60초 중앙값, FPS 85%)의 적정성. 끄려면 `.env`에서 값을 지우고 retrace를 재생성한다. 꺼진 상태에서는 컨트롤러가 없어
`committed.json`을 읽지 않으므로 원래 운영 모델(`VEHICLE_INFER_CONFIG`)로 시작한다(전체 복구 경로). 다시 켜면 마지막 committed 모델로 시작한다.

## 6. 남은 일·한계

- 사용자 계정 인증은 없다. admin 토큰은 공유 비밀번호 수준이며 브라우저 sessionStorage에만 둔다. 계정/권한이 도입되면
  `require_admin`을 교체한다.
- worker는 GPU 1장·job 1개를 가정한다. 여러 worker를 띄워도 DB 제약으로 job은 1개만 실행된다.
- 웹 배포(§4-2)는 640px의 COCO80·1클래스(car) 모델을 지원한다. 960 등 다른 입력 크기는 training/README.md §8 수동 절차를 따른다.
- 배포 기능의 실제 GPU 검증(engine 생성, DS 7.0 config OTA, 실시간 추론 판정, 복구)은 아직 하지 않았다(Not verified).
  활성화 절차는 §5-1.
