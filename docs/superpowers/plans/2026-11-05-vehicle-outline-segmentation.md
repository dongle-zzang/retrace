# 주차장 차량 윤곽선(instance mask) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 주차장 카메라에 이식 seg nvinfer를 선택적으로 붙여 영상에 차량 윤곽선을 그리고 `/api/cameras/{id}/vehicles`에 `outline`을 싣는다.

**Architecture:** det nvinfer 뒤에 같은 ROI를 추론하는 seg nvinfer(gie 3)를 두고, seg src probe에서 det↔seg를 IoU로 짝지어 seg mask를 det 박스 격자의 테두리 mask로 옮긴다(`alloc_mask_array`). nvdsosd `display-mask`가 테두리를 그리고, tracker 뒤 `on_batch`가 같은 mask에서 numpy 행 스캔으로 최대 32점 윤곽을 만들어 metadata → backend → API로 보낸다. `VEHICLE_SEG_INFER_CONFIG`가 비면 seg element가 없다.

**Tech Stack:** Python 3.10(컨테이너)/3.11(CPU venv), numpy, PyDS 1.1.11, DeepStream 7.0 nvinfer/nvdsosd, FastAPI/Pydantic 2, ultralytics(training venv만).

**Spec:** [docs/superpowers/specs/2026-11-05-vehicle-outline-segmentation-design.md](../specs/2026-11-05-vehicle-outline-segmentation-design.md)

## Global Constraints

- seg 끔(`VEHICLE_SEG_INFER_CONFIG` 비어 있음/없음): element 순서·probe 위치·OSD 속성·det 생성 config 텍스트가 지금과 같다. JSON에는 `outline: null`만 추가된다.
- det nvinfer, `DeploymentController`의 OTA·롤백 동작, det `contract()`는 바꾸지 않는다.
- 상수: `VEHICLE_SEG_GIE_ID = 3`, mask 격자 칸 `OUTLINE_CELL = 4` mux px, `OUTLINE_MAX_POINTS = 32`(16행 × 2변), 짝짓기 `MASK_MATCH_IOU = 0.5`, 최소 채움 `MASK_MIN_FILL = 0.15`.
- `VEHICLE_OUTLINE_BOX_WIDTH`: 정수 0–3, 기본 1, 그 밖은 1. 윤곽 없는 차량은 3.
- 윤곽 좌표: mux pixel(bbox와 같음), 소수 첫째 자리 반올림, 화면 기준 시계 방향, 첫 점 반복 없음, 3–32점 또는 null.
- 새 dependency 없음(컨테이너는 numpy 1.26.1만 있음, cv2 없음). 모델·engine·clone은 `models/yolo11m-seg/`(gitignore)와 `.cache/`에만.
- mask 처리 예외는 차량 단위로 삼키고 probe 밖으로 내지 않는다(차량 probe 예외는 `fail()` → pipeline 전체 정지).
- 로그·문서에 경로 원문, RTSP, 서버 주소를 남기지 않는다. 사용자 메시지는 "details hidden" 형식을 따른다.
- 커밋은 모든 CPU 테스트가 끝난 뒤 Task 8에서 한다(사용자가 정한 순서).

## Review Focus

- backend poller는 `/metadata.json` frame 하나라도 검증에 실패하면 모든 카메라 frame을 버린다(`ingest.py` `poll_once`). 잘못된 `outline`(비유한 값, 2점, 33점, 문자열)은 frame 거부가 아니라 그 차량의 `outline=None`이어야 한다. → Task 4 테스트.
- nvinfer/파서가 `mask_params.size`와 `width*height`가 맞지 않는 mask를 줄 수 있다(외부 파서). reshape 실패는 그 짝만 건너뛰고 다른 짝은 처리돼야 한다. → Task 3 테스트.
- 화면 가장자리 차량: det 박스가 seg 박스보다 크거나 어긋나면 격자 칸이 seg mask 밖을 가리킨다. 범위 밖 칸은 비어야 하고 예외가 없어야 한다. → Task 1 테스트.
- 아주 작은 박스(폭·높이 < 4 px, 격자 1×1): 윤곽은 None이고 예외가 없어야 한다. → Task 1 테스트.
- seg를 켠 pipeline의 비주차장 카메라 frame과 seg 객체만 있고 det 차량이 없는 frame: mask를 붙이지 않고 seg 객체는 모두 지워져야 한다(tracker로 가면 안 됨). → Task 3 테스트.

---

### Task 1: mask 격자·윤곽 numpy helper

**Files:**
- Modify: `vehicle_detection.py` (상수와 함수 추가, `import numpy as np`)
- Test: `tests/test_vehicle_detection.py`

**Interfaces:**
- Produces:
  - `outline_from_grid(grid: np.ndarray, left: float, top: float, width: float, height: float) -> tuple[tuple[float, float], ...] | None` — `grid`는 bool (h, w)이고 박스 전체를 덮는다.
  - `mask_grid(mask_params) -> np.ndarray | None` — `get_mask_array()`를 `(height, width)`로 reshape해 `> threshold` 복사. size 0이면 None. 크기 불일치는 예외(호출자가 처리).
  - `resample_mask(seg_grid: np.ndarray, seg_rect: tuple, det_rect: tuple) -> np.ndarray` — rect는 `(left, top, width, height)`. 결과 크기 `(max(1, ceil(h/4)), max(1, ceil(w/4)))`, 칸 중심 최근접, seg 박스 밖 False.
  - `ring_mask(grid: np.ndarray) -> np.ndarray` — 4-이웃이 모두 True인 내부를 뺀 테두리.
  - `vehicle_outline(obj) -> tuple | None` — `mask_grid(obj.mask_params)` → `outline_from_grid(..., obj.rect_params 값)`. 어떤 예외도 None.

- [ ] **Step 1: Write the failing tests** (`OutlineTests` 클래스, `import numpy as np`)

```python
def car_grid():  # 50 x 75, oblique body + roof, like the design prototype
    yy, xx = np.mgrid[0:50, 0:75]
    car = (xx > 5 + yy * .3) & (xx < 60 + yy * .25) & (yy > 8) & (yy < 48)
    return car | (((xx - 35) ** 2 / 400 + (yy - 14) ** 2 / 60) < 1)

def test_outline_follows_mask_within_box(self):
    grid, rect = car_grid(), (800.0, 500.0, 300.0, 200.0)
    points = outline_from_grid(grid, *rect)
    self.assertLessEqual(len(points), 32); self.assertGreaterEqual(len(points), 3)
    self.assertNotEqual(points[0], points[-1])
    self.assertTrue(all(800 <= x <= 1100 and 500 <= y <= 700 for x, y in points))
    self.assertTrue(all(round(v, 1) == v for p in points for v in p))
    self.assertGreater(signed_area(points), 0)          # clockwise on screen (y down) = positive shoelace
    self.assertGreaterEqual(polygon_mask_iou(points, grid, rect), 0.9)

def test_ring_outline_equals_filled_outline(self):
    self.assertEqual(outline_from_grid(ring_mask(car_grid()), 0, 0, 300, 200),
                     outline_from_grid(car_grid(), 0, 0, 300, 200))

def test_empty_single_row_and_tiny_masks_have_no_outline(self):
    self.assertIsNone(outline_from_grid(np.zeros((5, 5), bool), 0, 0, 20, 20))
    one = np.zeros((5, 5), bool); one[2, 1:4] = True
    self.assertIsNone(outline_from_grid(one, 0, 0, 20, 20))
    self.assertIsNone(outline_from_grid(resample_mask(np.ones((160, 160), bool), (0, 0, 3, 3), (0, 0, 3, 3)), 0, 0, 3, 3))

def test_resample_maps_cells_and_leaves_outside_seg_box_empty(self):
    seg = np.ones((160, 160), bool)
    grid = resample_mask(seg, (100.0, 100.0, 40.0, 40.0), (80.0, 100.0, 80.0, 40.0))  # det wider, edge vehicle
    self.assertEqual(grid.shape, (10, 20))
    self.assertFalse(grid[:, :5].any()); self.assertFalse(grid[:, 15:].any()); self.assertTrue(grid[:, 5:15].all())

def test_vehicle_outline_reads_mask_params_and_never_raises(self):
    data = car_grid().astype(np.float32).ravel()
    obj = NS(mask_params=NS(size=data.nbytes, width=75, height=50, threshold=.5, get_mask_array=lambda: data),
             rect_params=NS(left=800.0, top=500.0, width=300.0, height=200.0))
    self.assertEqual(vehicle_outline(obj), outline_from_grid(car_grid(), 800.0, 500.0, 300.0, 200.0))
    obj.mask_params.width = 74          # size mismatch from a broken parser
    self.assertIsNone(vehicle_outline(obj))
    obj.mask_params.size = 0
    self.assertIsNone(vehicle_outline(obj))
```

`signed_area`(신발끈 공식)와 `polygon_mask_iou`(칸 중심의 짝홀 판정, 설계 프로토타입과 같음)는 테스트 파일의 작은 helper로 둔다.

- [ ] **Step 2: Run to verify failure**

Run: `/tmp/retrace-shared-venv/bin/python -m unittest discover -s tests -p 'test_vehicle_detection.py'`
Expected: ImportError (`outline_from_grid` 없음)

- [ ] **Step 3: Implement the five helpers and constants in `vehicle_detection.py`**

행 스캔(설계 §4-6): mask가 있는 행 `rows`가 2개 미만이면 None. `pick = rows[linspace(0, len(rows)-1, min(16, len(rows))).round()]`. 행마다 `lo = argmax`, `hi = w - argmax(reversed)`(바깥 경계). x는 `left + col * width / w`. y는 첫 행 `top + pick[0]*sy`, 마지막 행 `top + (pick[-1]+1)*sy`, 나머지는 칸 중심. 오른쪽 변 위→아래, 왼쪽 변 아래→위. 점이 3개 미만이면 None. `vehicle_outline`은 `except Exception: return None`.

- [ ] **Step 4: Run to verify pass**

Run: 같은 명령. Expected: OK

### Task 2: seg config 준비, 박스 두께 설정, 이식 출처 상태

**Files:**
- Modify: `vehicle_detection.py` (`prepare_vehicle_detector`의 공통 부분을 helper로 추출, 새 함수·dataclass)
- Test: `tests/test_vehicle_detection.py`

**Interfaces:**
- Consumes: 없음(Task 1과 독립)
- Produces:
  - `@dataclass(frozen=True) class VehicleSegmenter: cameras: frozenset[str]; config_path: str; vehicle_class_ids: frozenset[int]; det_weights_sha256: str | None = None; box_width: int = 1`
  - `prepare_vehicle_segmenter(detector, cache_dir, environ=None) -> VehicleSegmenter | None` — env `VEHICLE_SEG_INFER_CONFIG`가 비면 None. 실패하면 `print("Vehicle segmentation unavailable; details hidden; outlines disabled", flush=True)` 후 None(예외를 내지 않음).
  - `outline_box_width(environ) -> int`
  - `graft_status(segmenter, det_config_path) -> str | None` — segmenter None → None. `Path(det_config_path).parent / "manifest.json"`의 `weights_sha256`와 `segmenter.det_weights_sha256` 비교: 둘 다 있고 같으면 `"match"`, 다르면 `"stale"`, 하나라도 없거나 읽기 실패면 `"unknown"`.
  - 내부 helper `_generate_config(path, cache_dir, namespace, prefix, overrides, required=(), threshold_override=None) -> tuple[str, int, list[int]]` (생성 config 경로, car id, 차량 class id들). det는 namespace `"vehicle"`/prefix `"vehicle"`, seg는 `"vehicle-seg"`/`"vehicle_seg"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_detector_generated_config_is_unchanged_by_refactor(self):
    # Expected text captured from the pre-refactor implementation with the existing fixture.
    self.assertEqual(generated_detector_text(), DETECTOR_GOLDEN)

def test_segmenter_disabled_without_env_and_forces_secondary_instance_seg(self):
    detector = fixture_detector()                       # cameras {'p1','p2'}
    self.assertIsNone(prepare_vehicle_segmenter(detector, cache, {}))
    seg = prepare_vehicle_segmenter(detector, cache, {'VEHICLE_SEG_INFER_CONFIG': str(seg_cfg)})
    props = read(seg.config_path)['property']
    for key, value in {'process-mode': '2', 'network-type': '3', 'output-instance-mask': '1', 'cluster-mode': '4',
                       'gie-unique-id': '3', 'operate-on-gie-id': '90', 'operate-on-class-ids': '0',
                       'batch-size': '2', 'interval': '0', 'gpu-id': '0'}.items():
        self.assertEqual(props[key], value)
    self.assertEqual(props['filter-out-class-ids'], non_vehicle_ids)
    self.assertIn('vehicle-seg', seg.config_path)
    self.assertEqual((seg.cameras, seg.box_width, seg.det_weights_sha256), (detector.cameras, 1, None))

def test_segmenter_reads_graft_manifest_and_rejects_bad_config_without_raising(self):
    write_json(seg_dir / 'manifest.json', {'det_weights_sha256': 'a' * 64})
    self.assertEqual(prepare_vehicle_segmenter(...).det_weights_sha256, 'a' * 64)
    for broken in ('parse-bbox-instance-mask-func-name', 'custom-lib-path'):   # removed from the config
        with patch('builtins.print') as printed:
            self.assertIsNone(prepare_vehicle_segmenter(...))
        self.assertIn('Vehicle segmentation unavailable', printed.call_args.args[0])

def test_outline_box_width(self):
    for raw, expected in ((None, 1), ('0', 0), ('3', 3), ('4', 1), ('-1', 1), ('abc', 1), ('1.5', 1)):
        self.assertEqual(outline_box_width({} if raw is None else {'VEHICLE_OUTLINE_BOX_WIDTH': raw}), expected)

def test_graft_status(self):
    seg = VehicleSegmenter(frozenset({'p1'}), 'unused', frozenset({2}), det_weights_sha256='a' * 64)
    self.assertIsNone(graft_status(None, det_cfg))
    self.assertEqual(graft_status(seg, det_cfg), 'unknown')                 # no manifest beside det config
    write_json(det_cfg.parent / 'manifest.json', {'weights_sha256': 'a' * 64})
    self.assertEqual(graft_status(seg, det_cfg), 'match')
    write_json(det_cfg.parent / 'manifest.json', {'weights_sha256': 'b' * 64})
    self.assertEqual(graft_status(seg, det_cfg), 'stale')
    self.assertEqual(graft_status(replace(seg, det_weights_sha256=None), det_cfg), 'unknown')
```

`DETECTOR_GOLDEN`: Step 1에서 **현재 코드**로 기존 fixture(`test_config_is_camera_roi_detector_and_reuses_native_cache`와 같은 파일 구성, engine runtime은 `patch('vehicle_detection.prepare_engine_cache', ...)` 대신 기존 테스트가 쓰는 방식 그대로)를 돌려 얻은 텍스트를 붙여 넣는다. 임시 경로는 placeholder로 바꿔 비교한다.

- [ ] **Step 2: Run to verify failure** — golden 테스트만 통과하고 나머지는 ImportError.
- [ ] **Step 3: Extract `_generate_config` (det golden이 계속 통과하는지 먼저 확인), then add `VehicleSegmenter`, `prepare_vehicle_segmenter`, `outline_box_width`, `graft_status`.**
  seg 강제 값은 det와 같은 키 순서에 `network-type=3`, `gie-unique-id=3`, `output-instance-mask=1`, `cluster-mode=4`를 더한다. `required`에 없으면 `ValueError`. `VEHICLE_MIN_CONFIDENCE`는 det에만 쓴다. 출처는 원본 seg config 디렉터리의 `manifest.json`에서 `det_weights_sha256`를 읽는다(없으면 None).
- [ ] **Step 4: Run to verify pass** — `test_vehicle_detection.py` OK, 그리고 `-p 'test_vehicle_deployment.py'`, `-p 'test_pgie_cache.py'` OK.

### Task 3: det↔seg 짝짓기·mask 이식과 seg 객체 제거

**Files:**
- Modify: `vehicle_detection.py` (`attach_vehicle_masks` 추가, `prepare_vehicle_tracks` 수정)
- Test: `tests/test_vehicle_detection.py`

**Interfaces:**
- Consumes: Task 1 `mask_grid`, `resample_mask`, `ring_mask`, 상수; Task 2 `VehicleSegmenter`
- Produces:
  - `attach_vehicle_masks(batch, detector, segmenter, camera_for_frame, pyds) -> int` — 실패한 짝 수를 돌려준다. 예외를 내지 않는다. 주차장 카메라 frame에서 det(gie 2, `detector.vehicle_class_ids`)와 seg(gie 3, `segmenter.vehicle_class_ids`, `mask_params.size > 0`)를 IoU 내림차순 1:1로 짝짓는다(≥ 0.5). 짝마다 `resample_mask` → 채움 < 0.15면 건너뜀 → `ring_mask` → det `mask_params.width/height/threshold(0.5)` 설정 → `alloc_mask_array()`가 준 view에 float 1/0 기록.
  - `prepare_vehicle_tracks(...)`: 기존 동작 + gie 3 객체 제거, ROI(gie 90)는 제거 목록 맨 뒤.

- [ ] **Step 1: Write the failing tests** (fake pyds: `NvDsFrameMeta/NvDsObjectMeta.cast` 항등, `nvds_remove_obj_meta_from_frame`가 목록에 기록; fake `mask_params`는 `alloc_mask_array`가 `np.zeros(width*height, np.float32)`를 만들어 보관하고 돌려준다)

```python
def test_masks_attach_to_best_iou_det_one_to_one(self):
    det_a, det_b = det(800, 500, 300, 200), det(1200, 500, 300, 200)
    seg_a, seg_far = seg(805, 505, 295, 195, full=True), seg(100, 100, 50, 50, full=True)
    errors = attach_vehicle_masks(batch_with(parking=[det_a, det_b, seg_a, seg_far]), DETECTOR, SEGMENTER, cam, pyds)
    self.assertEqual(errors, 0)
    self.assertEqual((det_a.mask_params.width, det_a.mask_params.height), (75, 50))
    self.assertEqual(det_a.mask_params.threshold, .5)
    grid = det_a.mask_params.allocated.reshape(50, 75)
    self.assertTrue(grid[0].all()); self.assertFalse(grid[25, 30])          # ring: border set, interior empty
    self.assertIsNone(getattr(det_b.mask_params, 'allocated', None))        # no seg with IoU >= 0.5

def test_low_fill_and_non_parking_frames_get_no_mask(self):
    sparse = seg(800, 500, 300, 200, fill=.1)
    attach_vehicle_masks(batch_with(parking=[d := det(800, 500, 300, 200), sparse], lobby=[l := det(0, 0, 300, 200), seg(0, 0, 300, 200, full=True)]), ...)
    self.assertIsNone(getattr(d.mask_params, 'allocated', None)); self.assertIsNone(getattr(l.mask_params, 'allocated', None))

def test_broken_pair_is_skipped_and_others_still_attach(self):
    bad = seg(800, 500, 300, 200, full=True); bad.mask_params.width = 7     # size != width*height
    good_det, good_seg = det(1200, 500, 300, 200), seg(1200, 500, 300, 200, full=True)
    errors = attach_vehicle_masks(batch_with(parking=[det(800, 500, 300, 200), bad, good_det, good_seg]), ...)
    self.assertEqual(errors, 1); self.assertIsNotNone(good_det.mask_params.allocated)

def test_prepare_removes_seg_objects_and_roi_last(self):
    # parking frame: ROI, det car (parent ROI), seg car, seg-only frame without det
    prepare_vehicle_tracks(batch, DETECTOR, cam, pyds)
    self.assertIn(seg_car, removed); self.assertEqual(removed[-1], roi); self.assertIsNone(car.parent)
```

- [ ] **Step 2: Run to verify failure** — ImportError / assertion on `removed`.
- [ ] **Step 3: Implement `attach_vehicle_masks` and the `prepare_vehicle_tracks` change.** 짝마다 `try/except Exception: errors += 1; continue`. IoU는 `rect_params` 값으로 계산한다.
- [ ] **Step 4: Run to verify pass** — `test_vehicle_detection.py` OK.

### Task 4: metadata `outline`과 backend 계약

**Files:**
- Modify: `person_metadata.py` (`VehicleMetadata.outline`, `vehicle_from_object(..., outline=None)`)
- Modify: `backend/app/schemas.py` (`VehicleIn.outline`)
- Test: `tests/test_vehicle_detection.py`, `backend/tests/test_occupancy.py`, `backend/tests/test_preview_contract.py`

**Interfaces:**
- Produces:
  - `VehicleMetadata(..., outline: tuple[tuple[float, float], ...] | None = None)`; `to_dict()`의 `"outline"`은 `[[x, y], ...]` 또는 None(JSON null).
  - `vehicle_from_object(camera_id, timestamp, obj, class_id, outline=None) -> VehicleMetadata`
  - `VehicleIn.outline: list[tuple[float, float]] | None = None` — `field_validator("outline", mode="before")`: list/tuple, 3–32개, 각 원소가 숫자 2개이고 유한하면 통과, 그 밖은 None.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vehicle_detection.py
def test_detached_vehicle_serializes_outline(self):
    vehicle = vehicle_from_object('parking', TS, obj, 0, outline=((1.0, 2.0), (3.0, 4.0), (5.0, 6.5)))
    frame = FrameMetadata(0, 'parking', 1, TS, None, True, (), vehicles=(vehicle,)).to_dict()
    self.assertEqual(json.loads(json.dumps(frame))['vehicles'][0]['outline'], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.5]])
    self.assertIsNone(vehicle_from_object('parking', TS, obj, 0).to_dict()['outline'])

# backend/tests/test_occupancy.py (기존 service/observation helper 사용)
def test_vehicles_api_passes_outline_and_nulls_invalid_values(service, monkeypatch):
    square = [[100.0, 200.0], [400.0, 200.0], [400.0, 500.0], [100.0, 500.0]]
    vehicles = [vehicle(track=1) | {"outline": square}, vehicle(track=2),            # missing -> null
                vehicle(track=3) | {"outline": [[1, 2], [3, float("nan")], [5, 6]]},
                vehicle(track=4) | {"outline": [[1, 2]] * 33}, vehicle(track=5) | {"outline": [[1, 2], [3, 4]]},
                vehicle(track=6) | {"outline": "bad"}]
    # poll once like test_polling_db_events_live_vehicles_and_unknown_failure, NaN sent as JSON NaN
    body = client.get("/api/cameras/first/vehicles").json()["vehicles"]
    assert [v["outline"] for v in body] == [square, None, None, None, None, None]
    assert all("bbox" in v for v in body) and len(body) == 6        # one bad outline never drops the frame
```

`test_preview_contract.py`: 기존 `/metadata.json` 테스트의 frame에 outline 있는 차량 하나를 넣고 응답에 `[[x, y], ...]`가 그대로 있는지 단언한다.

- [ ] **Step 2: Run to verify failure**
  Run: `/tmp/retrace-shared-venv/bin/python -m unittest discover -s tests -p 'test_vehicle_detection.py'` 및 `/tmp/retrace-shared-venv/bin/python -m pytest -q backend/tests/test_occupancy.py backend/tests/test_preview_contract.py`
  Expected: TypeError(`outline` 인자) / KeyError `outline`
- [ ] **Step 3: Implement the dataclass field, the parameter and the lenient validator.**
- [ ] **Step 4: Run to verify pass** — 위 두 명령 OK, `python -m pytest -q backend/tests` 전체 OK(166 + 새 테스트).

### Task 5: preview pipeline 통합

**Files:**
- Modify: `preview.py` (`run()`의 seg 준비, `build_output(..., display_mask=False)`, `run_shared_pipeline`의 seg element·probe·진단·시작 로그, `on_batch`)
- Test: `tests/test_camera_shared_pipeline.py`

**Interfaces:**
- Consumes: Task 1 `vehicle_outline`; Task 2 `prepare_vehicle_segmenter`, `graft_status`, `VEHICLE_SEG_GIE_ID`; Task 3 `attach_vehicle_masks`, `prepare_vehicle_tracks`; Task 4 `vehicle_from_object(..., outline=)`; Task 6 `DeploymentController(..., segmenter=)`
- Produces:
  - `args.vehicle_segmenter: VehicleSegmenter | None` (`run()`에서 det가 준비된 뒤 `prepare_vehicle_segmenter(args.vehicle_detector, cache)`, det가 없으면 None)
  - `build_output(index, args, store, fail, display_mask=False)` — `osd.set_property("display-mask", display_mask)`
  - pipeline element 이름 `vehicle-segmentation`(factory `nvinfer`), timing stage `vehicle-seg-batch`
  - 시작 로그 `vehicle_segmentation graft=<status>`; `stale`이면 `WARNING vehicle_segmentation graft=stale; re-run the graft for the active detector; details hidden`
  - 진단 줄 `vehicle_masks {"cameras": {...}, "probe_ms_per_s": x, "probe_max_ms": y, "errors": e, "graft": s}` (seg 켬 + `--diagnostics`일 때만)

배치(설계 §4-3): det sink `add_vehicle_rois`; det src `observe_batch`(배포 켬) + seg 끔일 때만 `prepare_vehicle_tracks`; seg src `attach_vehicle_masks` → `prepare_vehicle_tracks`. `on_batch`에서 seg 켬이면 `outline = vehicle_outline(obj)`, `border_width = 3 if outline is None else segmenter.box_width`. 시작 로그의 det config는 배포 켬이면 `ROOT / deployment.active["config"]`, 아니면 `vehicle_detector.config_path`. 진단의 probe 시간은 `attach_vehicle_masks` 호출과 윤곽 계산을 `time.perf_counter()`로 감싸 `counts_lock` 아래 합산한다.

- [ ] **Step 1: Write the failing tests** — harness `construct(..., segmenter_enabled=False)` 인자 추가. seg 켬이면 두 번째 config 파일과 `args.vehicle_segmenter = NS(cameras=frozenset({'camera_0', 'camera_1'}), config_path=..., vehicle_class_ids=frozenset({0}), det_weights_sha256=None, box_width=1)`. harness의 `build_output` lambda는 `(index, *args, display_mask=False)`로 받아 `display_masks[index]`에 기록한다. namespace에 Task 1–3 함수, `graft_status`, `ROOT`, `time.perf_counter`를 넣는다.

```python
def test_segmentation_off_keeps_pipeline_identical(self):
    result = self.construct(vehicle_enabled=True)
    self.assertNotIn('vehicle-segmentation', result['elements'])
    self.assertEqual(result['display_masks'], {0: False, 1: False, 2: False})
    self.assertIn(('vehicle-inference', 'src', 'vehicle-tracker', 'sink'), result['links'])
    self.assertNotIn('vehicle_segmentation', result['captured'])

def test_segmentation_on_inserts_seg_after_detector_and_moves_track_preparation(self):
    result = self.construct(vehicle_enabled=True, segmenter_enabled=True, count=3)
    seg = result['elements']['vehicle-segmentation']
    self.assertEqual(seg.props['batch-size'], 2)
    for link in (('vehicle-inference', 'src', 'vehicle-segmentation', 'sink'),
                 ('vehicle-segmentation', 'src', 'vehicle-tracker', 'sink')):
        self.assertIn(link, result['links'])
    self.assertIn(('vehicle-segmentation', 'src'), result['probes'])
    self.assertEqual(result['display_masks'], {0: True, 1: True, 2: False})   # camera_2 is not a parking camera
    self.assertIn('vehicle_segmentation graft=unknown', result['captured'])
```

probe 순서는 `vehicle_probe`에 넘긴 단계 목록을 harness가 기록하게 해서 단언한다: det src 단계에 `prepare_vehicle_tracks`가 없고 seg src 단계가 `[attach_vehicle_masks, prepare_vehicle_tracks]`. `build_output` 단독 테스트(`test_camera_shared_pipeline.py` 185행 근처)에 `display-mask` 기본 False, `display_mask=True`면 True 단언을 더한다. `on_batch` 테두리·윤곽: 기존 AST 추출 방식이 없으므로, 공유 pipeline harness의 tracker src probe callback을 fake batch(차량 2대: mask 있음/없음)로 한 번 호출해 `border_width`가 1/3이고 저장된 frame의 `outline`이 tuple/None인지 단언한다.

- [ ] **Step 2: Run to verify failure** — `-p 'test_camera_shared_pipeline.py'`: KeyError `vehicle-segmentation` 등.
- [ ] **Step 3: Implement in `preview.py`.** `vehicle_probe`는 단계 함수 목록을 받아 차례로 부르게 바꾸고(예외 시 기존 `fail("Vehicle metadata preparation failed; details hidden")` 유지), `attach_vehicle_masks`의 오류 수는 진단 카운터에 더한다. `make_element("nvinfer", "vehicle-segmentation")` + `configure_pgie(seg, segmenter.config_path, len(segmenter.cameras))`.
- [ ] **Step 4: Run to verify pass** — `-p 'test_camera_*.py'` 전체 OK(기존 92 + 새 테스트), `-p 'test_rtsp_inputs.py'`, `-p 'test_preview_capture.py'`, `-p 'test_preview_realtime.py'` OK.

### Task 6: 배포 상태의 `segmentation_graft`

**Files:**
- Modify: `vehicle_deployment.py` (`PUBLIC`, `DeploymentController.__init__(..., segmenter=None)`, `publish()`)
- Test: `tests/test_vehicle_deployment.py`

**Interfaces:**
- Consumes: Task 2 `graft_status`, `VehicleSegmenter`
- Produces: `status.json`과 `public_status()`의 `segmentation_graft: "match" | "stale" | "unknown" | None`. `publish()`가 `graft_status(self.segmenter, local_path(self.active['config']))`로 계산한다.

- [ ] **Step 1: Write the failing tests**

```python
def test_segmentation_graft_follows_the_active_detector(self):
    controller = self.controller()                                       # no segmenter
    self.assertIsNone(vd.public_status(self.directory, wall=lambda: self.wall[0])['segmentation_graft'])
    segmenter = NS(det_weights_sha256='a' * 64)
    controller = self.controller(segmenter=segmenter)                    # original model: no manifest
    self.assertEqual(vd.read_json(self.directory / 'status.json')['segmentation_graft'], 'unknown')
    self.deploy(controller, weights_sha256='a' * 64); self.advance(controller, vd.SETTLE_SECONDS + vd.MEASURE_SECONDS)
    self.assertEqual(controller.state['phase'], 'completed')
    self.assertEqual(vd.read_json(self.directory / 'status.json')['segmentation_graft'], 'match')
    self.deploy(controller, model='m2', job='22222222-2222-4222-8222-222222222222', weights_sha256='b' * 64)
    self.advance(controller, vd.SETTLE_SECONDS + vd.MEASURE_SECONDS)
    self.assertEqual(vd.public_status(self.directory, wall=lambda: self.wall[0])['segmentation_graft'], 'stale')
```

`self.controller(...)`에 `segmenter=None` 인자를 더해 그대로 넘긴다. `deploy()` 이후 commit까지의 advance 길이는 기존 `test_deploy_hot_updates_config_verifies_inference_and_commits`와 같게 맞춘다.

- [ ] **Step 2: Run to verify failure** — `-p 'test_vehicle_deployment.py'`: TypeError `segmenter`.
- [ ] **Step 3: Implement.** `segmenter=None` keyword를 끝에 추가하고 `publish()`의 `state.update`에 한 키를 더하며 `PUBLIC`에 `'segmentation_graft'`를 추가한다.
- [ ] **Step 4: Run to verify pass** — `-p 'test_vehicle_deployment.py'` OK.

### Task 7: 모델 준비 도구

**Files:**
- Create: `training/graft_seg.py`, `training/tests/test_graft_seg.py`
- Create: `tools/build_vehicle_seg_engine.py`
- Modify: `tools/build_vehicle_engine.py` (PAUSED 직렬화 부분을 `serialize_engine(config_path: Path, timeout: int = BUILD_TIMEOUT) -> bool`로 추출, `main()` 동작은 같게)
- Create: `tools/deepstream-yolo-seg-a8ec7ec.patch`
- Modify: `.gitignore` (`models/yolo11m-seg/`)

**Interfaces:**
- Consumes: Task 2 `prepare_vehicle_detector`, `prepare_vehicle_segmenter`, `graft_status`; `vehicle_deployment.digest`, `write_json`; `camera_config.load_public_cameras`
- Produces:
  - `graft_state(seg_state: dict, det_state: dict) -> dict` — det 키가 seg에 없거나 모양이 다르면 `ValueError("det head layout differs from seg")`. seg dtype으로 변환해 덮어쓴 새 dict.
  - `write_manifest(out_dir: Path, det: Path, seg: Path) -> dict` — `{"det_weights_sha256": digest(det), "seg_weights_sha256": digest(seg)}`를 `out_dir/manifest.json`에 쓰고 돌려준다.
  - CLI `training/.venv/bin/python training/graft_seg.py --seg <yolo11m-seg.pt> --det <best.pt> --out-dir models/yolo11m-seg` → `graft.pt`, `manifest.json`, det sha256 출력. ultralytics `YOLO`는 `main()` 안에서만 import한다(테스트는 torch만 필요).
  - CLI `python3 tools/build_vehicle_seg_engine.py` (컨테이너) → enabled 카메라로 det·seg 준비, `serialize_engine`, `vehicle_segmentation graft=<status>`와 `Engine ready` 출력. 실패하면 0이 아닌 코드.

- [ ] **Step 1: Write the failing test** (`training/tests/test_graft_seg.py`, `test_deploy_export.py`처럼 `sys.path`에 training과 repo root 추가)

```python
def test_graft_copies_det_tensors_and_keeps_seg_only_tensors(self):
    seg = {'a': torch.zeros(2, dtype=torch.float16), 'mask': torch.ones(3)}
    det = {'a': torch.tensor([1.0, 2.0])}
    out = graft_state(seg, det)
    self.assertTrue(torch.equal(out['a'], torch.tensor([1.0, 2.0], dtype=torch.float16)))
    self.assertTrue(torch.equal(out['mask'], seg['mask'])); self.assertEqual(seg['a'].sum(), 0)   # input untouched

def test_graft_rejects_a_different_head_layout(self):
    with self.assertRaises(ValueError):
        graft_state({'a': torch.zeros(2)}, {'a': torch.zeros(3)})
    with self.assertRaises(ValueError):
        graft_state({'a': torch.zeros(2)}, {'b': torch.zeros(2)})

def test_manifest_records_both_hashes(self):
    manifest = write_manifest(out, det_file, seg_file)
    self.assertEqual(json.loads((out / 'manifest.json').read_text()), manifest)
    self.assertEqual(manifest['det_weights_sha256'], hashlib.sha256(det_file.read_bytes()).hexdigest())
```

- [ ] **Step 2: Run to verify failure** — `cd training && .venv/bin/python -m unittest discover -s tests -p 'test_graft_seg.py'`: ImportError.
- [ ] **Step 3: Implement `graft_seg.py`, extract `serialize_engine`, write `build_vehicle_seg_engine.py`.**
- [ ] **Step 4: Create the patch** in a scratch clone of DeepStream-Yolo-Seg at `a8ec7ec`:
  1. `nvdsinfer_custom_impl_Yolo_seg/nvdsparseseg_Yolo.cpp`: `addBBoxProposal`이 `bool`을 반환(1 미만이면 false, 끝에서 true)하고 호출부가 false면 `continue`(mask 할당 전).
  2. `utils/export_yolo11_seg.py`: `selected_boxes = boxes[batch_index, box_index]` 다음 줄에서 x는 `[0, w_protos * 4]`, y는 `[0, h_protos * 4]`로 clamp(박스는 xyxy, `box_coding=0`).
  `git diff > tools/deepstream-yolo-seg-a8ec7ec.patch`, 같은 clone에서 `git apply --check`로 재적용 확인.
- [ ] **Step 5: Run to verify pass** — Step 2 명령 OK, `python3 -m py_compile tools/build_vehicle_seg_engine.py tools/build_vehicle_engine.py`, `-p 'test_vehicle_deployment.py'` OK(빌더 추출 회귀).

### Task 8: 문서, 인수인계, 전체 검증, 커밋

**Files:**
- Modify: `docs/preview-realtime.md`, `backend/README.md`, `docs/parking-occupancy.md`, `docs/agent/architecture.md`, `docs/operations.md`, `training/README.md`, `docs/training.md`, `docs/agent/testing.md`, `docs/project-status.md`, `.ai/WORK.md`, 설계 문서 상태 줄

- [ ] **Step 1: 문서 갱신** — 설계 §8 목록. API 계약(§4-7)과 알려진 한계(행 스캔이 조각 사이를 잇고 가린 물체를 가로지를 수 있음, 하단 경계 의미)는 `backend/README.md` API 절과 `docs/preview-realtime.md`의 `/metadata.json` 설명에 같은 문장으로 둔다. invalid outline은 null이 된다는 점도 적는다. 운영 절차(G1–G6 명령)는 `docs/operations.md`에 두고 나머지 문서는 링크한다.
- [ ] **Step 2: 설계 문서 갱신** — 상태 줄을 "구현됨, GPU 검증 대기"로 바꾸고 구현 중 달라진 점을 "구현 메모"로 남긴다.
- [ ] **Step 3: 전체 CPU 검증**

```bash
P=/tmp/retrace-shared-venv/bin/python
$P -m unittest discover -s tests -p 'test_camera_*.py'
$P -m unittest discover -s tests -p 'test_vehicle_detection.py'
$P -m unittest discover -s tests -p 'test_vehicle_deployment.py'
$P -m unittest discover -s tests -p 'test_rtsp_inputs.py'
$P -m unittest discover -s tests -p 'test_pgie_cache.py'
$P -m unittest discover -s tests -p 'test_preview_realtime.py'
$P -m unittest discover -s tests -p 'test_preview_capture.py'
$P -m unittest discover -s tests            # 기존 오류 1(GPU 전용 test_preview.py import)만 허용
$P -m pytest -q backend/tests
(cd training && .venv/bin/python -m unittest discover -s tests)
git diff --check
git grep -nE 'rtsp://|[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' -- ':!*.lock' ':!package-lock.json'   # 새 내부 값 없음
```

Expected: 모두 OK, 전체 discover는 기존 오류 1개만.

- [ ] **Step 4: `.ai/WORK.md`** — 두 번째 작업 단락을 "구현·main 커밋, GPU 검증(G1–G6) 대기"로 바꾸고 측정 기록 자리와 다음 행동(사용자 실행 명령)을 남긴다.
- [ ] **Step 5: 커밋** (`docs/report_20261020.md`는 사용자 파일이라 넣지 않는다)

```bash
git add vehicle_detection.py person_metadata.py preview.py vehicle_deployment.py backend/app/schemas.py \
  tools/build_vehicle_engine.py tools/build_vehicle_seg_engine.py tools/deepstream-yolo-seg-a8ec7ec.patch \
  training/graft_seg.py training/tests/test_graft_seg.py tests/test_vehicle_detection.py \
  tests/test_camera_shared_pipeline.py tests/test_vehicle_deployment.py backend/tests/test_occupancy.py \
  backend/tests/test_preview_contract.py .gitignore
git commit -m "feat: 주차장 카메라 차량 윤곽선(이식 seg nvinfer)과 vehicles API outline"
git add docs backend/README.md training/README.md .ai/WORK.md
git commit -m "docs: 차량 윤곽선 설계·구현 계획·운영 절차와 API 계약"
```
