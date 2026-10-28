# ReTrace 현재 확정 상태 (2026-10-28 기준)

> 근거: 프로젝트 스레드 기록, 공유 폴더의 AI 문서 백업(`/mnt/project-files/ai-docs-backup/`, 서버 WORK.md는 2026-10-21까지 기록), GitHub 저장소 git log.
> 대화에서 확인되지 않은 내용은 **확인 필요**로 표시했습니다.

---

## 1. 프로젝트 개요

**목적**
- 여러 대의 RTSP CCTV 카메라 영상을 웹에서 모니터링하고, 주차장 카메라에서 차량을 검출해 주차면 점유 상태를 보여주는 웹 앱.
- 초기 README는 "사람 검출·추적"으로 소개했지만, 2026-10-13에 PeopleNet을 제거했습니다. 지금은 YOLO11m으로 주차장 카메라의 차량만 검출합니다.
- 최종 목표 문서는 아직 없습니다 (확인 필요).

**기술 스택**

| 영역 | 내용 |
| --- | --- |
| 영상 파이프라인 (루트) | Python, DeepStream 7.0 (`preview.py`, `app.py`), nvinfer SGIE (YOLO11m), NvDCF tracker, TensorRT |
| 영상 전송 | 모든 카메라의 JPEG 프레임을 단일 `/ws` WebSocket으로 브라우저에 전송 (MJPEG over WS) |
| 백엔드 (`backend/`) | FastAPI, PostgreSQL |
| 배포 | Docker Compose (`retrace`, `backend`, `postgres` 서비스) |
| 프론트엔드 (`frontend/`) | Nuxt 4.4.8, `ssr:false`(CSR), Vue 3 `<script setup>`, shadcn-vue(Reka UI), Tailwind, three.js |
| 학습 (`training/`) | 호스트의 별도 venv: Python 3.11.4, torch 2.6.0+cu124, Ultralytics 8.4.176 |
| GPU | Tesla V100 16GB 1장 (운영 추론과 공유) |
| 저장소 | https://github.com/dongle-zzang/retrace (`main` 브랜치만 운영) |

**주요 화면**
- `/` → `/parking`으로 리다이렉트.
- `/parking`: 주차 현황.
- `/monitoring`: CCTV 모니터링.
- `/cameras/[id]/parking`: 주차면 영역 편집기.
- `/datasets`: 탭 두 개 (이미지 수집, 라벨링·내보내기).
- `/datasets/images/[id]`: bbox 편집기.
- `/training`: 탭 네 개 (새 학습, 진행·결과, 새 이미지 평가, 기록·모델).

---

## 2. 확정된 결정 사항 (+ 이유)

**영상과 모델**

| 결정 | 이유 |
| --- | --- |
| WebRTC를 빼고, 단일 `/ws` WebSocket으로 JPEG를 전송 | 다중 카메라를 연결 하나로 단순하게 다루기 위해서입니다. 세부 비교 근거는 확인 필요. |
| PeopleNet 제거, YOLO11m 차량 검출로 운영 전환 | 주차 점유 판단이 핵심 용도라서입니다. |
| 주차장 카메라 5대만 대상 (b1_parking_entrance, overview, gate, inner, 1f_outdoor_parking) | `b1_parking_tower`는 사용자가 제외하기로 결정했습니다. |
| 프리뷰 튜닝: mux timeout 33333, 디코더 low-latency, `/ws` 버스트 8 토큰 버킷, `previewVideoFps` 30 | 프리뷰 지연을 줄이기 위해서입니다. 서버 측에서 `main`에 5e0183b로 반영했습니다. |
| 사람 박스는 서버가 그림 | 프론트가 프레임 위에 따로 그리지 않도록 하는 설계입니다. |

**주차 판정**

| 결정 | 이유 |
| --- | --- |
| 판정 모드는 두 가지(색상, 차량 검출)이고 기본값은 차량 검출 | 색상 방식은 기준 이미지(baseline)에 민감해서입니다. `/parking` 카드에서 전환할 수 있습니다 (PR #6, 827cf6a). |
| 차량 검출 판정은 브라우저에서 수행 | 작업 범위가 프론트엔드로 한정되어 있었기 때문입니다. |

차량 검출 판정 규칙:
- `/api/cameras/{id}/vehicles`를 2초마다 폴링합니다.
- 기준점(앵커)은 박스 하단 25% 띠의 중심입니다.
- 주차면과 55% 이상 겹치면 후보로 봅니다.
- 2회 연속 검출되면 "주차됨", 3회 연속 비면 "빈자리"가 됩니다.
- confidence 0.4 미만은 무시합니다.
- 주차면별 임계값 슬라이더는 브라우저에만 저장되고, 판정 결과는 DB에 저장하지 않습니다.

색상 모드 규칙:
- baseline이 없으면 HSV 엔트로피로 판정합니다.
- baseline이 있으면 Hellinger 거리로 판정합니다. 0.35 이상이면 점유, 0.25 이하가 3프레임 이어지면 빈자리입니다.
- baseline은 실제로 빈 상태일 때 등록해야 합니다.

**학습과 배포**

| 결정 | 이유 |
| --- | --- |
| 학습은 웹에서 `/api/training`으로 요청하고, 호스트 워커(`systemctl --user retrace-training-worker`)가 실행 | 백엔드 컨테이너에 GPU가 없기 때문입니다. |
| 학습 작업은 한 번에 하나만 실행 | GPU를 운영 추론과 공유하기 때문입니다. |
| 관리자 토큰과 워커 토큰을 분리하고, 각각 24자 이상 (`TRAINING_ADMIN_TOKEN`, `TRAINING_WORKER_TOKEN`) | 공유 토큰 기반의 최소 보호입니다. |
| 운영 모델은 자동으로 교체하지 않음 | 운영 안정성 때문입니다. 교체는 사람이 결정합니다. |
| Train All 모드: train+val+test 전체 사용, val=False, patience 0, best.pt = last.pt | 데이터가 적어 전부 학습에 쓰기 위해서입니다. |
| 새 CCTV 이미지 평가에서는 학습에 쓴 이미지와 같은 촬영 세션을 제외 | 평가가 과대평가되지 않게 하기 위해서입니다. |
| UI의 클래스는 vehicle3(car/truck/bus)로 고정하고, 단일 클래스 옵션은 UI에서 제거 | 단일 클래스 모델의 head가 학습되지 않는 문제가 있었기 때문입니다. |
| DeepStream 모델 배포(OTA + 롤백)는 기본 비활성 (`VEHICLE_DEPLOYMENT_ENABLED`) | 실환경 검증이 끝나지 않았기 때문입니다. |

**자동 라벨링** (PR #9, 2f3a9af)

| 결정 | 이유 |
| --- | --- |
| 학습 작업 슬롯을 `kind: 'autolabel'`로 재사용 | GPU 작업을 하나로 직렬화하기 위해서입니다. |
| 자동 초안은 `source:'auto'` 박스로 만들고, 사람이 단 라벨은 덮어쓰지 않음 | 수작업 라벨을 보호하기 위해서입니다. |
| 일괄 실행(목록 버튼)과 단일 이미지 실행(편집기, `image_ids` + `apply:false`)을 모두 제공 | 한 장씩 검증할 수 있게 하기 위해서입니다. |

- 데모는 dev 환경에서만 `/datasets?demo=autolabel`로 열립니다.
- COCO의 car, bus, truck을 모두 `car`로 매핑했습니다. 이 매핑은 스레드에서 "추측"으로 표시된 상태입니다 (확인 필요).

**저장소 운영**

| 결정 | 이유 |
| --- | --- |
| AI/에이전트 문서는 저장소에 올리지 않고 로컬에만 둠 (`.ai/`, `AGENTS.md`, `CLAUDE.md`, `.claude/`, `/docs/agent/`는 gitignore) | 공개 저장소에 에이전트 문서와 서버 경로가 노출되지 않게 하기 위해서입니다. |
| 2026-10-23에 저장소를 정리된 히스토리로 재생성 (62커밋, `main` 06956b5) | 과거 커밋에 남은 AI 문서와 서버 경로를 제거하기 위해서입니다. |
| 브랜치는 `main`만 유지 | 사용자 결정입니다. |
| README는 기능·구조·스택 소개만 담고, 운영 내용은 `docs/operations.md`로 분리 | 소개 문서를 깔끔하게 하기 위해서입니다. |
| 푸터 "© ReTrace · made by ohjoo" 유지 (`frontend/app/layouts/default.vue`) | 사용자 결정입니다. |

**i18n**

| 결정 | 이유 |
| --- | --- |
| vue-i18n 없이 자체 구현 | 이유는 확인 필요. |

- 영어 원문을 키로 씁니다 (`t('...')`).
- 한국어는 `app/lib/i18n.ko.ts`에 두고, 숫자가 들어간 문장은 패턴으로 처리합니다.
- 언어 설정은 localStorage `retrace.locale`에 저장합니다.

---

## 3. 코드/디자인 규칙

**공통**
- 추측하지 말고, 모르는 것은 Unknown이나 "확인 필요"로 표시합니다.
- 사용자가 이미 해 둔 변경은 보존합니다.
- 시크릿, 토큰, RTSP 주소, 서버 주소는 문서와 커밋에 남기지 않습니다.
- 커밋 메시지와 문서에 "포트폴리오"라는 표현을 쓰지 않습니다 (대신 "프로젝트 소개").
- 커밋은 사용자가 승인했을 때만 합니다. AI 문서는 절대 커밋하지 않습니다.
- Python 의존성은 pip로 관리합니다. npm 파일은 Codex 도구용입니다.
- DeepStream은 자동으로 중지하거나 재시작하지 않습니다.
- 운영 모델은 자동으로 교체하지 않습니다.
- sudo나 docker가 필요한 작업은 사용자가 SSH로 직접 실행합니다.

**프론트엔드** (`frontend/AGENTS.md`, `frontend/.ai/MEMORY.md`)
- 프론트 작업에서는 `frontend/`만 수정하고 서버 영역은 건드리지 않습니다. 주차 작업 때 사용자가 요청한 규칙입니다: "작업시에는 프론트엔드 파일만 수정해 주세요".
- API는 상대 경로 `/api`, `/ws`만 씁니다. 백엔드 origin은 `.env`의 `NUXT_PUBLIC_BACKEND_BASE_URL`에만 둡니다.
- `/ws` dev 프록시는 `nuxt.config`의 listen hook에서 처리합니다.
- shadcn `ui/` 컴포넌트는 공용입니다. 수정하지 말고 호출하는 쪽에서 class로 스타일을 덮어씁니다.
- 편집용 ID에는 `crypto.randomUUID`가 없을 때를 대비한 `getRandomValues` 기반 UUID fallback을 둡니다 (일반 HTTP에서는 `randomUUID`가 없습니다).
- 영상 소켓은 화면당 하나이고, 카메라마다 대기 슬롯은 하나입니다. 디코딩은 offscreen에서 끝낸 뒤 교체합니다.
- `CameraVideo.vue`에서는 원시값(primitive)만 watch합니다.
- 편집기에서 영상 위에 컨트롤을 겹쳐 올리지 않습니다.
- `frontend/.gitignore`의 `!*.ts`를 삭제하지 않고, `main.css`의 `@source "../../";`를 유지합니다.
- 모달 위에 뜨는 확인 다이얼로그는 `z-[60]`과 `overlayClass` prop을 씁니다.
- 새 문구는 반드시 `i18n.ko.ts`에 번역을 추가합니다. `tests/i18n.test.cjs`가 누락을 검사합니다.
- 검증은 `npm test`, `npm run typecheck`, `npm run build`로 합니다.
- 배포는 `npm run generate`를 실행한 뒤 `.output/public` 내용을 서버의 `frontend-dist/`에 복사합니다.
- Node 22 이상이 필요합니다.
- 코드 스타일은 Composition API와 `<script setup>`입니다.

**로컬 PC 작업 방식**
- 대상: Windows `C:\Users\wkit\Desktop\retrace`.
- Claude가 PC 파일을 직접 수정하고, 사용자는 `npm run dev` HMR로 결과를 확인합니다. 결과가 괜찮다고 할 때만 커밋합니다.
- PC 파일은 CRLF 줄바꿈이므로 유지합니다.
- dev 서버 포트를 지정할 때는 `npx nuxt dev --host 0.0.0.0 --port 3333`을 씁니다.
- PC에서 `git checkout -- .`은 실행하지 않습니다. 커밋되지 않은 작업이 지워질 수 있습니다.

---

## 4. 완료된 작업

**주차와 모델**
- PeopleNet 제거, YOLO11m 운영 전환, 주차장 카메라 5대 구성.
- MJPEG over WS 배포, PGIE 엔진 캐시.
- 색상 기반 주차 MVP.
- 주차 차량 검출 모드와 모드 전환 (PR #6, 827cf6a).
- 주차 라벨 PATCH/DELETE API.

**데이터셋과 학습**
- 데이터셋 기능 (344428e/369ed49). 별도 worktree에서 운영 반영되었습니다.
- 학습 UI와 워커: 새 학습, 진행·결과, 새 이미지 평가, 기록·모델.
- Train All 모드와 vehicle3 고정.
- 데이터셋 자동 라벨링 (일괄과 단일 이미지), 무한 스크롤, 내보내기 모달 (PR #9, 2026-10-21, 2f3a9af). 코드 병합까지만 끝났고 서버 배포는 아직입니다.

**배포 기능과 튜닝**
- DeepStream 모델 배포(OTA + 롤백) 소스 (7957590, 5c4da59). 기본 비활성 상태입니다.
- 프리뷰 지연 튜닝 (5e0183b).

**저장소와 문서**
- `/`를 `/parking`으로 리다이렉트하고, 모니터링을 `/monitoring`으로 이동.
- 자체 i18n(한국어)과 번역 누락 테스트.
- 저장소 재생성 (2026-10-23):
  - AI 문서를 히스토리에서 제거했습니다.
  - 서버 경로를 `~/retrace`로 바꿨습니다.
  - 커밋 메시지의 "포트폴리오"를 "프로젝트 소개"로 바꿨습니다.
  - AI 문서 15개는 `/mnt/project-files/ai-docs-backup/`에 백업했습니다.
- README를 프로젝트 소개용으로 다시 쓰고 `docs/operations.md`를 분리 (c587bc9 #10, a52ddd2 #11).

---

## 5. 미해결 이슈 / 다음 할 일

**배포와 검증**
- [ ] 자동 라벨링 서버 배포. 배포 후 이미지 한 장으로 먼저 테스트합니다. 실제 GPU, DB, 프록시에서는 아직 검증하지 않았습니다.
- [ ] DeepStream 모델 배포의 실환경 검증. 실제 실행 결과는 `rollback_failed`나 `verification_failed`였습니다. FPS 검증 개선은 소스에만 있고, 서버 적용 여부는 확인 필요입니다.
- [ ] 업데이트된 워커로 Train All 재실행. 기존 job f4818a65는 standard 모드로 돌아서 실제 Train All 결과가 아닙니다.
- [ ] 저장소 재생성 이후 PC와 서버를 맞추기: `git fetch origin` 다음 `git reset origin/main` (hard 아님). 실행 여부는 확인 필요.
- [ ] 서버가 최신 `main`인지, `frontend-dist`가 동기화됐는지 확인 필요.

**데이터와 모델 품질**
- [ ] 학습 데이터 추가 수집. 날짜와 세션을 다양하게 해야 합니다. 첫 내보내기는 6장뿐이라 너무 적었습니다.
- [ ] 모델이 P202 주차면을 검출하지 못합니다.
- [ ] 색상 baseline이 차량이 있을 때 등록됐을 가능성이 있습니다 (P101, O-DP1, P202). 재등록이 필요한지는 확인 필요.
- [ ] 단일 클래스 모델 884b0145와 a2fd4884는 head가 학습되지 않았으니 사용하지 않습니다.
- [ ] COCO car/bus/truck을 `car`로 매핑한 것이 맞는지 확인 필요.

**기능과 운영**
- [ ] 영상이 나오지 않는 카메라 3대: 3f_terrace_4k, 5f_server_room, 7f_rooftop. 원인은 확인 필요.
- [ ] 모델 삭제 API가 없습니다.
- [ ] 차량 검출 모드 판정 결과와 임계값이 브라우저에만 저장되고 서버에는 저장되지 않습니다.
- [ ] 인증이 공유 토큰 방식뿐입니다.
- [ ] PC에 커밋되지 않은 작업이 남아 있는지 확인 필요. 자동 라벨링 코드는 PR #9로 병합됐습니다.
- [ ] 프로젝트 최종 목표와 로드맵 문서가 없습니다 (확인 필요).

**운영 중 알려진 문제와 해결법**

| 증상 | 해결 |
| --- | --- |
| 백엔드 `PermissionError`로 크래시 루프가 생기고 502 발생 | `chmod -R a+rX backend/app camera_config.py` 실행 후 재빌드. `frontend-dist`와 `training`도 같은 조치가 필요합니다. |
| evaluate 작업에서 `KeyError 'epochs'` | 예전 워커가 떠 있는 상태이므로 워커를 재시작합니다. |
| `/api/training/auth`가 415 반환 | `{}` 본문과 `Content-Type: application/json`을 함께 보냅니다. |

---

## 6. 시도했다가 버린 방법 (반복 금지)

**영상, 모델, 학습**

| 버린 방법 | 대체 / 이유 |
| --- | --- |
| WebRTC 영상 전송 | 제거하고 단일 `/ws` WebSocket으로 JPEG를 전송합니다. |
| PeopleNet 사람 검출 | 제거하고 YOLO11m 차량 검출만 씁니다. |
| UI의 단일 클래스 학습 옵션 | head가 학습되지 않아서 제거했습니다. vehicle3로 고정합니다. |
| 학습 모델로 운영 모델을 자동 교체 | 하지 않습니다. 수동으로 결정합니다. |

**저장소와 문서**

| 버린 방법 | 대체 / 이유 |
| --- | --- |
| AI 문서를 저장소에 커밋해서 관리 | 로컬에만 두고 gitignore 처리합니다. 추적 해제만 하던 옛 PR #12는 저장소 재생성으로 의미가 없어졌습니다. |
| 예전 저장소 히스토리(옛 PR #1–12) | 저장소를 재생성해서 폐기했습니다. |
| README에 스크린샷과 "현재 상태와 한계" 섹션 넣기 | 제거했습니다. 운영 내용은 `docs/operations.md`에 둡니다. |
| 클라우드 세션에서 원격 브랜치 삭제 | 403으로 막힙니다. PC의 Remote Control로 해야 하고, 사용자가 직접 요청한 경우에만 가능합니다. |

**로컬 PC와 도구 사용**

| 버린 방법 | 대체 / 이유 |
| --- | --- |
| `npm run dev -- --port ...` | `--host`와 함께 쓰면 포트가 무시됩니다. `npx nuxt dev --host 0.0.0.0 --port 3333`을 씁니다. |
| `/mnt/project-files`의 같은 경로를 재사용해 PC로 파일 푸시 (`device_commit_files`) | 네트워크 마운트라서 옛 사본이 푸시됐습니다. 푸시마다 새 폴더를 쓰고, 10초 기다린 뒤 diff로 확인합니다. |
| 구버전 Node로 PC에서 dev 실행 | `trustedFunctions.difference` 오류가 납니다. Node 22 이상을 씁니다. |
| rows 파일만 diff해서 DB 덤프 비교 | rows 파일이 비어 있어 비교가 되지 않았습니다. pg_restore manifest로 비교합니다. |
