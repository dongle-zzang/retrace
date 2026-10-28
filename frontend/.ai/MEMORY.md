# Frontend memory

비자명한 장기 제약과 그 이유만 둔다.

## HTTP 배포에서도 편집 ID를 생성할 수 있어야 한다

- 이유: localhost에서는 crypto.randomUUID가 제공되지만 일반 HTTP 서버에서는 없어 박스 pointerdown이 실패한다. 데이터셋 박스는 서버가 UUID로 검증하므로 datasetBoxId의 getRandomValues 기반 UUID v4 fallback을 사용한다.
- 변경 시 확인: randomUUID가 없는 환경의 tests/datasetBoxId.test.cjs와 실제 HTTP 배포 화면에서 그리기·저장을 확인한다. 문자열 임시 ID로 대체하면 저장 API가 거부한다.

## `/ws` dev 프록시는 `nuxt.config.ts`의 `listen` hook이 직접 처리한다

- 이유: 서버는 같은 origin의 WebSocket만 허용한다(Origin 검사). Nitro `devProxy`는 HTTP만 전달하고 upgrade는 Nitro worker로 보내며, Vite `server.proxy`는 Nuxt의 middleware mode에서 upgrade를 받지 못한다. 그래서 `listen` hook에서 `/ws` upgrade만 가로채 Node `http`로 전달하고 `Origin`을 서버 origin으로 바꾼다. 서버의 `PREVIEW_WS_ORIGINS`는 compose에 연결돼 있지 않아 쓰지 않는다.
- 변경 시 확인: dev server에서 `ws://localhost:<port>/ws`로 연결해 `hello` 수신과 binary 프레임 수신을 확인한다. Nuxt/Nitro 업그레이드 후 기존 upgrade listener(HMR 등) 위임이 유지되는지 본다.
- 잘못 바꾸면: dev에서 영상이 403이거나 소켓이 Nitro worker로 가서 끊긴다(운영 빌드에는 영향 없음).

## 코드는 상대 경로 `/api`, `/ws`만 사용한다

- 이유: 운영 빌드는 서버 `frontend/`에 올라가 같은 origin에서 제공된다. `runtimeConfig.public`에 base URL 키를 두면 `.env`의 `NUXT_PUBLIC_*` 값이 `nuxt generate` 산출물에 절대 주소로 들어가, 다른 인터페이스(LAN/Tailscale)로 접속할 때 cross-origin이 되고 서버 주소가 산출물에 노출된다.
- 변경 시 확인: generate 산출물에서 서버 주소가 검색되지 않는지 확인한다.

## 영상은 화면당 소켓 1개, 카메라별 단일 대기 슬롯으로 표시한다

- 이유: 카메라마다 MJPEG `<img>`를 열면 브라우저의 origin당 HTTP/1.1 연결 6개 제한으로 6대에서 멈춘다. 느린 클라이언트에서 프레임을 큐에 쌓으면 지연·메모리가 늘어나므로 decode 중 도착한 프레임은 하나의 슬롯에 덮어쓴다. 화면의 `<img>.src`를 바로 바꾸면 빈 프레임이 보여 깜빡이므로 offscreen `Image.decode()` 후 교체한다.
- 변경 시 확인: `tests/realtimePreviewClient.test.cjs`, DevTools에서 blob URL 누적 여부.

## `CameraVideo.vue`는 카메라 객체가 아니라 원시 값만 watch한다

- 이유: `/api/cameras`가 10초마다 새 객체 배열로 교체된다. 객체 참조를 watch하면 10초마다 구독 해제/화면 비움/재구독이 일어나 영상이 깜빡인다.
- 변경 시 확인: watch source가 `camera_id`, 표시 여부, client 같은 원시 값인지 확인한다.

## 사람 박스는 클라이언트에서 그리지 않는다

- 이유: 서버가 JPEG에 bbox와 Track ID를 이미 그려 보낸다. `detections`로 SVG를 다시 그리면 박스가 이중으로, 시간차를 두고 보인다. `CameraOverlay.vue`는 선/영역 표시용으로 남겨 두었다.

## 빌드는 서버 `frontend-dist/`, 소스는 `frontend/`

- 이유: 예전에는 서버의 `frontend/`가 Git 소스 폴더이면서 빌드 업로드 폴더(`FRONTEND_DIST_DIR`)였다. 빌드를 올리면 소스가 `git status`에서 삭제로 보였고(커밋하면 원격 소스가 지워짐), 소스를 복원하면 `/README.md` 같은 소스가 40225에서 열람됐다. 서버 커밋 `2a9254e`(2026-10-07)부터 빌드는 Git에서 무시되는 `frontend-dist/`에서 제공되고, 루트 `.gitignore`는 더 이상 `frontend/`를 무시하지 않는다.
- 변경 시 확인: 배포 안내에 `frontend/`로 업로드하라는 문구가 다시 생기지 않게 한다. 업로드 후 `/`가 200, `/README.md`가 404인지 본다.

## `frontend/.gitignore`의 `!*.ts`를 지우지 않는다

- 이유: 루트 `.gitignore`(서버 소유)가 MPEG-TS 녹화 파일 때문에 `*.ts`를 무시한다. 그래서 `frontend/`에 새로 만든 TypeScript 파일이 `git status`에 나타나지 않고 커밋에서 조용히 빠졌다(기존 `.ts`는 이미 추적 중이라 드러나지 않았음). 2026-10-08 주차면 작업에서 발견해 `frontend/.gitignore`에 `!*.ts`를 두었다.
- 변경 시 확인: 새 `.ts` 파일을 만든 뒤 `git status --short -uall`에 보이는지, `git check-ignore -v`로 무시 규칙이 걸리지 않는지 본다.
- 같은 이유로 Tailwind 자동 소스 탐지도 `.ts`를 건너뛴다(중첩 `.gitignore`의 `!*.ts`로는 풀리지 않음). shadcn variant 클래스(`components/ui/*/index.ts`의 `[&_svg...]:size-4`, `size-8` 등)가 CSS에서 빠져 버튼·입력창 아이콘이 24px로 커지고 버튼 모양이 깨졌다. 그래서 `main.css`에 `@source "../../";`를 둔다. 확인: 빌드 CSS에 `svg:not`이 여러 개 있는지.

## 영상 위 편집 overlay에는 컨트롤을 겹치지 않는다

- 이유: 주차면 편집 툴바를 영상 위에 띄웠더니 영상 위쪽 가장자리 클릭이 버튼에 먹혀 꼭짓점이 빠졌다. 편집 화면은 툴바를 영상 위 별도 줄에 두고 그 wrapper를 전체 화면으로 만든다(`useVideoStage`의 `fullscreenElement`).
