# Preview 첫 영상 시작 계측

`preview.py --diagnostics`에서만 `startup`, `startup_config`, `startup_summary`
JSON 로그를 추가한다. source/element 생성, 연결, batch 설정, RTSP latency,
차량 detector, NvDCF, JPEG/MJPEG, Backend 동작은 바꾸지 않는다(PeopleNet은 2026-10-13 제거).
카메라 ID와 숫자만 기록하며 RTSP 주소, SDP, native progress text는 출력하지 않는다.
streaming callback에서는 최초 timestamp를 메모리에 저장하고 main loop가 로그를 출력한다.
따라서 로그 줄 순서보다 `elapsed_ms`로 비교한다. 재연결 이후에는 최초 기록을 덮어쓰지 않는다.

## 기준 시각과 관측 위치

`origin=pid1_start`는 Linux PID namespace의 PID 1 시작 시각이다. fresh container에서는
컨테이너 init 시작에 해당하며 Docker create/image pull 이전 시간은 포함하지 않는다.
PID 1이 systemd/init이거나 `/proc`을 읽지 못하면 `python_entry`로 fallback한다.
`docker compose exec`로 오래 실행된 컨테이너에서 실행하면 PID 1 기준에 기존 uptime이
포함되므로 container startup 비교에는 fresh `compose run`을 사용한다.

| stage | 실제 관측 |
| --- | --- |
| python_entry / run_entered | Python 진입(import 이전) / config parsing 이후 run 진입 |
| infer_config_start / infer_config_ready | 차량 detector 설정 파일 준비 전후; TensorRT engine 초기화 완료를 뜻하지 않음 |
| source_create_start / source_created | source helper 호출 전후(NULL 상태); 연결 완료를 뜻하지 않음 |
| all_source_branches_built | 모든 source·입력 정규화·JPEG branch 구성 완료 |
| playing_requested / playing_call_returned | 공유 pipeline set_state(PLAYING) 호출 전후; return만으로 PLAYING 확정 불가 |
| pipeline_ready / pipeline_paused / pipeline_playing | native STATE_CHANGED가 bus에 게시되는 시점(sync handler) |
| vehicle_ready / vehicle_paused / vehicle_playing | 공유 차량 nvinfer의 native 상태 변화(차량 탐지 활성 시); 정확한 CUDA 초기화 시작/끝은 아님 |
| tracker_ready / tracker_paused / tracker_playing | 공유 nvtracker의 native 상태 변화; 내부 per-stream context 초기화는 별도 |
| pipeline_playing_bus_observed | 기존 main loop가 PLAYING 메시지를 읽은 시점; 위 native timestamp와 구분 |
| rtsp_source_created | 내부 rtspsrc child 생성 관측 |
| rtsp_open_start / rtsp_connect_start | rtspsrc native progress의 open START / connect CONTINUE; plugin이 메시지를 게시하는 경우 |
| rtsp_request_start | 첫 before-send 신호; TCP connect 시작/성공 자체가 아님 |
| rtsp_sdp_received | on-sdp 신호; DESCRIBE 이후 SDP 수신, 전체 handshake 완료 자체가 아님 |
| rtsp_open_complete | native progress open COMPLETE; stream open/SETUP 경계, PLAY 응답 완료와 다름 |
| rtsp_video_pad_added / first_rtp_buffer | rtspsrc video RTP pad 생성 / 해당 pad의 첫 buffer; jitter buffer 영향도 포함 |
| pad_added | source helper의 video pad-added(NVMM gate 이전) |
| decoder_created | nvv4l2decoder child 생성; decoder 준비 완료를 뜻하지 않음 |
| first_decoder_input / first_decoded_frame | 실제 nvv4l2decoder sink/src의 첫 buffer |
| first_source_output | source bin ghost src의 첫 buffer |
| first_mux_input / first_mux_output | camera별 mux sink buffer / mux src batch metadata에 해당 camera 등장 |
| first_infer_input / first_infer_output | 차량 nvinfer sink/src batch에 해당 camera 등장(모든 camera가 통과, 추론은 주차장만) |
| first_inference_done | 차량 nvinfer src에서 bInferDone=true인 주차장 camera 첫 frame; 검출된 차량이 없어도 기록 |
| first_tracker_input / first_tracker_output | tracker sink/src batch에 해당 camera 등장 |
| first_demux_output | 해당 demux request src pad 첫 buffer |
| first_jpeg / first_appsink_buffer | nvjpegenc src / appsink sink의 첫 encoded buffer |
| first_mjpeg_available | 세대 검사에서 수락된 JPEG를 FrameStore에 저장한 시점(HTTP waiter notify 이전) |
| first_http_request / first_http_frame_flushed | 해당 MJPEG 요청 handler 진입 / JPEG write·flush 완료; 브라우저 렌더링·실제 수신 완료는 아님 |
| online | runtime snapshot에서 최초 online 관측; 약 100ms bus polling 및 진단 보고 지연이 포함될 수 있음 |

RTSP progress 코드와 신호 의미는 [GStreamer rtspsrc 문서](https://gstreamer.freedesktop.org/documentation/rtsp/rtspsrc.html)와
[공식 소스](https://github.com/GStreamer/gstreamer/blob/1.24/subprojects/gst-plugins-good/gst/rtsp/gstrtspsrc.c)를
참조했다. 서버의 실제 plugin 버전에서 신호/progress가 없으면 해당 stage는 누락되며
`source_observer_unavailable` / `bus_observer_unavailable` / `batch_observer_unavailable`을 확인한다.
TCP connect 성공만을 나타내는 별도 callback은 계측하지 않는다.

first-arrival probe는 해당 camera/단계의 최초 기록만 저장한다. 단일 pad probe는 첫 buffer 이후 제거하며,
batch probe는 모든 configured camera가 관측되면 제거한다(offline source가 있으면 유지).
buffer가 있는 단계에는 가능한 범위에서 `pts_ns`도 기록한다.
각 단계의 최초 frame이 서로 다를 수 있으므로 elapsed 차이는 startup 구간이며,
동일 프레임의 순수 처리시간으로 해석하려면 PTS도 일치하는지 확인한다.
서로 다른 camera의 PTS는 공통 wall clock으로 비교하지 않는다.

## 코드에서 확인된 사실

`run_shared_pipeline()`은 모든 source와 output을 NULL에서 생성·연결한 뒤 공유 pipeline에
한 번만 PLAYING을 요청한다. source loop에는 camera별 set_state/get_state/wait가 없다.
Python element 생성은 순차적이지만 카메라별 RTSP/decoder 준비를 기다리는 구조는 아니다.
native plugin 내부 초기화가 병렬인지 직렬인지는 실제 timestamp로 확인해야 한다.
차량 nvinfer, nvtracker/NvDCF는 각각 한 개이며 camera별 재생성하지 않는다.

RTSP는 nvurisrcbin, TCP, latency 기본 1000ms, drop-on-latency=false,
rtsp-reconnect-attempts=-1, reconnect interval은 max(1, int(camera_offline_after))로 기본 10초다.
mux는 batch-size=enabled source 수, live-source=1, timeout=40000us, sync-inputs=false다.
이 설정만으로 offline source가 전체 시작을 지연시키지 않는다고 확정하지 않는다.
runtime online은 기본 2초 연속 프레임 조건 등을 적용하므로 JPEG availability와 다르다.

## 서버 비교 실험

기존 retrace가 중지된 점검 시간에 실행한다. 동시 GPU pipeline/RTSP 세션을 만들면
측정 조건이 달라진다. Backend/PostgreSQL은 변경할 필요 없다.
아래 명령은 서비스 포트를 publish하지 않는 fresh container를 사용한다.
source subset은 config의 enabled 카메라 앞 N개이며, slot/source ID 순서를 유지한다.
batch 크기는 기존 규칙대로 실제 선택된 source 수이며 임의로 override하지 않는다.

```bash
mkdir -p .cache/startup
for n in 1 2 4 8; do
  docker compose run --rm --no-deps --name "retrace-startup-$n" retrace \
    timeout --signal=INT --kill-after=15s 180s \
    python3 -u /workspace/ReTrace/preview.py \
    --cameras /workspace/ReTrace/configs/cameras.yaml \
    --env-file /workspace/ReTrace/.env \
    --diagnostics --startup-source-count "$n" \
    2>&1 | rg '^startup(_config|_summary)? ' | tee ".cache/startup/sources-$n.jsonl"
done
```

각 N을 같은 조건에서 두 번 이상 실행한다. engine cache는 batch 크기별로 구분되어 있으므로
최초 engine build가 포함된 cold run과 기존 engine load를 사용하는 warm run을 분리한다.
cache를 삭제하거나 생성된 engine/config를 편집하지 않는다. 180초 안에 startup이 끝나지 않으면
시간 제한을 늘려 재측정한다. timeout의 정상 시간 제한 종료 코드는 124일 수 있다.

HTTP 경계도 확인하려면 특정 N 실행 중 별도 터미널에서 다음을 실행한다.
컨테이너가 HTTP listen을 시작한 뒤 가능한 일찍 요청한다.
기존 도구는 모든 광고된 MJPEG에 동시에 요청하지만 첫 payload timeout은 5초이므로,
startup이 그보다 길면 재실행하여 early timeout과 영상 장애를 구분한다.

```bash
# 위 loop에서 현재 실행 중인 N으로 지정
n=4
docker exec "retrace-startup-$n" python3 tools/measure_preview.py --seconds 30
```

`startup_summary`는 PLAYING, 첫 MJPEG camera, 마지막 관측 camera, first-frame 분산,
camera별 모든 단계 및 online 관측 수를 제공한다. `all_configured_online_ms`는 모두 online이
관측될 때만 값이 있고, 나머지는 null이다. 각 camera가 최초로 online이 된 기록이며
현재 동시에 모두 online임을 보장하지 않는다. 연결 가능한 집합은 자동 판정할 수 없으므로
`missing_frames`, `missing_online`, camera_status/error_reason 및 운영자가 아는 offline 목록으로
비교한다. `last_observed_online_ms`를 무조건 “모든 연결 가능한 camera 완료”로 단정하지 않는다.
GPU memory는 별도 스레드가 약 5초마다 수집한 device 전체 used MiB와 sample 시각이며,
다른 GPU 프로세스도 포함한다. null은 측정 불가/첫 sample 이전이고 정확한 peak 측정은 아니다.

## 가설 판별

| 가설 | 비교할 경계와 판단 한계 |
| --- | --- |
| A RTSP | connect_start → request_start → SDP → open_complete → RTP; 인증/SETUP/첫 keyframe 등의 차이는 추가 관측 없이는 완전히 분리되지 않음 |
| B decoder 직렬 초기화 | camera별 decoder_created, first_decoder_input, first_decoded_frame 분포; input 자체가 늦으면 decoder만의 지연으로 단정 불가 |
| C mux 대기 | 빠른 camera의 first_mux_input과 first_mux_output, 느린 camera의 첫 input 비교; 느린 input 전 빠른 camera batch가 출력되면 해당 run에서 전체 대기 가설 배제 |
| D TensorRT/차량 detector | vehicle READY/PAUSED, playing 호출 구간, first_infer_input → output/inference_done; engine build/load와 첫 실행을 cold/warm run으로 분리 |
| E NvDCF | tracker 상태 전환 및 first_tracker_input → output; 내부 per-stream context 생성 횟수/정확한 초기화 시간은 이 계측만으로 확정 불가 |
| F HTTP | JPEG → appsink → MJPEG available 및 HTTP request → flush; 요청이 늦었다면 서버 저장 지연과 분리, 브라우저 connection limit/rendering은 별도 확인 |
| G Python 직렬 대기 | 코드상 per-source wait 없음; source_created 분포와 playing_requested로 구성 loop 비용 확인 |

실측 전에는 가장 느린 구간/순차 표시 원인을 확정하거나 latency·architecture를 변경하지 않는다.
RTSP→RTP 지연이 확인된 경우에만 같은 source/engine warm 조건에서 1000ms와 더 낮은
latency의 별도 A/B 실험을 제안한다. 기본값 변경은 이번 작업에 포함하지 않는다.
