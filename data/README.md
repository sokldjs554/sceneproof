# 데이터 출처·사용 조건·분할

이 프로젝트는 **공개 사용 조건이 확인되는 영상만** 사용합니다. 원본은 `scripts/fetch_data.sh`가 받아
`data/videos/raw/`에 두고, 작업용 사본(H.264, 최대 1280px, 고정 프레임레이트)을 `data/videos/`에 만듭니다.
영상 파일은 저장소에 커밋하지 않습니다(`.gitignore`). `data/videos/SHA256SUMS`로 재현을 확인합니다.

## 출처와 라이선스

| 작업용 파일 | 원본 | 출처 | 라이선스 | 해상도·fps·길이 | 내용 |
|---|---|---|---|---|---|
| `worker-zone.mp4` | `worker-zone-detection.mp4` | [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) | **CC BY 4.0** (저장소 LICENSE) | 1280×720 · 30fps(원본 1920×1080 · 60fps) · 76s | 안전조끼·안전모를 착용한 작업자 1명이 바닥의 적색·황색 테이프 구획 사이를 이동 |
| `one-by-one.mp4` | `one-by-one-person-detection.mp4` | 같은 저장소 | CC BY 4.0 | 768×432 · 10fps · 139s | 사람이 한 명씩 작업대(테이블)로 다가와 머물다 떠남 |
| `people-indoor.mp4` | `people-detection.mp4` | 같은 저장소 | CC BY 4.0 | 768×432 · 12fps · 50s | 실내 공간을 여러 사람이 통행 |
| `parking-lot.mp4` | `person-bicycle-car-detection.mp4` | 같은 저장소 | CC BY 4.0 | 768×432 · 12fps · 54s | 주차장 상공 시점. 보행자·자전거·차량 (사람이 아닌 이동체 비교용) |
| `store-aisle.mp4` | `store-aisle-detection.mp4` | 같은 저장소 | CC BY 4.0 | 720×404 · 30fps(원본 60fps) · 65s | 매장 통로. 선반에 의한 가림·근접 통행 |
| `campus-walkway.mp4` | `vtest.avi` | [opencv/opencv `samples/data/vtest.avi`](https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi) | **Apache-2.0** (OpenCV 저장소 라이선스) | 768×576 · 10fps · 80s | 옥외 보행로. 다수 보행자, 교차·가림, 콘과 통제 테이프로 막힌 구역, 잔디 |

- CC BY 4.0: 출처 표기 조건으로 사용·수정·재배포 가능. 이 문서와 README에 출처를 표기합니다.
- Apache-2.0: 사용·수정·재배포 가능. OpenCV 저장소의 NOTICE/LICENSE를 따릅니다.
- 사람 얼굴이 식별될 수 있는 공개 샘플 영상입니다. 본 프로젝트는 신원 식별 기능(얼굴 인식·재식별)을 포함하지 않습니다.

## 개발용(dev)·평가용(eval) 분할

같은 영상의 인접 프레임이 양쪽에 섞이지 않도록 **영상(촬영 장소·카메라) 단위**로 나눕니다.

| 분할 | 영상 | 용도 |
|---|---|---|
| dev / demo | `worker-zone.mp4`, `one-by-one.mp4` | 구역 설계, 디바운스·체류 임계값 조정, 데모 화면, RTSP 시뮬레이터 |
| eval (held-out) | `campus-walkway.mp4`, `people-indoor.mp4`, `store-aisle.mp4`, `parking-lot.mp4` | 사건 단위 지표. 임계값 조정에 쓰지 않음 |
| synthetic (eval 파생) | `scripts/make_synthetic.py`로 eval 영상에서 생성: 조명 변화, 부분 가림, 과입력(배속), 스트림 단절 | 기능·복구 검증. 실데이터 성능과 **분리해 보고** |

평가 영상의 정답(사건 시작·종료, 구역 정의, 정상 상황 기준)은 `data/ground_truth/`에 있으며,
작성 방법과 남은 검수 작업은 `docs/eval/ground-truth.md`에 적습니다.

## 연구용 벤치마크 (사람이 손으로 표시한 정답)

위 6편과 별도로, **사람이 프레임마다 손으로 그린 사람 상자**가 있는 연구용 공개 데이터로 구역 사건 성능과 추가 학습 효과를 잽니다.
목록·라이선스·제외 이유는 `configs/bench/sources.yaml`, 분할은 `configs/bench/split.yaml`(측정 전에 고정), 구역은 `configs/bench/zones.yaml`,
결과는 `docs/eval/bench/README.md`에 있습니다.

| 데이터 | 장소(카메라) | 편수 | 라이선스 | 정답 |
|---|---|---|---|---|
| CAVIAR 1차 (INRIA 로비) | 실내 로비, 위에서 내려다보는 광각 384×288·25fps | 27 | Creative Commons BY-SA, 출처 표기: EC Funded CAVIAR project/IST 2001 37540 | 손으로 그린 상자(움직이지 않는 사람은 표시 안 됨) |
| CAVIAR 2차 (리스본 쇼핑몰) | 복도 방향·매장 정면 두 시점, 384×288·25fps | 26×2 | 같음 | 손으로 그린 상자 |
| MOT16-02·04·09 | 광장·밤거리(높은 시점)·거리(낮은 각도), 1920×1080·30fps | 3 | CC BY-NC-SA 3.0 | MOT16 규약에 따른 수작업 상자 |
| MOT20-01·02 | 혼잡한 실내 기차역, 1920×1080·25fps | 2 | CC BY-NC-SA 3.0 | 같음 |

- 원본은 저장소에 올리지 않습니다. `python scripts/fetch_bench.py`가 원래 사이트에서 받아 `data/bench/`에 두고, 각 영상의 다운로드 해시와 프레임 수를 `data/bench/manifest.json`에 적습니다.
- MOT 데이터는 비상업(NC) 조건이므로 이 데이터로 추가 학습한 가중치도 비상업·연구 용도로만 씁니다.
- 제외: CHAD(상자를 사람이 아니라 YOLOv4가 만듦), VIRAT(사용 동의서 수락 필요 — 사용자 결정), CAVIAR `Fight_OneManDown`(정답 파일 3개가 서로 다름), MOT 이동 카메라 시퀀스(고정 구역을 정의할 수 없음).

## 재현

```bash
bash scripts/fetch_data.sh        # 원본 다운로드 + 트랜스코딩 (ffmpeg 필요)
python scripts/fetch_bench.py     # 연구용 벤치마크 (약 2.3 GB 전송, MOT는 필요한 시퀀스만 범위 요청)
sha256sum -c data/videos/SHA256SUMS
```
