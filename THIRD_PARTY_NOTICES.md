# Third-party notices

| 구성요소 | 라이선스 | 출처 | 사용 방식 |
|---|---|---|---|
| YOLOX 모델 코드 (`src/zonewatch/edge/yolox_model/`) | Apache-2.0 | https://github.com/Megvii-BaseDetection/YOLOX | 추론 전용으로 손질해 벤더링(파일 머리말에 표기, `LICENSE.YOLOX` 동봉) |
| YOLOX COCO 가중치 (`models/*.pth`, 커밋 안 함) | Apache-2.0 | 같은 저장소 릴리스 0.1.1rc0 | `scripts/fetch_models.sh`로 다운로드 |
| Intel IoT DevKit sample videos | CC BY 4.0 | https://github.com/intel-iot-devkit/sample-videos | `scripts/fetch_data.sh`로 다운로드, 트랜스코딩, 데모 번들에 일부 포함 |
| OpenCV `vtest.avi` | Apache-2.0 | https://github.com/opencv/opencv | 같음 |
| ByteTrack (알고리즘) | MIT (원 구현) | https://github.com/ifzhang/ByteTrack | 코드 복사 없이 알고리즘을 재구현 |
| Python 패키지 | 각 패키지 라이선스 | PyPI | `pyproject.toml` |
| Go 모듈 (grpc, protobuf) | Apache-2.0 / BSD | proxy.golang.org | `gateway/go.mod` |
