# Third-party notices

| 구성요소 | 라이선스 | 출처 | 사용 방식 |
|---|---|---|---|
| YOLOX 모델 코드 (`src/sceneproof/edge/yolox_model/`) | Apache-2.0 | https://github.com/Megvii-BaseDetection/YOLOX | 모델 정의와 학습 손실(`losses.py`: SimOTA·IoU 손실)을 손질해 벤더링(파일 머리말에 표기, `LICENSE.YOLOX` 동봉). 공식 학습기·데이터 증강은 가져오지 않음 |
| 추가 학습한 가중치 (`models/finetune/`, 커밋 안 함) | Apache-2.0 가중치에서 시작, 학습 데이터는 CAVIAR(CC BY-SA)·MOT16(CC BY-NC-SA 3.0) | 이 저장소 `sceneproof.train.finetune` | 학습 데이터의 비상업 조건 때문에 **비상업·연구 용도로만** 사용 |
| YOLOX COCO 가중치 (`models/*.pth`, 커밋 안 함) | Apache-2.0 | 같은 저장소 릴리스 0.1.1rc0 | `scripts/fetch_models.sh`로 다운로드 |
| Intel IoT DevKit sample videos | CC BY 4.0 | https://github.com/intel-iot-devkit/sample-videos | `scripts/fetch_data.sh`로 다운로드, 트랜스코딩, 데모 번들에 일부 포함 |
| OpenCV `vtest.avi` | Apache-2.0 | https://github.com/opencv/opencv | 같음 |
| CAVIAR 영상·정답 (EC Funded CAVIAR project/IST 2001 37540) | Creative Commons BY-SA (CAVIAR 데이터 페이지 표기) | http://homepages.inf.ed.ac.uk/rbf/CAVIAR/ | `scripts/fetch_bench.py`로 다운로드·재인코딩, 커밋 안 함. 연구용 벤치마크와 추가 학습에 사용 |
| MOT16 / MOT20 영상·정답 | CC BY-NC-SA 3.0 | https://motchallenge.net | 같음. 정지 카메라 5개 시퀀스만, 비상업 연구 용도 |
| ByteTrack (알고리즘) | MIT (원 구현) | https://github.com/ifzhang/ByteTrack | 코드 복사 없이 알고리즘을 재구현 |
| Python 패키지 | 각 패키지 라이선스 | PyPI | `pyproject.toml` |
| Go 모듈 (grpc, protobuf) | Apache-2.0 / BSD | proxy.golang.org | `gateway/go.mod` |
