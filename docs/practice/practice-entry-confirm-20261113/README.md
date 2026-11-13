# 과제 2: 들어감 확인 시간 비교

- 평가 영상 4개에서 0.3초 → 1.0초로 바꾸자 일치 사건은 12 → 9, 놓친 사건은 0 → 3, 중복은 6 → 2로 바뀌었다.
- 확인 시간을 늘리면 짧게 보인 사람의 사건이 열리지 않거나 시작이 늦어질 수 있다. 이 결과는 네 영상의 관찰이며 원인에 대한 설명은 동작 방식에 근거한 해석이다.
- 이 평가에서는 놓친 사건이 없는 0.3초를 우선 선택하되, 실제 현장 영상과 위험도에 맞춰 중복과 지연을 함께 확인한다.

| 영상 | 시작 시각 MAE: 0.3초 | 시작 시각 MAE: 1.0초 | 놓침: 0.3초 → 1.0초 | 중복: 0.3초 → 1.0초 |
| --- | ---: | ---: | ---: | ---: |
| eval-campus | 0.23초 | 0.83초 | 0 → 1 | 2 → 1 |
| eval-indoor | 0.24초 | 0.91초 | 0 → 1 | 0 → 0 |
| eval-store | 0.13초 | 0.83초 | 0 → 0 | 3 → 1 |
| eval-parking | 0.45초 | 1.84초 | 0 → 1 | 1 → 0 |

시작 시각 MAE는 각 설정에서 정답과 연결된 사건을 기준으로 하므로 두 설정의 대상 수가 다를 수 있다.

`base.json`과 `slow.json`은 각 실행의 원본 결과다. `slow.yaml`은 `configs/eval.yaml`에서 각 구역의 `entry_confirm_seconds`만 1.0으로 바꾼 설정이다. YOLOX Tiny ONNX CPU, Python 3.12.14, OpenCV 5.0.0, NumPy 2.3.5, ONNX Runtime 1.31.1에서 실행했다. 두 결과 파일에 기록된 Git 식별자는 실행 작업공간의 로컬 HEAD `aad33d0`이며, 작업 트리에 커밋되지 않은 변경이 있었으므로 깨끗한 해당 커밋에서 실행한 결과를 뜻하지 않는다.

입력 SHA-256:

- `models/yolox_tiny.onnx`: `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`
- `campus-walkway.mp4`: `fe5aeb64ce444386a97060a268bc41ce55156660729cd6e515ee17dfb0938658`
- `people-indoor.mp4`: `b14c9fafc89249462d68f98afe44740d87dc6353b02a300ca458d4c7a432ca83`
- `store-aisle.mp4`: `da9291acf846aca2902974be66f4c2a04a04e04a124aba68b9291ac57a92f8ee`
- `parking-lot.mp4`: `8649d95fb692aa39e2ecbc7d87c1b44e2145ad0f5fd824c538c770e3be73d6ae`
