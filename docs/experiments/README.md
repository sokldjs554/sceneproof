# 설정 실험

제품 설정 하나를 바꿔 평가 영상 4편(`configs/eval.yaml`, 정답 `data/ground_truth/`)의 사건 결과가 어떻게 달라지는지 본 기록입니다. 폴더마다 설정 파일, 원본 결과(JSON), 실행 환경과 입력 파일 해시가 들어 있습니다. 표본이 영상 4편이라 방향을 보는 용도입니다.

| 실험 | 바꾼 것 | 결과 | 결론 |
|---|---|---|---|
| [들어감 확인 시간](entry-confirm-20261113/README.md) | `entry_confirm_seconds` 0.3초 → 1.0초 | 찾은 침입 12 → 9건, 놓침 0 → 3건, 중복 6 → 2건, 시작 시각 오차 증가(주차장 0.45 → 1.84초) | 기본값 0.3초 유지. 길게 잡으면 짧은 침입과 추적이 끊긴 조각을 놓친다 |
| [최소 사람 상자 높이](min-box-height-20261113/README.md) | `detector.min_box_height_px` 0(끔) → 40 | 찾음·놓침·중복은 같고 주차장 종료 시각 오차만 1.69 → 1.33초. 기본 3회·40px 2회 반복 실행에서 같은 차이가 재현됨 | 기본은 끔. 표본에서 40px 미만 상자는 드물었고(캠퍼스 168개 중 4개), 주차장의 5개는 모두 화면 위쪽 가장자리에 닿아 있었다 |

| [작은 VLM 입력 영역](vlm-crop/README.md) (ERGO 적용, 2026-11-24) | 로컬 VLM에 보내는 화면: 전체 한 조각 · 여러 조각 · 사람 주변만 | 46문항 맞음 19 · 9 · 8, 원래·옮긴 구역 쌍을 둘 다 맞힌 경우 0/23 | 이 작은 모델은 발과 구역의 관계를 판단하지 못함. 제품 설정은 그대로. [논문 요약](vlm-crop/ergo-summary.md) |

**다시 돌리기(2026-11-21)**: 두 실험은 처음 실행 때 커밋되지 않은 변경이 있어 확정된 코드에서 다시 돌렸습니다([기록](rerun-20261121/README.md)). 들어감 확인 시간은 환경이 달라도 같은 결론이었고(놓침은 3건이 아니라 2건), 최소 상자 높이 40px의 효과는 ONNX Runtime 1.19인 Mac에서 반대로 나와(끝 시각 오차 1.69 → 1.97초) 실행 환경의 차이보다 작았습니다. 평가 영상을 컴퓨터마다 다시 인코딩하기 때문에 영상 파일 자체도 환경마다 다릅니다.

**원인과 고정(2026-11-23)**: 환경마다 결과가 달랐던 원인은 영상을 읽는 방식(GStreamer/OpenCV), ffmpeg 스레드 수, 인코더가 쓰는 CPU 명령어였습니다. 평가 환경을 이미지(`deploy/docker/Dockerfile.eval`)로 고정하고 평가 영상은 릴리스 파일을 그대로 쓰게 해, 서로 다른 두 컴퓨터에서 같은 결과(값 291개)를 확인했습니다([기록](rerun-20261121/README.md#다시-확인한-원인-2026-11-23)).

관련 코드와 테스트: 구역 판정 `src/sceneproof/edge/zones.py`(`tests/test_zones.py`, `tests/test_zones_flicker_exclude.py`), 탐지기 `src/sceneproof/edge/detector.py`(`tests/test_min_box_height.py`, `tests/test_min_box_metadata.py`). 설계 결정: `docs/design/decisions.md` D15.

다시 돌리기(예: 들어감 확인 시간):

```bash
python -m sceneproof.eval.run_eval --out /tmp/ev_base.json
python -m sceneproof.eval.run_eval --config docs/experiments/entry-confirm-20261113/slow.yaml --out /tmp/ev_slow.json
```
