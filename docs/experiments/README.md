# 설정 실험

제품 설정 하나를 바꿔 평가 영상 4편(`configs/eval.yaml`, 정답 `data/ground_truth/`)의 사건 결과가 어떻게 달라지는지 본 기록입니다. 폴더마다 설정 파일, 원본 결과(JSON), 실행 환경과 입력 파일 해시가 들어 있습니다. 표본이 영상 4편이라 방향을 보는 용도입니다.

| 실험 | 바꾼 것 | 결과 | 결론 |
|---|---|---|---|
| [들어감 확인 시간](entry-confirm-20261113/README.md) | `entry_confirm_seconds` 0.3초 → 1.0초 | 찾은 침입 12 → 9건, 놓침 0 → 3건, 중복 6 → 2건, 시작 시각 오차 증가(주차장 0.45 → 1.84초) | 기본값 0.3초 유지. 길게 잡으면 짧은 침입과 추적이 끊긴 조각을 놓친다 |
| [최소 사람 상자 높이](min-box-height-20261113/README.md) | `detector.min_box_height_px` 0(끔) → 40 | 찾음·놓침·중복은 같고 주차장 종료 시각 오차만 1.69 → 1.33초. 기본 3회·40px 2회 반복 실행에서 같은 차이가 재현됨 | 기본은 끔. 표본에서 40px 미만 상자는 드물었고(캠퍼스 168개 중 4개), 주차장의 5개는 모두 화면 위쪽 가장자리에 닿아 있었다 |

관련 코드와 테스트: 구역 판정 `src/sceneproof/edge/zones.py`(`tests/test_zones.py`, `tests/test_zones_flicker_exclude.py`), 탐지기 `src/sceneproof/edge/detector.py`(`tests/test_min_box_height.py`, `tests/test_min_box_metadata.py`). 설계 결정: `docs/design/decisions.md` D15.

다시 돌리기(예: 들어감 확인 시간):

```bash
python -m sceneproof.eval.run_eval --out /tmp/ev_base.json
python -m sceneproof.eval.run_eval --config docs/experiments/entry-confirm-20261113/slow.yaml --out /tmp/ev_slow.json
```
