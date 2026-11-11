# 두 사람 표시 일치도 결과를 두는 곳

1. 브라우저로 `tools/event_labeler/index.html`을 열고 `data/bench/videos/`의 영상을 고릅니다(최종 분할 영상 권장: `configs/bench/split.yaml`의 `test`).
2. 두 사람이 서로의 결과를 보지 않고 따로 표시한 뒤 각자 **내보내기**를 누릅니다. 영상을 **재생하면서**, 발이 구역 안으로 들어간 순간 I, 밖으로 나온 순간 O를 누릅니다(예시: [`how-to-mark.png`](how-to-mark.png) — 표시 대상이 아닌 조정 분할 영상 `caviar-lobby-browse3`의 28–32초). 파일 이름은 `<영상>.<표시자>.json`으로 저장됩니다.
3. 영상마다 계산해 이 폴더에 저장합니다.

```bash
python -m sceneproof.eval.bench.agreement A.json B.json --with-gt --out docs/eval/bench/agreement/<영상>.json
python scripts/bench_report.py    # docs/eval/bench/README.md §7이 이 폴더의 결과로 채워짐
```

`--with-gt`는 각 사람의 표시를 데이터셋 상자로 만든 정답과도 비교합니다. 사람끼리의 일치도가 사람–정답 일치도보다 크게 높다면, 정답을 만드는 규칙(발 위치·확정 시간)이 사람의 판단과 다르다는 뜻입니다.

## 영상 늘리기 (2차, 6편 · 약 4분 · 데이터셋 기준 출입 24건)

**2026-11-11 완료**: 새로운 두 사람(표시자3·4)이 6편을 표시했습니다. 결과는 `docs/eval/bench/README.md` §7, 해석은 `docs/eval/bench/findings.md`.

1차 4편에 더해, 같은 최종 분할에서 로비·복도 영상 6편을 골랐습니다. 매장 정면 화면은 1차에서 사람도 출입을 판단하지 못했으므로 뺐습니다.

| 영상 | 길이 | 데이터셋 기준 출입 |
|---|---|---|
| caviar-lobby-fight-run-away2 | 22초 | 2 |
| caviar-lobby-rest-slump-on-floor | 36초 | 3 |
| caviar-lobby-left-bag-picked-up | 54초 | 5 |
| caviar-mall-cor-enter-exit-crossing-paths2 | 19초 | 2 |
| caviar-mall-cor-one-stop-no-enter2 | 60초 | 5 |
| caviar-mall-cor-shop-assistant1 | 67초 | 7 |

받기(이미 받은 영상은 건너뜀):

```bash
python scripts/fetch_bench.py --ids caviar-lobby-fight-run-away2,caviar-lobby-rest-slump-on-floor,caviar-lobby-left-bag-picked-up,caviar-mall-cor-enter-exit-crossing-paths2,caviar-mall-cor-one-stop-no-enter2,caviar-mall-cor-shop-assistant1
```

두 사람이 1차와 같은 방법으로 따로 표시하고, 각자 내보낸 파일 12개(영상 6 × 2명)를 주면 영상마다 계산해 이 폴더와 `docs/eval/bench/README.md` §7에 반영합니다. 1차와 같은 두 사람이면 "사람끼리" 비교가 영상 수만 늘어나고, 다른 사람이면 표시하는 사람에 따라 얼마나 달라지는지도 볼 수 있습니다.
