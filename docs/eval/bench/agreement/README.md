# 두 사람 표시 일치도 결과를 두는 곳

1. 브라우저로 `tools/event_labeler/index.html`을 열고 `data/bench/videos/`의 영상을 고릅니다(최종 분할 영상 권장: `configs/bench/split.yaml`의 `test`).
2. 두 사람이 서로의 결과를 보지 않고 따로 표시한 뒤 각자 **내보내기**를 누릅니다. 파일 이름은 `<영상>.<표시자>.json`으로 저장됩니다.
3. 영상마다 계산해 이 폴더에 저장합니다.

```bash
python -m sceneproof.eval.bench.agreement A.json B.json --with-gt --out docs/eval/bench/agreement/<영상>.json
python scripts/bench_report.py    # docs/eval/bench/README.md §6이 이 폴더의 결과로 채워짐
```

`--with-gt`는 각 사람의 표시를 데이터셋 상자로 만든 정답과도 비교합니다. 사람끼리의 일치도가 사람–정답 일치도보다 크게 높다면, 정답을 만드는 규칙(발 위치·확정 시간)이 사람의 판단과 다르다는 뜻입니다.
