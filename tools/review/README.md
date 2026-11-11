# 확인용 페이지 (설치 없이 브라우저로 열기)

| 페이지 | 하는 일 | 끝나면 |
|---|---|---|
| [`gt_review.html`](gt_review.html) | 처음 만든 평가 영상 4편 정답 초안의 확인할 곳 14개. 사진에 구역(빨간 칸)이 그려져 있고 버튼으로 답합니다. | **내보내기**로 받은 `gt_review.<이름>.json`을 올리면 정답 파일(`data/ground_truth/*.yaml`)을 고치고 다시 채점합니다. |
| [`vlm_rubric.html`](vlm_rubric.html) | Claude 해석 19건을 같은 순간의 사진과 나란히 보고 4가지(관찰이 맞는지, 지어낸 내용, 불확실한 점의 타당성, 읽기 쉬움)를 채점합니다. | `python scripts/apply_rubric_scores.py vlm_rubric.<이름>.json` → `docs/eval/raw/vlm_eval-rubric.csv`의 채점 칸이 채워집니다. |

답은 브라우저에 자동으로 남아서, 중간에 닫았다가 다시 열어도 이어서 할 수 있습니다. 페이지는 `python scripts/build_review_pages.py`로 다시 만듭니다
(Claude가 본 사진의 순간을 찾는 방법은 스크립트 첫머리 설명 참고).
