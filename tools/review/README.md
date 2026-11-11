# 확인용 페이지 (설치 없이 브라우저로 열기)

**2026-11-11 두 페이지 모두 완료·반영**: 정답 검수 결과는 [`docs/eval/ground-truth.md`](../../docs/eval/ground-truth.md), 채점 결과는 [`docs/eval/vlm-findings.md`](../../docs/eval/vlm-findings.md) §1-2. 페이지 안의 "지금 정답"은 검수 전 초안 기준입니다. 다른 사람이 한 번 더 채점하려면 같은 페이지를 그대로 쓰면 됩니다.

## 다른 사람이 한 번 더 채점하기 (채점이 얼마나 일치하는지)

1. `vlm_rubric.html` 파일 하나만 전달합니다(메신저·메일, 약 5 MB). 같은 컴퓨터에서 하면 먼저 오른쪽 위 **새로 시작 (다른 사람)** 을 누릅니다 — 앞 사람 답이 지워지니 앞 사람은 내보내기를 먼저 해 두세요.
2. 채점한 사람이 **내보내기**로 받은 `vlm_rubric.<이름>.json`을 받아서 실행합니다:
   `python scripts/rubric_agreement.py vlm_rubric.<이름>.json`
3. 저장소에는 이름 없이 `grader2`로 저장되고(`docs/eval/raw/vlm_rubric_grader2.json`), 질문별로 두 사람이 같은 답을 한 수와 다른 답 목록이 `docs/eval/raw/vlm_rubric_agreement.json`에 나옵니다. `python scripts/eval_report.py`를 다시 돌리면 평가 보고서에도 들어갑니다.

부탁할 때는 "사진에 보이는 것과 Claude 글이 맞는지 보는 것"이라고만 설명하고, 첫 번째 채점 결과는 보여주지 않는 것이 좋습니다.

| 페이지 | 하는 일 | 끝나면 |
|---|---|---|
| [`gt_review.html`](gt_review.html) | 처음 만든 평가 영상 4편 정답 초안의 확인할 곳 14개. 사진에 구역(빨간 칸)이 그려져 있고 버튼으로 답합니다. | **내보내기**로 받은 `gt_review.<이름>.json`을 올리면 정답 파일(`data/ground_truth/*.yaml`)을 고치고 다시 채점합니다. |
| [`vlm_rubric.html`](vlm_rubric.html) | Claude 해석 19건을 같은 순간의 사진과 나란히 보고 4가지(관찰이 맞는지, 지어낸 내용, 불확실한 점의 타당성, 읽기 쉬움)를 채점합니다. | `python scripts/apply_rubric_scores.py vlm_rubric.<이름>.json` → `docs/eval/raw/vlm_eval-rubric.csv`의 채점 칸이 채워집니다. |

답은 브라우저에 자동으로 남아서, 중간에 닫았다가 다시 열어도 이어서 할 수 있습니다. 페이지는 `python scripts/build_review_pages.py`로 다시 만듭니다
(Claude가 본 사진의 순간을 찾는 방법은 스크립트 첫머리 설명 참고).
