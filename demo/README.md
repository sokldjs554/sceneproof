# 공개 데모 번들

`bundle/`은 개발용 샘플 2편(`worker-zone.mp4`, `one-by-one.mp4`)을 `configs/edge.yaml` 설정으로 **오프라인 무손실 분석**한 결과입니다
(`events.jsonl`, `explanations.jsonl`, `evidence/`). `videos/`는 "원본 영상에서 보기"용 원본 사본(CC BY 4.0, 출처 `data/README.md`).

- 공개 데모(Render, `render.yaml`)와 정적 재생(GitHub Pages, `scripts/build_static_demo.py`)은 이 번들을 **재생 모드**로 보여 줍니다.
  콘솔 상단에 "저장된 실행 결과 재생" 배지가 뜨고, 사건마다 "재생" 태그가 붙습니다. 실시간 추론이 아닙니다.
- 생성 커밋·실행 조건은 `events.jsonl`의 `run` 필드(모델·백엔드·설정 해시·edge_version)에 들어 있습니다.
- `explanations.jsonl`이 비어 있으면 VLM 해석이 포함되지 않은 번들입니다(콘솔에 "해석이 요청되지 않았습니다"로 표시).

재생성: `zonewatch-edge offline --camera cam-worker --out /tmp/w.jsonl` 등으로 다시 분석한 뒤 서버에 import → `zonewatch-server export demo/bundle`.
