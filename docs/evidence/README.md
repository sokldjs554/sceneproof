# 증적(evidence) 디렉터리

| 경로 | 내용 | 생성 주체 |
|---|---|---|
| `failure-scenarios/*.json`, `summary.json` | 장애 시나리오 6종의 상태 스냅샷·검사 결과(`scripts/failure_scenarios.py`). 각 JSON에 git 해시·시각 포함. `logs/`에 각 프로세스 로그 | 이 세션에서 실행 |
| `k8s-local-attempt/` | 이 개발 샌드박스에서 kind·k3s를 시도했을 때의 runc 오류와 파드 상태(중첩 컨테이너 런타임 제약으로 실패) | 이 세션에서 실행 |
| `k8s/` | `scripts/k8s_verify.sh`가 남기는 kind 배포·복구 증적(노드·파드·API 스냅샷·로그). **GitHub Actions(`k8s-runtime.yml`)에서 생성** — 이 세션에서는 비어 있음 | CI |
| `../eval/raw/vlm_eval*.json`·`*-rubric.csv`, `../../demo/bundle/explanations.jsonl`, `../images/replay-vlm/` | Claude Opus 5.5 CV+VLM 비교(평가 19건, 정답 매칭 포함; 프롬프트 v1.2 → v1.3 두 실행)·데모 번들 해석 15건·재생 모드 화면. **GitHub Actions(`vlm-eval.yml`) https://github.com/sokldjs554/sceneproof/actions/runs/42460838313(v1.2)·https://github.com/sokldjs554/sceneproof/actions/runs/42470298972(v1.3)·https://github.com/sokldjs554/sceneproof/actions/runs/42458789215(데모)에서 생성**, 소견은 `../eval/vlm-findings.md` | CI |

수치 결과(평가·벤치마크)는 `docs/eval/raw/`에 있습니다.
