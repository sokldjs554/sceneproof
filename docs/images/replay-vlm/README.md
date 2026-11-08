# 화면 캡처·시연 영상

생성 커밋: `477730a` · 생성 시각: 2026-11-08T08:07:51.882Z · 대상: http://127.0.0.1:8000 (실시간 모드, VLM=mock)

| 파일 | 내용 |
|---|---|
| 01_overview.png | 첫 화면: 카메라 상태 · 선택 사건 영상(구역·추적 상자 오버레이) · 타임라인 · 사건 목록 |
| 02_event_cv_evidence.png | 사건 상세 — CV 근거(구역·트랙·시각·키프레임) |
| 03_event_vlm.png | 사건 상세 — VLM 설명(관찰·판단의 한계; 캡처는 mock 제공자) |
| 04_event_review.png | 처리 상태 전이와 처리 이력 |
| 05_original_video.png | 원본 영상에서 사건 구간 보기 |
| 06_engineer_details.png | 상세 정보(모델·버전·지연·원시 응답) |
| 07_audit.png | 처리 기록(감사 로그) |
| demo.webm | 위 흐름을 녹화한 짧은 시연 영상 (Playwright 녹화) |

`node scripts/capture_demo.mjs <server-url> docs/images` 로 재생성합니다. 콘솔 오류: 0건
