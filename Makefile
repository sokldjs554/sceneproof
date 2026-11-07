# SceneProof — common tasks. `make help` lists them.
PY ?= .venv/bin/python
SHELL := /bin/bash

help:  ## 이 도움말
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup:  ## venv + 의존성 + 데이터 + 모델 + 게이트웨이 빌드
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e ".[server,dev,torch]" && (pip install -e ".[gst]" || echo "PyGObject skipped (OpenCV backend will be used)")
	bash scripts/fetch_data.sh
	. .venv/bin/activate && bash scripts/fetch_models.sh
	$(MAKE) -C gateway build

test:  ## 파이썬 단위 테스트 + Go 테스트
	$(PY) -m pytest -q
	$(MAKE) -C gateway test

lint:  ## ruff + go vet
	$(PY) -m ruff check src tests scripts
	$(MAKE) -C gateway vet

proto:  ## protobuf 코드 재생성
	bash scripts/gen_proto.sh

server:  ## 검토 서버 (mock VLM) — http://127.0.0.1:8000
	SP_SERVER_VLM_PROVIDER=$${SP_SERVER_VLM_PROVIDER:-mock} .venv/bin/sceneproof-server run

gateway-run:  ## 엣지 게이트웨이
	gateway/bin/sceneproof-gateway --listen :50051 --server 127.0.0.1:50052 --device-id edge-01 --queue-dir data/gateway --evidence-root data/evidence --http :9100 --insecure

edge:  ## 엣지 분석기 (샘플 cam-bench 실시간)
	.venv/bin/sceneproof-edge --config configs/edge.yaml run --camera cam-bench

rtsp-sim:  ## RTSP 카메라 시뮬레이터 (rtsp://127.0.0.1:8554/<video>)
	$(PY) -m sceneproof.tools.rtsp_sim --dir data/videos --port 8554

replay:  ## 분석기 없이 데모 번들 재생 서버
	.venv/bin/sceneproof-server run --mode replay --replay-bundle demo/bundle --vlm none

eval:  ## 사건 단위 평가 (3 모델) + 보고서
	for v in yolox_nano yolox_tiny yolox_s; do $(PY) -m sceneproof.eval.run_eval --variant $$v --out docs/eval/raw/event_eval_$$v.json; done
	$(PY) scripts/eval_report.py

bench:  ## 추론 성능 벤치마크 (조용한 CPU에서)
	$(PY) -m sceneproof.eval.bench_detector --video data/videos/campus-walkway.mp4 --frames 150 --repeats 3 --threads 4,1 --out docs/eval/raw/bench_detector.json
	$(PY) scripts/eval_report.py

scenarios:  ## 장애 시나리오 6종
	$(PY) scripts/failure_scenarios.py --out docs/evidence/failure-scenarios
	$(PY) scripts/eval_report.py

capture:  ## 문서용 캡처·시연 영상 (서버가 떠 있어야 함)
	node scripts/capture_demo.mjs http://127.0.0.1:8000 docs/images

static-demo:  ## GitHub Pages용 정적 재생 사이트 생성 (site/)
	$(PY) scripts/build_static_demo.py --bundle demo/bundle --out site

.PHONY: help setup test lint proto server gateway-run edge rtsp-sim replay eval bench scenarios capture static-demo
