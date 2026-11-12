#!/usr/bin/env bash
# CV+VLM on videos the prompt never saw (plan, fixed before the paid run: configs/bench/vlm_new_videos.yaml).
#   1) fetch the 12 DEV-split sequences, build the edge config and the ground-truth YAMLs
#   2) product analyser on every video -> events_all.jsonl; select the events of the plan -> bundle
#   3) review server with the VLM provider and the plan's budget (hard stop), wait for every explanation
#   4) match against the ground truth with ALL CV events and write docs/eval/raw/vlm_eval_new_videos.json
# Paid: run it in CI (vlm-eval workflow with new_videos=true). Locally: SP_SERVER_VLM_PROVIDER=mock (free) to test.
# Env: WORK (default data/runs/vlm-new), OUT (default docs/eval/raw/vlm_eval_new_videos.json), SKIP_CV=1 reuses
# events_all.jsonl. Nothing is deleted: every run gets its own server data directory.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -x .venv/bin/python ] && export PATH="$ROOT/.venv/bin:$PATH"  # a local venv if there is one (CI installs globally)
PROVIDER="${SP_SERVER_VLM_PROVIDER:-anthropic}"
WORK="${WORK:-data/runs/vlm-new}"
OUT="${OUT:-docs/eval/raw/vlm_eval_new_videos.json}"
DATA_DIR="data/server-vlm-new-${PROVIDER}-$(date +%Y%m%d-%H%M%S)"
PORT="${SP_EVAL_PORT:-8023}"
PLAN=configs/bench/vlm_new_videos.yaml
VIDEOS=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('$PLAN'))['videos']))")
BUDGET=$(python -c "import yaml; print(yaml.safe_load(open('$PLAN'))['budget_usd'])")

echo "[1/4] data + inputs"
python scripts/fetch_bench.py --ids "$(echo $VIDEOS | tr ' ' ',')"
python scripts/vlm_new_videos.py prepare --work "$WORK"

echo "[2/4] product analyser"
if [ "${SKIP_CV:-0}" != 1 ]; then
  : > "$WORK/events_all.jsonl"
  for cam in $VIDEOS; do
    sceneproof-edge --config "$WORK/edge.yaml" --log-level WARNING offline --camera "$cam" --out "$WORK/events_all.jsonl"
  done
fi
python scripts/vlm_new_videos.py select --work "$WORK"
ln -sfn "$(cd "$WORK" && pwd)/evidence" "$WORK/bundle/evidence"

echo "[3/4] server provider=${PROVIDER} budget=\$${BUDGET} on :${PORT} (data: ${DATA_DIR})"
SP_SERVER_DATA_DIR="$DATA_DIR" SP_SERVER_HTTP_PORT="$PORT" SP_SERVER_GRPC_LISTEN="0.0.0.0:50073" SP_SERVER_MODE=replay \
  SP_SERVER_REPLAY_BUNDLE="$WORK/bundle" SP_SERVER_VLM_PROVIDER="$PROVIDER" SP_SERVER_VLM_BUDGET_USD="$BUDGET" \
  SP_SERVER_VLM_MAX_CONCURRENCY="${SP_SERVER_VLM_MAX_CONCURRENCY:-2}" \
  sceneproof-server run > "$WORK/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 60); do curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1 && break; sleep 1; done
for _ in $(seq 1 600); do
  st=$(curl -fsS "http://127.0.0.1:${PORT}/api/status" | python -c 'import json,sys; d=json.load(sys.stdin)["vlm"]["db"]; print(d.get("queued",0)+d.get("running",0), json.dumps(d))')
  echo "   pending=${st%% *} ${st#* }"
  [ "${st%% *}" = "0" ] && break
  sleep 5
done
kill $SRV; wait $SRV 2>/dev/null || true

echo "[4/4] matching (all CV events) + report -> $OUT"
python -m sceneproof.eval.vlm_eval --data-dir "$DATA_DIR" --events-jsonl "$WORK/events_all.jsonl" --gt "$WORK/gt" --out "$OUT"
cp "$WORK/selection.json" "${OUT%.json}-selection.json"
echo "done"
