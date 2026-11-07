#!/usr/bin/env bash
# CV+VLM evaluation in one command (run only after the paid-API/network decision):
#   1) analyse the 4 held-out videos offline (CV path, evidence written to data/evidence-eval)
#   2) start a review server with the chosen VLM provider on a dedicated data dir and port
#   3) wait until every closed event has a final explanation status
#   4) compute CV-only vs CV+VLM numbers + write the human rubric CSV, regenerate docs/eval/README.md
# Usage: ZW_SERVER_VLM_PROVIDER=anthropic ZW_SERVER_VLM_BUDGET_USD=5 bash scripts/run_vlm_eval.sh [yolox_tiny]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VARIANT="${1:-yolox_tiny}"
PROVIDER="${ZW_SERVER_VLM_PROVIDER:-anthropic}"
DATA_DIR="data/server-eval-${PROVIDER}"
PORT="${ZW_EVAL_PORT:-8020}"
BUNDLE="data/runs/eval-bundle-${VARIANT}"
. .venv/bin/activate 2>/dev/null || true

echo "[1/4] offline CV runs (${VARIANT}) -> ${BUNDLE}"
rm -rf "$BUNDLE" && mkdir -p "$BUNDLE"
sed -e "s/variant: yolox_tiny/variant: ${VARIANT}/" -e "s#models/yolox_tiny.onnx#models/${VARIANT}.onnx#" configs/eval.yaml > "$BUNDLE/eval.yaml"
for cam in eval-campus eval-indoor eval-store eval-parking; do
  zonewatch-edge --config "$BUNDLE/eval.yaml" --log-level WARNING offline --camera "$cam" --out "$BUNDLE/events.jsonl"
done
ln -sfn "$ROOT/data/evidence-eval" "$BUNDLE/evidence"

echo "[2/4] server with provider=${PROVIDER} on :${PORT} (data: ${DATA_DIR})"
rm -rf "$DATA_DIR"
ZW_SERVER_DATA_DIR="$DATA_DIR" ZW_SERVER_HTTP_PORT="$PORT" ZW_SERVER_GRPC_LISTEN="0.0.0.0:50072" ZW_SERVER_MODE=replay \
  ZW_SERVER_REPLAY_BUNDLE="$BUNDLE" ZW_SERVER_VLM_PROVIDER="$PROVIDER" ZW_SERVER_VLM_MAX_CONCURRENCY="${ZW_SERVER_VLM_MAX_CONCURRENCY:-2}" \
  zonewatch-server run > "$BUNDLE/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 60); do curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1 && break; sleep 1; done

echo "[3/4] waiting for explanations"
for _ in $(seq 1 600); do
  st=$(curl -fsS "http://127.0.0.1:${PORT}/api/status" | python -c 'import json,sys; d=json.load(sys.stdin)["vlm"]["db"]; print(d.get("queued",0)+d.get("running",0), json.dumps(d))')
  pending=${st%% *}
  echo "   pending=${pending} ${st#* }"
  [ "$pending" = "0" ] && break
  sleep 5
done
kill $SRV; wait $SRV 2>/dev/null || true

echo "[4/4] metrics + report"
python -m zonewatch.eval.vlm_eval --data-dir "$DATA_DIR" --event-eval "docs/eval/raw/event_eval_${VARIANT}.json" --out docs/eval/raw/vlm_eval.json
python scripts/eval_report.py
echo "done. rubric: docs/eval/raw/vlm_eval-rubric.csv"
