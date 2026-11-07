#!/usr/bin/env bash
# Regenerate demo/bundle from the two dev samples, optionally with real VLM explanations.
# Usage: bash scripts/rebuild_demo_bundle.sh [provider]   (provider default: none)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PROVIDER="${1:-none}"
WORK="data/runs/demo-build"
. .venv/bin/activate 2>/dev/null || true
rm -rf "$WORK" && mkdir -p "$WORK/bundle"
for cam in cam-worker cam-bench; do
  sceneproof-edge --config configs/edge.yaml --log-level WARNING offline --camera "$cam" --out "$WORK/bundle/events.jsonl"
done
ln -sfn "$ROOT/data/evidence" "$WORK/bundle/evidence"
SP_SERVER_DATA_DIR="$WORK/server" SP_SERVER_HTTP_PORT=8021 SP_SERVER_MODE=replay SP_SERVER_REPLAY_BUNDLE="$WORK/bundle" \
  SP_SERVER_VLM_PROVIDER="$PROVIDER" sceneproof-server run > "$WORK/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:8021/api/health >/dev/null 2>&1 && break; sleep 1; done
if [ "$PROVIDER" != "none" ]; then
  for _ in $(seq 1 600); do
    p=$(curl -fsS http://127.0.0.1:8021/api/status | python -c 'import json,sys; d=json.load(sys.stdin)["vlm"]["db"]; print(d.get("queued",0)+d.get("running",0))')
    [ "$p" = "0" ] && break; sleep 5
  done
fi
kill $SRV; wait $SRV 2>/dev/null || true
rm -rf demo/bundle
SP_SERVER_DATA_DIR="$WORK/server" sceneproof-server export demo/bundle
echo "demo/bundle rebuilt (provider=${PROVIDER}); commit it together with docs/images"
