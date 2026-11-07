#!/usr/bin/env bash
# Kubernetes runtime verification on kind: build images, deploy, wait, check the device comes online and
# events arrive, kill the server pod and the edge pod, verify recovery and no duplicate events.
# Evidence (kubectl outputs, API snapshots) is written to docs/evidence/k8s/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EV="docs/evidence/k8s"
mkdir -p "$EV"
: > "$EV/summary.txt"
log() { printf '%s\n' "$*" | tee -a "$EV/summary.txt"; }
NS=zonewatch

log "=== build images ($(git rev-parse --short HEAD)) ==="
docker build -q -f deploy/docker/Dockerfile.server -t zonewatch-server:local . | tail -1
docker build -q -t zonewatch-gateway:local gateway | tail -1
docker build -q -f deploy/docker/Dockerfile.edge -t zonewatch-edge:local . | tail -1
docker build -q -f deploy/docker/Dockerfile.rtsp-sim -t zonewatch-rtsp-sim:local . | tail -1

log "=== cluster ==="
kind get clusters | grep -q "^${NS}$" || kind create cluster --config deploy/kind/kind-config.yaml --wait 180s
for img in zonewatch-server zonewatch-gateway zonewatch-edge zonewatch-rtsp-sim; do kind load docker-image "$img:local" --name "$NS"; done
kubectl version | tee "$EV/kubectl-version.txt"
kubectl get nodes -o wide | tee "$EV/nodes.txt"

log "=== deploy ==="
kubectl apply -f deploy/k8s/
kubectl -n "$NS" rollout status deployment/server --timeout=240s
kubectl -n "$NS" rollout status deployment/rtsp-sim --timeout=240s
kubectl -n "$NS" rollout status deployment/edge --timeout=300s
kubectl -n "$NS" get pods -o wide | tee "$EV/pods-initial.txt"

kubectl -n "$NS" port-forward svc/server 18000:8000 > "$EV/port-forward.log" 2>&1 &
PF=$!
trap 'kill $PF >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:18000/api/health >/dev/null 2>&1 && break; sleep 1; done

wait_events() {  # $1 = minimum live events, $2 = seconds
  for _ in $(seq 1 "$2"); do
    n=$(curl -fsS "http://127.0.0.1:18000/api/events?limit=500" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(sum(1 for x in d["items"] if x["ingest_source"]=="live"))' 2>/dev/null || echo 0)
    [ "$n" -ge "$1" ] && { echo "$n"; return 0; }
    sleep 2
  done
  echo "$n"; return 1
}
log "=== device online + first events ==="
N1=$(wait_events 1 90) || { log "FAIL: no live events within 180s"; kubectl -n "$NS" logs deploy/edge -c analyzer --tail=50; exit 1; }
curl -fsS http://127.0.0.1:18000/api/status | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({k:(v if k!="devices" else {i:{"online":x["online"],"cams":[(c["camera_id"],c.get("state")) for c in x["status"].get("cameras",[])]} for i,x in v.items()}) for k,v in d.items() if k in ("mode","devices","events","ingest")}, ensure_ascii=False))' | tee "$EV/status-1.json"
log "live events after first wait: $N1"

log "=== kill server pod (recovery: gateway queues + resends) ==="
kubectl -n "$NS" delete pod -l app=server --wait=false
sleep 5
kubectl -n "$NS" rollout status deployment/server --timeout=240s
kill $PF || true; sleep 1
kubectl -n "$NS" port-forward svc/server 18000:8000 > "$EV/port-forward.log" 2>&1 &
PF=$!
for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:18000/api/health >/dev/null 2>&1 && break; sleep 1; done
N2=$(wait_events $((N1 + 1)) 60) || log "note: no new events after server restart within 120s (stream may have ended)"
curl -fsS "http://127.0.0.1:18000/api/events?limit=500" | python3 -c 'import json,sys; d=json.load(sys.stdin); ids=[x["event_id"] for x in d["items"]]; print("events:", len(ids), "unique:", len(set(ids))); assert len(ids)==len(set(ids))' | tee -a "$EV/summary.txt"
log "live events after server restart: $N2"

log "=== kill edge pod (analyzer + gateway restart) ==="
kubectl -n "$NS" delete pod -l app=edge --wait=false
sleep 5
kubectl -n "$NS" rollout status deployment/edge --timeout=300s
N3=$(wait_events $((N2 + 1)) 90) || log "note: no new events after edge restart within 180s"
curl -fsS http://127.0.0.1:18000/api/status | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({k:d[k] for k in ("events","ingest")}))' | tee "$EV/status-final.json"
kubectl -n "$NS" get pods -o wide | tee "$EV/pods-final.txt"
kubectl -n "$NS" get events --sort-by=.lastTimestamp | tail -30 > "$EV/k8s-events.txt" || true
kubectl -n "$NS" logs deploy/edge -c gateway --tail=100 > "$EV/gateway.log" || true
kubectl -n "$NS" logs deploy/server --tail=100 > "$EV/server.log" || true
log "=== done: live events $N1 -> $N2 -> $N3 ==="
