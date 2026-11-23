#!/usr/bin/env bash
# Checks the evaluation as a Kubernetes Job: how the cluster files are written, that the Job runs on a kind cluster
# and gives the recorded result, that the CPU-limit sweep script works, and that the committed measurement agrees
# with the result files it points to.
#
#   bash scripts/check_k8s_eval.sh [outDir]
#
#   STATIC_ONLY=1    only parts [1] and [4]; needs no Docker and no cluster
#   SKIP_BUILD=1     use the image sceneproof-eval:local that is already built
#   SWEEP_CHECK=0    skip part [3]
#   KEEP_CLUSTER=1   leave the cluster running (a cluster that was already there is always left alone)
#   JOB_TIMEOUT=2400 seconds to wait for the Job
#
# Why this exists: the analyser's pod has a CPU limit (deploy/k8s/30-edge.yaml) that was never measured. Offline, the
# evaluation analyses every frame, so the result is the same however little CPU it gets. In real time the video plays
# at its own speed and frames are dropped when the analyser falls behind: there the CPU limit decides the result.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
KIND_CFG=deploy/kind/kind-eval.yaml
K8S_DIR=deploy/k8s/eval
SWEEP=scripts/k8s_cpu_sweep.sh
MEASURED=docs/eval/raw/k8s_cpu_limits.json
RECORDED=docs/experiments/rerun-20261121/session_base.json
DF=deploy/docker/Dockerfile.eval
IMAGE=sceneproof-eval:local
CLUSTER=sceneproof-eval
NS=sceneproof-eval
JOB=eval-offline
HOST_OUT=/tmp/sceneproof-eval-out
JOB_TIMEOUT="${JOB_TIMEOUT:-2400}"
OUT="${1:-$(mktemp -d)}"
mkdir -p "$OUT"
fails=0
created_cluster=0
ok()  { echo "  ok   $*"; }
bad() { echo "  FAIL $*"; fails=$((fails + 1)); }
cleanup() {
  if [ "$created_cluster" = 1 ] && [ "${KEEP_CLUSTER:-0}" != 1 ]; then kind delete cluster --name "$CLUSTER" > /dev/null 2>&1 || true; fi
}
finish() { cleanup; if [ "$fails" -eq 0 ]; then echo; echo "all checks passed"; exit 0; fi; echo; echo "$fails check(s) failed"; exit 1; }
K() { kubectl --context "kind-$CLUSTER" "$@"; }
diagnose() {   # what the cluster says about the Job, for a person to read
  echo "       --- kubectl get job,pods"
  K -n "$NS" get job,pods -o wide 2>&1 | sed 's/^/       /'
  echo "       --- kubectl describe job (events)"
  K -n "$NS" describe job "$JOB" 2>&1 | sed -n '/^Events:/,$p' | tail -8 | sed 's/^/       /'
  echo "       --- kubectl describe pods (state and events)"
  K -n "$NS" describe pods -l "job-name=$JOB" 2>&1 | grep -E '^ +(State|Reason|Message|Exit Code):|^Events:|^ +(Warning|Normal) ' | tail -14 | sed 's/^/       /'
  echo "       --- kubectl logs (last lines)"
  K -n "$NS" logs "job/$JOB" --tail=12 2>&1 | sed 's/^/       /'
}

if ! "$PY" -c 'import yaml' 2>/dev/null; then echo "this check reads YAML with Python: run 'pip install pyyaml' (or set PYTHON=...)"; exit 2; fi

# The committed measurement, and the summary the sweep script writes, are checked by the same rules.
#   check_summary <summary.json> <fewest different limits> <fewest runs per limit> <videos, comma list or ''>
count_fails() {   # shows the lines a Python part wrote to $OUT/part.txt and counts the FAIL lines
  local n
  cat "$OUT/part.txt"
  n="$(grep -c '^  FAIL' "$OUT/part.txt")"
  rm -f "$OUT/part.txt"
  fails=$((fails + n))
  [ "$n" -eq 0 ]
}
check_summary() {
  "$PY" - "$1" "$2" "$3" "$4" "$RECORDED" > "$OUT/part.txt" <<'PY'
import json, sys
from pathlib import Path

sys.path.insert(0, "scripts")
from compare_event_eval import different_keys

path, min_limits, min_runs, cameras, recorded_path = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
fails = 0


def check(good, text, detail=""):
    global fails
    print(("  ok   " if good else "  FAIL ") + text)
    if not good:
        fails += 1
        if detail:
            print("       " + detail)


if not path.is_file():
    check(False, f"{path} exists")
    sys.exit(1)
try:
    summary = json.loads(path.read_text())
    runs = summary["runs"]
    assert isinstance(runs, list) and runs and all(isinstance(r, dict) for r in runs)
except Exception as e:  # noqa: BLE001
    check(False, f'{path} is JSON with a list "runs"', repr(e))
    sys.exit(1)
check(True, f"{path} lists {len(runs)} run(s)")
recorded = json.loads(Path(recorded_path).read_text())
wanted = set(cameras.split(",")) if cameras else set(recorded["per_video"])
per_limit, problems = {}, []
for i, r in enumerate(runs):
    where = f"runs[{i}]"
    missing = [k for k in ("cpu_limit", "threads", "run", "result", "frames", "dropped_frames", "values_different") if k not in r]
    if missing:
        problems.append(f"{where}: missing {', '.join(missing)}")
        continue
    if isinstance(r["cpu_limit"], bool) or not isinstance(r["cpu_limit"], (int, float)) or r["cpu_limit"] <= 0:
        problems.append(f"{where}: cpu_limit must be a number of CPUs such as 0.5 or 2, not {r['cpu_limit']!r}")
        continue
    where += f" (cpu {r['cpu_limit']}, run {r['run']})"
    file = path.parent / str(r["result"])
    if not file.is_file():
        problems.append(f"{where}: result file {file} is not there (the path is read from the summary's folder)")
        continue
    result = json.loads(file.read_text())
    per_limit.setdefault(float(r["cpu_limit"]), []).append(r)
    if result.get("realtime") is not True:
        problems.append(f"{where}: the result was not made in real time (realtime is {result.get('realtime')!r})")
    seen = (result.get("environment") or {}).get("cpu_limit")
    if seen is None or abs(seen - r["cpu_limit"]) > 1e-6:
        problems.append(f"{where}: the container saw a CPU limit of {seen!r}")
    threads = ((result.get("config") or {}).get("detector") or {}).get("intra_op_threads")
    if threads != r["threads"]:
        problems.append(f"{where}: threads is {r['threads']!r}, the result used {threads!r}")
    videos = result.get("per_video") or {}
    if set(videos) != wanted:
        problems.append(f"{where}: videos are {sorted(videos)}, expected {sorted(wanted)}")
    dropped = sum((v.get("run") or {}).get("dropped", 0) for v in videos.values())
    if not (r["dropped_frames"] == dropped == result.get("dropped_frames")):
        problems.append(f"{where}: dropped_frames is {r['dropped_frames']!r}, the result says {result.get('dropped_frames')!r} and its videos add up to {dropped}")
    frames = sum((v.get("run") or {}).get("frames", 0) for v in videos.values())
    if r["frames"] != frames:
        problems.append(f"{where}: frames is {r['frames']!r}, the result's videos add up to {frames}")
    if not result.get("git"):
        problems.append(f"{where}: the result does not say which commit it was made from")
    if set(videos) == set(recorded["per_video"]):
        different = len(different_keys(recorded, result)[0])
        if r["values_different"] != different:
            problems.append(f"{where}: values_different is {r['values_different']!r}, comparing with {recorded_path} gives {different}")
    elif not isinstance(r["values_different"], int):
        problems.append(f"{where}: values_different must be a whole number")
check(not problems, "every run agrees with the result file it points to", "\n       ".join(problems[:12]))
check(len(per_limit) >= min_limits, f"at least {min_limits} different CPU limit(s) (found: {sorted(per_limit)})")
short = {limit: len(rs) for limit, rs in per_limit.items() if len(rs) < min_runs}
check(not short, f"at least {min_runs} run(s) for every CPU limit", f"too few: {short}")
numbers = [(limit, r["run"]) for limit, rs in per_limit.items() for r in rs]
check(len(numbers) == len(set(numbers)), "no run is listed twice (cpu_limit, run)")
if not problems:
    print("       cpu  threads  run  frames  dropped  dropped %  values different")
    for limit in sorted(per_limit):
        for r in sorted(per_limit[limit], key=lambda r: r["run"]):
            total = r["frames"] + r["dropped_frames"]
            share = f"{100 * r['dropped_frames'] / total:.1f}" if total else "-"
            print(f"       {limit:<4} {r['threads']:<8} {r['run']:<4} {r['frames']:<7} {r['dropped_frames']:<8} {share:<10} {r['values_different']}")
PY
  count_fails
}

echo "[1] how the cluster files are written"
"$PY" - "$KIND_CFG" "$K8S_DIR" "$NS" "$JOB" "$IMAGE" "$CLUSTER" "$HOST_OUT" > "$OUT/part.txt" <<'PY'
import re, sys
from pathlib import Path

import yaml

kind_cfg, k8s_dir, ns, job_name, image, cluster, host_out = sys.argv[1:8]
fails = 0


def check(good, text, detail=""):
    global fails
    print(("  ok   " if good else "  FAIL ") + text)
    if not good:
        fails += 1
        if detail:
            print("       " + detail)
    return good


def cpu(value):  # "500m" -> 0.5, "2" -> 2.0, 1.5 -> 1.5
    text = str(value)
    return float(text[:-1]) / 1000 if text.endswith("m") else float(text)


def load(path):
    try:
        return [d for d in yaml.safe_load_all(Path(path).read_text()) if isinstance(d, dict)]
    except yaml.YAMLError as e:
        check(False, f"{path} is valid YAML", str(e).splitlines()[0])
        return []


# --- the kind cluster
if check(Path(kind_cfg).is_file(), f"{kind_cfg} exists"):
    docs = load(kind_cfg)
    cfg = docs[0] if docs else {}
    check(cfg.get("kind") == "Cluster" and cfg.get("name") == cluster, f"the cluster is named {cluster} (the existing one, sceneproof, is left alone)")
    mounts = [m for node in cfg.get("nodes") or [] for m in (node.get("extraMounts") or [])]
    check(any(m.get("hostPath") == host_out for m in mounts), f"a node mounts {host_out} from this computer (extraMounts): results come out there")

# --- the Kubernetes objects
files = sorted(p for p in Path(k8s_dir).glob("*") if p.suffix in (".yaml", ".yml")) if Path(k8s_dir).is_dir() else []
if not check(bool(files), f"{k8s_dir}/ holds YAML files"):
    sys.exit(1)
docs = [d for f in files for d in load(f)]
namespaces = [d for d in docs if d.get("kind") == "Namespace" and (d.get("metadata") or {}).get("name") == ns]
if check(bool(namespaces), f"Namespace {ns}"):
    labels = (namespaces[0].get("metadata") or {}).get("labels") or {}
    check(labels.get("pod-security.kubernetes.io/enforce") == "restricted",
          "the namespace enforces the 'restricted' Pod Security Standard (label pod-security.kubernetes.io/enforce)")
jobs = [d for d in docs if d.get("kind") == "Job" and (d.get("metadata") or {}).get("name") == job_name]
if not check(len(jobs) == 1, f"one Job named {job_name}"):
    sys.exit(1)
job = jobs[0]
check((job.get("metadata") or {}).get("namespace") == ns, f"the Job is in the namespace {ns}")
spec = job.get("spec") or {}
pod = (spec.get("template") or {}).get("spec") or {}
containers = pod.get("containers") or []
check(spec.get("backoffLimit") == 0, "backoffLimit: 0 (an evaluation that failed stays failed; it is not quietly run again)")
deadline = spec.get("activeDeadlineSeconds")
check(isinstance(deadline, int) and deadline > 0, "activeDeadlineSeconds is set (a Job that hangs is stopped)")
check(pod.get("restartPolicy") == "Never", "restartPolicy: Never")
if not check(len(containers) == 1, "the pod has one container"):
    sys.exit(1)
c = containers[0]
check(c.get("image") == image, f"the container runs the image {image}")
check(c.get("imagePullPolicy") in ("Never", "IfNotPresent"), "imagePullPolicy is Never or IfNotPresent (the image is loaded into the cluster, not pulled)")
args = [str(a) for a in (c.get("args") or [])]
check("--out" in args and args[args.index("--out") + 1:][:1] == ["/out/offline.json"], "args write the result to /out/offline.json (--out)")
check("--realtime" not in args, f"{job_name} runs offline (no --realtime)")
res = c.get("resources") or {}
have = all((res.get(kind) or {}).get(what) is not None for kind in ("requests", "limits") for what in ("cpu", "memory"))
if check(have, "CPU and memory each have a request and a limit"):
    try:
        check(cpu(res["requests"]["cpu"]) <= cpu(res["limits"]["cpu"]), "the CPU request is not larger than the CPU limit")
    except ValueError:
        check(False, "the CPU request and limit are written as Kubernetes quantities (for example 500m or 2)")
psc, csc = pod.get("securityContext") or {}, c.get("securityContext") or {}
check(csc.get("runAsNonRoot", psc.get("runAsNonRoot")) is True, "runAsNonRoot: true")
check(csc.get("allowPrivilegeEscalation") is False, "allowPrivilegeEscalation: false")
check("ALL" in ((csc.get("capabilities") or {}).get("drop") or []), "capabilities: drop ALL")
check(((csc.get("seccompProfile") or psc.get("seccompProfile") or {}).get("type")) in ("RuntimeDefault", "Localhost"), "seccompProfile: RuntimeDefault")
volumes = pod.get("volumes") or []
check(not any("hostPath" in v for v in volumes), "the pod does not mount a folder of the node directly (no hostPath volume in the pod)")
check(any(m.get("mountPath") == "/out" for m in (c.get("volumeMounts") or [])), "a volume is mounted at /out")
others = [f"{d.get('kind')}/{(d.get('metadata') or {}).get('name')}" for d in docs if d.get("kind") == "Job" and d is not job]
check(not others, f"{k8s_dir}/ holds no other Job (this check applies the whole folder)", ", ".join(others))
unfilled = [str(f) for f in files if re.search(r"\$\{|\{\{|<[A-Z_]+>", f.read_text())]
check(not unfilled, f"no file in {k8s_dir}/ has a placeholder to fill in", ", ".join(unfilled))
PY
count_fails
if [ -f "$SWEEP" ]; then
  ok "$SWEEP exists"
  if bash -n "$SWEEP" 2>/dev/null; then ok "$SWEEP has no syntax error (bash -n)"; else bad "$SWEEP has no syntax error (bash -n)"; fi
else
  bad "$SWEEP exists"
fi
static_fails=$fails

if [ "${STATIC_ONLY:-0}" != 1 ]; then
  echo
  echo "[2] the Job on a kind cluster"
  if [ "$static_fails" -ne 0 ]; then echo "  skip: fix part [1] first"; else
    for tool in docker kind kubectl; do command -v "$tool" > /dev/null || { bad "$tool is installed"; finish; }; done
    COMMIT=""
    if [ "${SKIP_BUILD:-0}" = 1 ]; then
      echo "  skip the build (SKIP_BUILD=1), using $IMAGE"
    else
      COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
      if docker build --build-arg GIT_COMMIT="$COMMIT" -f "$DF" -t "$IMAGE" . > "$OUT/build.log" 2>&1; then ok "docker build $IMAGE"; else bad "docker build $IMAGE (see $OUT/build.log)"; tail -12 "$OUT/build.log" | sed 's/^/       /'; finish; fi
    fi
    # the folder results come out in: made before the cluster so that it is not created as root, and writable by
    # the container's user, who is not the user of this computer
    mkdir -p "$HOST_OUT" && chmod 777 "$HOST_OUT" 2>/dev/null
    rm -f "$HOST_OUT/offline.json"
    if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
      ok "cluster $CLUSTER is already running"
    elif kind create cluster --config "$KIND_CFG" --wait 180s > "$OUT/kind-create.log" 2>&1; then
      created_cluster=1; ok "kind create cluster --config $KIND_CFG"
    else
      bad "kind create cluster --config $KIND_CFG (see $OUT/kind-create.log)"; tail -8 "$OUT/kind-create.log" | sed 's/^/       /'; finish
    fi
    if kind load docker-image "$IMAGE" --name "$CLUSTER" > "$OUT/kind-load.log" 2>&1 \
       || { docker save "$IMAGE" -o "$OUT/image.tar" && kind load image-archive "$OUT/image.tar" --name "$CLUSTER" >> "$OUT/kind-load.log" 2>&1; }; then
      ok "the image is loaded into the cluster"
    else
      bad "the image is loaded into the cluster (see $OUT/kind-load.log)"; tail -6 "$OUT/kind-load.log" | sed 's/^/       /'; finish
    fi
    rm -f "$OUT/image.tar"
    K -n "$NS" delete job "$JOB" --ignore-not-found --wait > /dev/null 2>&1    # a finished Job cannot be applied again
    if K apply -f "$K8S_DIR/" > "$OUT/apply.log" 2>&1; then ok "kubectl apply -f $K8S_DIR/"; else bad "kubectl apply -f $K8S_DIR/"; sed 's/^/       /' "$OUT/apply.log" | tail -8; finish; fi
    grep -i warning "$OUT/apply.log" | sed 's/^/       /'
    echo "  waiting for the Job (up to ${JOB_TIMEOUT}s; offline, all four videos)"
    start=$SECONDS; state=""; last=""
    while [ $((SECONDS - start)) -lt "$JOB_TIMEOUT" ]; do
      succeeded="$(K -n "$NS" get job "$JOB" -o jsonpath='{.status.succeeded}' 2>/dev/null)"
      failed="$(K -n "$NS" get job "$JOB" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)"
      if [ "${succeeded:-0}" = 1 ]; then state=complete; break; fi
      if [ "$failed" = True ]; then state=failed; break; fi
      now="$(K -n "$NS" get pods -l "job-name=$JOB" --no-headers 2>/dev/null | awk '{print $3}' | head -1)"
      waited=$((SECONDS - start))
      if [ "${now:-none}" != "$last" ]; then echo "       ${waited}s: pod ${now:-not created yet}"; last="${now:-none}"; fi
      # a pod that is not running after three minutes will not start by itself
      if [ "$waited" -ge 180 ] && [ "${now:-none}" != Running ] && [ "${now:-none}" != Completed ]; then state="stuck (pod: ${now:-not created})"; break; fi
      sleep 5
    done
    K -n "$NS" get job,pods -o wide > "$OUT/job.txt" 2>&1
    K -n "$NS" logs "job/$JOB" > "$OUT/job.log" 2>&1
    if [ "$state" = complete ]; then
      ok "the Job finished ($((SECONDS - start))s)"
    else
      bad "the Job finished (${state:-not within ${JOB_TIMEOUT}s})"; diagnose; finish
    fi
    limit="$(K -n "$NS" get job "$JOB" -o jsonpath='{.spec.template.spec.containers[0].resources.limits.cpu}' 2>/dev/null)"
    if [ -s "$HOST_OUT/offline.json" ]; then
      ok "the result came out at $HOST_OUT/offline.json"
      cp "$HOST_OUT/offline.json" "$OUT/offline.json"
      if COMMIT="$COMMIT" "$PY" - "$OUT/offline.json" "$limit" <<'PY'
import json, os, sys
run, limit = json.load(open(sys.argv[1])), sys.argv[2]
limit = float(limit[:-1]) / 1000 if limit.endswith("m") else float(limit)
env = run.get("environment") or {}
problems = []
if run.get("realtime") is not False or run.get("dropped_frames") != 0:
    problems.append(f"realtime is {run.get('realtime')!r} and dropped_frames is {run.get('dropped_frames')!r}; offline they are false and 0")
if env.get("cpu_limit") is None or abs(env["cpu_limit"] - limit) > 1e-6:
    problems.append(f"environment.cpu_limit is {env.get('cpu_limit')!r}, the Job's limit is {limit}")
commit = os.environ.get("COMMIT")
if not run.get("git"):
    problems.append("git is empty: the result does not say which commit it was made from")
elif commit and run["git"] != commit:
    problems.append(f"git is {run['git']!r}, but the image was built from {commit!r}")
for p in problems:
    print("       " + p)
print(f"       commit {run.get('git')}, cpu limit {env.get('cpu_limit')} of {env.get('cpu_count')} CPUs, "
      f"threads {((run.get('config') or {}).get('detector') or {}).get('intra_op_threads')}")
sys.exit(1 if problems else 0)
PY
      then ok "the result says how it was made: offline, its commit, and the Job's CPU limit ($limit)"; else bad "the result says how it was made: offline, its commit, and the Job's CPU limit ($limit)"; fi
      if "$PY" scripts/compare_event_eval.py "$RECORDED" "$OUT/offline.json" > "$OUT/recorded.log" 2>&1; then ok "same as the recorded result ($(head -1 "$OUT/recorded.log"))"; else bad "same as the recorded result"; sed 's/^/       /' "$OUT/recorded.log" | head -10; fi
    else
      bad "the result came out at $HOST_OUT/offline.json"; diagnose
      echo "       (a cluster made from an older $KIND_CFG keeps its old mounts: kind delete cluster --name $CLUSTER)"
    fi

    echo
    echo "[3] the CPU-limit sweep script: one short real-time run"
    if [ "${SWEEP_CHECK:-1}" = 0 ]; then echo "  skip (SWEEP_CHECK=0)"; elif [ "$fails" -ne 0 ]; then echo "  skip: fix part [2] first"; else
      mkdir -p "$OUT/sweep"; rm -f "$OUT/sweep/summary.json"
      echo "  bash $SWEEP --limits \"1\" --runs 1 --cameras eval-indoor --out $OUT/sweep/summary.json"
      if bash "$SWEEP" --limits "1" --runs 1 --cameras eval-indoor --out "$OUT/sweep/summary.json" > "$OUT/sweep/sweep.log" 2>&1; then
        ok "the sweep script ran (log in $OUT/sweep/sweep.log)"
        check_summary "$OUT/sweep/summary.json" 1 1 eval-indoor
      else
        bad "the sweep script ran (see $OUT/sweep/sweep.log)"; tail -12 "$OUT/sweep/sweep.log" | sed 's/^/       /'
      fi
    fi
  fi
fi

echo
echo "[4] the committed measurement ($MEASURED)"
check_summary "$MEASURED" 3 2 ""
[ "${STATIC_ONLY:-0}" = 1 ] || echo "  results kept in $OUT"
finish
