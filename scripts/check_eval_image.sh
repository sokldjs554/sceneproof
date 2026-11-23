#!/usr/bin/env bash
# Checks the evaluation image (deploy/docker/Dockerfile.eval): how it is written, that it builds, and that a
# container made from it reads exactly the recorded input files and gives the same events on every run.
#
#   bash scripts/check_eval_image.sh [outDir]
#
#   SKIP_BUILD=1   use an image that is already built (IMAGE, default sceneproof-eval:check)
#   STATIC_ONLY=1  only part [1]; needs no Docker
#
# Why this exists: the same settings gave different results on different computers
# (docs/experiments/rerun-20261121). ffmpeg writes a different working video depending on the number of CPU
# cores unless the thread count is fixed; the analyser reads video through GStreamer when it is installed and
# through OpenCV when it is not; and library versions differed. The image fixes all three, and the result file
# names the inputs (SHA-256) and the tools it was made with.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DF=deploy/docker/Dockerfile.eval
CONS=deploy/docker/eval.constraints.txt
LOCK=docs/eval/inputs.lock.json
RECORDED=docs/experiments/rerun-20261121/session_base.json
IMAGE="${IMAGE:-sceneproof-eval:check}"
OUT="${1:-$(mktemp -d)}"
mkdir -p "$OUT"
fails=0
ok()  { echo "  ok   $*"; }
bad() { echo "  FAIL $*"; fails=$((fails + 1)); }
check() { if eval "$1" >/dev/null 2>&1; then ok "$2"; else bad "$2"; fi; }
finish() { if [ "$fails" -eq 0 ]; then echo; echo "all checks passed"; exit 0; fi; echo; echo "$fails check(s) failed"; exit 1; }

echo "[1] how the image is written"
if [ ! -f "$DF" ]; then bad "$DF exists"; finish; fi
ok "$DF exists"
# every base image is named by digest: a tag such as :24.04 can point at a different image next month
stages="$(grep -iE '^FROM ' "$DF" | sed -nE 's/.* [Aa][Ss] ([A-Za-z0-9._-]+)$/\1/p')"
unpinned=0
while read -r _ image _; do
  [ -n "$image" ] || continue
  if echo "$stages" | grep -qx "$image"; then continue; fi          # FROM <an earlier stage>
  echo "$image" | grep -qE '@sha256:[0-9a-f]{64}$' || { unpinned=$((unpinned + 1)); echo "       not pinned by digest: $image"; }
done < <(grep -iE '^FROM ' "$DF")
[ "$unpinned" -eq 0 ] && ok "every base image is pinned by digest (@sha256:…)" || bad "every base image is pinned by digest (@sha256:…)"
check "! grep -qE ':latest( |$)' '$DF'" "no ':latest' tag"
last_user="$(grep -iE '^USER ' "$DF" | tail -1 | awk '{print $2}')"
if [ -n "$last_user" ] && [ "$last_user" != root ] && [ "${last_user%%:*}" != 0 ]; then ok "the container does not run as root (USER $last_user)"; else bad "the container does not run as root (last USER: '${last_user:-none}')"; fi
if [ -s "$CONS" ]; then
  loose="$(grep -vE '^\s*(#|$)' "$CONS" | grep -vE '^[A-Za-z0-9._-]+(\[[A-Za-z0-9._,-]+\])?==[A-Za-z0-9.+!_-]+$' || true)"
  [ -z "$loose" ] && ok "$CONS pins every package to one exact version (name==version)" || { bad "$CONS pins every package to one exact version"; echo "$loose" | head -5 | sed 's/^/       /'; }
  for pkg in numpy onnxruntime opencv-python-headless scipy; do
    check "grep -qiE '^$pkg==' '$CONS'" "$CONS pins $pkg"
  done
else
  bad "$CONS exists and is not empty"
fi
check "grep -q 'eval.constraints.txt' '$DF'" "the image installs Python packages with the constraints file"
check "grep -qE -- '-threads' scripts/fetch_data.sh" "scripts/fetch_data.sh can fix the ffmpeg thread count (-threads)"
check "grep -qE 'provenance +verify' '$DF'" "the build stops when an input differs from $LOCK (provenance verify)"
check "[ -s '$LOCK' ]" "$LOCK exists"
if [ "${STATIC_ONLY:-0}" = 1 ]; then finish; fi

echo
echo "[2] build"
if [ "${SKIP_BUILD:-0}" = 1 ]; then
  echo "  skip (SKIP_BUILD=1), using $IMAGE"
else
  if docker build -f "$DF" -t "$IMAGE" . > "$OUT/build.log" 2>&1; then ok "docker build ($(grep -c . "$OUT/build.log") log lines in $OUT/build.log)"; else bad "docker build (see $OUT/build.log)"; tail -15 "$OUT/build.log" | sed 's/^/       /'; finish; fi
fi

echo
echo "[3] a container made from the image"
uid="$(docker run --rm --entrypoint id "$IMAGE" -u 2>/dev/null || echo '?')"
if [ "$uid" != 0 ] && [ "$uid" != '?' ]; then ok "runs as a non-root user (uid $uid)"; else bad "runs as a non-root user (uid $uid)"; fi
if docker run --rm --entrypoint python "$IMAGE" -m sceneproof.eval.provenance verify --lock "$LOCK" > "$OUT/verify.log" 2>&1; then ok "the files inside the image match $LOCK"; else bad "the files inside the image match $LOCK"; sed 's/^/       /' "$OUT/verify.log" | head -8; fi
chmod 777 "$OUT"
for n in 1 2; do
  if docker run --rm -v "$OUT:/out" "$IMAGE" --out "/out/run$n.json" > "$OUT/run$n.log" 2>&1 && [ -s "$OUT/run$n.json" ]; then ok "evaluation run $n wrote run$n.json"; else bad "evaluation run $n wrote run$n.json (see $OUT/run$n.log)"; tail -6 "$OUT/run$n.log" | sed 's/^/       /'; finish; fi
done
if python3 - "$OUT/run1.json" "$LOCK" <<'PY'
import json, sys
run, lock = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))["files"]
env, inputs = run.get("environment") or {}, run.get("inputs") or {}
problems = [f"environment.{k} missing" for k in ("python", "platform", "numpy", "opencv", "onnxruntime", "ffmpeg") if not env.get(k)]
backend = ((run.get("config") or {}).get("pipeline") or {}).get("source_backend")
if backend not in ("gstreamer", "opencv"):
    problems.append(f"config.pipeline.source_backend is {backend!r}: the image must name the way video is read, not leave it to 'auto'")
if "gstreamer" not in env or (backend == "gstreamer" and not env.get("gstreamer")):
    problems.append("environment.gstreamer must be the GStreamer version when video is read through GStreamer (None when it is absent)")
problems += [f"inputs: {name} is {(inputs.get(name) or {}).get('sha256')}, lock says {sha}" for name, sha in lock.items() if (inputs.get(name) or {}).get("sha256") != sha]
for p in problems:
    print("       " + p)
print("       tools: " + ", ".join(f"{k} {env.get(k)}" for k in ("python", "numpy", "opencv", "onnxruntime")))
print("       ffmpeg: " + str(env.get("ffmpeg")))
print("       video read through: " + str(backend) + " (gstreamer " + str(env.get("gstreamer")) + ")")
sys.exit(1 if problems else 0)
PY
then ok "the result names its tools and its inputs, and the inputs are the locked files"; else bad "the result names its tools and its inputs, and the inputs are the locked files"; fi
if python3 scripts/compare_event_eval.py "$OUT/run1.json" "$OUT/run2.json" > "$OUT/repeat.log" 2>&1; then ok "two runs give the same events ($(head -1 "$OUT/repeat.log"))"; else bad "two runs give the same events"; sed 's/^/       /' "$OUT/repeat.log" | head -8; fi

echo
echo "[4] against the recorded result ($RECORDED)"
# The recorded result was made outside any image, reading video through GStreamer 1.24. An image with the same
# inputs, the same library versions and the same way of reading video gives exactly the same events, so a result
# from the image can stand in for the recorded one. ALLOW_DIFFERENT=1 reports a difference without failing.
if python3 scripts/compare_event_eval.py "$RECORDED" "$OUT/run1.json" > "$OUT/recorded.log" 2>&1; then
  ok "same as the recorded result ($(head -1 "$OUT/recorded.log"))"
else
  if [ "${ALLOW_DIFFERENT:-0}" = 1 ]; then echo "  note different from the recorded result (not counted: ALLOW_DIFFERENT=1)"; else bad "same as the recorded result"; fi
  sed 's/^/       /' "$OUT/recorded.log" | head -12
fi
echo "  results kept in $OUT"
finish
