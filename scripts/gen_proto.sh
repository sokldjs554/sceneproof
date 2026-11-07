#!/usr/bin/env bash
# Regenerates Python and Go code from proto/. Run from the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[gen_proto] python -> src/zonewatch/proto"
python -m grpc_tools.protoc -I proto \
  --python_out=src/zonewatch/proto \
  --grpc_python_out=src/zonewatch/proto \
  --pyi_out=src/zonewatch/proto \
  proto/zonewatch/v1/zonewatch.proto
# grpc_tools emits absolute imports (zonewatch.v1....); rewrite to package-relative.
sed -i 's/^from zonewatch\.v1 import zonewatch_pb2 as zonewatch_dot_v1_dot_zonewatch__pb2/from zonewatch.proto.zonewatch.v1 import zonewatch_pb2 as zonewatch_dot_v1_dot_zonewatch__pb2/' \
  src/zonewatch/proto/zonewatch/v1/zonewatch_pb2_grpc.py
touch src/zonewatch/proto/zonewatch/__init__.py src/zonewatch/proto/zonewatch/v1/__init__.py

echo "[gen_proto] go -> gateway/gen/zonewatchv1"
mkdir -p gateway/gen/zonewatchv1
protoc -I proto \
  --go_out=gateway/gen/zonewatchv1 --go_opt=paths=source_relative \
  --go-grpc_out=gateway/gen/zonewatchv1 --go-grpc_opt=paths=source_relative \
  proto/zonewatch/v1/zonewatch.proto
# flatten zonewatch/v1/* into the package dir
mv -f gateway/gen/zonewatchv1/zonewatch/v1/*.go gateway/gen/zonewatchv1/ 2>/dev/null || true
rm -rf gateway/gen/zonewatchv1/zonewatch
echo "[gen_proto] done"
