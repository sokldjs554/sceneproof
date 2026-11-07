#!/usr/bin/env bash
# Regenerates Python and Go code from proto/. Run from the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[gen_proto] python -> src/sceneproof/proto"
python -m grpc_tools.protoc -I proto \
  --python_out=src/sceneproof/proto \
  --grpc_python_out=src/sceneproof/proto \
  --pyi_out=src/sceneproof/proto \
  proto/sceneproof/v1/sceneproof.proto
# grpc_tools emits absolute imports (sceneproof.v1....); rewrite to package-relative.
sed -i 's/^from sceneproof\.v1 import sceneproof_pb2 as sceneproof_dot_v1_dot_sceneproof__pb2/from sceneproof.proto.sceneproof.v1 import sceneproof_pb2 as sceneproof_dot_v1_dot_sceneproof__pb2/' \
  src/sceneproof/proto/sceneproof/v1/sceneproof_pb2_grpc.py
touch src/sceneproof/proto/sceneproof/__init__.py src/sceneproof/proto/sceneproof/v1/__init__.py

echo "[gen_proto] go -> gateway/gen/sceneproofv1"
mkdir -p gateway/gen/sceneproofv1
protoc -I proto \
  --go_out=gateway/gen/sceneproofv1 --go_opt=paths=source_relative \
  --go-grpc_out=gateway/gen/sceneproofv1 --go-grpc_opt=paths=source_relative \
  proto/sceneproof/v1/sceneproof.proto
# flatten sceneproof/v1/* into the package dir
mv -f gateway/gen/sceneproofv1/sceneproof/v1/*.go gateway/gen/sceneproofv1/ 2>/dev/null || true
rm -rf gateway/gen/sceneproofv1/sceneproof
echo "[gen_proto] done"
