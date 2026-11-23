#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON:-python3}" "$(dirname "$0")/k8s_cpu_sweep.py" "$@"
