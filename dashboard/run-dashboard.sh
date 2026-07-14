#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec env \
  DASHBOARD_BIND="${DASHBOARD_BIND:-127.0.0.1}" \
  DASHBOARD_PORT="${DASHBOARD_PORT:-11002}" \
  DS4_METRICS_URL="${DS4_METRICS_URL:-http://127.0.0.1:8000/metrics}" \
  python3 "$ROOT/server.py"
