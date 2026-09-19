#!/usr/bin/env bash
# Local benchmark sweep. Raw measurements are written to benchmarks/raw/ and
# are the only input accepted by the plotting scripts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -c "import aegisq.benchmark.runner" >/dev/null 2>&1; then
    echo "benchmark runner is not implemented yet in this checkout" >&2
    exit 1
fi

exec "$PYTHON" -m aegisq.benchmark.runner "$@"
