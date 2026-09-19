#!/usr/bin/env bash
# Run the full local verification sweep: C++ tests, Python tests, MPI tests.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
PYTHON="${PYTHON:-python3}"

echo "==> C++ unit tests"
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo "==> Python tests (single process)"
"$PYTHON" -m pytest tests -m "not slow and not mpi and not gpu"

if command -v mpirun >/dev/null 2>&1; then
    echo "==> MPI tests"
    ./scripts/run_mpi_tests.sh
else
    echo "==> MPI tests skipped (no mpirun on PATH)"
fi
