#!/usr/bin/env bash
# Configure and build the AegisQ native core plus the Python extension.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
PYTHON="${PYTHON:-python3}"

echo "==> configuring ($BUILD_TYPE) in $BUILD_DIR"
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DPython_EXECUTABLE="$("$PYTHON" -c 'import sys; print(sys.executable)')" \
    "$@"

echo "==> building"
cmake --build "$BUILD_DIR" --parallel

echo "==> done"
