#!/usr/bin/env bash
# Execute the distributed correctness suite at several rank counts.
#
# MPI tests cannot run inside a normal pytest process: pytest itself must be
# launched by mpirun so that every rank executes the same test body.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
RANKS="${MPI_RANKS:-4}"

if ! command -v mpirun >/dev/null 2>&1; then
    echo "mpirun not found on PATH; skipping distributed tests" >&2
    exit 0
fi

# Homebrew/OpenMPI on laptops frequently oversubscribes the physical cores.
OVERSUBSCRIBE_FLAG=""
if mpirun --help 2>&1 | grep -q -- "--oversubscribe"; then
    OVERSUBSCRIBE_FLAG="--oversubscribe"
fi

status=0
for np in $(echo "$RANKS" | tr ',' ' '); do
    echo "==> mpirun -np $np pytest tests/mpi"
    if ! mpirun $OVERSUBSCRIBE_FLAG -np "$np" "$PYTHON" -m pytest tests/mpi -q -p no:cacheprovider; then
        status=1
    fi
done
exit $status
