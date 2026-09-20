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

# Correctness tests routinely ask for more ranks than the machine has cores
# (a 2-core CI runner still has to prove the 4-rank code paths work). OpenMPI
# refuses that without --oversubscribe; MPICH does not accept the flag at all.
#
# The capability is probed by *running* a trivial job rather than by grepping
# `mpirun --help`: OpenMPI 5 lists the flag in its top-level help and OpenMPI 4
# does not, so parsing the help text silently drops the flag on older
# installations and the 4-rank run then dies on slot exhaustion.
OVERSUBSCRIBE_FLAG=""
if mpirun --oversubscribe -np 1 true >/dev/null 2>&1; then
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
