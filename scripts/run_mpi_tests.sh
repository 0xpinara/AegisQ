#!/usr/bin/env bash
# Execute the distributed correctness suite at several rank counts.
#
# MPI tests cannot run inside a normal pytest process: pytest itself must be
# launched by mpirun so that every rank executes the same test body.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
RANKS="${MPI_RANKS:-4}"

if ! command -v mpirun >/dev/null 2>&1; then
    echo "mpirun not found on PATH; skipping distributed tests" >&2
    exit 0
fi

# Refuse to launch an interpreter that cannot run these tests. `python3` on
# a developer machine is often the system one rather than the project's
# virtualenv, and mpirun will happily start four ranks of it.
#
# The probe asks for the MPI-enabled native core, not for `import aegisq`.
# Importing the package proves nothing: the working directory is the source
# tree, so `import aegisq` succeeds under any interpreter at all, while the
# compiled extension is tagged for one Python version. A mismatched
# interpreter therefore imports the package, gets `None` for the core, and
# skips all 166 distributed tests -- exiting zero, having proved nothing.
core_probe='from aegisq.runtime.distributed import mpi_compiled; raise SystemExit(0 if mpi_compiled() else 1)'
if ! "$PYTHON" -c "$core_probe" >/dev/null 2>&1; then
    if [ -x "$ROOT/.venv/bin/python" ] && "$ROOT/.venv/bin/python" -c "$core_probe" >/dev/null 2>&1; then
        PYTHON="$ROOT/.venv/bin/python"
    else
        echo "$PYTHON has no MPI-enabled aegisq core." >&2
        echo "Rebuild with 'pip install -e .' (CMake needs to find MPI), or set PYTHON." >&2
        exit 1
    fi
fi

# Correctness tests routinely ask for more ranks than the machine has cores
# (a 2-core CI runner still has to prove the 4-rank code paths work). OpenMPI
# refuses that without --oversubscribe; MPICH does not accept the flag and
# oversubscribes anyway.
#
# The probe launches a real program and checks its output, rather than only
# checking the exit status. A launcher that mis-parses the flag can still
# exit zero while running nothing, and an earlier version of this probe was
# fooled exactly that way: every subsequent run produced no output at all.
OVERSUBSCRIBE_FLAG="${AEGISQ_MPI_FLAGS-}"
if [ -z "${AEGISQ_MPI_FLAGS+x}" ]; then
    OVERSUBSCRIBE_FLAG=""
    probe="$(mpirun --oversubscribe -np 1 "$PYTHON" -c 'print("aegisq-probe-ok")' 2>/dev/null || true)"
    if [ "$probe" = "aegisq-probe-ok" ]; then
        OVERSUBSCRIBE_FLAG="--oversubscribe"
    fi
fi

echo "interpreter: $PYTHON"
echo "launcher: $(command -v mpirun)"
echo "extra flags: ${OVERSUBSCRIBE_FLAG:-<none>}"

status=0
for np in $(echo "$RANKS" | tr ',' ' '); do
    echo "==> mpirun ${OVERSUBSCRIBE_FLAG} -np $np pytest tests/mpi"
    # shellcheck disable=SC2086
    # -u keeps stdout unbuffered. Without it, a rank that dies takes the
    # whole run's buffered output with it, which is how a CI failure once
    # arrived with no diagnostic text at all.
    if ! mpirun $OVERSUBSCRIBE_FLAG -np "$np" "$PYTHON" -u -m pytest tests/mpi -q -p no:cacheprovider; then
        echo "    FAILED at $np rank(s)" >&2
        status=1
    fi
done
exit $status
