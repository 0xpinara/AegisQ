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

# Where a failing run's output is kept. Writing to a file rather than
# relying on the pipe matters: when a rank dies from a signal, some process
# managers discard whatever it had buffered, and this suite has twice
# produced a CI failure whose entire diagnostic content was the word
# FAILED. Anything written before the crash survives on disk.
LOGDIR="$(mktemp -d)"
trap 'rm -rf "$LOGDIR"' EXIT

describe_status() {
    # Distinguish "pytest reported failures" from "the rank was killed",
    # which are different bugs and were indistinguishable before.
    local code="$1"
    if [ "$code" -gt 128 ]; then
        echo "exit $code (killed by signal $((code - 128)))"
    else
        echo "exit $code"
    fi
}

status=0
for np in $(echo "$RANKS" | tr ',' ' '); do
    echo "==> preflight: can $np rank(s) start at all?"
    # shellcheck disable=SC2086
    if ! mpirun $OVERSUBSCRIBE_FLAG -np "$np" "$PYTHON" -u -c \
        'from aegisq.runtime import distributed as d; print(f"  rank {d.rank()} of {d.world_size()} up", flush=True)'; then
        echo "    FAILED to start $np rank(s); the launcher or the build is the problem" >&2
        status=1
        continue
    fi

    echo "==> mpirun ${OVERSUBSCRIBE_FLAG} -np $np pytest tests/mpi"
    log="$LOGDIR/ranks-$np.log"
    # -u keeps stdout unbuffered, so a rank that dies still leaves behind
    # everything it had printed up to that point.
    # shellcheck disable=SC2086
    mpirun $OVERSUBSCRIBE_FLAG -np "$np" "$PYTHON" -u -m pytest tests/mpi \
        -q -p no:cacheprovider >"$log" 2>&1
    code=$?
    cat "$log"
    if [ "$code" -ne 0 ]; then
        echo "    FAILED at $np rank(s): $(describe_status $code)" >&2
        echo "    output above is $(wc -l <"$log" | tr -d ' ') line(s); empty output with a" >&2
        echo "    signal means the ranks died outside pytest -- start or teardown." >&2
        status=1
    fi
done
exit $status
