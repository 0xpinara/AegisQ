#!/usr/bin/env bash
# A small end-to-end benchmark sweep sized for a laptop.
#
# Writes raw measurements to benchmarks/raw/ and regenerates every processed
# table, plot and README figure from them. Roughly two minutes on an 8-core
# machine.
#
#   ./scripts/benchmark_local.sh             # default sizes
#   QUBITS=22 RANKS=2,4 ./scripts/benchmark_local.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
QUBITS="${QUBITS:-20}"
RANKS="${RANKS:-2,4,8}"
SCALING_QUBITS="${SCALING_QUBITS:-22}"
SCALING_RANKS="${SCALING_RANKS:-1,2,4,8}"
REPEATS="${REPEATS:-3}"

if ! "$PYTHON" -c "import aegisq.benchmark.runner" >/dev/null 2>&1; then
    echo "aegisq is not importable; run 'pip install -e .' first" >&2
    exit 1
fi

echo "==> harness calibration: the same run against itself (${TRIALS:-10} trials/cell)"
# Measured first and under the same conditions as the placement sweep that
# follows, because it is the null those results are judged against.
"$PYTHON" -m aegisq.cli.main benchmark calibrate \
    --circuits ghz,qft,ising,grover,random \
    --qubits "$QUBITS" --ranks "$RANKS" --trials "${TRIALS:-10}" \
    --option grover:iterations=2

echo "==> optimisation levers (${QUBITS} qubits, ranks ${RANKS})"
# --launches is the estimator the calibration suite argues for: the minimum
# over independent launches rather than over repeats inside one, because
# interference here only ever slows a run down. Five launches take the
# median resolution from about 27% to about 6%, which is the difference
# between reporting wall time and not being able to.
"$PYTHON" -m aegisq.cli.main benchmark mapping \
    --circuits ghz,qft,ising,random,grover \
    --qubits "$QUBITS" --ranks "$RANKS" --repeats "$REPEATS" \
    --launches "${LAUNCHES:-5}" \
    --levers placement,windowed,fusion \
    --option grover:iterations=2

echo "==> strong scaling (${SCALING_QUBITS} qubits, ranks ${SCALING_RANKS})"
for circuit in qft ising; do
    "$PYTHON" -m aegisq.cli.main benchmark strong \
        --circuit "$circuit" --qubits "$SCALING_QUBITS" --ranks "$SCALING_RANKS" \
        --repeats "$REPEATS" --thread-policy one-thread-per-rank
done

echo "==> weak scaling"
"$PYTHON" -m aegisq.cli.main benchmark weak \
    --circuit ising --qubits $((SCALING_QUBITS - 2)) --ranks "$SCALING_RANKS" \
    --repeats "$REPEATS" --thread-policy one-thread-per-rank

echo "==> local kernel bandwidth"
"$PYTHON" -m aegisq.cli.main benchmark kernels --qubits "$SCALING_QUBITS" --threads 1,2,4,8

echo "==> placement search quality"
"$PYTHON" -m aegisq.cli.main benchmark placement --qubits 18 --ranks 4,8,16 --samples 30

echo "==> single-precision error"
"$PYTHON" -m aegisq.cli.main benchmark precision --qubits 18 --depths 2,8,32,128,512

echo "==> post-quantum primitives and envelope"
"$PYTHON" -m aegisq.cli.main benchmark pqc --iterations 1000

echo "==> Grover query scaling"
"$PYTHON" -m aegisq.cli.main benchmark search

echo "==> regenerating tables, plots and the README block"
"$PYTHON" scripts/generate_report.py

echo
echo "Done. Raw data in benchmarks/raw/, derived output in benchmarks/processed/."
