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

echo "==> optimisation levers (${QUBITS} qubits, ranks ${RANKS})"
"$PYTHON" -m aegisq.cli.main benchmark mapping \
    --circuits ghz,qft,ising,random,grover \
    --qubits "$QUBITS" --ranks "$RANKS" --repeats "$REPEATS" \
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

echo "==> post-quantum primitives and envelope"
"$PYTHON" -m aegisq.cli.main benchmark pqc --iterations 1000

echo "==> Grover query scaling"
"$PYTHON" -m aegisq.cli.main benchmark search

echo "==> regenerating tables, plots and the README block"
"$PYTHON" scripts/generate_report.py

echo
echo "Done. Raw data in benchmarks/raw/, derived output in benchmarks/processed/."
