#!/bin/bash
# The placement experiment: the same circuit under the default placement and
# under the communication-aware one, measured on the same allocation.
#
#   sbatch slurm/mapping_comparison.sbatch
#SBATCH --job-name=aegisq-mapping
#SBATCH --output=logs/aegisq-mapping-%j.out
#SBATCH --error=logs/aegisq-mapping-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G

set -euo pipefail

FAMILIES="${FAMILIES:-ghz qft ising random grover}"
QUBITS="${QUBITS:-28}"
REPEATS="${REPEATS:-5}"
NP="${NP:-8}"
OUTPUT="${OUTPUT:-benchmarks/raw/mapping_comparison_${SLURM_JOB_ID:-local}.csv}"

module load openmpi 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true

mkdir -p "$(dirname "$OUTPUT")" logs

for family in $FAMILIES; do
    for mapping in default optimized; do
        srun --ntasks="$NP" --cpus-per-task=1 \
            python -m aegisq.benchmark.runner \
                --circuit "$family" \
                --qubits "$QUBITS" \
                --experiment mapping_comparison \
                --mapping "$mapping" \
                --repeats "$REPEATS" \
                --threads 1 \
                --thread-policy one-thread-per-rank \
                --output "$OUTPUT"
    done
done
