#!/bin/bash
# Weak scaling: one extra qubit per doubling of ranks, so the shard size is
# constant and ideal behaviour is constant wall time.
#
#   sbatch --array=0-3 slurm/weak_scaling.sbatch
#SBATCH --job-name=aegisq-weak
#SBATCH --output=logs/aegisq-weak-%A_%a.out
#SBATCH --error=logs/aegisq-weak-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

set -euo pipefail

RANKS=(1 2 4 8)
INDEX=${SLURM_ARRAY_TASK_ID:-0}
NP=${RANKS[$INDEX]}

CIRCUIT="${CIRCUIT:-ising}"
BASE_QUBITS="${BASE_QUBITS:-26}"
REPEATS="${REPEATS:-5}"
OUTPUT="${OUTPUT:-benchmarks/raw/weak_scaling_${SLURM_JOB_ID:-local}.csv}"

# log2(NP) extra qubits keeps 2^(n-p) amplitudes per rank constant.
QUBITS=$((BASE_QUBITS + INDEX))

module load openmpi 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true

mkdir -p "$(dirname "$OUTPUT")" logs

srun --ntasks="$NP" --cpus-per-task=1 \
    python -m aegisq.benchmark.runner \
        --circuit "$CIRCUIT" \
        --qubits "$QUBITS" \
        --experiment weak_scaling \
        --repeats "$REPEATS" \
        --threads 1 \
        --thread-policy one-thread-per-rank \
        --output "$OUTPUT"
