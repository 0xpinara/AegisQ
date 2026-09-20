#!/bin/bash
# Strong scaling: one circuit, increasing rank count.
#
#   sbatch --array=0-3 slurm/strong_scaling.sbatch
#
# Each array task runs one rank count so the queue can schedule them
# independently; all tasks append to the same raw CSV.
#SBATCH --job-name=aegisq-strong
#SBATCH --output=logs/aegisq-strong-%A_%a.out
#SBATCH --error=logs/aegisq-strong-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

set -euo pipefail

RANKS=(1 2 4 8)
NP=${RANKS[${SLURM_ARRAY_TASK_ID:-0}]}

CIRCUIT="${CIRCUIT:-qft}"
QUBITS="${QUBITS:-28}"
REPEATS="${REPEATS:-5}"
OUTPUT="${OUTPUT:-benchmarks/raw/strong_scaling_${SLURM_JOB_ID:-local}.csv}"

module load openmpi 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true

mkdir -p "$(dirname "$OUTPUT")" logs

# One thread per rank: total compute grows with the rank count, which is the
# configuration a strong-scaling curve is supposed to describe.
srun --ntasks="$NP" --cpus-per-task=1 \
    python -m aegisq.benchmark.runner \
        --circuit "$CIRCUIT" \
        --qubits "$QUBITS" \
        --experiment strong_scaling \
        --repeats "$REPEATS" \
        --threads 1 \
        --thread-policy one-thread-per-rank \
        --output "$OUTPUT"
