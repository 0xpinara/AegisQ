#!/bin/bash
# Execute an authenticated, encrypted AegisQ job bundle.
#
#   sbatch --export=ALL,JOB=qft24.aqjob slurm/secure_job.sbatch
#
# The bundle is verified and decrypted on the compute node; the signed result
# is written next to it. See docs/pqc-protocol.md.
#SBATCH --job-name=aegisq-secure
#SBATCH --output=logs/aegisq-secure-%j.out
#SBATCH --error=logs/aegisq-secure-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

set -euo pipefail

JOB="${JOB:?set JOB=<bundle>.aqjob}"
CLUSTER_KEYS="${CLUSTER_KEYS:-keys/cluster}"
TRUSTED="${TRUSTED:-keys/trusted}"
RESULT="${RESULT:-${JOB%.aqjob}.aqresult}"

module load openmpi 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true

mkdir -p logs

srun --ntasks="${SLURM_NTASKS:-8}" --cpus-per-task=1 \
    aegisq secure-run "$JOB" \
        --cluster "$CLUSTER_KEYS" \
        --trusted "$TRUSTED" \
        --output "$RESULT"
