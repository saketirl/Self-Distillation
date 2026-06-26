#!/bin/bash
# Submit all Hessian eigenspectrum analysis jobs for Qwen3-4B continual learning checkpoints.
#
# Chain structure:
#   base  (array[0-35], runs once)
#      └── dft_e06_tooluse → dft_e06_cotmath → dft_e06_spider
#      └── sft_tooluse     → sft_cotmath     → sft_spider
#      └── sdft_tooluse    → sdft_cotmath    → sdft_spider
#
# Layers are parallelised via SLURM job arrays (--array=0-35).
# Each array element skips its layer if summary.json already exists.
# Usage:
#   bash scripts/sbatch/eigenspectrum/submit_spectrum_chains.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/home/guests/saket/continual/Self-Distillation"
LOGS="${BASE_DIR}/logs"
mkdir -p "${LOGS}"

# sbatch args common to all jobs
SBATCH_COMMON="--parsable"

echo "=================================================="
echo "Submitting Hessian eigenspectrum analysis chains"
echo "Base model: Qwen/Qwen3-4B"
echo "Methods: base, dft_e06, sft, sdft"
echo "=================================================="

# ── Base model (shared across all methods, analyzed once) ────────────────────
BASE_JOB=$(sbatch ${SBATCH_COMMON} "${SCRIPT_DIR}/base.sbatch")
echo "Submitted base:              array job ${BASE_JOB}"

# ── DFT e06 chain (sequential tasks, parallel layers within each) ─────────────
DFT_T0=$(sbatch ${SBATCH_COMMON} \
    --dependency=afterok:${BASE_JOB} \
    "${SCRIPT_DIR}/dft_e06_tooluse.sbatch")
echo "Submitted dft_e06 task_0:    array job ${DFT_T0}  [after base ${BASE_JOB}]"

DFT_T1=$(sbatch ${SBATCH_COMMON} \
    --dependency=afterok:${DFT_T0} \
    "${SCRIPT_DIR}/dft_e06_cotmath.sbatch")
echo "Submitted dft_e06 task_1:    array job ${DFT_T1}  [after dft_t0 ${DFT_T0}]"

DFT_T2=$(sbatch ${SBATCH_COMMON} \
    --dependency=afterok:${DFT_T1} \
    "${SCRIPT_DIR}/dft_e06_spider.sbatch")
echo "Submitted dft_e06 task_2:    array job ${DFT_T2}  [after dft_t1 ${DFT_T1}]"

# ── SFT chain ─────────────────────────────────────────────────────────────────
SFT_T0=$(sbatch ${SBATCH_COMMON} \
    --dependency=afterok:${BASE_JOB} \
    "${SCRIPT_DIR}/sft_tooluse.sbatch")
echo "Submitted sft task_0:        array job ${SFT_T0}  [after base ${BASE_JOB}]"

SFT_T1=$(sbatch ${SBATCH_COMMON} \
    --dependency=afterok:${SFT_T0} \
    "${SCRIPT_DIR}/sft_cotmath.sbatch")
echo "Submitted sft task_1:        array job ${SFT_T1}  [after sft_t0 ${SFT_T0}]"

SFT_T2=$(sbatch ${SBATCH_COMMON} \
    --dependency=afterok:${SFT_T1} \
    "${SCRIPT_DIR}/sft_spider.sbatch")
echo "Submitted sft task_2:        array job ${SFT_T2}  [after sft_t1 ${SFT_T1}]"

# ── SDFT chain ────────────────────────────────────────────────────────────────
SDFT_T0=$(sbatch ${SBATCH_COMMON} \
    --dependency=afterok:${BASE_JOB} \
    "${SCRIPT_DIR}/sdft_tooluse.sbatch")
echo "Submitted sdft task_0:       array job ${SDFT_T0}  [after base ${BASE_JOB}]"

SDFT_T1=$(sbatch ${SBATCH_COMMON} \
    --dependency=afterok:${SDFT_T0} \
    "${SCRIPT_DIR}/sdft_cotmath.sbatch")
echo "Submitted sdft task_1:       array job ${SDFT_T1}  [after sdft_t0 ${SDFT_T0}]"

SDFT_T2=$(sbatch ${SBATCH_COMMON} \
    --dependency=afterok:${SDFT_T1} \
    "${SCRIPT_DIR}/sdft_spider.sbatch")
echo "Submitted sdft task_2:       array job ${SDFT_T2}  [after sdft_t1 ${SDFT_T1}]"

echo ""
echo "=================================================="
echo "All jobs queued. Monitor with:"
echo "  squeue -u \$(whoami)"
echo ""
echo "All job IDs:"
echo "  base:           ${BASE_JOB}"
echo "  dft_e06:        ${DFT_T0}, ${DFT_T1}, ${DFT_T2}"
echo "  sft:            ${SFT_T0}, ${SFT_T1}, ${SFT_T2}"
echo "  sdft:           ${SDFT_T0}, ${SDFT_T1}, ${SDFT_T2}"
echo ""
echo "Outputs land in:"
echo "  ${BASE_DIR}/eigenspectrum/outputs/qwen3/{base,dft_e06,sft,sdft}/{task}/layer_{N}/"
echo "=================================================="
