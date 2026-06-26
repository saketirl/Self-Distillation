#!/bin/bash
# Submit 9-job continual learning chains with SLURM dependencies.
# DFT, SDFT, and SFT chains run in parallel; within each chain jobs run sequentially.
#
# Usage: bash submit_continual_chains.sh
#
# Job graph:
#   dft_task1_tooluse  ──afterok──> dft_task2_cotmath  ──afterok──> dft_task3_spider
#   sdft_task1_tooluse ──afterok──> sdft_task2_cotmath ──afterok──> sdft_task3_spider
#   sft_task1_tooluse  ──afterok──> sft_task2_cotmath  ──afterok──> sft_task3_spider

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Submitting continual learning chains"
echo "DFT:  tooluse -> cot_math -> spider  (ProjectedGradient, frozen teacher)"
echo "SDFT: tooluse -> cot_math -> spider  (AdamW, EMA teacher — baseline)"
echo "SFT:  tooluse -> cot_math -> spider  (AdamW, no distillation)"
echo "Each job: 12 hr, 1 GPU, 80G RAM"
echo "=============================================="

# ── DFT chain ─────────────────────────────────────────────────────────────────
DFT1=$(sbatch --parsable ${SCRIPT_DIR}/dft_task1_tooluse.sbatch)
echo "Submitted DFT  task 1 (tooluse):   job ${DFT1}"

DFT2=$(sbatch --parsable --dependency=afterok:${DFT1} ${SCRIPT_DIR}/dft_task2_cotmath.sbatch)
echo "Submitted DFT  task 2 (cot_math):  job ${DFT2}  [after ${DFT1}]"

DFT3=$(sbatch --parsable --dependency=afterok:${DFT2} ${SCRIPT_DIR}/dft_task3_spider.sbatch)
echo "Submitted DFT  task 3 (spider):    job ${DFT3}  [after ${DFT2}]"

# ── SDFT chain ────────────────────────────────────────────────────────────────
SDFT1=$(sbatch --parsable ${SCRIPT_DIR}/sdft_task1_tooluse.sbatch)
echo "Submitted SDFT task 1 (tooluse):   job ${SDFT1}"

SDFT2=$(sbatch --parsable --dependency=afterok:${SDFT1} ${SCRIPT_DIR}/sdft_task2_cotmath.sbatch)
echo "Submitted SDFT task 2 (cot_math):  job ${SDFT2}  [after ${SDFT1}]"

SDFT3=$(sbatch --parsable --dependency=afterok:${SDFT2} ${SCRIPT_DIR}/sdft_task3_spider.sbatch)
echo "Submitted SDFT task 3 (spider):    job ${SDFT3}  [after ${SDFT2}]"

# ── SFT chain ─────────────────────────────────────────────────────────────────
SFT1=$(sbatch --parsable ${SCRIPT_DIR}/sft_task1_tooluse.sbatch)
echo "Submitted SFT  task 1 (tooluse):   job ${SFT1}"

SFT2=$(sbatch --parsable --dependency=afterok:${SFT1} ${SCRIPT_DIR}/sft_task2_cotmath.sbatch)
echo "Submitted SFT  task 2 (cot_math):  job ${SFT2}  [after ${SFT1}]"

SFT3=$(sbatch --parsable --dependency=afterok:${SFT2} ${SCRIPT_DIR}/sft_task3_spider.sbatch)
echo "Submitted SFT  task 3 (spider):    job ${SFT3}  [after ${SFT2}]"

echo ""
echo "=============================================="
echo "All 9 jobs queued. Monitor with:"
echo "  squeue -u \$(whoami)"
echo "  squeue -j ${DFT1},${DFT2},${DFT3},${SDFT1},${SDFT2},${SDFT3},${SFT1},${SFT2},${SFT3}"
echo "=============================================="
