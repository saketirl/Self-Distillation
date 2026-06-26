#!/bin/bash
# Submit SFT + AdamW continual learning chain with SLURM dependencies.
# Usage: bash submit_sft_chain.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Submitting SFT + AdamW continual chain"
echo "Tasks: tooluse -> cot_math -> spider"
echo "Each job: 12 hr, 1 GPU, 80G RAM"
echo "=============================================="

SFT1=$(sbatch --parsable ${SCRIPT_DIR}/sft_task1_tooluse.sbatch)
echo "Submitted SFT task 1 (tooluse):   job ${SFT1}"

SFT2=$(sbatch --parsable --dependency=afterok:${SFT1} ${SCRIPT_DIR}/sft_task2_cotmath.sbatch)
echo "Submitted SFT task 2 (cot_math):  job ${SFT2}  [after ${SFT1}]"

SFT3=$(sbatch --parsable --dependency=afterok:${SFT2} ${SCRIPT_DIR}/sft_task3_spider.sbatch)
echo "Submitted SFT task 3 (spider):    job ${SFT3}  [after ${SFT2}]"

echo ""
echo "=============================================="
echo "All 3 SFT jobs queued. Monitor with:"
echo "  squeue -j ${SFT1},${SFT2},${SFT3}"
echo "=============================================="
