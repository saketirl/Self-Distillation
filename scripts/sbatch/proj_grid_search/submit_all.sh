#!/bin/bash
# Submit all grid search jobs
# LR: 1e-3, 1e-4
# Energy: 0.9, 0.8

SCRIPT_DIR="$(dirname "$0")"

echo "Submitting Projected Gradient Grid Search Jobs"
echo "================================================"
echo "Grid: LR x Energy Threshold"
echo "  LR: 1e-3, 1e-4"
echo "  Energy: 0.9, 0.8"
echo "================================================"

sbatch "$SCRIPT_DIR/proj_lr1em3_energy09.sbatch"
sbatch "$SCRIPT_DIR/proj_lr1em3_energy08.sbatch"
sbatch "$SCRIPT_DIR/proj_lr1em4_energy09.sbatch"
sbatch "$SCRIPT_DIR/proj_lr1em4_energy08.sbatch"

echo "================================================"
echo "Submitted 4 jobs"
