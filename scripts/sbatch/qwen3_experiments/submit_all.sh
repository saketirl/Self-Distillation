#!/bin/bash
# Submit all Qwen3-4B DFT experiments
# Hyperparameters: LR=1e-3, Energy=0.5

SCRIPT_DIR="$(dirname "$0")"

echo "========================================"
echo "Qwen3-4B DFT Experiments"
echo "========================================"
echo "Tasks:"
echo "  - LiveCodeBench (competitive programming)"
echo "  - KernelBench (CUDA kernels)"
echo "  - MedQuad (medical QA)"
echo "  - Tooluse (tool selection)"
echo "  - MATH-Beyond (hard math)"
echo "  - POLARIS-53k (reasoning)"
echo "  - LIMO (less-is-more reasoning)"
echo "  - s1K-1.1 (DeepSeek r1 traces)"
echo "  - DAPO-Math-17k (verification-driven math)"
echo "  - OpenThoughts-10k (DeepSeek R1 reasoning traces)"
echo "Hyperparameters:"
echo "  LR: 1e-3"
echo "  Energy Threshold: 0.5"
echo "  Stiefel Update: projected_qr"
echo "  ADMM Steps: 20"
echo "========================================"

# Submit all jobs
sbatch "$SCRIPT_DIR/dft_livecodebench.sbatch"
sbatch "$SCRIPT_DIR/dft_kernelbench.sbatch"
sbatch "$SCRIPT_DIR/dft_medquad.sbatch"
sbatch "$SCRIPT_DIR/dft_tooluse.sbatch"
sbatch "$SCRIPT_DIR/dft_mathbeyond.sbatch"
sbatch "$SCRIPT_DIR/dft_polaris.sbatch"
sbatch "$SCRIPT_DIR/dft_limo.sbatch"
sbatch "$SCRIPT_DIR/dft_s1k.sbatch"
sbatch "$SCRIPT_DIR/dft_dapo.sbatch"
sbatch "$SCRIPT_DIR/dft_openthoughts10k.sbatch"

echo "========================================"
echo "Submitted 10 jobs"
echo "========================================"
