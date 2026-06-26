# DFT Experiments Notes

## Momentum-Based Methods for Projected Gradient

### Overview

The ProjectedGradientOptimizer uses Stiefel manifold constraints for the U and V matrices in low-rank weight decomposition (W = U @ B @ V.T). Key components:

### Stiefel Update Methods

| Method | Description |
|--------|-------------|
| `projected_qr` | QR-based projection onto Stiefel manifold (default, stable) |
| `cayley` | Cayley transform for orthogonal updates |
| `exponential` | Matrix exponential (expensive but exact) |

### ADMM-Based Manifold MUON

The optimizer uses ADMM (Alternating Direction Method of Multipliers) for gradient projection:

```python
--proj_admm_steps 20      # Number of ADMM iterations
--proj_grad_clip 1.0      # Gradient clipping before projection
--stiefel_max_rms 0.03    # Max RMS for Stiefel updates
```

### Learning Rate Scaling

Different components use scaled learning rates:

```python
--learning_rate 1e-3      # Base LR
--uv_lr_scale 0.3         # U/V matrices: 0.3 * base_lr
--b_lr_scale 0.3          # B matrix (Log-Cholesky): 0.3 * base_lr
```

### Energy-Based Rank Selection

SVD truncation based on cumulative energy:

```python
--proj_energy_threshold 0.5   # Keep singular values capturing 50% energy
```

### Key Hyperparameters (Current Experiments)

```bash
LR=1e-3
ENERGY_THRESHOLD=0.5
STIEFEL_UPDATE=projected_qr
ADMM_STEPS=20
STIEFEL_MAX_RMS=0.03
UV_LR_SCALE=0.3
B_LR_SCALE=0.3
```

---

## DFT Dataset Pipeline Testing

### How DFT Works

1. **Teacher** receives `teacher_prompt` (includes solution/demonstration)
2. **Student** receives `prompt` (problem only)
3. Student learns to match teacher's output distribution
4. Projected gradient constrains updates to preserve prior knowledge

### Dataset Requirements for DFT

For DFT to work effectively, datasets need:
- ✅ **Actual solutions/reasoning traces** for teacher demonstration
- ✅ **Clear problem-response pairs**
- ❌ Datasets with only problems (no solutions) don't work

---

## Dataset Evaluation Summary

### Working Datasets (Useful Benchmarks)

| Dataset | Size | Why It Works |
|---------|------|--------------|
| **tooluse** | ~100 | Clear tool selection with response demonstrations |
| **limo** | ~800 | High-quality step-by-step math solutions |

### Datasets with Issues

| Dataset | Size | Issue |
|---------|------|-------|
| **dapo** | 17k | No solutions - designed for RL verification only |
| **livecodebench** | varies | Solutions may lack quality for distillation |
| **kernelbench** | ~100 | CUDA-specific, reference code not ideal target |
| **medquad** | 16k | QA format, not reasoning traces |
| **mathbeyond** | 181 | Very small, hard problems |
| **polaris** | 53k | Large but mixed quality |
| **s1k** | 1k | Has DeepSeek traces - needs more testing |
| **openthoughts10k** | 10k | Has R1 traces - needs more testing |

### Promising (Needs Validation)

| Dataset | Size | Notes |
|---------|------|-------|
| **s1k** | 1,000 | DeepSeek r1 thinking trajectories |
| **openthoughts10k** | 10,000 | DeepSeek R1 reasoning + solutions |

These have proper reasoning traces but haven't been validated as useful benchmarks yet.

---

## Experiment Commands

### Submit All Jobs
```bash
bash scripts/sbatch/qwen3_experiments/submit_all.sh
```

### Run Individual Task
```bash
uv run python train_qwen3_dft.py \
    --method dft \
    --task limo \
    --model_name Qwen/Qwen3-4B \
    --output_dir outputs/qwen3/test \
    --learning_rate 1e-3 \
    --proj_energy_threshold 0.5 \
    --proj_no_resync \
    --proj_admm_steps 20 \
    --stiefel_update projected_qr \
    --skip_before_eval
```

---

## Key Findings

1. **Only tooluse and limo produce useful benchmarks** of all datasets tried
2. Datasets need explicit solutions for teacher prompts to work
3. RL-focused datasets (like DAPO) are not suitable for DFT
4. Reasoning trace datasets (s1k, openthoughts10k) are promising but unvalidated

---

## Next Steps

- [ ] Validate s1k and openthoughts10k as DFT benchmarks
- [ ] Consider removing dapo from pipeline (no solutions)
- [ ] Focus experiments on tooluse + limo for reliable comparisons
- [ ] Test continual learning across tooluse → limo sequence
