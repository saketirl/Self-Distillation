# CLAUDE.md

This file provides guidance to Claude Code when working with code in this directory.

## Project Overview

This repository implements the Self-Distillation Fine-Tuning setup from the paper **"Self-Distillation Enables Continual Learning"**. 

Code in this directory deals with the spectru of the Hessian as described in the paper CL_NOTE.pdf

My goal in this directory is:

1. Estimate the spectrum of deep neural networks, specifically LLMs
2. I study the models trained using the file ../train_single_task.py
3. **Improve** the method with careful ablations, better evaluation, and more robust training infrastructure. The goal is to get an accurate eigen spectrum while speeding up the computation.

The paper evaluates Hessian for the whole network which are relatively small but for LLMs we would like to study the spectrum by each layer.

## Repository Structure

### Original JAX Implementation
- `lanczos.py` - JAX Lanczos algorithm (for reference)
- `density.py` - JAX density estimation (for reference)
- `hessian_computation.py` - JAX full Hessian computation (for reference)

### PyTorch Implementation (Block-wise Analysis)
- `lanczos_torch.py` - PyTorch Lanczos with memory-efficient option
- `hessian_torch.py` - PyTorch HVP using torch.func, single-param analysis
- `density_numpy.py` - NumPy density estimation (vectorized)
- `spectrum_analyzer.py` - High-level API for block-wise spectrum analysis
- `example_analyze.py` - Example script for analyzing Qwen models

## Usage

```python
from eigenspectrum import SpectrumAnalyzer

# Initialize
analyzer = SpectrumAnalyzer(model, dataloader, device="cuda")

# List params in a layer
analyzer.list_params(layer_idx=0)

# Analyze a single weight matrix
result = analyzer.analyze_param(
    "model.layers.0.self_attn.q_proj.weight",
    lanczos_order=50,
    num_samples=5
)

# Plot density
plt.plot(result.grids, result.density)
```

## Memory Estimates (Qwen2.5-7B)

| Weight Matrix | Params | Lanczos Memory (order=50, fp32) |
|--------------|--------|--------------------------------|
| Q projection | 12.8M  | 2.6 GB |
| K/V projection | 1.8M | 0.4 GB |
| MLP gate/up/down | 67.9M | 13.6 GB |

## Memory Optimization: Gradient Checkpointing

The default HVP computation uses forward-over-reverse mode autodiff (`jvp` of `grad`), which is fast but memory-intensive because it stores all intermediate activations.

**New option**: `--use_gradient_checkpointing` enables reverse-over-reverse mode HVP (`grad(grad(L)·v)`) which:
- Is compatible with gradient checkpointing (recomputes activations during backward pass)
- Uses significantly less memory
- Is ~2x slower due to recomputation

```bash
# Memory-efficient analysis
python -m eigenspectrum.example_analyze \
    --model_path /path/to/model \
    --layer 0 \
    --use_bfloat16 \
    --use_gradient_checkpointing
```

### HVP Mode Comparison

| Mode | Function | Memory | Speed | Checkpointing Compatible |
|------|----------|--------|-------|--------------------------|
| Forward-over-reverse | `jvp(grad(L))` | High | Fast | No |
| Reverse-over-reverse | `grad(grad(L)·v)` | Low | ~2x slower | Yes |

## Known Limitations

### SDFT Loss (KL Divergence) Memory Issue
Computing the Hessian eigenspectrum using SDFT loss (KL divergence) requires loading both the student and reference models simultaneously. For Qwen2.5-3B with ~80GB GPU memory, this can cause OOM errors.

**Solution**: Use `--use_gradient_checkpointing` flag which enables gradient checkpointing and reduces memory significantly.

**Alternative**: Use SFT loss (cross-entropy) for analyzing SDFT-trained checkpoints. This only loads one model and captures the Hessian structure of the trained weights, though computed with a different loss function.

Memory breakdown:
1. Model weights: ~6GB each in bfloat16
2. HVP activations: Variable, reduced significantly with checkpointing
3. Lanczos vectors: ~(num_params × lanczos_order × 4 bytes)

Note: Gradients are only computed for the student model. The teacher model is frozen and only provides reference logits for KL divergence.

## Analysis Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_LANCZOS_SAMPLES` | 10 | Number of independent Lanczos runs |
| `LANCZOS_ORDER` | 50 | Lanczos iterations per run |
| `NUM_DATA_SAMPLES` | 500 | Data samples from tooluse/gsm8k/mbpp test sets |
| `BATCH_SIZE` | 1 | Batch size for HVP computation |
| `MAX_SEQ_LEN` | 512 | Maximum sequence length |
| `num_batches_per_hvp` | 5 | Random batches sampled per HVP call |

**Total eigenvalues per parameter**: 10 × 50 = 500

### Data for Hessian Estimation

The Hessian is estimated using real data from the continual learning task sequence:
- **tooluse**: Tool selection task (all available eval samples)
- **gsm8k**: Math word problems (split from remaining)
- **mbpp**: Python code generation (split from remaining)

Total: 500 samples combined, shuffled with seed=42.

### Models Analyzed

| Model Type | Task | Checkpoint Path | Output Path |
|------------|------|-----------------|-------------|
| **SFT Base** | - | `models/qwen2.5-3b-instruct` | `outputs/qwen2.5-3b-instruct/` |
| **SDFT Base** | - | `models/qwen2.5-3b-instruct` (same model, SDFT loss) | `outputs/qwen2.5-3b-instruct_sdft/` |
| SFT | task_0_tooluse | `outputs/sft_full_mbpp/task_0_tooluse/checkpoint-127` | `outputs/sft_full_mbpp/task_0_tooluse/checkpoint-127/` |
| SDFT | task_0_tooluse | `outputs/sdft_full_mbpp/task_0_tooluse/checkpoint-1011` | `outputs/sdft_full_mbpp/task_0_tooluse/checkpoint-1011/` |
| SFT | task_1_gsm8k | `outputs/sft_full_mbpp/task_1_gsm8k/checkpoint-234` | `outputs/sft_full_mbpp/task_1_gsm8k/checkpoint-234/` |
| SDFT | task_1_gsm8k | `outputs/sdft_full_mbpp/task_1_gsm8k/checkpoint-1868` | `outputs/sdft_full_mbpp/task_1_gsm8k/checkpoint-1868/` |
| SFT | task_2_mbpp | `outputs/sft_full_mbpp/task_2_mbpp/checkpoint-12` | `outputs/sft_full_mbpp/task_2_mbpp/checkpoint-12/` |
| SDFT | task_2_mbpp | `outputs/sdft_full_mbpp/task_2_mbpp/checkpoint-93` | `outputs/sdft_full_mbpp/task_2_mbpp/checkpoint-93/` |

Each model is analyzed across all 36 layers (layer 0-35).

### Parameters Analyzed Per Layer

Only large weight matrices (>100k params) are analyzed:
- `self_attn.q_proj.weight`
- `self_attn.k_proj.weight`
- `self_attn.v_proj.weight`
- `self_attn.o_proj.weight`
- `mlp.gate_proj.weight`
- `mlp.up_proj.weight`
- `mlp.down_proj.weight`

### Memory Optimization Flags

All analysis scripts use:
```bash
--use_bfloat16              # Use bfloat16 precision
--use_gradient_checkpointing # Reverse-mode HVP with gradient checkpointing
--batch_size 1              # Minimize batch memory
```

### Running Analysis Scripts

Scripts are located in `eigenspectrum/scripts/`:

```bash
# Base model (SFT loss)
bash scripts/analyze_all_layers.sh

# Base model (SDFT loss - same model as student and teacher)
bash scripts/analyze_base_sdft.sh

# SFT checkpoints
bash scripts/analyze_sft_checkpoint_tooluse.sh  # GPU 3
bash scripts/analyze_sft_checkpoint_gsm8k.sh    # GPU 1
bash scripts/analyze_sft_checkpoint_mbpp.sh     # GPU 2

# SDFT checkpoints (using SFT loss for Hessian)
bash scripts/analyze_sdft_checkpoint_tooluse.sh # GPU 2
bash scripts/analyze_sdft_checkpoint_gsm8k.sh   # GPU 0
bash scripts/analyze_sdft_checkpoint_mbpp.sh    # GPU 3
```

## Visualization Web Application

A Flask web app for comparing SFT vs SDFT eigenspectrums side-by-side.

### Running the Webapp

```bash
cd eigenspectrum/webapp
pip install flask numpy
python app.py
# Open http://localhost:5000
```

### Features

1. **Side-by-Side View**: SFT and SDFT histograms with shared axes for direct comparison
2. **Overlay View**: Both histograms overlaid with transparency
3. **Cross-Layer View**: Top eigenvalue, effective rank, and trace across all 36 layers

### Controls

- **Task selector**: Base Model, Task 0 (ToolUse), Task 1 (GSM8K), Task 2 (MBPP)
- **Layer slider**: Select layer 0-35
- **Parameter selector**: Choose weight matrix (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)

### Histogram Configuration

- **Number of bins**: 200
- **Shared axes**: X and Y axes are synchronized between SFT and SDFT plots
- **Log density**: Second row shows log10(density) for better visualization of tail behavior

## Output Format

Each layer produces:
- `*_spectrum.npz`: Contains eigenvalues, density, grids, and summary statistics
- `*_density.png`: Density plot visualization
- `summary.json`: Summary statistics for all parameters in the layer

### NPZ File Contents

| Key | Shape | Description |
|-----|-------|-------------|
| `eigenvalues` | (10, 50) | Eigenvalues from each Lanczos sample |
| `weights` | (10, 50) | Weights for density estimation |
| `density` | (10000,) | Smoothed density estimate |
| `grids` | (10000,) | Eigenvalue grid points |
| `effective_rank` | scalar | Effective rank of Hessian |
| `top_eigenvalue` | scalar | Maximum eigenvalue |
| `trace_estimate` | scalar | Trace estimate (sum of eigenvalues)
