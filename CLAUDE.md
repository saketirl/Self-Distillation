# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

This repository implements the Self-Distillation Fine-Tuning (SDFT) setup from the paper **"Self-Distillation Enables Continual Learning"**. The codebase uses TRL for training with a custom `DistilTrainer` and `DistilConfig`. The default model is `Qwen/Qwen2.5-3B-Instruct`.

Goals:
1. **Reproduce** the reported results as faithfully as possible.
2. **Improve** the method with careful ablations, better evaluation, and more robust training infrastructure.
3. Keep changes **measurable, reversible, and well-documented**.

## Continual Learning Task Sequence

We train sequentially on three tasks to study catastrophic forgetting:

| Order | Task | Description | Train Size | Eval Size |
|-------|------|-------------|------------|-----------|
| 1 | `tooluse` | Tool selection from descriptions | local JSON | 10 samples |
| 2 | `gsm8k` | Math word problems | local JSON | 10 samples |
| 3 | `mbpp` | Python code generation (MBPP) | 374 (HuggingFace) | 10 samples |

Task order is defined in `data_loaders.py`:
```python
TASK_ORDER = ["tooluse", "gsm8k", "mbpp"]
```

## Repository Structure

- `train_single_task.py` - Main entry point. Trains on one task, evaluates on all tasks before/after.
- `distil_trainer.py` - Custom SDFT trainer with EMA reference model sync.
- `distil_config.py` - Config class extending `TrainingArguments`.
- `data_loaders.py` - Unified data loaders for all tasks. MBPP loads from HuggingFace.
- `evaluate.py` - Standalone evaluation script with execution-based MBPP eval.
- `scripts/` - Shell scripts for running experiments.

## Running Experiments

### Quick Test (64 train samples, 10 eval samples per task)
```bash
bash scripts/run_test.sh
```

### Full Experiment
```bash
bash scripts/run_fullexp.sh
```

### Hyperparameters (from `run_fullexp.sh`)
```bash
MODEL_BASE="Qwen/Qwen2.5-3B-Instruct"
LEARNING_RATE="5e-6"
NUM_EPOCHS=1
REF_MODEL_MIXUP_ALPHA=0.05
--gradient_accumulation_steps 32
--vllm_gpu_memory_utilization 0.3
```

### Evaluate MBPP Baseline (no training)
```bash
python scripts/eval_mbpp_baseline.py --max_samples 50 --verbose
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/run_test.sh` | Quick test with 64 samples (EXP_ID: `exp_mbpp`) |
| `scripts/run_fullexp.sh` | Full experiment (EXP_ID: `full_mbpp`) |
| `scripts/run_ablations.sh` | Ablation study across alpha and lr |
| `scripts/eval_mbpp_baseline.py` | Standalone MBPP evaluation |

## Evaluation

- **tooluse**: Checks if correct tool name appears in response
- **gsm8k**: Extracts numbers from response, checks against expected answer
- **mbpp**: Execution-based - runs generated code against test cases

Eval samples per task: **10** (configured in `train_single_task.py:evaluate_all_tasks`)

## Environment Setup

```bash
python3.12 -m venv distillation
source distillation/bin/activate
pip install -r requirements.txt
```

## Data Sources

- `data/tooluse_data/` - Local JSON files
- `data/gsm8k/` - Local JSON files
- `mbpp` - Loaded from HuggingFace `datasets` (no local files needed)

## DFT Datasets (Qwen3-4B experiments)

The three datasets validated to work well for DFT with Qwen3-4B are `tooluse`,
`spider`, and `cot_math`. All loaded via `load_task_dataset(task_name)` in
`data_loaders.py`.

### Simplified ToolUse (`tooluse`)

- **Source**: Local JSON — `data/tooluse_data/train_data_simple.json` / `eval_data_simple.json`
- **Task**: Single-turn function/tool selection. Given a natural-language query and
  tool descriptions, the model picks the correct tool and explains why.
- **Format**:
  - `prompt`: tool description + user question
  - `response`: natural-language explanation of which tool to use
  - `teacher_prompt`: same prompt with a worked demonstration prepended
- **Size**: local files (check JSON directly for exact counts)
- **Prompt template**: plain user message; no system turn
- **Evaluation**: exact/fuzzy match on tool name in generated response

### Spider (`spider`)

- **Source**: Local JSON — `data/spider/train_data.json` / `eval_data.json`
  (derived from the Spider benchmark, Yu et al. EMNLP 2018)
- **Task**: Cross-domain text-to-SQL. Model generates a syntactically correct SQL
  query given a natural-language question and a database identifier.
- **Format**:
  - `prompt`: `"You are a SQL expert. ... Database: {db_id}\nQuestion: {question}\n..."`
  - `response`: raw SQL query string
  - `teacher_prompt`: prompt + example SQL query as demonstration
- **Key fields in JSON**: `db_id`, `question`, `query`
- **Size**: ~7,000 train / ~1,034 eval (standard Spider split)
- **Evaluation**: exact-match SQL or execution accuracy

### CoT Math (`cot_math`)

- **Source**: HuggingFace — `Open-COT-Data/COT-Dataset-Math` (train split only)
- **Task**: Multi-step math reasoning with full chain-of-thought. Solutions end with
  the phrase `"The final answer is [X]"`.
- **Format**:
  - `prompt`: `"Solve ... Show your step-by-step reasoning and end with 'The final answer is [answer]'.\n\n{problem}"`
  - `response`: full CoT solution string (from `output` field)
  - `teacher_prompt`: prompt + example solution
- **Key fields in HF dataset**: `input` (problem), `output` (CoT solution)
- **Size**: ~7,500 total; 90/10 train/eval split applied at load time
- **Evaluation**: extract final answer token after `"The final answer is"`,
  compare numerically

### Prompt structure shared across all three

Each example has three fields used during DFT:

```
prompt         → fed to the student model for generation
teacher_prompt → fed to the teacher model; includes a worked example
                 so the teacher's output distribution is conditioned
                 on a demonstration (used as KL target)
response       → ground-truth string (used for SFT baseline or eval)
```

The `teacher_prompt` pattern is:
```
{original prompt}

This is an example [response/query/solution]:
{ground_truth}

Now [solve/answer/write] ...
```

---

## Qwen3-4B Continual Learning Experiments

### Methods compared

| Method | Script | Optimizer | Teacher | Notes |
|--------|--------|-----------|---------|-------|
| **DFT + Proj** | `train_qwen3_dft.py --method dft` | ProjectedGradient | Frozen (`sync_ref_model=False`) | Our method |
| **SDFT** | `train_qwen3_dft.py --method sdft --adamw_only` | AdamW | EMA (`sync_ref_model=True, α=0.01`) | Baseline paper |
| **SFT** | `train_qwen3_dft.py --method sft --adamw_only` | AdamW | None | Fine-tuning baseline |

**SDFT baseline** uses the same `DistilTrainer` and KL distillation loss as DFT, but the
reference model slowly follows the student via EMA at every step:
`π_ref ← 0.01 · π_θ + 0.99 · π_ref_prev`. This prevents the teacher from becoming stale
as the student adapts to new tasks. Optimizer is plain AdamW at `lr=5e-5` — no projection.

### Task sequence (Qwen3-4B)

| Order | Task | Train | Full Eval |
|-------|------|-------|-----------|
| 1 | `tooluse` | 4,046 | 68 |
| 2 | `cot_math` | 6,750 | 750 |
| 3 | `spider` | 7,000 | 1,034 |

### Evaluation

During training, `--eval_max_samples 50` is used for speed. Full eval is run separately
after training completes via `scripts/eval_continual_checkpoints.py`, which evaluates
every checkpoint (including the Qwen3-4B base model as reference) on the complete eval
sets for all three tasks.

```bash
# Full eval on a finished experiment
python scripts/eval_continual_checkpoints.py \
    --exp_dir outputs/qwen3/qwen3_cl_dft_proj_e06 \
    --base_model Qwen/Qwen3-4B \
    --max_samples 10000   # effectively the full set
```

Always pass `--eval_tasks spider,tooluse,cot_math` (both during training and eval) to
avoid iterating over the full `TASK_ORDER_QWEN3` list which includes unrelated tasks
such as `livecodebench` and `kernelbench`.

### Energy threshold and memory

The `--proj_energy_threshold` controls what fraction of singular value energy is retained
when factoring each weight matrix `W = UΣV^T`. A threshold of **0.6** keeps the top-`r`
singular values that capture 60% of the total energy, yielding a low-rank `r` that fits
comfortably within 80 GB GPU RAM for Qwen3-4B. Higher thresholds retain more singular
values (higher rank), requiring more memory and compute per step.

Validated thresholds for Qwen3-4B on 80 GB:

| Energy | Approx rank (typical layer) | Fits in 80 GB |
|--------|-----------------------------|---------------|
| 0.6 | low | yes |
| 0.7 | low-medium | yes |
| 0.8 | medium | yes |
| 0.9 | medium-high | yes |

The energy sweep scripts are in `scripts/sbatch/continual/submit_dft_proj_single.sh`:
```bash
bash scripts/sbatch/continual/submit_dft_proj_single.sh tooluse 0.7
bash scripts/sbatch/continual/submit_dft_proj_single.sh tooluse 0.8
bash scripts/sbatch/continual/submit_dft_proj_single.sh tooluse 0.9
```

### Continual learning sbatch scripts

All scripts live in `scripts/sbatch/continual/`. Submit individual chains with:
```bash
bash scripts/sbatch/continual/submit_dft_proj_chain.sh 0.6   # DFT at any energy
bash scripts/sbatch/continual/submit_sdft_chain.sh            # SDFT baseline
bash scripts/sbatch/continual/submit_sft_chain.sh             # SFT baseline
```

Or submit all three in parallel:
```bash
bash scripts/sbatch/continual/submit_continual_chains.sh
```

---

## Projected Gradient Optimizer — Technical Notes

### Fixed-Rank Quotient Factorization

A weight matrix `W ∈ R^{n×k}` is factored via its rank-`r` SVD as:

```
W ≈ XΓY   →   W' = ⟨X', Γ', Y'⟩
```

where `X' = XO ∈ St(n,r)`, `Y' = YO ∈ St(k,r)` are Stiefel matrices, `Γ' ∈ R^{r×r}` is
symmetric positive definite, and `O` is an `r×r` orthogonal matrix (following Mishra 2014).

Implementation: `frozen_residual_ubv/factored_linear.py` (`FactoredLinear`).

The effective learning rates (relative to base lr) are:
- **X', Y'** (Stiefel):  `lr × uv_scale`  — e.g. `5e-3 × 0.3 = 1.5e-3`
- **Γ'** (SPD):          `lr × beta`       — e.g. `5e-3 × 0.1 = 5e-4`

### Manifold Muon with ADMM (for X', Y')

The Stiefel factors are updated via Manifold Muon (Bernstein 2025), which at each step solves:

```
min_{A}  ⟨A, G⟩   s.t.  ‖A‖_* ≤ η,  A^T W + W^T A = 0
```

then applies `W ← msign(W + A)` where `msign(X) = X(X^T X)^{-1/2}`.

The original inner loop uses subgradient descent on the dual (`O(1/√k)` convergence).
Buchanan 2025 replaces it with **ADMM + variable splitting** (`O(1/k)`, ~2× faster wall-clock):

Introduce slack `X` with constraint `X = G + W(Λ + Λ^T)`. One ADMM iteration:

```
Λ_{k+1} = ½ sym( W^T (Ω_k/ρ + X_k − G) )
M_{k+1} = 2W Λ_{k+1} + G − Ω_k/ρ
X_{k+1} = ½ (M_{k+1} − msign(M_{k+1})/ρ)(I + msign(M_{k+1}^T M_{k+1} − I/ρ²))
Ω_{k+1} = Ω_k + ρ(X_{k+1} − 2W Λ_{k+1} − G)
```

where `sym(M) = ½(M + M^T)` and `ρ > 0` is the ADMM penalty (robust to choice of ρ).
After K iterations: `A* = −η · msign(G + 2W Λ_K)`.

Both Λ and X subproblems have closed-form solutions using only matrix multiplies and
`msign`, making the update GPU-efficient.

Implementation: `frozen_residual_ubv/projected_gradient_optimizer.py`.
Reference: https://sdbuchanan.com/blog/manifold-muon/

### Riemannian Gradient on the SPD Manifold (for Γ')

`Γ' ∈ S_{++}^r` lives on the symmetric positive definite manifold with the
**affine-invariant metric** at point B:

```
⟨U, V⟩_B = tr(B^{-1} U B^{-1} V)
```

Given Euclidean gradient `∇_B L`, the Riemannian gradient is:

```
grad_B L = B · sym(∇_B L) · B
```

The update retracts back onto S_{++}^r via the exponential map:

```
B ← B^{1/2} exp(−α · B^{1/2} sym(∇_B L) B^{1/2}) B^{1/2}
```

This guarantees B remains symmetric and positive definite throughout training.
The matrix exponential is approximated with Newton–Schulz or Padé iterations.

Implementation: `frozen_residual_ubv/ubv_optimizer.py` (`UBVOptimizer`).

### LR Scaling Conventions

For the Stiefel factors, the correct scaling with architecture size is:
- **Width d**: `lr ∝ 1/√d` (Riemannian gradient Frobenius norm scales as O(√d))
- **Depth L**: `lr ∝ 1/L`  (gradient magnitude compounds across layers)

The scaling sweep in `scaling-data/wandb-export-scaling.csv` used the **opposite**
direction (`lr × √d` and `lr × L`), so larger/deeper configs ran with too-high LRs.
Results from that sweep are confounded — re-run with corrected scaling before
drawing architecture conclusions.
