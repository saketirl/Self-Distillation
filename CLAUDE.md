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

---

## Test Suite — Optimizers and Continual Learning

All tests live in `tests/`. Run with `python -m pytest tests/ -s`.

### Optimizer Intuitions

#### FeaturePreservingMuon (FPMuon)
`frozen_residual_ubv/feature_preserving_muon.py`

**Idea**: Each weight W encodes task features primarily in its dominant right singular
vectors V_r (the directions of input space that W responds to most strongly). After
training task 0, snap V_r from the weights and force all future gradient updates to
satisfy `ΔW V_r = 0` — i.e., the update lands entirely in null(V_r) and cannot
overwrite the stored features.

**ur_alpha=0** (V only): protects right singular directions only. Works per-matrix.
Each new task re-snaps V_r from the current weights, so after many tasks V_r rotates
away from T00's subspace and T00 features gradually drift — the "V_r rotation problem".

**ur_alpha=1** (UV double projection): additionally enforces `U_r^T ΔW = 0`. Because
both left and right singular directions are frozen to their T00 values, the mathematical
guarantee is `U_r_0^T (W_0 + ΔW_1 + … + ΔW_t) V_r_0 = Σ_r_0` exactly for all t.
T00's singular core is preserved indefinitely regardless of how many tasks follow.

**fixed_subspaces**: Pass U_r/V_r computed once from T0 weights to every subsequent
FPMuon instance. This prevents intra-task rotation — without it, even UV protection
drifts because each task re-snaps slightly different U_r/V_r.

**Ascending energy schedule**: `threshold(t) = 0.25 + (t-1)*0.075`. Early tasks
protect a small subspace (low r, lots of free gradient) so later tasks can still
learn. Later tasks protect more. This helps the chain learn all 10 tasks without
starving any task of gradient freedom.

#### SoftCompPreservingMuon (SCM)
`frozen_residual_ubv/soft_comp_preserving_muon.py`

**Idea**: For transformers, what matters for a learned task is not individual W_Q or
W_K weights in isolation, but their **composition** M = W_Q^T W_K — the bilinear form
that computes attention scores. FPMuon protects each matrix separately, which is a
weaker guarantee: you could rotate W_Q and W_K in opposite directions and destroy M
while leaving both V_r constraints satisfied.

SCM directly constrains: `U_k^T (ΔW_Q^T W_K + W_Q^T ΔW_K) V_k ≈ 0`, where U_k, V_k
are the top-k singular vectors of M. This is a Sylvester-type equation solved via
damped least-squares. The constraint means new-task updates can only change M in
directions orthogonal to its dominant subspace, preserving the attention pattern that
encodes the old task.

**GQA support**: In grouped query attention (multiple Q heads per KV head), the
correction to the shared K head is accumulated as a mean over all Q heads in its
group, then applied once.

**Fallback hierarchy**: (1) ADMM dual ascent, (2) Sylvester correction, (3) zero the
update if both fail and `hard_fallback_tol` is exceeded.

#### CompositionPreservingMuon (CompMuon)
`frozen_residual_ubv/composition_preserving_muon.py`

Earlier implementation of the same composition-preservation idea as SCM. Uses a
strict algebraic projection (strict_composition_fallback) rather than ADMM. Produces
near-zero residuals (~1e-7) via direct projection onto the null space of the constraint.
SCM replaced it as the primary optimizer but CompMuon tests document the fallback
algebra and GQA accumulation logic that SCM inherited.

#### ProjectedGradientOptimizer
`frozen_residual_ubv/projected_gradient_optimizer.py`

**Idea**: Factor W = UΓV^T (Stiefel × SPD × Stiefel). Update U and V on the Stiefel
manifold via Manifold Muon + ADMM; update Γ on the SPD manifold via Riemannian
gradient. The factorization constrains the optimizer to a smooth manifold, avoiding
the unconstrained weight drift that causes catastrophic forgetting.

This is the optimizer used in `--method dft` (DFT + Proj) for the Qwen3 experiments.

---

### Test Files

#### `tests/test_feature_preserving_muon.py`
Unit and integration tests for FPMuon.

| Test | What it checks |
|------|---------------|
| `test_update_orthogonal_to_vr` | Single step: `‖ΔW @ V_r‖ < 1e-3` |
| `test_constraint_holds_over_multiple_steps` | Leakage stays near zero over 5 steps |
| `test_vr_frozen` | V_r does not change between steps (snapped once at init) |
| `test_energy_threshold_determines_rank` | Higher threshold → higher r |
| `test_1d_bias_fallback` | 1-D params (biases) handled without crash |
| `test_momentum_orthogonality` | With momentum, the *update direction* (not the buffer) stays in null(V_r) |
| `test_debug_flag` | Debug mode prints constraint residuals |
| `test_conv_kernel` | 3-D conv weight flattened correctly; V_r lives in the right space |
| `test_loss_decreases_mlp` | FPMuon actually reduces MSE on 2-layer MLP |
| `test_loss_decreases_deep_linear` | Same check for 4-layer deep linear network |
| `test_fp_muon_10task_chain_mlp` | **10-task MLP chain** (V only, ascending energy). Shows leakage≈0 per task but T00 drifts to 0.16 by task 9 due to V_r rotation. Diagonal (new task acc) stays high. |
| `test_fp_muon_double_projection_mlp` | **UV vs V-only comparison** on same 10-task MLP. UV holds T00 substantially longer at midchain (after T05). Key assertion: `T00_mid_UV > T00_mid_V + 0.2`. |
| `test_fp_muon_uv_lr_ablation` | **LR sweep** (1e-3, 5e-4, 3e-4, 1e-4, 3e-5) for UV double projection. Best: lr=3e-4 (avg T00 retention ≈ 0.80). |

**Key empirical result from the MLP chain**: In MLP (no attention), the ReLU
nonlinearity breaks the per-layer singular-vector guarantee across layers. fc2's V_r
anchors to old hidden representations that fc1's null-space updates later invalidate.
This means T00 is NOT specially retained in the MLP case — it drifts like any other
task. FPMuon on MLP retains a sliding window of the most recent ~6 tasks, not a
fixed anchor.

#### `tests/test_soft_comp_continual.py`
Integration tests for FPMuon and SCM on attention-based continual learning tasks.

**Task construction**: token-pair matching. Each task uses a 2-dim signal subspace
in a 64-dim input. Task t places the signal in dims `[t*k : (t+1)*k]`, so all 10
tasks use orthogonal subspaces. Label is 1 iff both tokens have the same latent class
(same-class pairs have positive QK dot product). The QK bilinear form `x_0^T (W_Q^T W_K) x_1`
is the ideal classifier — protecting its dominant subspace is exactly what SCM does.

| Test | What it checks |
|------|---------------|
| `test_soft_comp_muon_retains_task_a_after_task_b` | 2-task 1-layer: SCM retains Task A ≥ 0.80 after Task B; Adam forgets |
| `test_feature_preserving_muon_retains_task_a_after_task_b` | Same but FPMuon; weaker guarantee ≥ 0.70 |
| `test_feature_preserving_muon_10task_chain` | 1-layer, FPMuon 10-task chain; ≥ 4/10 tasks retained (Task 0 ≥ 0.80) |
| `test_feature_preserving_muon_10task_chain_2layer` | 2-layer, FPMuon 10-task chain; harder (≥ 5/10) |
| `test_soft_comp_muon_10task_chain` | 1-layer, SCM 10-task chain; ≥ 7/10 tasks retained — best result |
| `test_feature_preserving_muon_10task_chain_2layer_last_only` | **Negative result**: FPMuon on last layer only of 2-layer model. Free AdamW on layer 0 overwrites intermediate representations, making last-layer V_r useless for old tasks. T00 ≈ 0.5 (chance). |
| `test_soft_comp_muon_10task_chain_2layer_last_only` | Same negative result for SCM: protecting only the last layer fails when intermediate layers are free. |

**Key insight from negative results**: Applying the constraint only to the last layer
is insufficient when earlier layers are free to change. The constraint protects
singular directions of the last layer's weights, but those directions become
meaningless for old tasks once the upstream representation has been overwritten by
a free optimizer. You must constrain all layers or accept forgetting through the
unconstrained path.

#### `tests/test_soft_comp_preserving_muon.py`
Unit tests for the SCM optimizer internals (10 tests).

| Test | What it checks |
|------|---------------|
| `test_dual_ascent_reduces_H_norm` | ADMM inner loop reduces `‖U_k^T (ΔW_Q^T W_K + W_Q^T ΔW_K) V_k‖` vs unconstrained |
| `test_core_drift_small_after_step` | After one step, `‖core − Σ‖ / ‖Σ‖ < 10` (doesn't blow up) |
| `test_k0_recovers_plain_muon` | Energy threshold=0 → k=0 → no constraint → same update as plain msign Muon |
| `test_no_nans_with_degenerate_gradients` | Handles zero, 1e-15, and NaN gradients gracefully |
| `test_gqa_k_update_is_averaged` | Shared KV head gets accumulated update from all Q heads in its group |
| `test_factored_svd_matches_direct` | Low-rank SVD of A@B matches direct SVD(A@B) to 1e-4 |
| `test_lambda_warmup_across_steps` | Lambda accumulation across 5 steps keeps H_norm finite (< 10.0) |
| `test_sylvester_correction_reduces_h_norm` | Sylvester solve reduces H_norm by ≥ 10× |
| `test_sylvester_fallback_zeroes_updates_on_failure` | When hard_fallback_tol=0, all updates are zeroed and weights unchanged |
| `test_sylvester_always_applied_and_reduces_h_norm` | Sylvester reduces H_norm by ≥ 100× on a well-conditioned problem |

#### `tests/test_composition_preserving_muon.py`
15 tests for the older CompositionPreservingMuon (CompMuon). Covers the same
ideas as SCM but with the strict algebraic fallback (strict_composition_fallback)
that directly projects (ΔA, ΔB) onto the null space of the constraint using
Gram-Schmidt style projection. Key correctness properties:

- Head splitting shapes match energy-chosen rank k
- Low-rank SVD of product A@B matches full SVD
- QK strict fallback achieves residual < 1e-3
- OV strict fallback achieves residual < 1e-3
- Shared KV fallback updates accumulate from all Q heads
- k=0 / preserve=False recovers plain msign Muon
- Stored subspace tensors have `requires_grad=False`
- `constraint_residual` returns ≈ 0 for a known feasible update
- `skip_if_fallback_fails=True` zeroes the update when both methods fail
- QK transpose mapping: `ΔW_Q_h = ΔA^T` (A = W_Q_h^T)
- OV direct mapping: `ΔW_O_h = ΔA` (no transpose)

#### `tests/test_projected_optimizer_regression.py`
Tests ProjectedGradientOptimizer on a 2-layer linear classifier `f(x) = U @ W @ x`
where W is updated with ProjectedGradient and U with Adam. Verifies that the
projected optimizer achieves > 65% accuracy on a binary Gaussian classification
task (class means differ in first 10 dims of 128). Also sweeps energy thresholds
and compares against Adam baseline.

#### `tests/test_two_layer_gradient.py`
Sanity checks for ProjectedGradientOptimizer: gradient flow through the network,
W actually changes after a step (ΔW > 0.01), single-layer baseline with direct
parameter access.

#### `tests/test_rectangular_init.py`
Tests UBV initialization for common transformer weight shapes: square attention
(4096×4096), tall MLP up/gate proj (11008×4096), wide MLP down proj (4096×11008),
and smaller variants. Verifies the factorization is well-formed for each shape.

#### `tests/test_rectangular_regression.py`
Same regression task as `test_projected_optimizer_regression.py` but with a
rectangular W ∈ R^{64×128} (d' ≠ d), exercising the tall/wide code paths that
transformer MLP matrices use.

#### `tests/test_admm_rectangular.py`
Tests `manifold_muon_admm` (the ADMM inner loop for Manifold Muon) on tall
(m > n) and wide (m < n) matrices. Checks that the returned direction is
orthonormal and that the ADMM constraint is satisfied.
