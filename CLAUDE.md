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
