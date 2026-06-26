#!/usr/bin/env python3
"""
Evaluate all task checkpoints from a continual learning run on the full eval sets.

Usage:
    python scripts/eval_continual_checkpoints.py \
        --exp_dir outputs/qwen3/qwen3_cl_dft_proj_e06 \
        --exp_id qwen3_cl_dft_proj_e06
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loaders import load_task_dataset
from evaluate_extended import evaluate_task

EVAL_TASKS = ["spider", "tooluse", "cot_math"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Base experiment dir containing task_0_*, task_1_*, task_2_* subdirs")
    parser.add_argument("--exp_id", type=str, default=None,
                        help="W&B run name prefix (defaults to exp_dir basename)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B",
                        help="Base model to evaluate first as reference point")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Max eval samples per task (default 10000 = full set)")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Batch size for generation during evaluation")
    return parser.parse_args()


def load_eval_datasets():
    datasets = {}
    for task in EVAL_TASKS:
        _, eval_ds = load_task_dataset(task)
        datasets[task] = eval_ds
        print(f"  {task}: {len(eval_ds)} eval examples")
    return datasets


def evaluate_checkpoint(ckpt_path, eval_datasets, max_samples, eval_batch_size=8):
    print(f"\nLoading model from: {ckpt_path}")
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    results = {}

    for task in EVAL_TASKS:
        print(f"  Evaluating {task} ({len(eval_datasets[task])} samples)...")
        result = evaluate_task(
            task, model, tokenizer, eval_datasets[task], device,
            max_samples=max_samples, verbose=False, batch_size=eval_batch_size,
        )
        results[task] = result
        print(f"    {task}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_table(all_results):
    print("\n" + "=" * 60)
    print("FULL EVAL RESULTS")
    print("=" * 60)
    header = f"{'Checkpoint':<25}" + "".join(f"{t:>12}" for t in EVAL_TASKS)
    print(header)
    print("-" * 60)
    for ckpt_name, results in all_results.items():
        row = f"{ckpt_name:<25}"
        for task in EVAL_TASKS:
            acc = results[task]["accuracy"] if task in results else float("nan")
            row += f"{acc:>11.1%} "
        print(row)
    print("=" * 60)


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    exp_id = args.exp_id or exp_dir.name

    # Find checkpoints in sorted order (task_0_*, task_1_*, task_2_*)
    checkpoints = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("task_")])
    if not checkpoints:
        print(f"No task_* subdirectories found in {exp_dir}")
        sys.exit(1)

    print(f"Experiment: {exp_id}")
    print(f"Base model: {args.base_model}")
    print(f"Checkpoints found: {[c.name for c in checkpoints]}")
    print(f"Max samples per task: {args.max_samples}")

    print("\nLoading eval datasets...")
    eval_datasets = load_eval_datasets()

    # Optionally log to wandb
    try:
        import wandb
        wandb.init(project="qwen3-continual-learning", name=f"{exp_id}_full_eval")
        use_wandb = True
    except Exception:
        use_wandb = False
        print("W&B not available, skipping logging")

    all_results = {}

    # Evaluate base model first as reference
    print(f"\n{'='*50}")
    print(f"[0/{len(checkpoints)}] Base model: {args.base_model}")
    results = evaluate_checkpoint(args.base_model, eval_datasets, args.max_samples, args.eval_batch_size)
    all_results["base_model"] = results
    if use_wandb:
        wandb.log({f"base_model/{task}": results[task]["accuracy"] for task in EVAL_TASKS})

    # Evaluate each task checkpoint
    for i, ckpt in enumerate(checkpoints):
        print(f"\n{'='*50}")
        print(f"[{i+1}/{len(checkpoints)}] Checkpoint: {ckpt.name}")
        results = evaluate_checkpoint(str(ckpt), eval_datasets, args.max_samples, args.eval_batch_size)
        all_results[ckpt.name] = results

        if use_wandb:
            log_dict = {f"{ckpt.name}/{task}": results[task]["accuracy"] for task in EVAL_TASKS}
            wandb.log(log_dict)

    print_table(all_results)

    # Save JSON
    out_file = exp_dir / "full_eval_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_file}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
