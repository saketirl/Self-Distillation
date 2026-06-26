#!/usr/bin/env python3
"""
Evaluate a single model checkpoint on all continual learning tasks.

Usage:
    python scripts/eval_single_checkpoint.py \
        --model_path outputs/qwen3/my_exp/task_0_tooluse \
        --tag after_tooluse \
        --exp_id my_exp
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loaders import load_task_dataset
from evaluate_extended import evaluate_task

EVAL_TASKS = ["spider", "tooluse", "cot_math"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint to evaluate")
    parser.add_argument("--tag", type=str, default=None,
                        help="Descriptive tag for this checkpoint, e.g. after_tooluse")
    parser.add_argument("--exp_id", type=str, default=None,
                        help="W&B run name prefix")
    parser.add_argument("--eval_tasks", type=str, default="spider,tooluse,cot_math",
                        help="Comma-separated list of tasks to evaluate on")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Max eval samples per task (default 10000 = full set)")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path
    tag = args.tag or Path(model_path).name
    exp_id = args.exp_id or tag
    eval_tasks = [t.strip() for t in args.eval_tasks.split(",")]

    print(f"Model:  {model_path}")
    print(f"Tag:    {tag}")
    print(f"Tasks:  {eval_tasks}")
    print(f"Max samples per task: {args.max_samples}")

    # Load eval datasets
    print("\nLoading eval datasets...")
    eval_datasets = {}
    for task in eval_tasks:
        _, eval_ds = load_task_dataset(task)
        eval_datasets[task] = eval_ds
        print(f"  {task}: {len(eval_ds)} examples")

    # Load model once
    print(f"\nLoading model from: {model_path}")
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {model_path}")
    weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("pytorch_model*.bin"))
    if not weight_files:
        contents = sorted(p.name for p in model_dir.iterdir())
        raise FileNotFoundError(
            f"No model weights found in {model_path}\n"
            f"Directory contains: {contents}\n"
            "Training job may have been killed before save completed."
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device

    # Evaluate
    results = {}
    print()
    for task in eval_tasks:
        print(f"Evaluating {task}...")
        result = evaluate_task(
            task, model, tokenizer, eval_datasets[task], device,
            max_samples=args.max_samples, verbose=False,
        )
        results[task] = result
        print(f"  {task}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 50)
    print(f"RESULTS — {tag}")
    print("=" * 50)
    for task in eval_tasks:
        r = results[task]
        print(f"  {task:<12} {r['accuracy']:.2%}  ({r['correct']}/{r['total']})")
    print("=" * 50)

    # Log to wandb
    try:
        import wandb
        wandb.init(project="qwen3-continual-learning", name=f"{exp_id}_eval_{tag}")
        wandb.log({f"eval/{task}": results[task]["accuracy"] for task in eval_tasks})
        wandb.run.summary.update({f"eval/{task}": results[task]["accuracy"] for task in eval_tasks})
        wandb.finish()
    except Exception as e:
        print(f"W&B logging skipped: {e}")

    # Save JSON next to the checkpoint
    out_file = Path(model_path) / f"eval_results_{tag}.json"
    with open(out_file, "w") as f:
        json.dump({"tag": tag, "model_path": model_path, "results": results}, f, indent=2, default=str)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
