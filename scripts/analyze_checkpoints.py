#!/usr/bin/env python3
"""
Comprehensive checkpoint analysis script for continual learning experiments.

Evaluates all trained checkpoints (SFT and SDFT) plus the base model on all three
tasks (tooluse, gsm8k, mbpp) using the full test + validation datasets.

Models evaluated:
- Base Qwen model (no training)
- SDFT after task 0 (tooluse)
- SDFT after task 1 (gsm8k)
- SDFT after task 2 (mbpp)
- SFT after task 0 (tooluse)
- SFT after task 1 (gsm8k)
- SFT after task 2 (mbpp)

Total: 7 models x 3 tasks = 21 evaluations

Usage:
    python scripts/analyze_checkpoints.py --output_base /data/saket/continual/Self-Distillation/outputs
    python scripts/analyze_checkpoints.py --exp_id full_mbpp --output_base ./outputs
"""

import argparse
import csv
import gc
import json
import os
import re
import signal
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loaders import TASK_ORDER


class ExecutionTimeout(Exception):
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager for timing out code execution."""
    def handler(signum, frame):
        raise ExecutionTimeout(f"Execution timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    # Try to find code in markdown blocks
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()

    # Try to find a function definition
    func_match = re.search(r"(def\s+\w+.*?)(?=\n\S|\Z)", response, re.DOTALL)
    if func_match:
        return func_match.group(1).strip()

    # Return the whole response as a fallback
    return response.strip()


def execute_code_with_tests(code: str, test_cases: List[str], timeout_sec: int = 5) -> bool:
    """Execute generated code against test cases. Returns True if all pass."""
    namespace = {}

    # Execute the generated code
    try:
        with timeout(timeout_sec):
            exec(code, namespace)
    except:
        return False

    # Run test cases
    for test in test_cases:
        try:
            with timeout(timeout_sec):
                exec(test, namespace)
        except:
            return False

    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze checkpoints across all tasks")

    parser.add_argument("--output_base", type=str, required=True,
                        help="Base output directory (containing sdft_<exp_id> and sft_<exp_id>)")
    parser.add_argument("--exp_id", type=str, default="full_mbpp",
                        help="Experiment ID (default: full_mbpp)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model name")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory to save results (default: output_base/analysis)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens for generation")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed results")

    return parser.parse_args()


def load_full_eval_dataset_tooluse(data_dir: Path, seed: int = 42) -> Dataset:
    """Load full tooluse eval dataset."""
    from string import Template

    eval_path = data_dir / "tooluse_data" / "eval_data_simple.json"
    with open(eval_path) as f:
        eval_data = json.load(f)

    def format_example(example):
        teacher_prompt = Template("""$prompt

This is an example for a response to the question:
$response

Now answer with a response of your own.""")

        return {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=example["prompt"],
                response=example["response"]
            )}],
            "response": example["response"],
            "task": "tooluse"
        }

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])
    return eval_dataset


def load_full_eval_dataset_gsm8k(data_dir: Path, seed: int = 42) -> Dataset:
    """Load full gsm8k eval dataset."""
    from string import Template

    eval_path = data_dir / "gsm8k" / "eval_data.json"
    with open(eval_path) as f:
        eval_data = json.load(f)

    def format_example(example):
        question = example["question"]
        answer = example["answer"]
        solution = example.get("solution", answer)

        prompt_text = f"""Solve this math problem. Give only the final numerical answer.

Question: {question}

Answer:"""

        teacher_prompt = Template("""$prompt

This is an example solution:
$solution

Now solve the problem and give only the final numerical answer.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                solution=solution
            )}],
            "response": answer,
            "task": "gsm8k"
        }

    eval_dataset = Dataset.from_list([format_example(ex) for ex in eval_data])
    return eval_dataset


def load_full_eval_dataset_mbpp(seed: int = 42) -> Dataset:
    """Load full mbpp test + validation dataset from HuggingFace."""
    from string import Template

    hf_dataset = load_dataset("mbpp")

    # Combine test and validation splits
    test_data = list(hf_dataset["test"])
    val_data = list(hf_dataset["validation"])
    all_eval_data = test_data + val_data

    def format_example(example):
        text = example["text"]
        code = example["code"]
        test_list = example["test_list"]

        test_examples = "\n".join(test_list[:2])

        prompt_text = f"""Write a Python function to solve the following problem.

Problem: {text}

Example test cases:
{test_examples}

Write only the Python function, no explanations."""

        teacher_prompt = Template("""$prompt

Here is an example solution:
```python
$code
```

Now write your own Python function to solve the problem.""")

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(
                prompt=prompt_text,
                code=code
            )}],
            "response": code,
            "test_list": test_list,
            "task_id": example["task_id"],
            "task": "mbpp"
        }

    eval_dataset = Dataset.from_list([format_example(ex) for ex in all_eval_data])
    return eval_dataset


def load_all_full_eval_datasets(data_dir: Path, seed: int = 42) -> Dict[str, Dataset]:
    """Load all evaluation datasets with full test + validation sets."""
    return {
        "tooluse": load_full_eval_dataset_tooluse(data_dir, seed),
        "gsm8k": load_full_eval_dataset_gsm8k(data_dir, seed),
        "mbpp": load_full_eval_dataset_mbpp(seed),
    }


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer from path."""
    print(f"  Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    return model, tokenizer


def evaluate_task(
    model,
    tokenizer,
    eval_dataset: Dataset,
    task_name: str,
    max_new_tokens: int = 512,
    verbose: bool = False
) -> Dict:
    """Evaluate model on a single task."""

    device = next(model.parameters()).device

    results = {
        "task": task_name,
        "total": len(eval_dataset),
        "correct": 0,
        "errors": []
    }

    for idx, example in enumerate(tqdm(eval_dataset, desc=f"    Evaluating {task_name}", leave=False)):
        prompt = example["prompt"]
        expected = example["response"]

        # Format as chat
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Task-specific evaluation
        is_correct = False

        if task_name == "tooluse":
            exp_lower = expected.lower()
            gen_lower = generated.lower()

            if "tool" in exp_lower:
                tool_match = re.search(r'use the (\w+) tool', exp_lower)
                if tool_match:
                    tool_name = tool_match.group(1)
                    is_correct = tool_name in gen_lower
                else:
                    is_correct = any(word in gen_lower for word in exp_lower.split() if len(word) > 4)
            else:
                is_correct = expected.lower().strip() in gen_lower

        elif task_name == "gsm8k":
            expected_num = expected.strip()
            numbers = re.findall(r'[-+]?\d*\.?\d+', generated)
            if numbers:
                is_correct = expected_num in numbers or (numbers and numbers[-1] == expected_num)

        elif task_name == "mbpp":
            code = extract_code(generated)
            test_list = example.get("test_list", [])
            if test_list:
                is_correct = execute_code_with_tests(code, test_list, timeout_sec=5)
            else:
                is_correct = code.strip() == expected.strip()

        if is_correct:
            results["correct"] += 1
        elif verbose and len(results["errors"]) < 3:
            results["errors"].append({
                "idx": idx,
                "expected": expected[:100],
                "generated": generated[:100]
            })

    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0

    return results


def get_checkpoint_configs(args) -> List[Dict]:
    """Build list of checkpoint configurations to evaluate."""
    base_sdft = Path(args.output_base) / f"sdft_{args.exp_id}"
    base_sft = Path(args.output_base) / f"sft_{args.exp_id}"

    checkpoints = [
        {
            "name": "Base Model",
            "short_name": "base",
            "method": "none",
            "task_trained": "none",
            "task_idx": -1,
            "path": args.base_model,
        },
        # SDFT checkpoints
        {
            "name": "SDFT after tooluse",
            "short_name": "sdft_t0",
            "method": "sdft",
            "task_trained": "tooluse",
            "task_idx": 0,
            "path": str(base_sdft / "task_0_tooluse"),
        },
        {
            "name": "SDFT after gsm8k",
            "short_name": "sdft_t1",
            "method": "sdft",
            "task_trained": "gsm8k",
            "task_idx": 1,
            "path": str(base_sdft / "task_1_gsm8k"),
        },
        {
            "name": "SDFT after mbpp",
            "short_name": "sdft_t2",
            "method": "sdft",
            "task_trained": "mbpp",
            "task_idx": 2,
            "path": str(base_sdft / "task_2_mbpp"),
        },
        # SFT checkpoints
        {
            "name": "SFT after tooluse",
            "short_name": "sft_t0",
            "method": "sft",
            "task_trained": "tooluse",
            "task_idx": 0,
            "path": str(base_sft / "task_0_tooluse"),
        },
        {
            "name": "SFT after gsm8k",
            "short_name": "sft_t1",
            "method": "sft",
            "task_trained": "gsm8k",
            "task_idx": 1,
            "path": str(base_sft / "task_1_gsm8k"),
        },
        {
            "name": "SFT after mbpp",
            "short_name": "sft_t2",
            "method": "sft",
            "task_trained": "mbpp",
            "task_idx": 2,
            "path": str(base_sft / "task_2_mbpp"),
        },
    ]

    return checkpoints


def print_results_table(all_results: List[Dict], task_names: List[str]):
    """Print results as a formatted table."""
    print("\n" + "=" * 90)
    print("RESULTS TABLE")
    print("=" * 90)

    # Header
    header = f"{'Model':<25} | {'Method':<6} | {'Trained On':<10}"
    for task in task_names:
        header += f" | {task:<12}"
    header += f" | {'Avg':<8}"
    print(header)
    print("-" * 90)

    # Data rows
    for result in all_results:
        row = f"{result['short_name']:<25} | {result['method']:<6} | {result['task_trained']:<10}"
        accs = []
        for task in task_names:
            acc = result['task_results'].get(task, {}).get('accuracy', 0)
            accs.append(acc)
            row += f" | {acc*100:>10.2f}%"
        avg_acc = sum(accs) / len(accs) if accs else 0
        row += f" | {avg_acc*100:>6.2f}%"
        print(row)

    print("=" * 90)


def save_results_csv(all_results: List[Dict], task_names: List[str], output_path: Path):
    """Save results to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['model_name', 'short_name', 'method', 'task_trained', 'task_idx']
        for task in task_names:
            header.extend([f'{task}_accuracy', f'{task}_correct', f'{task}_total'])
        header.append('avg_accuracy')
        writer.writerow(header)

        # Data
        for result in all_results:
            row = [
                result['name'],
                result['short_name'],
                result['method'],
                result['task_trained'],
                result['task_idx'],
            ]
            accs = []
            for task in task_names:
                task_res = result['task_results'].get(task, {})
                acc = task_res.get('accuracy', 0)
                correct = task_res.get('correct', 0)
                total = task_res.get('total', 0)
                row.extend([acc, correct, total])
                accs.append(acc)

            avg_acc = sum(accs) / len(accs) if accs else 0
            row.append(avg_acc)
            writer.writerow(row)

    print(f"\nResults saved to: {output_path}")


def save_results_json(all_results: List[Dict], output_path: Path):
    """Save detailed results to JSON file."""
    # Remove non-serializable items
    clean_results = []
    for result in all_results:
        clean = {
            'name': result['name'],
            'short_name': result['short_name'],
            'method': result['method'],
            'task_trained': result['task_trained'],
            'task_idx': result['task_idx'],
            'path': result['path'],
            'task_results': {
                task: {
                    'accuracy': tr['accuracy'],
                    'correct': tr['correct'],
                    'total': tr['total'],
                }
                for task, tr in result['task_results'].items()
            }
        }
        clean_results.append(clean)

    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': clean_results,
        }, f, indent=2)

    print(f"Detailed results saved to: {output_path}")


def main():
    args = parse_args()

    print("=" * 70)
    print("CHECKPOINT ANALYSIS - Continual Learning Experiments")
    print("=" * 70)
    print(f"Experiment ID: {args.exp_id}")
    print(f"Output base: {args.output_base}")
    print(f"Base model: {args.base_model}")
    print()

    # Setup paths
    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(args.results_dir) if args.results_dir else Path(args.output_base) / "analysis"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load all evaluation datasets
    print("Loading evaluation datasets (full test + validation)...")
    eval_datasets = load_all_full_eval_datasets(data_dir, seed=args.seed)

    for task_name, ds in eval_datasets.items():
        print(f"  {task_name}: {len(ds)} samples")
    print()

    # Get checkpoint configurations
    checkpoints = get_checkpoint_configs(args)

    # Check which checkpoints exist
    print("Checking checkpoint availability...")
    valid_checkpoints = []
    for ckpt in checkpoints:
        path = ckpt['path']
        if path.startswith("Qwen/") or Path(path).exists():
            print(f"  [OK] {ckpt['name']}: {path}")
            valid_checkpoints.append(ckpt)
        else:
            print(f"  [MISSING] {ckpt['name']}: {path}")
    print()

    if not valid_checkpoints:
        print("ERROR: No valid checkpoints found. Please check your output_base path.")
        sys.exit(1)

    # Evaluate each checkpoint
    all_results = []
    task_names = list(eval_datasets.keys())

    for ckpt_idx, ckpt in enumerate(valid_checkpoints):
        print(f"\n[{ckpt_idx + 1}/{len(valid_checkpoints)}] Evaluating: {ckpt['name']}")
        print("-" * 50)

        # Load model
        try:
            model, tokenizer = load_model_and_tokenizer(ckpt['path'])
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue

        # Evaluate on all tasks
        task_results = {}
        for task_name, eval_ds in eval_datasets.items():
            result = evaluate_task(
                model, tokenizer, eval_ds, task_name,
                max_new_tokens=args.max_new_tokens,
                verbose=args.verbose
            )
            task_results[task_name] = result
            print(f"    {task_name}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

        # Store results
        ckpt['task_results'] = task_results
        all_results.append(ckpt)

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Print summary table
    print_results_table(all_results, task_names)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"checkpoint_analysis_{args.exp_id}_{timestamp}.csv"
    json_path = results_dir / f"checkpoint_analysis_{args.exp_id}_{timestamp}.json"

    save_results_csv(all_results, task_names, csv_path)
    save_results_json(all_results, json_path)

    # Also save a latest version without timestamp
    csv_latest = results_dir / f"checkpoint_analysis_{args.exp_id}_latest.csv"
    json_latest = results_dir / f"checkpoint_analysis_{args.exp_id}_latest.json"
    save_results_csv(all_results, task_names, csv_latest)
    save_results_json(all_results, json_latest)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Total evaluations: {len(all_results) * len(task_names)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
